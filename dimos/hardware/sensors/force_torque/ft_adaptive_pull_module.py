# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General door/drawer pull: a quick probe, then a calibrated smooth pull.

Uses admittance_pull_law.compute_twist (see that file for the control-law
design) in two passes through the same _run_phase loop, not two separate
control laws:

  1. Probe -- short, a few cm, at probe_speed. Purpose is only to observe this
     specific door's peak resistance/torque before committing to anything.
  2. Execute -- slow, human-like pace (execute_speed), running out to
     execute_target_m ("almost open"). Its force_cutoff/torque_cutoff are
     calibrated from what the probe just measured (peak * safety margin, with
     a floor), instead of one fixed guess that's either too tight for a heavy
     door or too loose for a light one.

Separate module from FTPullModule, not a rewrite -- the microwave path stays
available while this is validated. Same plumbing (FK setup, joint-limit
safety check, wrench/joint_state subscription, Enter-key start trigger).

V1 scope, deliberately: proportional admittance only, no oscillation damping,
no sensor-delay prediction filter -- add complexity only once real hardware
data shows it's needed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pinocchio
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.force_torque.admittance_pull_law import (
    SENSOR_FORCE_OVERLOAD_N,
    SENSOR_TORQUE_OVERLOAD_NM,
    AdmittanceConfig,
    arc_waypoints,
    compute_hybrid_twist,
    compute_twist,
    fit_hinge,
    singularity_speed_scale,
    slew_limit,
)
from dimos.hardware.sensors.force_torque.ft_conditioning import tool_gravity_wrench
from dimos.manipulation.planning.utils.mesh_utils import prepare_urdf_for_drake
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.manipulators.common.topics import EEF_TWIST_TASK_NAME
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class FTAdaptivePullConfig(ModuleConfig):
    hardware_id: str = "arm"
    num_arm_joints: int = 7
    # Same model the coordinator's own IK backend uses -- see FTPullConfig's
    # docstring in ft_pull_module.py, identical reasoning, copied here.
    model_path: Path
    package_paths: dict[str, Path] = {}
    xacro_args: dict[str, str] = {}
    tool_frame_name: str
    task_name: str = EEF_TWIST_TASK_NAME

    # Local (tool-frame) pull direction, recomputed against the CURRENT ee_rot
    # every tick -- not fixed at grasp time -- so it follows the gripper as
    # compliance rotates it, same as the old law's pull_local convention.
    local_drive_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Shared reaction parameters (k_trans, k_rot, saturation caps, speed_bands,
    # decel_floor_scale) -- same for both phases. drive_speed/force_cutoff/
    # torque_cutoff/decel_start_m/decel_full_m are overridden per-phase below,
    # so their values here are unused defaults.
    admittance: AdmittanceConfig = AdmittanceConfig()

    # Phase 1: probe. Short and a little brisker than the real pull -- just
    # enough motion to see what this door actually resists with.
    probe_speed: float = 0.04  # m/s
    # 8 cm, not 4: on a 0.30 m door 4 cm is only 7.6 degrees of arc, which constrains a circle
    # far too weakly -- two hardware runs fitted radii of 0.274 m and 0.109 m for the same door.
    probe_distance_m: float = 0.08  # m -- stop the probe here even if never resisted
    probe_min_ticks: int = 20  # enough force samples to average before trusting the direction
    probe_max_duration: float = 8.0  # s, safety net independent of distance

    # A fitted radius outside this range is not believable for an appliance door; the pull
    # proceeds without door-following rotation rather than turning on a bad number.
    min_hinge_radius_m: float = 0.15
    max_hinge_radius_m: float = 0.80

    # Refit the hinge as the pull proceeds. The probe sees ~15 degrees of arc, which pins a
    # circle weakly -- three hardware runs fitted 0.274, 0.109 and 0.191 m for the same door.
    # By mid-pull there is 40+ degrees to fit, and since the commanded rotation is v/r, a
    # radius that is too small over-rotates the gripper and pushes the door shut.
    refit_hinge_every_ticks: int = 25

    # Hinge-to-handle distance, measured on the door with a ruler. The probe spans only ~15
    # degrees, where the fitted radius scatters over [0.03, 0.28] on a true 0.30m door, and
    # omega = v/r turns a low radius straight into over-rotation that shuts the door. When set,
    # the fitted direction is kept and only its length is replaced.
    door_radius_m: float | None = None
    # Ease following in over this much travel. Distance, not swept angle: gating on arc was a
    # deadlock, since the arc only appears once following is already on.
    follow_ramp_m: float = 0.02
    # Accept the fitted hinge direction only this close to perpendicular (cos of the angle off).
    max_hinge_direction_cos: float = 0.5
    # Unit vector from the grasp toward the hinge, in base frame, and the hinge axis. When both
    # this and door_radius_m are set the arc is fully known and the probe fit is not used at
    # all -- an 8cm probe cannot resolve which side the hinge is on, but you can just look.
    hinge_direction_world: tuple[float, float, float] | None = None
    # Constraint force needed before the hinge direction is trusted, well over the sensor floor.
    min_hinge_force_n: float = 8.0
    hinge_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Ceiling on the adaptive cutoff, so a probe that already fought hard cannot authorise a
    # force that damages the door.
    max_force_cutoff_n: float = 60.0

    # Phase 2: execute. Deliberately slow -- "human-like," not scaled by how
    # heavy the door is; the calibrated cutoffs below are what adapts to the
    # door, not the pace.
    execute_speed: float = 0.02  # m/s
    # This is LINEAR drive-direction travel, not an opening angle -- arc length
    # ~= handle_radius * angle, so the same distance target covers proportionally
    # less angle on a wide-swing door (found in sim_multi_door_test.py: a
    # large-radius door stalled around half-open at the old 0.45m default while
    # a smaller one finished fine). 0.8m is a generous ceiling for now;
    # max_duration is still the real backstop regardless. A proper fix would
    # size this off a measured radius (constraint_estimator.py could supply
    # one) rather than one fixed guess -- not done yet.
    execute_target_m: float = 0.8  # m
    max_duration: float = 30.0  # s, safety net for the execute phase

    # Velocity on the tangent, force regulated on the radial. See compute_hybrid_twist.
    use_hybrid_law: bool = True

    # Checked between probe and execute: the probe's motion is what reveals the hinge.
    check_arc_before_execute: bool = True
    target_open_angle_deg: float = 90.0
    arc_check_steps: int = 19
    arc_min_sigma: float = 0.05          # manipulability floor along the arc
    arc_min_margin_rad: float = 0.10     # joint-limit floor along the arc
    arc_reach_tolerance_m: float = 0.005
    # False by default: a partial open is usually still wanted, and the jam angle is logged.
    refuse_if_arc_blocked: bool = False

    # Execute-phase cutoffs = max(probe_peak * cutoff_safety_margin, min_*) --
    # scales the safety envelope to what THIS door demonstrably needs instead
    # of one fixed guess. All four values here are starting points, not
    # validated against real hardware yet.
    cutoff_safety_margin: float = 2.5
    min_force_cutoff: float = 30.0  # N
    min_torque_cutoff: float = 4.0  # N*m

    # Tool weight hanging off the sensor. The DIY sensor has no gravity compensation, so a
    # mounted gripper reads as a standing force that TILTS INTO Fx/Fy as the wrist rotates
    # through the door arc -- indistinguishable from real resistance unless removed per tick.
    # 0.0 disables it (the uFactory already compensates its own stream).
    tool_mass_kg: float = 0.0
    tool_com_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # See AdmittanceConfig.force_axis_weights. Vertical-hinge doors only.
    force_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    control_rate_hz: float = 25.0
    auto_run: bool = False


@dataclass
class PhaseStats:
    peak_resistance_force: float = 0.0
    peak_torque: float = 0.0
    distance_covered: float = 0.0
    rotation_covered: float = 0.0
    ticks: int = 0
    safety_tripped: bool = False
    stop_reason: str = ""
    min_sigma: float = float("inf")  # smallest manipulability seen this phase
    tool_path: list = dataclass_field(default_factory=list)  # EE positions, for fit_hinge
    force_world_sum: list = dataclass_field(default_factory=lambda: [0.0, 0.0, 0.0])


class FTAdaptivePullModule(Module):
    """Reads ext_wrench + coordinator_joint_state, publishes coordinator_ee_twist_command."""

    config: FTAdaptivePullConfig

    ext_wrench: In[WrenchStamped]
    coordinator_joint_state: In[JointState]
    coordinator_ee_twist_command: Out[TwistStamped]
    start_pull_command: In[Bool]
    stop_pull_command: In[Bool]

    _lock: threading.Lock
    _latest_wrench: np.ndarray | None = None  # [Fx,Fy,Fz,Mx,My,Mz], tool frame
    _latest_q: np.ndarray | None = None
    _following_logged: bool = False
    _hinge_centre: Any = None
    _hinge_axis: Any = None
    _pin_model: Any = None
    _pin_data: Any = None
    _frame_id: int = -1
    _running: bool = False
    _stop_requested: bool = False
    _phase: str = ""

    total_pull_distance: float = 0.0
    total_rotation: float = 0.0  # integral of |omega|*dt, a magnitude not a signed angle -- law has no fixed axis
    motion_count: int = 0
    peak_resistance_force: float = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        self._lock = threading.Lock()

        model_path = Path(self.config.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"FTAdaptivePullModule: robot model not found: {model_path}")
        if model_path.suffix == ".xml":
            self._pin_model = pinocchio.buildModelFromMJCF(str(model_path))
        else:
            prepared_path = prepare_urdf_for_drake(
                urdf_path=model_path,
                package_paths=self.config.package_paths,
                xacro_args=self.config.xacro_args,
            )
            self._pin_model = pinocchio.buildModelFromUrdf(str(prepared_path))
        self._pin_data = self._pin_model.createData()
        if not self._pin_model.existFrame(self.config.tool_frame_name):
            raise ValueError(
                f"FTAdaptivePullModule: no frame '{self.config.tool_frame_name}' in {model_path}"
            )
        self._frame_id = int(self._pin_model.getFrameId(self.config.tool_frame_name))
        self._q_lower = np.array(self._pin_model.lowerPositionLimit)
        self._q_upper = np.array(self._pin_model.upperPositionLimit)
        self._local_drive = np.array(self.config.local_drive_direction)
        self._local_drive /= np.linalg.norm(self._local_drive)

        self.ext_wrench.subscribe(self._on_wrench)
        self.coordinator_joint_state.subscribe(self._on_joint_state)
        self.start_pull_command.subscribe(self._on_start_pull_command)
        self.stop_pull_command.subscribe(self._on_stop_pull_command)
        logger.info("FTAdaptivePullModule ready (tool_frame=%s)", self.config.tool_frame_name)
        if self.config.auto_run:
            self.spawn(self._pull_loop())

    def _forward_kinematics(self, q: np.ndarray) -> pinocchio.SE3:
        pinocchio.forwardKinematics(self._pin_model, self._pin_data, q)
        pinocchio.updateFramePlacements(self._pin_model, self._pin_data)
        return self._pin_data.oMf[self._frame_id]

    def _near_joint_limit(self, q: np.ndarray, margin_rad: float = 0.1) -> int | None:
        too_low = q <= self._q_lower + margin_rad
        too_high = q >= self._q_upper - margin_rad
        hits = np.where(too_low | too_high)[0]
        return int(hits[0]) if len(hits) else None

    def _ik_position(self, target: np.ndarray, seed: np.ndarray, iters: int = 120) -> np.ndarray:
        """DLS IK, position only -- asking whether the arm can be near there at all, not
        reproducing Pink. Demanding orientation would reject arcs the real solver can follow."""
        q = seed.copy()
        for _ in range(iters):
            pose = self._forward_kinematics(q)
            error = target - np.asarray(pose.translation)
            if np.linalg.norm(error) < 1e-4:
                break
            pinocchio.computeJointJacobians(self._pin_model, self._pin_data, q)
            jac = pinocchio.getFrameJacobian(
                self._pin_model, self._pin_data, self._frame_id, pinocchio.LOCAL_WORLD_ALIGNED
            )[:3]
            u, s, vt = np.linalg.svd(jac, full_matrices=False)
            q = np.clip(q + vt.T @ ((s / (s**2 + 0.01**2)) * (u.T @ error)),
                        self._q_lower + 0.02, self._q_upper - 0.02)
        return q

    def _check_arc(self, q_now: np.ndarray, grasp: np.ndarray, hinge: np.ndarray,
                   axis: np.ndarray, max_angle_rad: float) -> dict:
        """Walk the whole arc before committing. Whether the arm can follow it is decided by
        where it grabbed, so no controller rescues a grasp that runs out of travel at 50 deg."""
        points = arc_waypoints(grasp, hinge, axis, max_angle_rad, self.config.arc_check_steps)
        angles = np.degrees(np.linspace(0.0, max_angle_rad, self.config.arc_check_steps))
        q = q_now.copy()
        worst_sigma, worst_margin, blocked_at = np.inf, np.inf, None
        for angle, point in zip(angles, points, strict=True):
            q = self._ik_position(point, q)
            reach_error = float(np.linalg.norm(point - np.asarray(self._forward_kinematics(q).translation)))
            sigma = self._manipulability(q)
            margin = float(min(np.min(q - self._q_lower), np.min(self._q_upper - q)))
            worst_sigma, worst_margin = min(worst_sigma, sigma), min(worst_margin, margin)
            if blocked_at is None and (
                reach_error > self.config.arc_reach_tolerance_m
                or sigma < self.config.arc_min_sigma
                or margin < self.config.arc_min_margin_rad
            ):
                blocked_at = float(angle)
        return {"worst_sigma": worst_sigma, "worst_margin": worst_margin, "blocked_at": blocked_at}

    def _manipulability(self, q: np.ndarray) -> float:
        """Smallest singular value of the tool-frame Jacobian -- backstop only, see AdmittanceConfig."""
        pinocchio.computeJointJacobians(self._pin_model, self._pin_data, q)
        jac = pinocchio.getFrameJacobian(self._pin_model, self._pin_data, self._frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
        return float(np.linalg.svd(jac, compute_uv=False)[-1])

    def _on_start_pull_command(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self._running:
            logger.info("Pull already running, ignoring start_pull_command")
            return
        logger.info("start_pull_command received -- starting adaptive pull")
        self.spawn(self._pull_loop())

    def _on_stop_pull_command(self, msg: Bool) -> None:
        if not msg.data:
            return
        logger.info("stop_pull_command received -- stopping pull")
        self._stop_requested = True

    def _on_wrench(self, msg: WrenchStamped) -> None:
        wrench = np.array([msg.force.x, msg.force.y, msg.force.z, msg.torque.x, msg.torque.y, msg.torque.z])
        with self._lock:
            self._latest_wrench = wrench

        # Fast-path hardware backstop: checked at sensor rate (up to 1000Hz), not the 25Hz
        # control loop -- a real overload transient can develop faster than one control tick,
        # so waiting for _run_phase's own cutoff check can be too late (this is what a real
        # error-53 fault on real hardware showed). Magnitude is rotation-invariant, no FK needed.
        # 0.7x, not 0.8x: this path exists to catch what the slower reactive cutoff (0.75x
        # on torque) might miss between ticks, so it must trip at least as early, not later.
        force_mag = float(np.linalg.norm(wrench[:3]))
        torque_mag = float(np.linalg.norm(wrench[3:]))
        if force_mag > SENSOR_FORCE_OVERLOAD_N * 0.7 or torque_mag > SENSOR_TORQUE_OVERLOAD_NM * 0.7:
            if not self._stop_requested:
                logger.warning(
                    "Fast-path overload trip: force=%.1fN torque=%.2fNm -- stopping immediately.",
                    force_mag, torque_mag,
                )
            self._stop_requested = True
            self.coordinator_ee_twist_command.publish(
                TwistStamped(frame_id=self.config.task_name, linear=[0, 0, 0], angular=[0, 0, 0])
            )

    def _on_joint_state(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position, strict=True))
        prefix = f"{self.config.hardware_id}/joint"
        try:
            q = np.array([by_name[f"{prefix}{i}"] for i in range(1, self.config.num_arm_joints + 1)])
        except KeyError:
            return
        with self._lock:
            self._latest_q = q

    def _get_state(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if self._latest_wrench is None or self._latest_q is None:
                return None
            return self._latest_wrench.copy(), self._latest_q.copy()

    @rpc
    def pull_door(self) -> str:
        if self._running:
            return "Already running"
        self.spawn(self._pull_loop())
        return "Pull started"

    async def _run_phase(self, name: str, cfg: AdmittanceConfig, max_progress_m: float, max_duration: float) -> PhaseStats:
        """Runs compute_twist in a loop until max_progress_m, max_duration, a safety
        cutoff, or an external stop -- shared by both the probe and execute phases,
        so there is exactly one control loop, just called twice with different config."""
        self._phase = name
        stats = PhaseStats()
        dt = 1.0 / self.config.control_rate_hz
        start_time = time.time()
        prev_linear, prev_angular = np.zeros(3), np.zeros(3)
        previous_position: np.ndarray | None = None

        while self._running and not self._stop_requested:
            if time.time() - start_time > max_duration:
                stats.stop_reason = "max_duration"
                break
            if stats.distance_covered >= max_progress_m:
                stats.stop_reason = "reached target"
                break
            # The probe exists only to read which way the door resists. Once that force is
            # clear, every further centimetre of straight pull is fighting the hinge for
            # nothing -- force has been running to 75N by 6cm.
            if (name == "probe" and stats.ticks >= self.config.probe_min_ticks
                    and self._hinge_direction_from_force(stats) is not None):
                stats.stop_reason = "constraint force resolved"
                logger.info("[%s] Constraint direction resolved after %.1fcm -- probe done.",
                            name, stats.distance_covered * 100)
                break
            if (name == "execute" and self._hinge_centre is not None
                    and self._swept_angle_deg(stats.tool_path) >= self.config.target_open_angle_deg):
                stats.stop_reason = f"reached {self.config.target_open_angle_deg:.0f} deg"
                logger.info("[%s] Door is open to %.0f degrees -- stopping.",
                            name, self.config.target_open_angle_deg)
                break

            state = self._get_state()
            if state is None:
                await asyncio.sleep(dt)
                continue
            wrench, q = state
            force_tool, torque_tool = wrench[:3], wrench[3:]

            near_limit = self._near_joint_limit(q)
            if near_limit is not None:
                logger.warning(
                    "[%s] joint%d at %.3f rad is %.3f rad from its %s limit (%.3f..%.3f) "
                    "-- stopping.",
                    name, near_limit + 1, q[near_limit],
                    min(q[near_limit] - self._q_lower[near_limit],
                        self._q_upper[near_limit] - q[near_limit]),
                    "lower" if (q[near_limit] - self._q_lower[near_limit]
                                < self._q_upper[near_limit] - q[near_limit]) else "upper",
                    self._q_lower[near_limit], self._q_upper[near_limit],
                )
                stats.stop_reason = "joint limit"
                break

            pose = self._forward_kinematics(q)
            ee_rot = np.asarray(pose.rotation)
            if self.config.tool_mass_kg > 0.0:
                g = tool_gravity_wrench(ee_rot, self.config.tool_mass_kg,
                                        np.array(self.config.tool_com_m))
                force_tool, torque_tool = force_tool - g[:3], torque_tool - g[3:]
            if stats.ticks < 3:
                logger.info("[%s] EE pose tick %d: translation=%s -- sanity check this", name, stats.ticks, pose.translation)

            sigma_min = self._manipulability(q)
            sing_scale = singularity_speed_scale(sigma_min, cfg.singularity_sigma_caution, cfg.singularity_sigma_stop)
            if sing_scale <= 0.0:
                logger.warning("[%s] Near a kinematic singularity (sigma_min=%.4f) -- stopping.", name, sigma_min)
                stats.stop_reason = "near singularity"
                break

            drive_direction_world = ee_rot @ self._local_drive
            # This path is what fit_hinge later reads the door's circle from.
            position = np.asarray(pose.translation).copy()
            stats.tool_path.append(position)
            f_w = ee_rot @ np.asarray(force_tool, float)
            stats.force_world_sum = [a + b for a, b in zip(stats.force_world_sum, f_w)]
            velocity = ((position - previous_position) / dt
                        if previous_position is not None else np.zeros(3))
            previous_position = position

            swept = follow = 0.0
            if self.config.use_hybrid_law:
                # Known only after the probe, so the probe itself pulls straight and the
                # execute phase follows the arc.
                if (self._hinge_centre is not None and self.config.refit_hinge_every_ticks
                        and stats.ticks and stats.ticks % self.config.refit_hinge_every_ticks == 0):
                    self._refit_hinge(stats.tool_path)
                swept = self._swept_angle_deg(stats.tool_path)
                # The arc check already cleared the full sweep, so follow from the start.
                follow = np.clip(stats.distance_covered
                                 / max(self.config.follow_ramp_m, 1e-6), 0.0, 1.0)
                to_grasp = (position - np.asarray(self._hinge_centre)
                            if self._hinge_centre is not None and follow > 0.0 else None)
                if to_grasp is not None and not self._following_logged:
                    self._following_logged = True
                    logger.info(
                        "[%s] DOOR-FOLLOWING ENGAGED at %.0f deg swept, radius %.3fm -- the "
                        "gripper will now turn with the door.",
                        name, swept, float(np.linalg.norm(to_grasp)),
                    )
                result = compute_hybrid_twist(
                    force_tool, torque_tool, ee_rot, drive_direction_world, cfg,
                    measured_velocity_world=velocity, hinge_to_grasp_world=to_grasp,
                    follow_scale=float(follow),
                    progress_m=stats.distance_covered, singularity_scale=sing_scale,
                )
            else:
                result = compute_twist(
                    force_tool, torque_tool, ee_rot, drive_direction_world, cfg,
                    progress_m=stats.distance_covered, singularity_scale=sing_scale,
                )

            if result.safety_stop:
                logger.warning(
                    "[%s] Safety cutoff: resistance=%.1fN torque=%.1fNm -- stopping.",
                    name, result.resistance_force, result.torque_mag,
                )
                stats.safety_tripped = True
                stats.stop_reason = "safety cutoff"
                break

            # Jerk limit: cap how much the PUBLISHED twist can change from last tick,
            # regardless of what compute_twist just jumped to. Stats/progress track
            # the actual limited command, not the raw target.
            linear_cmd = slew_limit(prev_linear, result.linear, cfg.max_linear_accel * dt)
            angular_cmd = slew_limit(prev_angular, result.angular, cfg.max_angular_accel * dt)
            prev_linear, prev_angular = linear_cmd, angular_cmd

            self.coordinator_ee_twist_command.publish(
                TwistStamped(frame_id=self.config.task_name, linear=list(linear_cmd), angular=list(angular_cmd))
            )
            stats.distance_covered += float(np.linalg.norm(linear_cmd) * dt)
            stats.rotation_covered += float(np.linalg.norm(angular_cmd) * dt)
            stats.peak_resistance_force = max(stats.peak_resistance_force, result.resistance_force)
            stats.peak_torque = max(stats.peak_torque, result.torque_mag)
            stats.ticks += 1
            self.total_pull_distance += float(np.linalg.norm(linear_cmd) * dt)
            self.total_rotation += float(np.linalg.norm(angular_cmd) * dt)
            self.peak_resistance_force = max(self.peak_resistance_force, result.resistance_force)
            self.motion_count += 1

            stats.min_sigma = min(stats.min_sigma, sigma_min)
            if stats.ticks % 25 == 0:
                logger.info(
                    "[%s tick %d] resistance=%.1fN torque=%.1fNm |v|=%.3fm/s |omega|=%.3frad/s "
                    "covered=%.1fcm opened=%.0fdeg follow=%.2f sigma_min=%.4f margin=%.3f",
                    name, stats.ticks, result.resistance_force, result.torque_mag,
                    float(np.linalg.norm(linear_cmd)), float(np.linalg.norm(angular_cmd)),
                    stats.distance_covered * 100, swept, float(follow), sigma_min,
                    float(np.min(np.minimum(q - self._q_lower, self._q_upper - q))),
                )

            await asyncio.sleep(dt)

        logger.info(
            "[%s] done (%s): %d ticks, %.1fcm, peak_force=%.1fN peak_torque=%.1fNm min_sigma=%.4f",
            name, stats.stop_reason or "external stop", stats.ticks, stats.distance_covered * 100,
            stats.peak_resistance_force, stats.peak_torque, stats.min_sigma,
        )
        return stats

    def _hinge_direction_from_force(self, probe) -> np.ndarray | None:
        """Unit vector from grasp toward the hinge, read off the constraint force.

        Dragging a hinged handle along a straight line pulls it off its arc, and the door
        resists toward the hinge. That force runs to tens of newtons where the path curvature
        a circle fit needs is a fraction of a millimetre, so this is the far stronger signal.
        """
        if len(probe.tool_path) < 5 or not probe.ticks:
            return None
        axis = np.asarray(self.config.hinge_axis_world, float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        travel = np.asarray(probe.tool_path[-1], float) - np.asarray(probe.tool_path[0], float)
        travel = travel - np.dot(travel, axis) * axis
        force = np.asarray(probe.force_world_sum, float) / probe.ticks
        force = force - np.dot(force, axis) * axis
        if np.linalg.norm(travel) < 1e-6 or np.linalg.norm(force) < self.config.min_hinge_force_n:
            return None
        t_hat = travel / np.linalg.norm(travel)
        radial = force - np.dot(force, t_hat) * t_hat
        if np.linalg.norm(radial) < self.config.min_hinge_force_n:
            return None
        return radial / np.linalg.norm(radial)

    def _swept_angle_deg(self, path: list) -> float:
        """Angle turned about the hinge so far, from the first recorded point."""
        if self._hinge_centre is None or len(path) < 2:
            return 0.0
        axis = np.asarray(self._hinge_axis, float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        centre = np.asarray(self._hinge_centre, float)
        first, last = np.asarray(path[0]) - centre, np.asarray(path[-1]) - centre
        first, last = first - np.dot(first, axis) * axis, last - np.dot(last, axis) * axis
        n1, n2 = np.linalg.norm(first), np.linalg.norm(last)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(first, last) / (n1 * n2), -1.0, 1.0))))

    def _refit_hinge(self, path: list) -> None:
        """Re-estimate from the whole path so far. More arc pins the circle far better."""
        if len(path) < 20:
            return
        fit = fit_hinge(np.array(path))
        if fit is None:
            return
        centre, axis, radius = fit
        if not (self.config.min_hinge_radius_m <= radius <= self.config.max_hinge_radius_m):
            return
        self._hinge_centre, self._hinge_axis = centre, axis

    def _assess_arc(self, probe: PhaseStats) -> float | None:
        """Fit the hinge from the probe's motion, then check the arc.

        Returns the angle it expects to jam at, or None if the full swing is clear.
        """
        measured = self._hinge_direction_from_force(probe)
        configured = self.config.hinge_direction_world
        if measured is not None and configured:
            agree = float(np.dot(measured, np.asarray(configured, float)
                                 / max(np.linalg.norm(configured), 1e-12)))
            logger.info("Hinge direction: force says %s, config says %s -- %s (cos %.2f).",
                        np.round(measured, 3).tolist(), list(configured),
                        "AGREE" if agree > 0 else "DISAGREE", agree)
        chosen = np.asarray(configured, float) if configured else measured
        if chosen is not None and self.config.door_radius_m:
            d = np.asarray(chosen, float)
            axis = np.asarray(self.config.hinge_axis_world, float)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            d = d - np.dot(d, axis) * axis
            if np.linalg.norm(d) < 1e-6:
                logger.error("hinge_direction_world is parallel to the hinge axis -- ignoring it.")
            else:
                state = self._get_state()
                if state is None:
                    return None
                _, q = state
                grasp = np.asarray(self._forward_kinematics(q).translation)
                hinge = grasp + d / np.linalg.norm(d) * self.config.door_radius_m
                self._hinge_centre, self._hinge_axis = hinge, axis
                result = self._check_arc(q, grasp, hinge, axis,
                                         np.radians(self.config.target_open_angle_deg))
                logger.info(
                    "Hinge from %s: radius=%.3fm direction=%s axis=%s. Arc to %.0f deg "
                    "-> worst sigma %.4f, worst joint margin %.3f rad.",
                    "config" if configured else "measured force",
                    self.config.door_radius_m, np.round(d / np.linalg.norm(d), 3).tolist(),
                    np.round(axis, 3).tolist(), self.config.target_open_angle_deg,
                    result["worst_sigma"], result["worst_margin"],
                )
                if result["blocked_at"] is not None:
                    logger.warning("Arc blocked at %.0f deg (%s).",
                                   np.degrees(result["blocked_at"]), result["blocked_by"])
                else:
                    logger.info("Arc is clear for the full %.0f degrees.",
                                self.config.target_open_angle_deg)
                return result
        if len(probe.tool_path) < 5:
            logger.info("Arc check skipped: probe recorded only %d points.", len(probe.tool_path))
            return None
        fit = fit_hinge(np.array(probe.tool_path))
        if fit is None:
            logger.info("Arc check skipped: probe motion is not an arc (drawer, or too little travel).")
            return None
        hinge, axis, radius = fit
        if self.config.door_radius_m:
            # The fitted radius is discarded, so gate on what is actually used: the direction
            # from grasp to hinge, which must be roughly perpendicular to the way we travelled.
            grasp0 = np.asarray(probe.tool_path[-1], float)
            travel = grasp0 - np.asarray(probe.tool_path[0], float)
            radial = grasp0 - hinge
            radial -= np.dot(radial, axis) * axis
            travel -= np.dot(travel, axis) * axis
            if np.linalg.norm(radial) < 1e-6 or np.linalg.norm(travel) < 1e-6:
                logger.warning("Probe gives no usable hinge direction -- no door-following.")
                return None
            r_hat = radial / np.linalg.norm(radial)
            off = abs(float(np.dot(r_hat, travel / np.linalg.norm(travel))))
            if off > self.config.max_hinge_direction_cos:
                logger.warning(
                    "Hinge direction is %.0f deg from perpendicular to travel -- too far off to "
                    "trust, no door-following.", np.degrees(np.arcsin(min(off, 1.0))),
                )
                return None
            hinge = grasp0 - r_hat * self.config.door_radius_m
            logger.info("Probe radius %.3fm replaced with the measured %.3fm (direction kept, "
                        "%.0f deg off perpendicular).", radius, self.config.door_radius_m,
                        np.degrees(np.arcsin(off)))
            radius = self.config.door_radius_m
        elif not (self.config.min_hinge_radius_m <= radius <= self.config.max_hinge_radius_m):
            logger.warning(
                "Hinge radius %.3fm is outside the believable range %.2f-%.2fm -- ignoring the "
                "fit. The pull will proceed without door-following rotation.",
                radius, self.config.min_hinge_radius_m, self.config.max_hinge_radius_m,
            )
            return None
        self._hinge_centre, self._hinge_axis = hinge, axis

        state = self._get_state()
        if state is None:
            return None
        _, q = state
        grasp = np.asarray(self._forward_kinematics(q).translation)
        result = self._check_arc(q, grasp, hinge, axis, np.radians(self.config.target_open_angle_deg))
        logger.info(
            "Hinge fitted: radius=%.3fm axis=%s. Arc to %.0f deg -> worst sigma %.4f, "
            "worst joint margin %.3f rad.",
            radius, np.round(axis, 3).tolist(), self.config.target_open_angle_deg,
            result["worst_sigma"], result["worst_margin"],
        )
        if result["blocked_at"] is None:
            logger.info("Arc is clear for the full %.0f degrees.", self.config.target_open_angle_deg)
            return None
        logger.warning(
            "Arc is NOT clear: the arm runs out of reach or travel at about %.0f degrees "
            "of %.0f. Expect the pull to stall there.",
            result["blocked_at"], self.config.target_open_angle_deg,
        )
        return float(result["blocked_at"])

    async def _pull_loop(self) -> None:
        state = self._get_state()
        if state is None:
            logger.warning("No wrench or joint data yet -- is the FT sensor and coordinator running?")
            return

        try:
            self._running = True
            self._stop_requested = False
            self.total_pull_distance = 0.0
            self.total_rotation = 0.0
            self.motion_count = 0
            self.peak_resistance_force = 0.0
            base = replace(self.config.admittance,
                       force_axis_weights=self.config.force_axis_weights)

            logger.info("Probing door: speed=%.3fm/s over up to %.2fcm", self.config.probe_speed, self.config.probe_distance_m * 100)
            probe_cfg = replace(
                base,
                drive_speed=self.config.probe_speed,
                decel_start_m=self.config.probe_distance_m * 0.5,
                decel_full_m=self.config.probe_distance_m,
            )
            probe = await self._run_phase("probe", probe_cfg, self.config.probe_distance_m, self.config.probe_max_duration)

            arc_blocked_at = None
            if (self._running and not self._stop_requested and not probe.safety_tripped
                    and self.config.check_arc_before_execute):
                arc_blocked_at = self._assess_arc(probe)
                if arc_blocked_at is not None and self.config.refuse_if_arc_blocked:
                    logger.warning(
                        "Refusing to execute: the arm cannot follow this door past %.0f degrees "
                        "from where it is holding. Re-grasp or reposition, then retry.",
                        arc_blocked_at,
                    )
                    self._running = False

            if self._running and not self._stop_requested and not probe.safety_tripped:
                # Clamped below the sensor's own hardware overload rating (see admittance_pull_law.py)
                # -- a calibrated cutoff above that can never actually protect anything, the hardware
                # faults first regardless of what our software thinks is safe.
                force_cutoff = min(
                    max(probe.peak_resistance_force * self.config.cutoff_safety_margin, self.config.min_force_cutoff),
                    SENSOR_FORCE_OVERLOAD_N * 0.8,
                    self.config.max_force_cutoff_n,
                )
                torque_cutoff = min(
                    max(probe.peak_torque * self.config.cutoff_safety_margin, self.config.min_torque_cutoff),
                    SENSOR_TORQUE_OVERLOAD_NM * 0.8,
                )
                logger.info(
                    "Probe measured peak_force=%.1fN peak_torque=%.1fNm -- executing with cutoffs force<=%.1fN torque<=%.1fNm",
                    probe.peak_resistance_force, probe.peak_torque, force_cutoff, torque_cutoff,
                )
                exec_cfg = replace(
                    base,
                    drive_speed=self.config.execute_speed,
                    force_cutoff=force_cutoff,
                    torque_cutoff=torque_cutoff,
                    decel_start_m=self.config.execute_target_m * 0.5,
                    decel_full_m=self.config.execute_target_m,
                )
                await self._run_phase("execute", exec_cfg, self.config.execute_target_m, self.config.max_duration)
            elif probe.safety_tripped:
                logger.warning("Probe itself hit a safety cutoff -- door may be jammed or grasp is off. Not proceeding to execute.")

        finally:
            # Unconditional: any escape from the loop -- exception, cancellation, cutoff --
            # must leave the arm stopped, never holding the last non-zero twist.
            self.coordinator_ee_twist_command.publish(
                TwistStamped(frame_id=self.config.task_name, linear=[0, 0, 0], angular=[0, 0, 0])
            )
            self._running = False
            self._phase = ""
        logger.info(
            "Pull finished: %d steps, %.1fcm, %.1fdeg, peak_resistance=%.1fN",
            self.motion_count, self.total_pull_distance * 100, np.degrees(self.total_rotation), self.peak_resistance_force,
        )

    @rpc
    def stop_pull(self) -> str:
        self._stop_requested = True
        return "Stop requested"

    @rpc
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            wrench = self._latest_wrench.copy() if self._latest_wrench is not None else None
        return {
            "has_wrench_data": wrench is not None,
            "resistance_force": float(np.linalg.norm(wrench[:3])) if wrench is not None else 0.0,
            "torque_mag": float(np.linalg.norm(wrench[3:])) if wrench is not None else 0.0,
            "phase": self._phase,
            "total_pull_cm": self.total_pull_distance * 100,
            "total_rotation_deg": float(np.degrees(self.total_rotation)),
            "peak_resistance_force": self.peak_resistance_force,
            "motion_count": self.motion_count,
            "running": self._running,
        }
