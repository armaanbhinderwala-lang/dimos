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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pinocchio

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.force_torque.admittance_pull_law import (
    SENSOR_FORCE_OVERLOAD_N,
    SENSOR_TORQUE_OVERLOAD_NM,
    AdmittanceConfig,
    compute_twist,
    singularity_speed_scale,
    slew_limit,
)
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
    probe_distance_m: float = 0.04  # m -- stop the probe here even if never resisted
    probe_max_duration: float = 5.0  # s, safety net independent of distance

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

    # Execute-phase cutoffs = max(probe_peak * cutoff_safety_margin, min_*) --
    # scales the safety envelope to what THIS door demonstrably needs instead
    # of one fixed guess. All four values here are starting points, not
    # validated against real hardware yet.
    cutoff_safety_margin: float = 2.5
    min_force_cutoff: float = 30.0  # N
    min_torque_cutoff: float = 4.0  # N*m

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
        with self._lock:
            self._latest_wrench = np.array(
                [msg.force.x, msg.force.y, msg.force.z, msg.torque.x, msg.torque.y, msg.torque.z]
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

        while self._running and not self._stop_requested:
            if time.time() - start_time > max_duration:
                stats.stop_reason = "max_duration"
                break
            if stats.distance_covered >= max_progress_m:
                stats.stop_reason = "reached target"
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
                    "[%s] Joint %d at %.3f rad is within the safety margin of its limit -- stopping.",
                    name, near_limit, q[near_limit],
                )
                stats.stop_reason = "joint limit"
                break

            pose = self._forward_kinematics(q)
            ee_rot = np.asarray(pose.rotation)
            if stats.ticks < 3:
                logger.info("[%s] EE pose tick %d: translation=%s -- sanity check this", name, stats.ticks, pose.translation)

            sigma_min = self._manipulability(q)
            sing_scale = singularity_speed_scale(sigma_min, cfg.singularity_sigma_caution, cfg.singularity_sigma_stop)
            if sing_scale <= 0.0:
                logger.warning("[%s] Near a kinematic singularity (sigma_min=%.4f) -- stopping.", name, sigma_min)
                stats.stop_reason = "near singularity"
                break

            drive_direction_world = ee_rot @ self._local_drive
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
                    "[%s tick %d] resistance=%.1fN torque=%.1fNm |v|=%.3fm/s |omega|=%.3frad/s covered=%.1fcm sigma_min=%.4f",
                    name, stats.ticks, result.resistance_force, result.torque_mag,
                    float(np.linalg.norm(linear_cmd)), float(np.linalg.norm(angular_cmd)),
                    stats.distance_covered * 100, sigma_min,
                )

            await asyncio.sleep(dt)

        logger.info(
            "[%s] done (%s): %d ticks, %.1fcm, peak_force=%.1fN peak_torque=%.1fNm min_sigma=%.4f",
            name, stats.stop_reason or "external stop", stats.ticks, stats.distance_covered * 100,
            stats.peak_resistance_force, stats.peak_torque, stats.min_sigma,
        )
        return stats

    async def _pull_loop(self) -> None:
        state = self._get_state()
        if state is None:
            logger.warning("No wrench or joint data yet -- is the FT sensor and coordinator running?")
            return

        self._running = True
        self._stop_requested = False
        self.total_pull_distance = 0.0
        self.total_rotation = 0.0
        self.motion_count = 0
        self.peak_resistance_force = 0.0
        base = self.config.admittance

        logger.info("Probing door: speed=%.3fm/s over up to %.2fcm", self.config.probe_speed, self.config.probe_distance_m * 100)
        probe_cfg = replace(
            base,
            drive_speed=self.config.probe_speed,
            decel_start_m=self.config.probe_distance_m * 0.5,
            decel_full_m=self.config.probe_distance_m,
        )
        probe = await self._run_phase("probe", probe_cfg, self.config.probe_distance_m, self.config.probe_max_duration)

        if self._running and not self._stop_requested and not probe.safety_tripped:
            # Clamped below the sensor's own hardware overload rating (see admittance_pull_law.py)
            # -- a calibrated cutoff above that can never actually protect anything, the hardware
            # faults first regardless of what our software thinks is safe.
            force_cutoff = min(
                max(probe.peak_resistance_force * self.config.cutoff_safety_margin, self.config.min_force_cutoff),
                SENSOR_FORCE_OVERLOAD_N * 0.8,
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
