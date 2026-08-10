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

"""Step C: Yashas's FT-feedback door-pull control law, ported off Drake.

Same control law as the old dimos/hardware/ft_pull_skill.py compute_combined_motion:
adaptive rotation gain from lateral force error, force-adaptive pull speed,
oscillation damping, a safety cutoff at extreme force. Only the motion-execution
layer changed -- instead of solving Drake diff-IK to a joint-angle target each
tick, this publishes a TwistStamped on the same coordinator_ee_twist_command
channel keyboard teleop uses, and EEFTwistTask (already in the framework)
integrates it into motion. Gripper is NOT touched here -- close it by hand
first, same as the manual-pull workflow this replaces.

Frame check (verified against dimos/control/tasks/eef_twist_task/eef_twist_task.py
_prepare_target): both the linear and angular parts of the published twist are
WORLD-frame -- pose.translation += linear*dt, pose.rotation = exp3(angular*dt) @
pose.rotation is a left-multiply, i.e. a world-frame rotation update. Yashas's
original math is also entirely world-frame, so no frame conversion was needed --
just the pivot-rotation math re-expressed as an instantaneous twist (see
_pivot_rotation_to_twist below for the derivation).

NOT yet hardware-verified: `ee_joint_id` (which Pinocchio joint index is the
tool frame) and that PinocchioIK.forward_kinematics on this URDF returns a
sane pose. Print the computed EE position on the first few ticks and sanity
check it against a known pose before trusting the pull motion.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.manipulation.planning.kinematics.pinocchio_ik import PinocchioIK
from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


class FTPullConfig(ModuleConfig):
    """Same tuning knobs as the old FTPullModule.continuous_pull skill."""

    hardware_id: str = "arm"
    num_arm_joints: int = 7
    urdf_path: str | Path = "dimos/hardware/xarm7_openft_gripper.urdf"
    ee_joint_id: int = 7  # UNVERIFIED -- last arm joint before the tool, check on first run

    pivot_distance: float = 0.2
    force_threshold: float = 7.0
    rotation_gain: float = 0.01
    pull_speed: float = 0.015
    door_opens_clockwise: bool = True
    rotation_axis: str = "z"
    max_duration: float = 30.0
    end_angle_deg: float | None = None

    control_rate_hz: float = 25.0
    max_rotation_per_step: float = 0.2
    min_rotation_per_step: float = 0.005
    oscillation_damping: float = 0.5

    # Starts pulling automatically once force+joint data arrive, no RPC call
    # needed -- matches the old ft_pull_test.py --auto-run workflow. Gripper
    # is still yours to close by hand first; this only ever touches the twist
    # channel, never the gripper joint.
    auto_run: bool = False


class FTPullModule(Module):
    """Reads ext_wrench + coordinator_joint_state, publishes coordinator_ee_twist_command."""

    config: FTPullConfig

    ext_wrench: In[WrenchStamped]
    coordinator_joint_state: In[JointState]
    coordinator_ee_twist_command: Out[TwistStamped]
    # Same name as KeyboardTeleopModule's output -- autoconnect wires them
    # without remapping when both are in one blueprint.
    start_pull_command: In[Bool]

    _lock: threading.Lock
    _latest_force: np.ndarray | None = None
    _latest_q: np.ndarray | None = None
    _ik: PinocchioIK | None = None
    _running: bool = False
    _stop_requested: bool = False

    @rpc
    def start(self) -> None:
        super().start()
        self._lock = threading.Lock()
        self._ik = PinocchioIK.from_model_path(str(self.config.urdf_path), self.config.ee_joint_id)
        self.ext_wrench.subscribe(self._on_wrench)
        self.coordinator_joint_state.subscribe(self._on_joint_state)
        self.start_pull_command.subscribe(self._on_start_pull_command)
        logger.info("FTPullModule ready (ee_joint_id=%d, VERIFY this on first run)", self.config.ee_joint_id)
        if self.config.auto_run:
            self.spawn(self._pull_loop())

    def _on_start_pull_command(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self._running:
            logger.info("Pull already running, ignoring start_pull_command")
            return
        logger.info("start_pull_command received -- starting pull")
        self.spawn(self._pull_loop())

    def _on_wrench(self, msg: WrenchStamped) -> None:
        with self._lock:
            self._latest_force = np.array([msg.force.x, msg.force.y, msg.force.z])

    def _on_joint_state(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position, strict=True))
        prefix = f"{self.config.hardware_id}/joint"
        try:
            q = np.array([by_name[f"{prefix}{i}"] for i in range(1, self.config.num_arm_joints + 1)])
        except KeyError:
            return  # not all arm joints reported yet
        with self._lock:
            self._latest_q = q

    def _get_state(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            if self._latest_force is None or self._latest_q is None:
                return None
            return self._latest_force.copy(), self._latest_q.copy()

    def _compute_rotation_and_pull(
        self, force: np.ndarray, rotation_history: deque, oscillation_damping: float
    ) -> tuple[float, float]:
        """Yashas's original adaptive control law, unchanged. Returns (rotation_angle, pull_distance)."""
        force_x = force[0]
        lateral_force = float(np.linalg.norm(force[:2]))
        cfg = self.config

        rotation_angle = 0.0
        pull_safety_factor = 0.0 if lateral_force > 80 else 1.0

        if lateral_force > 60:
            adaptive_gain = cfg.rotation_gain * 0.3
        elif lateral_force > 40:
            adaptive_gain = cfg.rotation_gain * 0.5
        elif lateral_force > 20:
            adaptive_gain = cfg.rotation_gain * 0.7
        elif lateral_force > 10:
            adaptive_gain = cfg.rotation_gain * 0.9
        else:
            adaptive_gain = cfg.rotation_gain * 1.2

        if len(rotation_history) >= 2:
            recent_signs = [np.sign(r) for r in rotation_history if abs(r) > 0.01]
            if len(recent_signs) >= 2 and recent_signs[-1] * recent_signs[-2] < 0:
                adaptive_gain *= oscillation_damping

        in_success_zone = abs(force_x) <= cfg.force_threshold

        if cfg.door_opens_clockwise:
            if force_x > cfg.force_threshold:
                error = force_x - cfg.force_threshold
                if error > 30:
                    rotation_angle = -adaptive_gain * 15 * (1 - np.exp(-error / 30))
                elif error > 15:
                    rotation_angle = -adaptive_gain * error * 0.7
                else:
                    rotation_angle = -error * adaptive_gain
            elif force_x < -cfg.force_threshold:
                error = abs(force_x) - cfg.force_threshold
                if error > 8:
                    rotation_angle = adaptive_gain * min(error * 0.2, 5)
        else:
            if force_x < -cfg.force_threshold:
                error = abs(force_x) - cfg.force_threshold
                rotation_angle = (
                    adaptive_gain * 20 * (1 - np.exp(-error / 20))
                    if error > 20
                    else error * adaptive_gain
                )
            elif force_x > cfg.force_threshold:
                error = force_x - cfg.force_threshold
                if error > 5:
                    rotation_angle = -adaptive_gain * min(error * 0.3, 10)

        max_rotation = cfg.max_rotation_per_step
        if lateral_force > 50:
            max_rotation = min(cfg.max_rotation_per_step * 0.4, 0.08)
        elif lateral_force > 30:
            max_rotation = min(cfg.max_rotation_per_step * 0.6, 0.12)
        rotation_angle = float(np.clip(rotation_angle, -max_rotation, max_rotation))

        rotation_history.append(rotation_angle)
        if abs(rotation_angle) < cfg.min_rotation_per_step:
            rotation_angle = 0.0

        if in_success_zone and lateral_force < 10:
            pull_distance = cfg.pull_speed * 1.5
        elif lateral_force < 15:
            pull_distance = cfg.pull_speed
        elif lateral_force < 25:
            pull_distance = cfg.pull_speed * 0.8
        elif lateral_force < 40:
            pull_distance = cfg.pull_speed * 0.6
        elif lateral_force < 60:
            pull_distance = cfg.pull_speed * 0.4
        else:
            pull_distance = cfg.pull_speed * 0.2

        if abs(rotation_angle) > cfg.max_rotation_per_step * 0.5:
            pull_distance *= 0.7
        pull_distance *= pull_safety_factor

        return rotation_angle, pull_distance

    def _pivot_rotation_to_twist(
        self, ee_pos: np.ndarray, ee_rot: np.ndarray, rotation_angle: float, pull_distance: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one tick of (rotate about a pivot in front of the EE, then pull) into a twist.

        Small-angle equivalence: rotating a rigid body by angle theta about a
        pivot P moves a point at ee_pos by approximately omega x r * dt, where
        r = ee_pos - P and omega = (theta/dt) along the rotation axis. Valid
        here because max_rotation_per_step (0.2 rad ~= 11.5 deg) is small
        relative to a single control tick.
        """
        axis = _AXES[self.config.rotation_axis]
        omega = axis * (rotation_angle / dt)

        pivot = ee_pos + ee_rot @ np.array([0.0, 0.0, self.config.pivot_distance])
        r = ee_pos - pivot
        v_from_rotation = np.cross(omega, r)

        pull_local = np.array([0.0, 0.0, -pull_distance])
        v_from_pull = (ee_rot @ pull_local) / dt

        return v_from_rotation + v_from_pull, omega

    @rpc
    def pull_door(self) -> str:
        """Start the pull loop in the background. Poll get_stats() or watch the logs for progress."""
        if self._running:
            return "Already running"
        self.spawn(self._pull_loop())
        return "Pull started"

    async def _pull_loop(self) -> None:
        state = self._get_state()
        if state is None:
            logger.warning("No force or joint data yet -- is the FT sensor and coordinator running?")
            return

        self._running = True
        self._stop_requested = False
        dt = 1.0 / self.config.control_rate_hz
        rotation_history: deque = deque(maxlen=5)
        total_rotation = 0.0
        end_angle_rad = np.radians(self.config.end_angle_deg) if self.config.end_angle_deg else None
        start_time = time.time()
        ticks = 0

        logger.info(
            "Starting FT pull: threshold=%.1fN gain=%.4f axis=%s clockwise=%s",
            self.config.force_threshold,
            self.config.rotation_gain,
            self.config.rotation_axis,
            self.config.door_opens_clockwise,
        )

        while self._running and not self._stop_requested:
            if time.time() - start_time > self.config.max_duration:
                logger.info("Reached max_duration (%.1fs)", self.config.max_duration)
                break
            if end_angle_rad and abs(total_rotation) >= end_angle_rad:
                logger.info("Reached end_angle (%.1f deg)", self.config.end_angle_deg)
                break

            state = self._get_state()
            if state is None:
                await asyncio.sleep(dt)
                continue
            force, q = state

            rotation_angle, pull_distance = self._compute_rotation_and_pull(
                force, rotation_history, self.config.oscillation_damping
            )
            total_rotation += rotation_angle

            pose = self._ik.forward_kinematics(q)  # type: ignore[union-attr]
            if ticks < 3:
                logger.info("EE pose tick %d: translation=%s -- sanity check this", ticks, pose.translation)

            linear, angular = self._pivot_rotation_to_twist(
                np.asarray(pose.translation), np.asarray(pose.rotation), rotation_angle, pull_distance, dt
            )
            self.coordinator_ee_twist_command.publish(
                TwistStamped(frame_id="ft_pull", linear=list(linear), angular=list(angular))
            )

            ticks += 1
            if ticks % 25 == 0:
                logger.info(
                    "[tick %d] Fx=%.1fN lateral=%.1fN rot=%.2fdeg total=%.1fdeg",
                    ticks,
                    force[0],
                    float(np.linalg.norm(force[:2])),
                    np.degrees(rotation_angle),
                    np.degrees(total_rotation),
                )

            await asyncio.sleep(dt)

        # Zero twist clears EEFTwistTask's latched command (see on_ee_twist_command).
        self.coordinator_ee_twist_command.publish(
            TwistStamped(frame_id="ft_pull", linear=[0, 0, 0], angular=[0, 0, 0])
        )
        self._running = False
        logger.info("Pull finished: %d ticks, %.1f deg total rotation", ticks, np.degrees(total_rotation))

    @rpc
    def stop_pull(self) -> str:
        self._stop_requested = True
        return "Stop requested"

    @rpc
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            has_force = self._latest_force is not None
            has_q = self._latest_q is not None
        return {"running": self._running, "has_force": has_force, "has_joint_state": has_q}
