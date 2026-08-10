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

"""Keyboard teleop for xArm6/xArm7, with the arm's own built-in FT sensor graphed live.

Same hardware/control-task setup as keyboard_teleop_xarm6/7 (./teleop.py), with
XArmFTSensor and WrenchPlotter added into the same autoconnect(). Jog the arm
and close the gripper on the handle by hand, same as before -- but now
keyboard_teleop_xarm7_ft also has FTPullModule wired in: press ENTER to hand
off from manual WASD control to the automatic FT-feedback pull (Step C).
Don't drive WASD while a pull is active -- both publish to the same twist
channel, and whichever publishes most recently wins each tick.

Uses the xArm's built-in FT sensor (no serial port, just the arm's own IP) --
NOT the homemade OpenFT sensor (dimos.hardware.sensors.force_torque.openft_module),
which needs its MCU wired up over serial. Swap XArmFTSensor for OpenFTSensor
when that's plugged in -- same ext_wrench/raw_wrench ports, same plotter.
"""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.ft_pull_module import FTPullModule
from dimos.hardware.sensors.force_torque.read_FTModule import XArmFTSensor
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.robot.manipulators.common.blueprints import GripperTaskOverrides, eef_twist_task
from dimos.robot.manipulators.xarm.config import (
    XARM_GRIPPER_PARAMS,
    make_xarm6_model_config,
    make_xarm7_model_config,
    xarm6_hardware,
    xarm7_hardware,
)
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule
from dimos.visualization.wrench_plotter import WrenchPlotter

_gripper_params = XARM_GRIPPER_PARAMS  # type: GripperTaskOverrides

_ft_transports = {
    ("ext_wrench", WrenchStamped): LCMTransport("/ft/ext_wrench", WrenchStamped),
    ("raw_wrench", WrenchStamped): LCMTransport("/ft/raw_wrench", WrenchStamped),
}

_xarm7_hw = xarm7_hardware("arm", gripper=True, mock_without_address=True)
_xarm7_control_model = make_xarm7_model_config(add_gripper=False)

keyboard_teleop_xarm7_ft = autoconnect(
    KeyboardTeleopModule.blueprint(),
    ControlCoordinator.blueprint(
        tick_rate=100.0,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[_xarm7_hw],
        tasks=[
            # No gripper params on purpose: they make eef_twist claim the gripper
            # (claim_with_gripper) and hold it at gripper_open_pos every tick,
            # which overrides the one-shot `[`/`]` key commands.
            eef_twist_task(
                _xarm7_hw,
                robot_model=_xarm7_control_model,
                timeout=0.0,
            ),
            # `[`/`]` publish a JointState on `joint_command`, which routing only
            # delivers to `servo` tasks - eef_twist declares no such binding.
            TaskConfig(
                name="servo_gripper",
                type="servo",
                joint_names=_xarm7_hw.gripper_joints,
                priority=20,
                params={"timeout": 0.0, "default_positions": [0.0]},
            ),
        ],
    ),
    ManipulationModule.blueprint(
        robots=[make_xarm7_model_config(add_gripper=True)],
        visualization={"backend": "viser"},
    ),
    # Same physical arm as the hardware above -- reuse its IP rather than the
    # separate DIMOS_XARM_IP env var XArmFTSensorConfig defaults to.
    XArmFTSensor.blueprint(ip=global_config.xarm7_ip),
    # Same model config eef_twist_task above was built with -- add_gripper=False
    # there means its tip frame is "link7" (see xarm/config.py's tip_link logic),
    # not "link_tcp".
    FTPullModule.blueprint(
        hardware_id="arm",
        num_arm_joints=7,
        model_path=_xarm7_control_model.model_path,
        package_paths=_xarm7_control_model.package_paths,
        xacro_args=_xarm7_control_model.xacro_args,
        tool_frame_name="link7",
        auto_run=False,
    ),
    WrenchPlotter.blueprint(),
).transports(_ft_transports)

_xarm6_hw = xarm6_hardware("arm", gripper=True, mock_without_address=True)
_xarm6_control_model = make_xarm6_model_config(add_gripper=False)

keyboard_teleop_xarm6_ft = autoconnect(
    KeyboardTeleopModule.blueprint(),
    ControlCoordinator.blueprint(
        tick_rate=100.0,
        publish_joint_state=True,
        joint_state_frame_id="coordinator",
        hardware=[_xarm6_hw],
        tasks=[
            eef_twist_task(
                _xarm6_hw,
                robot_model=_xarm6_control_model,
                timeout=0.0,
                params=_gripper_params,
            )
        ],
    ),
    ManipulationModule.blueprint(
        robots=[make_xarm6_model_config(add_gripper=True)],
        visualization={"backend": "viser"},
    ),
    XArmFTSensor.blueprint(ip=global_config.xarm6_ip),
    WrenchPlotter.blueprint(),
).transports(_ft_transports)
