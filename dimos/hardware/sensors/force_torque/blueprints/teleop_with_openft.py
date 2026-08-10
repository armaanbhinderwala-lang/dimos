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

"""Keyboard teleop for xArm6/xArm7, with the homemade OpenFT sensor graphed live.

Same hardware/control-task setup as keyboard_teleop_xarm6/7
(dimos.robot.manipulators.xarm.blueprints.teleop), with OpenFTSensor and
WrenchPlotter added into the same autoconnect(). Jog the arm and close the
gripper on the handle by hand -- no perception, no pull logic yet -- while
watching live force/torque in the Rerun viewer WrenchPlotter opens.

To use the xArm's own built-in FT sensor instead of the homemade one, swap
OpenFTSensor for XArmFTSensor (dimos.hardware.sensors.force_torque.read_FTModule)
-- same ext_wrench/raw_wrench ports, same plotter, no other change needed.
"""

from __future__ import annotations

from dimos.control.coordinator import ControlCoordinator
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.openft_module import OpenFTSensor
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
            eef_twist_task(
                _xarm7_hw,
                robot_model=_xarm7_control_model,
                timeout=0.0,
                params=_gripper_params,
            )
        ],
    ),
    ManipulationModule.blueprint(
        robots=[make_xarm7_model_config(add_gripper=True)],
        visualization={"backend": "viser"},
    ),
    OpenFTSensor.blueprint(),
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
    OpenFTSensor.blueprint(),
    WrenchPlotter.blueprint(),
).transports(_ft_transports)
