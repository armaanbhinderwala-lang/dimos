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

"""Door opening on an xArm7, from either force-torque sensor.

    dimos run door-opener-xarm7        uFactory sensor via the arm's tool port
    dimos run door-opener-xarm7-diy    homemade 16-channel sensor over serial

Grip the handle, press ENTER: probe, fit the hinge from that motion, check the arc, pull.

Two blueprints rather than a CLI flag because blueprints resolve by name at `dimos run` time.
Shared wiring lives in _build() so they cannot drift.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.control.coordinator import ControlCoordinator
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.ft_adaptive_pull_module import FTAdaptivePullModule
from dimos.hardware.sensors.force_torque.ft_conditioner_module import FTConditionerModule
from dimos.hardware.sensors.force_torque.openft_module import OpenFTSensor
from dimos.hardware.sensors.force_torque.read_FTModule import XArmFTSensor
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.robot.manipulators.common.blueprints import eef_twist_task
from dimos.robot.manipulators.xarm.config import make_xarm7_model_config, xarm7_hardware
from dimos.teleop.keyboard.keyboard_teleop_module import KeyboardTeleopModule
from dimos.visualization.wrench_plotter import WrenchPlotter

DIY_TOOL_MASS_KG = 0.85  # gravity comp for sensors that do not do it themselves

_ft_transports = {
    ("ext_wrench", WrenchStamped): LCMTransport("/ft/ext_wrench", WrenchStamped),
    ("raw_wrench", WrenchStamped): LCMTransport("/ft/raw_wrench", WrenchStamped),
    ("clean_wrench", WrenchStamped): LCMTransport("/ft/clean_wrench", WrenchStamped),
}


def _build(sensor_module, profile: str, tool_mass_kg: float):
    hardware = xarm7_hardware("arm", gripper=True, mock_without_address=True)
    # add_gripper=False makes the tip frame "link7"; the pull module must name the same frame
    # or its FK describes a different point than the one being moved.
    control_model = make_xarm7_model_config(add_gripper=False)
    return autoconnect(
        KeyboardTeleopModule.blueprint(),
        ControlCoordinator.blueprint(
            tick_rate=100.0,
            publish_joint_state=True,
            joint_state_frame_id="coordinator",
            hardware=[hardware],
            # Pink IK lives here and owns the arm. The policy only publishes a TwistStamped;
            # solving IK itself would be a second writer and bypass joint-limit handling.
            tasks=[eef_twist_task(hardware, robot_model=control_model, timeout=0.0)],
        ),
        sensor_module,
        FTConditionerModule.blueprint(profile=profile, tool_mass_kg=tool_mass_kg),
        FTAdaptivePullModule.blueprint(
            hardware_id="arm",
            num_arm_joints=7,
            model_path=control_model.model_path,
            package_paths=control_model.package_paths,
            xacro_args=control_model.xacro_args,
            tool_frame_name="link7",
            auto_run=False,
        ),
        WrenchPlotter.blueprint(),
    ).transports(_ft_transports).remappings([
        # autoconnect matches by name, so without this the sensor's ext_wrench binds straight
        # to the policy and the conditioner's output goes nowhere.
        (FTAdaptivePullModule, "ext_wrench", "clean_wrench"),
        (FTConditionerModule, "raw_wrench", "raw_wrench"),
    ])


# uFactory is gravity-compensated upstream, hence tool_mass_kg=0.0.
door_opener_xarm7 = _build(
    XArmFTSensor.blueprint(ip=global_config.xarm7_ip),
    profile="factory",
    tool_mass_kg=0.0,
)

# Homemade sensor: 6x16 calibration applied in the driver, no gravity comp of its own.
door_opener_xarm7_diy = _build(
    OpenFTSensor.blueprint(),
    profile="diy",
    tool_mass_kg=DIY_TOOL_MASS_KG,
)
