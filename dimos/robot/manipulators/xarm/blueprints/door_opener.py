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
from dimos.control.coordinator import ControlCoordinator, TaskConfig
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
# Tool centre of mass from the sensor, along the tool axis. Left at zero the moment term
# cross(com, force) is identically zero, so the gripper and camera weight is removed from the
# force and none of it from the torque -- a pose-varying m*g*d bias that spends the torque
# safety budget before the door does. Measure it properly; this is a stand-in.
DIY_TOOL_COM_M = (0.0, 0.0, 0.06)

_ft_transports = {
    ("ext_wrench", WrenchStamped): LCMTransport("/ft/ext_wrench", WrenchStamped),
    ("raw_wrench", WrenchStamped): LCMTransport("/ft/raw_wrench", WrenchStamped),
    ("clean_wrench", WrenchStamped): LCMTransport("/ft/clean_wrench", WrenchStamped),
}


# Hinge-to-handle distance of the door being opened. Measure it; the probe cannot.
DOOR_RADIUS_M = 0.368  # 14.5 in, hinge pin to handle
# Unit vector from the handle toward the hinge, in the arm's base frame: +X points away from
# the arm, +Y is the arm's LEFT. Set explicitly because the force reading put the hinge on the
# opposite side to the real one, and a wrong side drives the door away from its hinge.
# Set to None to go back to reading it from the constraint force.
HINGE_DIRECTION_WORLD = (0.0, 1.0, 0.0)


def _build(sensor_module, profile: str, tool_mass_kg: float,
           tool_com_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
           safety_cutoffs_enabled: bool = True,
           force_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
           door_radius_m: float | None = DOOR_RADIUS_M,
           hinge_direction_world: tuple[float, float, float] | None = HINGE_DIRECTION_WORLD):
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
            tasks=[
                # No gripper params here on purpose: they make eef_twist claim the gripper and
                # hold it open every tick, which overrides the one-shot [ and ] key commands.
                # 7 joints track a 6-DOF twist, so one DOF is spare. Unused it lets a single
                # joint walk into its stop at ~60 deg of arc. posture_cost holds a fixed
                # reference posture and fights the redistribution, so hand the nullspace to
                # joint centering instead: it targets the middle of each range, and joint6's
                # middle is +39 deg, away from the -100.3 stop that keeps ending the pull.
                eef_twist_task(hardware, robot_model=control_model, timeout=0.0,
                               control_ik={"posture_cost": 0.0,
                                           "joint_centering_cost": 1e-2}),
                # [ and ] publish a JointState on joint_command, and routing delivers that
                # only to `servo` tasks. Without this the gripper keys do nothing.
                TaskConfig(
                    name="servo_gripper",
                    type="servo",
                    joint_names=hardware.gripper_joints,
                    priority=20,
                    params={"timeout": 0.0, "default_positions": [0.0]},
                ),
            ],
        ),
        sensor_module,
        # Tool mass goes to the PULL module, not here: gravity compensation needs the wrist
        # orientation, and only that module computes forward kinematics.
        FTConditionerModule.blueprint(profile=profile),
        FTAdaptivePullModule.blueprint(
            hardware_id="arm",
            num_arm_joints=7,
            model_path=control_model.model_path,
            package_paths=control_model.package_paths,
            xacro_args=control_model.xacro_args,
            tool_frame_name="link7",
            tool_mass_kg=tool_mass_kg,
            tool_com_m=tool_com_m,
            force_axis_weights=force_axis_weights,
            door_radius_m=door_radius_m,
            safety_cutoffs_enabled=safety_cutoffs_enabled,
            hinge_direction_world=hinge_direction_world,
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
# Fz weighted 0: this sensor's Fz drifts ~5 N, and a microwave/fridge/cabinet hinge is
# vertical so vertical force carries no door motion. Set it back to 1 for an oven.
door_opener_xarm7_diy = _build(
    OpenFTSensor.blueprint(),
    profile="diy",
    tool_mass_kg=DIY_TOOL_MASS_KG,
    tool_com_m=DIY_TOOL_COM_M,
    force_axis_weights=(1.0, 1.0, 0.0),
    # Off at the operator's request while tuning: force trips were ending runs long before
    # anything was actually at risk. Turn back on before leaving this unattended.
    safety_cutoffs_enabled=False,
)
