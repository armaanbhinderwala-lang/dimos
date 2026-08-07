# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

import time
from typing import BinaryIO, TypeAlias

from dimos_lcm.geometry_msgs import WrenchStamped as LCMWrenchStamped
from plum import dispatch

from dimos.msgs.geometry_msgs.Vector3 import VectorConvertable
from dimos.msgs.geometry_msgs.Wrench import Wrench
from dimos.types.timestamped import Timestamped

# Types that can be converted to/from WrenchStamped
WrenchConvertable: TypeAlias = (
    tuple[VectorConvertable, VectorConvertable] | LCMWrenchStamped | dict[str, VectorConvertable]
)


class WrenchStamped(Wrench, Timestamped):
    """A force/torque measurement with timestamp and frame_id.

    Follows the same pattern as TwistStamped(Twist, Timestamped) and
    PoseStamped(Pose, Timestamped). Inherits force/torque from Wrench
    (which inherits from LCMWrench).

    This is equivalent to ROS geometry_msgs/WrenchStamped.
    """

    msg_name = "geometry_msgs.WrenchStamped"
    ts: float
    frame_id: str

    @dispatch
    def __init__(self, ts: float = 0.0, frame_id: str = "", **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.frame_id = frame_id
        self.ts = ts if ts != 0 else time.time()
        super().__init__(**kwargs)

    @classmethod
    def from_force_torque_array(
        cls,
        ft_data: VectorConvertable,
        frame_id: str = "ft_sensor",
        ts: float | None = None,
    ) -> WrenchStamped:
        """
        Create WrenchStamped from a 6-element force/torque array.

        Args:
            ft_data: [fx, fy, fz, tx, ty, tz]
            frame_id: Reference frame
            ts: Timestamp (defaults to current time)

        Returns:
            WrenchStamped instance
        """
        if len(ft_data) != 6:
            raise ValueError(f"Expected 6 elements, got {len(ft_data)}")

        return cls(
            ts=ts if ts is not None else time.time(),
            frame_id=frame_id,
            force=ft_data[0:3],
            torque=ft_data[3:6],
        )

    # -- LCM encode / decode --

    def lcm_encode(self) -> bytes:
        """Encode to LCM binary format."""
        lcm_msg = LCMWrenchStamped()
        lcm_msg.wrench = self  # Works because Wrench inherits from LCMWrench
        [lcm_msg.header.stamp.sec, lcm_msg.header.stamp.nsec] = self.ros_timestamp()
        lcm_msg.header.frame_id = self.frame_id
        return lcm_msg.lcm_encode()  # type: ignore[no-any-return]

    @classmethod
    def lcm_decode(cls, data: bytes | BinaryIO) -> WrenchStamped:
        """Decode from LCM binary format."""
        lcm_msg = LCMWrenchStamped.lcm_decode(data)
        return cls(
            ts=lcm_msg.header.stamp.sec + (lcm_msg.header.stamp.nsec / 1_000_000_000),
            frame_id=lcm_msg.header.frame_id,
            force=[lcm_msg.wrench.force.x, lcm_msg.wrench.force.y, lcm_msg.wrench.force.z],
            torque=[lcm_msg.wrench.torque.x, lcm_msg.wrench.torque.y, lcm_msg.wrench.torque.z],
        )

    # -- String representations --

    def __str__(self) -> str:
        return (
            f"WrenchStamped(force=[{self.force.x:.3f}, {self.force.y:.3f}, {self.force.z:.3f}], "
            f"torque=[{self.torque.x:.3f}, {self.torque.y:.3f}, {self.torque.z:.3f}])"
        )

    def __repr__(self) -> str:
        return (
            f"WrenchStamped(ts={self.ts}, frame_id={self.frame_id!r}, "
            f"force={self.force!r}, torque={self.torque!r})"
        )
