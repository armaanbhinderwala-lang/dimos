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

from dimos_lcm.geometry_msgs import Wrench as LCMWrench
from plum import dispatch

from dimos.msgs.geometry_msgs.Vector3 import Vector3, VectorLike


class Wrench(LCMWrench):  # type: ignore[misc]
    """
    Represents a force and torque in 3D space.

    This is equivalent to ROS geometry_msgs/Wrench.
    """

    force: Vector3
    torque: Vector3
    msg_name = "geometry_msgs.Wrench"

    @dispatch
    def __init__(self) -> None:
        """Initialize a zero wrench (no force or torque)."""
        self.force = Vector3()
        self.torque = Vector3()

    @dispatch  # type: ignore[no-redef]
    def __init__(self, force: VectorLike, torque: VectorLike) -> None:
        """Initialize a wrench from force (N) and torque (Nm) vectors."""
        self.force = Vector3(force)
        self.torque = Vector3(torque)

    @dispatch  # type: ignore[no-redef]
    def __init__(self, wrench: Wrench) -> None:
        """Initialize from another Wrench (copy constructor)."""
        self.force = Vector3(wrench.force)
        self.torque = Vector3(wrench.torque)

    @dispatch  # type: ignore[no-redef]
    def __init__(self, lcm_wrench: LCMWrench) -> None:
        """Initialize from an LCM Wrench."""
        self.force = Vector3(lcm_wrench.force)
        self.torque = Vector3(lcm_wrench.torque)

    @dispatch  # type: ignore[no-redef]
    def __init__(self, **kwargs) -> None:
        """Handle keyword arguments for LCM compatibility."""
        force = kwargs.get("force", Vector3())
        torque = kwargs.get("torque", Vector3())

        self.__init__(force, torque)

    def __repr__(self) -> str:
        return f"Wrench(force={self.force!r}, torque={self.torque!r})"

    def __str__(self) -> str:
        return f"Wrench:\n  Force: {self.force}\n  Torque: {self.torque}"

    def __eq__(self, other) -> bool:  # type: ignore[no-untyped-def]
        """Check if two wrenches are equal."""
        if not isinstance(other, Wrench):
            return False
        return self.force == other.force and self.torque == other.torque

    @classmethod
    def zero(cls) -> Wrench:
        """Create a zero wrench (no force or torque)."""
        return cls()

    def is_zero(self) -> bool:
        """Check if this is a zero wrench (no force or torque)."""
        return self.force.is_zero() and self.torque.is_zero()

    def __sub__(self, other: Wrench) -> Wrench:
        """Component-wise subtraction: self - other."""
        if not isinstance(other, Wrench):
            return NotImplemented
        return Wrench(
            force=self.force - other.force,
            torque=self.torque - other.torque,
        )

    def __add__(self, other: Wrench) -> Wrench:
        """Component-wise addition: self + other."""
        if not isinstance(other, Wrench):
            return NotImplemented
        return Wrench(
            force=self.force + other.force,
            torque=self.torque + other.torque,
        )

    def __bool__(self) -> bool:
        """A Wrench is False when it carries no force or torque, True otherwise."""
        return not self.is_zero()
