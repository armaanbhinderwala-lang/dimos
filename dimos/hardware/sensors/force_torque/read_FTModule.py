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

"""Module wrapper for the xArm 6-Axis Force-Torque Sensor.

Binds the xArm Python SDK FT sensor interfaces to stream compensated external forces
and raw strain gauge readings as standardized WrenchStamped messages over the DimOS bus.
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import time

from pydantic import Field
from xarm.wrapper import XArmAPI

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Mirrors the SDK's own test for "is this an IP or a serial port" (xarm/x3/base.py).
_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


class XArmFTSensorConfig(ModuleConfig):
    """Pydantic configuration parameters for the xArm FT Sensor module."""

    # Hardware network config (falls back to env var if unset)
    ip: str | None = Field(default_factory=lambda: os.environ.get("DIMOS_XARM_IP", "192.168.1.197"))

    # Telemetry sampling rate (Hz)
    frequency: float = Field(default=100, gt=0.0)

    # Frame metadata. Both streams share it: raw and compensated are the same
    # sensor in the same frame, and the two topics already tell them apart.
    frame_id: str = "ft_sensor_link"

    # Streaming toggles
    publish_raw: bool = True
    publish_ext: bool = True


class XArmFTSensor(Module):
    """Dimensional OS driver for xArm Force-Torque Sensor."""

    config: XArmFTSensorConfig

    # Typed output streams exposed on the DimOS bus
    ext_wrench: Out[WrenchStamped]
    raw_wrench: Out[WrenchStamped]

    _arm: XArmAPI | None = None
    _running: bool = False

    @rpc
    def start(self) -> None:
        """Lifecycle hook to initialize hardware connection and start streaming loops."""
        ip = self._resolve_ip()
        super().start()

        # Connect to xArm hardware API
        arm = XArmAPI(ip)
        if not arm.connected:
            raise RuntimeError(f"XArmFTSensor: Could not connect to arm at IP {ip}")

        # Enable the physical FT sensor peripheral
        code = arm.set_ft_sensor_enable(1)
        if code != 0:
            arm.disconnect()
            raise RuntimeError(f"XArmFTSensor: set_ft_sensor_enable failed with code {code}")

        # The telemetry loop reads the SDK's report-thread caches, which never
        # error — they just sit at zero forever if the firmware omits FT from
        # its report frame. One command round-trip here turns that silent
        # stream of zero wrenches into a startup failure.
        code, _ = arm.get_ft_sensor_data()
        if code != 0:
            arm.set_ft_sensor_enable(0)
            arm.disconnect()
            raise RuntimeError(
                f"XArmFTSensor: FT sensor did not answer (code {code}). Check that the "
                "sensor is attached and the controller firmware supports it."
            )

        self._arm = arm
        self._running = True

        # Spawn background telemetry loop
        self.spawn(self._telemetry_loop())
        logger.info(
            "XArmFTSensor streaming at %.1f Hz from %s (frame %s)",
            self.config.frequency,
            ip,
            self.frame_id,
        )

    @rpc
    def stop(self) -> None:
        """Lifecycle hook for clean sensor teardown."""
        self._running = False
        arm, self._arm = self._arm, None
        if arm is not None:
            if arm.connected:
                arm.set_ft_sensor_enable(0)
            arm.disconnect()
        super().stop()

    @rpc
    def set_zero(self) -> bool:
        """RPC Endpoint: Tare / zero the FT sensor on-demand."""
        arm = self._arm
        if arm is None or not arm.connected:
            return False
        # Annotated because xarm is untyped (mypy ignores it), so an unannotated
        # `== 0` would leak Any out of a function declared to return bool.
        code: int = arm.set_ft_sensor_zero()
        return code == 0

    async def _telemetry_loop(self) -> None:
        """Continuous stream generator publishing WrenchStamped topics."""
        interval = 1.0 / self.config.frequency

        while self._running:
            # Bound once per tick: stop() clears self._arm from the RPC thread,
            # so re-reading it mid-tick can hand the second publish a None.
            arm = self._arm
            if arm is None:
                break

            now = time.time()

            # 1. Compensated External Force Stream
            if self.config.publish_ext:
                self.ext_wrench.publish(
                    WrenchStamped.from_force_torque_array(
                        arm.ft_ext_force, frame_id=self.frame_id, ts=now
                    )
                )

            # 2. Raw Unfiltered Force Stream
            if self.config.publish_raw:
                self.raw_wrench.publish(
                    WrenchStamped.from_force_torque_array(
                        arm.ft_raw_force, frame_id=self.frame_id, ts=now
                    )
                )

            await asyncio.sleep(interval)

    def _resolve_ip(self) -> str:
        """Return a literal IPv4 address for the configured host.

        The SDK only speaks TCP when handed ``localhost`` or a dotted-quad
        IPv4; anything else it treats as a serial device path and dies with
        "serial module is not found, ... pip install pyserial", which says
        nothing about the real problem. Resolving here keeps hostnames usable
        and turns a bad value into an error that names it.
        """
        ip = (self.config.ip or "").strip()
        if not ip:
            raise RuntimeError(
                "XArmFTSensor: ip not set. Set it in the config or via DIMOS_XARM_IP "
                "environment variable."
            )
        if ip == "localhost" or _IPV4_RE.match(ip):
            return ip

        try:
            resolved = socket.gethostbyname(ip)
        except OSError as error:
            raise RuntimeError(
                f"XArmFTSensor: {ip!r} is neither an IPv4 address nor a resolvable "
                f"hostname ({error}). The xArm SDK connects over TCP only to a literal "
                "IP; anything else it tries to open as a serial port. Set the arm's "
                "address via DIMOS_XARM_IP or --ip."
            ) from error

        logger.info("XArmFTSensor: resolved %s to %s", ip, resolved)
        return resolved
