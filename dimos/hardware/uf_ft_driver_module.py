#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
uFactory FT Sensor Driver Module for Dimos

Publishes the xArm's built-in FT sensor on the exact same LCM contract
FTDriverModule uses for the homemade sensor (`force`/`torque` as
geometry_msgs/Vector3). Any downstream module written against that contract
-- FTPullModule, FTLoggerModule, FTVisualizerModule -- consumes this driver
with no changes, because none of them depend on how the wrench was produced.

What's structurally different from FTDriverModule, and why:
  - No calibration matrix: the uFactory sensor reports (Fx,Fy,Fz,Mx,My,Mz)
    directly, factory-calibrated. There's no 16-raw-channel stage.
  - `axis_transform` replaces the calibration matrix's role of getting
    numbers into the *convention this codebase expects* (as opposed to the
    physical units, which the sensor already provides correctly). This is
    exactly the "change of axes only" wrench transform derived in
    FT_CALIBRATION_MATH.md section 2 (F_B = R^T F_A, tau_B = R^T tau_A) --
    it defaults to identity and MUST NOT be trusted as correct until the
    section 8c poke test has actually measured it.
  - Zeroing is a live hardware call (set_ft_sensor_zero), not a fitted
    bias term, and it's gated behind an explicit flag because this module's
    start() runs in a worker process, not a TTY -- it cannot prompt to
    confirm the sensor is unloaded, so it doesn't zero unless told to.
"""

import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from dimos.core import Module, Out, rpc
from dimos.msgs.geometry_msgs import Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class UFactoryFTDriverModule(Module):
    """Drives the uFactory xArm's built-in FT sensor; same output contract as FTDriverModule."""

    force: Out[Vector3] = None  # Force vector in Newtons
    torque: Out[Vector3] = None  # Torque vector in Newton-meters

    def __init__(
        self,
        xarm_ip: Optional[str] = None,
        arm: Optional[Any] = None,
        sample_rate_hz: float = 100.0,
        window_size: int = 3,
        axis_transform: Optional[np.ndarray] = None,
        zero_on_start: bool = False,
        verbose: bool = False,
        frame_id: str = "uf_ft_sensor",
    ):
        """
        Args:
            xarm_ip: xArm IP address, used to open a new XArmAPI connection.
                Required if `arm` is not supplied.
            arm: an already-connected XArmAPI instance to reuse instead of
                opening a new one -- e.g. one a caller such as FTPullModule
                already owns for joint control. When given, this module
                never connects or disconnects it; the caller owns that
                connection's lifecycle for its whole life, not just while
                this module borrows it.
            sample_rate_hz: how often to poll get_ft_sensor_data().
            window_size: moving-average window over the 6-vector wrench.
                Same smoothing role as FTDriverModule's per-channel average
                -- kept configurable so the pull skill's control gains
                (tuned against the DIY sensor's smoothing/latency) can be
                matched deliberately instead of silently changing when the
                sensor changes.
            axis_transform: 3x3 rotation matrix mapping the uFactory
                sensor's raw axes into this codebase's convention. See the
                module docstring -- defaults to identity, unverified.
            zero_on_start: if True, calls set_ft_sensor_zero() once inside
                start(). Only set this when you have independently
                confirmed the end effector is unloaded at that moment.
            frame_id: label only, used in log messages.
        """
        super().__init__()
        if arm is None and xarm_ip is None:
            raise ValueError("UFactoryFTDriverModule needs either xarm_ip or an existing arm")

        self.xarm_ip = xarm_ip
        self._owns_arm = arm is None
        self.arm = arm
        self.sample_rate_hz = sample_rate_hz
        self.window_size = window_size
        self.axis_transform = (
            np.asarray(axis_transform, dtype=float) if axis_transform is not None else np.eye(3)
        )
        self.zero_on_start = zero_on_start
        self.verbose = verbose
        self.frame_id = frame_id

        self._force_buf = deque(maxlen=window_size)
        self._torque_buf = deque(maxlen=window_size)

        self.message_count = 0
        self.error_count = 0
        self.latest_force_mag = 0.0
        self.latest_torque_mag = 0.0

        self.running = False
        self._thread = None

    def _connect(self) -> bool:
        """Open our own xArm connection, unless one was supplied to reuse."""
        if not self._owns_arm:
            logger.info("Reusing externally-supplied xArm connection")
            return self.arm is not None

        try:
            from xarm.wrapper import XArmAPI

            logger.info(f"Connecting to xArm at {self.xarm_ip}...")
            self.arm = XArmAPI(self.xarm_ip, do_not_open=False, is_radian=True)
            self.arm.clean_error()
            self.arm.clean_warn()
            logger.info("Connected.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to xArm at {self.xarm_ip}: {e}")
            return False

    def _enable_ft_sensor(self) -> bool:
        """
        Enable (and optionally zero) the FT sensor. Call shape here matches
        what ft_ground_truth_smoke_test.py checks -- if that script reports
        a different signature for your SDK version, update both together.
        """
        try:
            self.arm.set_ft_sensor_enable(1)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"set_ft_sensor_enable failed: {e}")
            return False

        if self.zero_on_start:
            logger.warning(
                "zero_on_start=True: zeroing the FT sensor now. This assumes the end "
                "effector is UNLOADED at this exact moment -- that has NOT been verified "
                "by this module, only by whatever told you to set this flag."
            )
            try:
                self.arm.set_ft_sensor_zero()
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"set_ft_sensor_zero failed: {e}")
                return False

        return True

    def _read_once(self) -> Optional[np.ndarray]:
        """One raw (Fx,Fy,Fz,Mx,My,Mz) reading in the sensor's own axes, or None on failure."""
        try:
            code, data = self.arm.get_ft_sensor_data()
        except Exception as e:
            if self.verbose:
                logger.warning(f"get_ft_sensor_data() raised: {e}")
            self.error_count += 1
            return None

        if code != 0 or data is None or len(data) < 6:
            self.error_count += 1
            return None

        return np.asarray(data[:6], dtype=float)

    def read_and_process(self):
        """Read once, apply the axis transform and moving average, publish."""
        raw = self._read_once()
        if raw is None:
            return

        force_raw, torque_raw = raw[:3], raw[3:]

        # FT_CALIBRATION_MATH.md section 2, pure-rotation case: F_B = R^T F_A.
        # Identity (no-op) until section 8c's poke test measures the real
        # axis correspondence.
        force = self.axis_transform.T @ force_raw
        torque = self.axis_transform.T @ torque_raw

        self._force_buf.append(force)
        self._torque_buf.append(torque)
        force_avg = np.mean(self._force_buf, axis=0)
        torque_avg = np.mean(self._torque_buf, axis=0)

        self.latest_force_mag = float(np.linalg.norm(force_avg))
        self.latest_torque_mag = float(np.linalg.norm(torque_avg))

        self.force.publish(Vector3(*force_avg))
        self.torque.publish(Vector3(*torque_avg))
        self.message_count += 1

        if self.verbose:
            logger.debug(
                f"F=({force_avg[0]:7.2f},{force_avg[1]:7.2f},{force_avg[2]:7.2f}) "
                f"T=({torque_avg[0]:7.4f},{torque_avg[1]:7.4f},{torque_avg[2]:7.4f})"
            )

    def _run_loop(self):
        period = 1.0 / self.sample_rate_hz
        logger.info(f"uFactory FT driver loop started ({self.sample_rate_hz:.0f} Hz target)")
        last_log = time.time()

        while self.running:
            t0 = time.time()
            self.read_and_process()

            if time.time() - last_log > 5.0:
                logger.info(
                    f"uFactory FT driver status: {self.message_count} reads, "
                    f"{self.error_count} errors"
                )
                last_log = time.time()

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        logger.info(f"uFactory FT driver loop stopped after {self.message_count} reads")

    @rpc
    def start(self):
        """Start the sensor driver."""
        if self.running:
            logger.warning("uFactory FT driver already running")
            return True

        logger.info("Starting uFactory FT driver module...")
        logger.info(f"  xArm IP: {self.xarm_ip or '<using supplied connection>'}")
        logger.info(f"  Sample rate: {self.sample_rate_hz} Hz, window: {self.window_size}")
        logger.info(f"  Zero on start: {self.zero_on_start}")

        if not self._connect():
            logger.error("CRITICAL: could not obtain an xArm connection")
            return False
        if not self._enable_ft_sensor():
            logger.error("CRITICAL: could not enable the FT sensor")
            return False

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.1)

        if not self._thread.is_alive():
            logger.error("uFactory FT driver thread failed to start!")
            self.running = False
            return False

        logger.info("uFactory FT driver started successfully")
        return True

    @rpc
    def zero(self) -> bool:
        """Zero the FT sensor now. Caller is responsible for confirming an unloaded state."""
        if not self.arm:
            logger.error("Cannot zero: no xArm connection")
            return False
        try:
            self.arm.set_ft_sensor_zero()
            logger.info("uFactory FT sensor zeroed")
            return True
        except Exception as e:
            logger.error(f"Failed to zero FT sensor: {e}")
            return False

    @rpc
    def stop(self):
        """Stop the sensor driver."""
        if not self.running:
            return

        logger.info("Stopping uFactory FT driver...")
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._owns_arm and self.arm:
            try:
                self.arm.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting xArm: {e}")

        logger.info(
            f"uFactory FT driver stopped. Reads={self.message_count}, Errors={self.error_count}"
        )

    @rpc
    def get_stats(self) -> Dict[str, Any]:
        """Get driver statistics."""
        return {
            "message_count": self.message_count,
            "error_count": self.error_count,
            "xarm_connected": self.arm is not None,
            "latest_force_magnitude": self.latest_force_mag,
            "latest_torque_magnitude": self.latest_torque_mag,
        }


if __name__ == "__main__":
    # For testing standalone
    import argparse

    parser = argparse.ArgumentParser(description="uFactory FT Driver Module")
    parser.add_argument("--xarm", required=True, help="xArm IP address")
    parser.add_argument("--rate", type=float, default=100.0, help="Sample rate (Hz)")
    parser.add_argument("--window", type=int, default=3, help="Moving average window size")
    parser.add_argument(
        "--zero-on-start",
        action="store_true",
        help="Zero the FT sensor at startup -- only pass this if the end effector is unloaded",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    from dimos.core import start

    dimos = start(1)
    driver = dimos.deploy(
        UFactoryFTDriverModule,
        xarm_ip=args.xarm,
        sample_rate_hz=args.rate,
        window_size=args.window,
        zero_on_start=args.zero_on_start,
        verbose=args.verbose,
    )

    driver.start()
