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

"""Module wrapper for the homemade OpenFT 16-channel magnetic force-torque sensor.

A microcontroller (Seeeduino XIAO) streams 16 comma-separated channel readings
over serial. This module averages them, maps them through a 6x16 calibration
matrix, and publishes the result as :class:`WrenchStamped` -- the same message
:class:`XArmFTSensor` emits, so both sensors feed the same recorder and plotter.

Ported from the older ``dimos/hardware/ft_driver_module.py``, which published a
pair of bare ``Vector3`` topics against a ``dimos.core`` namespace that no
longer exists. The serial framing, moving average and calibration maths are
carried over unchanged; only the module API and output message differ.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

CHANNELS = 16  # magnetic channels the MCU streams per line
AXES = 6  # fx, fy, fz, tx, ty, tz
AXIS_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")

# A calibration row this much smaller than the largest one contributes nothing
# measurable. Real rows sit within ~2 orders of magnitude of each other; an
# unexcited axis comes back ~20 orders down.
DEAD_ROW_RATIO = 1e-12


def parse_frame(line: str) -> list[float] | None:
    """Parse one serial frame into 16 channel readings, or None if malformed.

    Shared with the calibration collector so the driver and the data it is
    calibrated against can never disagree about the wire format.
    """
    # The firmware terminates each frame with a trailing comma. Newer firmware also
    # prepends a microsecond timestamp, so accept both widths -- a driver that rejects
    # every frame after a flash goes silent rather than failing, which is how a whole
    # session ran with a dead sensor.
    values = [v for v in line.strip().rstrip(",").split(",") if v]
    if len(values) == CHANNELS + 1:
        values = values[1:]
    if len(values) != CHANNELS:
        return None
    try:
        return [float(v) for v in values]
    except ValueError:
        return None


class OpenFTSensorConfig(ModuleConfig):
    """Pydantic configuration parameters for the OpenFT sensor module."""

    # Serial link to the MCU. The device needs to be readable by this user --
    # it is root:dialout by default, so the account must be in `dialout`.
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200

    # Serial read timeout (s). Bounds how long a stalled MCU blocks the worker
    # thread before the loop gets a chance to notice `_running` went false.
    timeout: float = 1.0

    # Moving-average window applied per channel before calibration.
    window_size: int = Field(default=3, ge=1)

    # 6x16 calibration matrix + optional 6-vector bias, as written by
    # dimos/hardware/calc_calibration_matrix.py. Without it there is no way to
    # turn 16 raw channels into a wrench, so start() refuses to run.
    calibration_file: str | Path = "dimos/hardware/ft_calibration.json"

    frame_id: str = "ft_sensor_link"


class OpenFTSensor(Module):
    """Streams calibrated wrenches from the homemade 16-channel FT sensor.

    Both ports carry the same calibration matrix applied to the same samples;
    they differ only in bias. ``raw_wrench`` is the bare matrix product and
    ``ext_wrench`` has the calibration bias removed, mirroring the raw vs
    compensated split :class:`XArmFTSensor` publishes so downstream modules
    (recorder, plotter) connect to either sensor unchanged.
    """

    config: OpenFTSensorConfig

    ext_wrench: Out[WrenchStamped]
    raw_wrench: Out[WrenchStamped]

    _serial: object | None = None
    _running: bool = False
    _dropped: int = 0

    @rpc
    def start(self) -> None:
        """Open the serial link, load calibration, and start streaming."""
        import serial

        matrix, bias = self._load_calibration()
        self._matrix, self._bias = matrix, bias
        self._buffers = [deque[float](maxlen=self.config.window_size) for _ in range(CHANNELS)]

        super().start()

        try:
            link = serial.Serial(
                self.config.serial_port, self.config.baud_rate, timeout=self.config.timeout
            )
        except serial.SerialException as error:
            raise RuntimeError(
                f"OpenFTSensor: could not open {self.config.serial_port} ({error}). Check the "
                "MCU is plugged in and that this user is in the 'dialout' group "
                "(sudo usermod -aG dialout $USER, then log back in)."
            ) from error

        link.reset_input_buffer()
        self._serial = link
        self._running = True

        self.spawn(self._telemetry_loop())
        logger.info(
            "OpenFTSensor streaming from %s at %d baud (frame %s, window %d)",
            self.config.serial_port,
            self.config.baud_rate,
            self.config.frame_id,
            self.config.window_size,
        )

    @rpc
    def stop(self) -> None:
        """Close the serial link."""
        self._running = False
        link, self._serial = self._serial, None
        if link is not None:
            link.close()  # type: ignore[attr-defined]
        super().stop()

    def _load_calibration(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Read the 6x16 matrix (and optional bias) from JSON or NPZ."""
        path = Path(self.config.calibration_file)
        if not path.exists():
            raise RuntimeError(
                f"OpenFTSensor: calibration file {path} not found. Generate one with "
                "dimos/hardware/calc_calibration_matrix.py; without it the 16 raw "
                "channels cannot be resolved into a wrench."
            )

        if path.suffix == ".npz":
            data = np.load(path)
            matrix = np.asarray(data["calibration_matrix"], dtype=float)
            raw_bias = data["bias_vector"] if "bias_vector" in data else None
        else:
            payload = json.loads(path.read_text())
            matrix = np.asarray(payload["calibration_matrix"], dtype=float)
            raw_bias = payload.get("bias_vector")

        if matrix.shape != (AXES, CHANNELS):
            raise RuntimeError(
                f"OpenFTSensor: calibration matrix in {path} is {matrix.shape}, "
                f"expected ({AXES}, {CHANNELS})."
            )

        bias = None if raw_bias is None else np.asarray(raw_bias, dtype=float)
        if bias is not None and bias.shape != (AXES,):
            raise RuntimeError(
                f"OpenFTSensor: bias vector in {path} is {bias.shape}, expected ({AXES},)."
            )

        # A vanishing row pins that axis to a constant for every input. lstsq
        # returns one whenever the calibration CSV never excited the axis, so
        # the output looks like a dead sensor channel when it is really missing
        # calibration data. Say so at startup rather than let it read as zero.
        #
        # Tested against the largest row rather than np.any(): lstsq leaves
        # denormal dust (~1e-22) rather than exact zeros, which np.any() reads
        # as a live row.
        norms = np.linalg.norm(matrix, axis=1)
        dead = [AXIS_NAMES[i] for i in range(AXES) if norms[i] <= DEAD_ROW_RATIO * norms.max()]
        if dead:
            logger.warning(
                "OpenFTSensor: calibration %s has all-zero row(s) for %s -- those axes will "
                "read a constant (bias only, 0.0 if the bias is also zero) no matter what the "
                "sensor does. Re-record the calibration CSV with load applied about %s and "
                "re-run calc_calibration_matrix.py.",
                path,
                ", ".join(dead),
                ", ".join(dead),
            )

        logger.info("OpenFTSensor: calibration loaded from %s (bias: %s)", path, bias is not None)
        return matrix, bias

    def _read_sample(self) -> np.ndarray | None:
        """Blocking read of one line; returns the per-channel moving average.

        Runs on a worker thread -- pyserial has no async API, and readline()
        would otherwise stall the event loop for up to `timeout` seconds.
        """
        link = self._serial
        if link is None:
            return None

        try:
            line = link.readline().decode("utf-8").strip()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("OpenFTSensor: serial read failed", exc_info=True)
            return None

        if not line:
            return None

        parsed = parse_frame(line)
        if parsed is None:
            # Warn once a run, not once a frame. Silent per-frame drops let a firmware
            # change take the sensor offline without anything appearing in the log.
            self._dropped += 1
            if self._dropped in (1, 100) or self._dropped % 1000 == 0:
                logger.warning(
                    "OpenFTSensor: %d frames rejected -- the sensor is not being read. "
                    "Wire format may have changed. Last line: %r",
                    self._dropped, line[:120],
                )
            return None
        self._dropped = 0

        for buffer, value in zip(self._buffers, parsed, strict=True):
            buffer.append(value)
        return np.array([float(np.mean(b)) for b in self._buffers])

    async def _telemetry_loop(self) -> None:
        """Publish a wrench per serial frame; the MCU sets the rate."""
        while self._running:
            channels = await asyncio.to_thread(self._read_sample)
            if channels is None:
                if not self._running:
                    break
                continue

            now = time.time()

            # F = C @ s, the direct matrix product, before bias removal.
            wrench = self._matrix @ channels
            self.raw_wrench.publish(
                WrenchStamped.from_array(wrench, frame_id=self.config.frame_id, ts=now)
            )

            if self._bias is not None:
                wrench = wrench + self._bias
            self.ext_wrench.publish(
                WrenchStamped.from_array(wrench, frame_id=self.config.frame_id, ts=now)
            )
