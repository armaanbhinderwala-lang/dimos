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

"""Record a calibration CSV for the OpenFT 16-channel sensor.

Pairs the sensor's 16 raw channels with a known applied wrench, one row per
sample, in the 22-column layout ``calc_calibration_matrix.py`` expects:

    sensor_1..sensor_16, force_local_x/y/z, torque_local_x/y/z

Workflow: load the sensor in a known way, type the applied wrench, and the
harness captures N settled samples. Repeat across enough distinct loadings to
span all six axes, then fit with::

    python dimos/hardware/calc_calibration_matrix.py \\
        --csv <this file> --out dimos/hardware/ft_calibration.json

On exit it reports per-axis excitation. This matters: the shipped
ft_calibration.json has an all-zero ``tz`` row because its source data never
twisted the sensor about Z, so ``tz`` reads a constant no matter what the
sensor does. An axis you never load here will be dead there too.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from dimos.hardware.sensors.force_torque.openft_module import (
    AXIS_NAMES,
    CHANNELS,
    parse_frame,
)

WRENCH_COLUMNS = [
    "force_local_x",
    "force_local_y",
    "force_local_z",
    "torque_local_x",
    "torque_local_y",
    "torque_local_z",
]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, CHANNELS + 1)]
COLUMNS = SENSOR_COLUMNS + WRENCH_COLUMNS

# Below this an axis is treated as never loaded. Sensor noise and a hand
# resting on the fixture produce far less than this in N / N·m.
EXCITED = 1e-6


def capture(link: Any, count: int, settle: float) -> list[list[float]]:
    """Return `count` good frames, discarding whatever arrived while settling."""
    time.sleep(settle)
    link.reset_input_buffer()

    rows: list[list[float]] = []
    dropped = 0
    while len(rows) < count:
        try:
            line = link.readline().decode("utf-8", errors="replace")
        except Exception as error:
            print(f"  serial read failed: {error}", file=sys.stderr)
            break
        if not line:
            print("  timed out waiting for a frame -- is the MCU streaming?", file=sys.stderr)
            break
        parsed = parse_frame(line)
        if parsed is None:
            dropped += 1
            continue
        rows.append(parsed)

    if dropped:
        print(f"  ({dropped} malformed frames skipped)")
    return rows


def prompt_wrench(index: int) -> list[float] | None:
    """Read one applied wrench from the operator. None ends the session."""
    print(f"\n--- point {index} ---")
    print("Apply a known load, then enter it as: fx fy fz tx ty tz")
    print("(forces N, torques N·m, sensor frame; blank line or 'q' to finish)")
    while True:
        try:
            raw = input("wrench> ").strip()
        except EOFError:
            return None
        if not raw or raw.lower() in {"q", "quit", "done"}:
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) != 6:
            print(f"  need 6 numbers, got {len(parts)}")
            continue
        try:
            return [float(p) for p in parts]
        except ValueError:
            print("  could not parse those as numbers")


def report_coverage(rows: list[list[float]]) -> None:
    """Warn about axes that were never meaningfully loaded."""
    if not rows:
        return
    wrenches = np.array([r[CHANNELS:] for r in rows], dtype=float)
    spans = np.abs(wrenches).max(axis=0)

    print("\nper-axis excitation across this file:")
    dead = []
    for name, span, col in zip(AXIS_NAMES, spans, wrenches.T, strict=True):
        distinct = len(np.unique(np.round(col, 6)))
        flag = ""
        if span <= EXCITED:
            dead.append(name)
            flag = "  <-- NEVER LOADED"
        elif distinct < 3:
            flag = f"  <-- only {distinct} distinct values"
        print(f"  {name}: max|applied|={span:9.4f}  distinct={distinct}{flag}")

    if dead:
        print(
            f"\nWARNING: {', '.join(dead)} was never loaded. Least squares will return an "
            f"all-zero row for {'it' if len(dead) == 1 else 'those'}, and the driver will "
            f"report a constant on {'that axis' if len(dead) == 1 else 'those axes'} "
            "forever. Add points that load them before fitting."
        )
    else:
        print("\nAll six axes were loaded.")


def monitor(link: Any, samples: int, noise_sigma: float) -> None:
    """Show live per-channel deviation from an unloaded baseline.

    Answers the question the calibrated output cannot: does the hardware
    respond to this load at all? The calibration matrix can force an axis to
    zero regardless of the sensor, so a dead axis has to be diagnosed on the
    raw channels, upstream of it.
    """
    print(f"Leave the sensor UNLOADED. Capturing {samples} baseline frames...")
    base_rows = capture(link, samples, 1.0)
    if not base_rows:
        print("no frames captured -- is the MCU streaming?", file=sys.stderr)
        return

    base = np.array(base_rows, dtype=float)
    mean, sigma = base.mean(axis=0), base.std(axis=0)
    # Floor the per-channel noise estimate: a channel that sat perfectly still
    # for the baseline would otherwise make every later wobble look significant.
    floor = np.maximum(sigma, noise_sigma)
    print(f"baseline captured (per-channel sigma {sigma.min():.4g}..{sigma.max():.4g})")
    print("\nNow apply the load you want to test -- e.g. twist about Z.")
    print("Channels moving well beyond their noise band are responding. Ctrl-C to stop.\n")

    try:
        while True:
            rows = capture(link, max(3, samples // 4), 0.0)
            if not rows:
                break
            now = np.array(rows, dtype=float).mean(axis=0)
            delta = now - mean
            ratio = np.abs(delta) / floor
            cells = " ".join(
                f"{i + 1:>2}:{d:+8.2f}{'*' if r > 5 else ' '}"
                for i, (d, r) in enumerate(zip(delta, ratio, strict=True))
            )
            live = int((ratio > 5).sum())
            print(f"[{live:2d}/16 responding] {cells}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="CSV to write (appends if it exists)")
    ap.add_argument(
        "--monitor",
        action="store_true",
        help="diagnostic: show raw per-channel response to a load instead of recording",
    )
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=1e-3,
        help="minimum per-channel sigma assumed in --monitor (default: %(default)s)",
    )
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial device (default: %(default)s)")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate (default: %(default)s)")
    ap.add_argument(
        "--samples", type=int, default=20, help="frames per point (default: %(default)s)"
    )
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to settle before capturing")
    ap.add_argument("--timeout", type=float, default=2.0, help="serial read timeout (s)")
    args = ap.parse_args()

    if not args.monitor and args.out is None:
        ap.error("--out is required unless --monitor is given")

    import serial

    try:
        link = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as error:
        print(f"could not open {args.port}: {error}", file=sys.stderr)
        print(
            "if this is a permission error, add yourself to the dialout group:\n"
            "  sudo usermod -aG dialout $USER   (then log out and back in)",
            file=sys.stderr,
        )
        return 1

    if args.monitor:
        try:
            monitor(link, args.samples, args.noise_floor)
        finally:
            link.close()
        return 0

    # Append so a session can be resumed, but only write the header once.
    existing: list[list[float]] = []
    if args.out.exists():
        with args.out.open() as fh:
            existing = [
                [float(v) for v in row]
                for row in csv.reader(fh)
                if row and not row[0].startswith("sensor_1")
            ]
        print(f"appending to {args.out} ({len(existing)} existing rows)")

    written = 0
    collected: list[list[float]] = list(existing)
    try:
        with args.out.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if not existing:
                writer.writerow(COLUMNS)

            index = 1
            while True:
                wrench = prompt_wrench(index)
                if wrench is None:
                    break
                print(f"  capturing {args.samples} frames...")
                frames = capture(link, args.samples, args.settle)
                if not frames:
                    print("  no frames captured, point skipped")
                    continue
                for frame in frames:
                    row = frame + wrench
                    writer.writerow([f"{v:.6f}" for v in row])
                    collected.append(row)
                fh.flush()
                written += len(frames)
                print(f"  wrote {len(frames)} rows (total this session: {written})")
                index += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        link.close()

    print(f"\n{written} rows appended to {args.out} ({len(collected)} total)")
    report_coverage(collected)
    if collected:
        print(
            "\nfit with:\n"
            f"  python dimos/hardware/calc_calibration_matrix.py --csv {args.out} "
            "--out dimos/hardware/ft_calibration.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
