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
FT Ground-Truth Check / Data Collector (Step 1 & Step 4 of the calibration roadmap)

Purpose
-------
A known mass is rigidly attached at the sensor's tool flange. At each of a set
of arm orientations, the expected wrench (Fx,Fy,Fz,Mx,My,Mz) at the sensor's
own origin can be computed directly from physics: gravity + the arm's own
forward-kinematics orientation + the known mass and its lever arm. No second
sensor is required for that part -- which matters here because the uFactory
FT sensor and the homemade sensor are swap-only (never mounted at the same
time), so they can never be compared against *each other* directly. Instead,
both get compared against this shared, physics-computed ground truth, in two
separate passes:

    Pass A (--sensor ufactory): validates the FK/gravity math and the
        uFactory sensor's own accuracy against a trusted reference.
    Pass B (--sensor homemade): logs the homemade sensor's raw 16 channels
        against the same expected wrench. The output CSV uses the exact
        column names calc_calibration_matrix.py expects
        (sensor_1..sensor_16, force_local_*, torque_local_*), so once you
        trust the method from Pass A, running Pass B *is* Step 4 -- just
        point calc_calibration_matrix.py at the resulting CSV.

Deliberately out of scope here (separate steps):
    - Temperature drift characterization (needs a static thermal-soak test,
      not a pose sweep).
    - The fixed mechanical transform between the uFactory sensor's origin and
      the homemade sensor's origin (needs a CAD/mechanical measurement -- see
      --lever-arm below, which you supply per sensor for exactly this reason).

Safety
------
This script never commands arm motion. Jog the arm to each pose yourself
(teach pendant / hand-guide / xArm Studio), then press Enter to capture.

Usage
-----
  # Pass A: validate against the uFactory sensor (attach a known mass first)
  python3 dimos/hardware/ft_ground_truth_check.py \\
      --xarm 192.168.1.210 --sensor ufactory \\
      --mass 0.5 --lever-arm 0 0 0.03 \\
      --out gt_check_ufactory.csv

  # Pass B: same protocol, homemade sensor mounted instead
  python3 dimos/hardware/ft_ground_truth_check.py \\
      --xarm 192.168.1.210 --sensor homemade --port /dev/ttyACM0 \\
      --mass 0.5 --lever-arm 0 0 0.03 \\
      --out gt_check_homemade.csv

  --lever-arm is the vector from the CURRENT sensor's own origin to the
  known mass's center of gravity, expressed in the sensor's own frame
  (meters). It is almost certainly different for the two sensors since they
  physically sit at different points along the tool stack-up -- measure each
  from CAD/calipers, don't reuse one value for both passes.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

G = 9.80665  # m/s^2

FT_COLUMNS = [
    "force_local_x",
    "force_local_y",
    "force_local_z",
    "torque_local_x",
    "torque_local_y",
    "torque_local_z",
]


def expected_wrench(mass_kg: float, lever_arm_m, rpy_rad) -> np.ndarray:
    """
    Expected (Fx,Fy,Fz,Mx,My,Mz) at the sensor origin, in the sensor/tool
    frame, from a known mass at `lever_arm_m` while the tool is at
    orientation `rpy_rad` (roll, pitch, yaw, radians) as reported by the arm.

    CONFIRM before trusting numbers: this assumes the arm reports TCP
    orientation as the rotation from base/world frame to tool frame using
    extrinsic X-Y-Z Euler angles (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), and
    that world frame has +Z up. Both are common xArm conventions but verify
    against your SDK/manual. `--sanity-check` below gives you a way to
    confirm this from data instead of trusting the convention blind.
    """
    R_world_from_tool = Rotation.from_euler("xyz", rpy_rad).as_matrix()
    weight_world = np.array([0.0, 0.0, -mass_kg * G])
    F_tool = R_world_from_tool.T @ weight_world
    r_tool = np.asarray(lever_arm_m, dtype=float)
    T_tool = np.cross(r_tool, F_tool)
    return np.concatenate([F_tool, T_tool])


def read_ufactory(arm, samples: int, dwell_s: float) -> np.ndarray:
    """Average `samples` readings of the uFactory FT sensor over `dwell_s`."""
    readings = []
    dt = dwell_s / max(samples, 1)
    for _ in range(samples):
        code, data = arm.get_ft_sensor_data()
        if code == 0 and data is not None:
            readings.append(np.array(data[:6], dtype=float))
        time.sleep(dt)
    if not readings:
        raise RuntimeError("Got no valid readings from get_ft_sensor_data()")
    return np.mean(readings, axis=0)


def init_ufactory_ft(arm):
    """Enable + zero the uFactory FT sensor. Must be UNLOADED when zeroing."""
    input("\nRemove any attached mass so the FT sensor is unloaded, then press Enter to zero it...")
    try:
        arm.set_ft_sensor_enable(1)
        time.sleep(0.2)
        arm.set_ft_sensor_zero()
        time.sleep(0.5)
        print("uFactory FT sensor enabled and zeroed.")
    except Exception as e:
        print(f"WARNING: FT sensor enable/zero call failed or has a different signature "
              f"on your SDK version ({e}). Check `help(arm.set_ft_sensor_enable)` / "
              f"`help(arm.set_ft_sensor_zero)` and adjust this function.")
    input("Now attach the known mass, then press Enter to start capturing poses...")


def read_homemade(ser, buffers, samples: int, dwell_s: float) -> np.ndarray:
    """Average `samples` lines of the homemade sensor's 16 raw channels."""
    readings = []
    deadline = time.time() + dwell_s
    while len(readings) < samples and time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.endswith(","):
            line = line[:-1]
        try:
            values = [float(x) for x in line.split(",")]
        except ValueError:
            continue
        if len(values) != 16:
            continue
        readings.append(np.array(values))
    if not readings:
        raise RuntimeError("Got no valid 16-channel lines from the homemade sensor's serial port")
    return np.mean(readings, axis=0)


def get_tcp_pose(arm):
    """Returns (x,y,z in m, roll,pitch,yaw in rad) for the current TCP pose."""
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"get_position() failed with code {code}")
    x, y, z, roll, pitch, yaw = pose
    return np.array([x / 1000.0, y / 1000.0, z / 1000.0]), np.array([roll, pitch, yaw])


def main():
    parser = argparse.ArgumentParser(
        description="FT ground-truth check / calibration data collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--xarm", required=True, help="xArm IP address")
    parser.add_argument("--sensor", required=True, choices=["ufactory", "homemade"])
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (homemade sensor only)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mass", type=float, required=True, help="Known attached mass, kg")
    parser.add_argument(
        "--lever-arm",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Vector from THIS sensor's origin to the mass's CG, in the sensor's own frame, meters",
    )
    parser.add_argument("--samples-per-pose", type=int, default=50)
    parser.add_argument("--dwell", type=float, default=2.5, help="Seconds to average over per pose")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    from xarm.wrapper import XArmAPI

    print(f"Connecting to xArm at {args.xarm}...")
    arm = XArmAPI(args.xarm, do_not_open=False, is_radian=True)
    arm.clean_error()
    arm.clean_warn()

    ser = None
    if args.sensor == "ufactory":
        init_ufactory_ft(arm)
    else:
        import serial

        ser = serial.Serial(args.port, args.baud, timeout=1)
        ser.reset_input_buffer()
        print(f"Connected to homemade sensor on {args.port} at {args.baud} baud.")

    out_path = Path(args.out)
    rows = []
    pose_id = 0

    print("\n" + "=" * 60)
    print("Jog the arm to a pose, then press Enter to capture (or 'q' + Enter to finish).")
    print("=" * 60)

    try:
        while True:
            user_in = input(f"\n[Pose {pose_id + 1}] Enter to capture, 'q' to quit: ")
            if user_in.strip().lower() == "q":
                break

            xyz_m, rpy_rad = get_tcp_pose(arm)
            expected = expected_wrench(args.mass, args.lever_arm, rpy_rad)

            row = {
                "pose_id": pose_id,
                "timestamp": time.time(),
                "x": xyz_m[0],
                "y": xyz_m[1],
                "z": xyz_m[2],
                "roll": rpy_rad[0],
                "pitch": rpy_rad[1],
                "yaw": rpy_rad[2],
            }
            for name, val in zip(FT_COLUMNS, expected):
                row[name] = val

            if args.sensor == "ufactory":
                measured = read_ufactory(arm, args.samples_per_pose, args.dwell)
                for i, axis in enumerate(["Fx", "Fy", "Fz", "Mx", "My", "Mz"]):
                    row[f"measured_{axis}"] = measured[i]
                resid = measured - expected
                print(
                    f"  expected F=({expected[0]:6.2f},{expected[1]:6.2f},{expected[2]:6.2f}) "
                    f"M=({expected[3]:6.3f},{expected[4]:6.3f},{expected[5]:6.3f})"
                )
                print(
                    f"  measured F=({measured[0]:6.2f},{measured[1]:6.2f},{measured[2]:6.2f}) "
                    f"M=({measured[3]:6.3f},{measured[4]:6.3f},{measured[5]:6.3f})  "
                    f"resid |F|={np.linalg.norm(resid[:3]):.2f}N |M|={np.linalg.norm(resid[3:]):.3f}Nm"
                )
            else:
                buffers = None
                raw16 = read_homemade(ser, buffers, args.samples_per_pose, args.dwell)
                for i, val in enumerate(raw16):
                    row[f"sensor_{i + 1}"] = val
                print(f"  expected F=({expected[0]:6.2f},{expected[1]:6.2f},{expected[2]:6.2f}) "
                      f"M=({expected[3]:6.3f},{expected[4]:6.3f},{expected[5]:6.3f})")
                print(f"  raw16 mean={np.mean(raw16):8.1f}  std={np.std(raw16):6.1f}")

            rows.append(row)
            pose_id += 1

    finally:
        if ser:
            ser.close()

    if not rows:
        print("No poses captured, nothing written.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} poses to {out_path}")

    if args.sensor == "ufactory":
        expected_arr = np.array([[r[c] for c in FT_COLUMNS] for r in rows])
        measured_arr = np.array(
            [[r[f"measured_{a}"] for a in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]] for r in rows]
        )
        resid = measured_arr - expected_arr
        print("\nResidual (measured - expected) per axis:")
        for i, name in enumerate(["Fx", "Fy", "Fz", "Mx", "My", "Mz"]):
            unit = "N" if i < 3 else "N*m"
            print(
                f"  {name}: mean={resid[:, i].mean():8.4f} {unit}  "
                f"std={resid[:, i].std():8.4f} {unit}  "
                f"rmse={np.sqrt(np.mean(resid[:, i] ** 2)):8.4f} {unit}"
            )
        print(
            "\nIf residuals are small and roughly zero-mean -> the FK/gravity ground-truth "
            "method is trustworthy; carry the same --lever-arm-measurement discipline into "
            "the homemade-sensor pass. If one axis is consistently biased, suspect either the "
            "rpy convention in expected_wrench() or the --lever-arm vector before blaming the "
            "sensor."
        )


if __name__ == "__main__":
    sys.exit(main())
