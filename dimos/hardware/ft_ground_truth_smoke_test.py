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
FT Ground-Truth Smoke Test (run BEFORE ft_ground_truth_check.py)

Confirms, on your actual SDK/firmware versions, the things
ft_ground_truth_check.py currently has to assume:
    - What arm.get_position(is_radian=True) actually returns (order, units).
    - The real call signatures of the ft_sensor_* methods on your installed
      xarm-python-sdk (printed via inspect, not guessed).
    - Whether get_ft_sensor_data() / set_ft_sensor_enable() / set_ft_sensor_zero()
      actually behave as expected on your arm.
    - (Optional, --port) whether the homemade sensor's serial stream parses
      as 16 comma-separated values in the firmware's documented 9000-21000
      raw range.

Does not move the arm. Does not require a mass attached. Prints a PASS/FAIL/
UNKNOWN checklist at the end -- fix anything FAILing before trusting
ft_ground_truth_check.py's output.

Usage:
  python3 dimos/hardware/ft_ground_truth_smoke_test.py --xarm 192.168.1.210
  python3 dimos/hardware/ft_ground_truth_smoke_test.py --xarm 192.168.1.210 --port /dev/ttyACM0
"""

import argparse
import inspect
import sys
import time


def check(results, name, fn):
    """Run fn(), print what happened, record PASS/FAIL in results."""
    print(f"\n--- {name} ---")
    try:
        fn()
        results[name] = "PASS"
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        results[name] = "FAIL"


def print_signature(label, method):
    try:
        print(f"  {label} signature: {inspect.signature(method)}")
    except (TypeError, ValueError):
        print(f"  {label} signature: <could not introspect, likely a C extension/builtin>")
    doc = inspect.getdoc(method)
    if doc:
        print(f"  {label} docstring:\n    " + doc.replace("\n", "\n    "))
    else:
        print(f"  {label} docstring: <none>")


def main():
    parser = argparse.ArgumentParser(description="FT ground-truth smoke test")
    parser.add_argument("--xarm", required=True, help="xArm IP address")
    parser.add_argument("--port", default=None, help="Homemade sensor serial port (optional)")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    results = {}

    from xarm.wrapper import XArmAPI

    print(f"Connecting to xArm at {args.xarm}...")
    arm = XArmAPI(args.xarm, do_not_open=False, is_radian=True)
    arm.clean_error()
    arm.clean_warn()
    results["connect"] = "PASS"
    print("  Connected.")
    print(f"  SDK version reported: {getattr(arm, 'version', '<no .version attribute>')}")

    def _position():
        print_signature("get_position", arm.get_position)
        code, pose_rad = arm.get_position(is_radian=True)
        print(f"  get_position(is_radian=True)  -> code={code}, pose={pose_rad}")
        code2, pose_deg = arm.get_position(is_radian=False)
        print(f"  get_position(is_radian=False) -> code={code2}, pose={pose_deg}")
        if code != 0:
            raise RuntimeError(f"non-zero return code {code}")
        print(
            "  CONFIRM: is this [x, y, z, roll, pitch, yaw]? Are x/y/z in mm? "
            "Does the radian version's roll/pitch/yaw look like the degree version / 57.3?"
        )

    check(results, "get_position", _position)

    def _ft_signatures():
        for name in [
            "set_ft_sensor_enable",
            "set_ft_sensor_mode",
            "get_ft_sensor_mode",
            "set_ft_sensor_zero",
            "get_ft_sensor_data",
            "get_ft_sensor_config",
            "get_ft_sensor_error",
        ]:
            method = getattr(arm, name, None)
            if method is None:
                print(f"  {name}: NOT FOUND on this XArmAPI instance")
                continue
            print_signature(name, method)

    check(results, "ft_sensor_signatures", _ft_signatures)

    def _ft_read_before_enable():
        code, data = arm.get_ft_sensor_data()
        print(f"  get_ft_sensor_data() before enable -> code={code}, data={data}")

    check(results, "ft_read_before_enable", _ft_read_before_enable)

    def _ft_enable():
        ret = arm.set_ft_sensor_enable(1)
        print(f"  set_ft_sensor_enable(1) -> {ret}")
        time.sleep(0.3)
        code, data = arm.get_ft_sensor_data()
        print(f"  get_ft_sensor_data() after enable -> code={code}, data={data}")
        if data is not None:
            print(f"  len(data)={len(data)} (expect 6: Fx,Fy,Fz,Mx,My,Mz)")

    check(results, "ft_enable_and_read", _ft_enable)

    def _ft_zero():
        confirm = input(
            "\n  Confirm the end effector is COMPLETELY UNLOADED (no mass, nothing touching "
            "it) before zeroing. Type 'yes' to proceed, anything else to skip: "
        )
        if confirm.strip().lower() != "yes":
            print("  Skipped by user.")
            return
        ret = arm.set_ft_sensor_zero()
        print(f"  set_ft_sensor_zero() -> {ret}")
        time.sleep(0.5)
        code, data = arm.get_ft_sensor_data()
        print(f"  get_ft_sensor_data() after zero -> code={code}, data={data}")
        print("  CONFIRM: values should now be near [0,0,0,0,0,0] within noise.")

    check(results, "ft_zero", _ft_zero)

    if args.port:
        def _homemade_serial():
            import serial

            ser = serial.Serial(args.port, args.baud, timeout=1)
            ser.reset_input_buffer()
            lines_read = 0
            for _ in range(20):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.endswith(","):
                    line = line[:-1]
                values = [float(x) for x in line.split(",")]
                lines_read += 1
                if lines_read <= 3:
                    print(f"  line: {values}")
                    if len(values) != 16:
                        print(f"    WARNING: expected 16 values, got {len(values)}")
                    out_of_range = [v for v in values if not (9000 < v < 21000)]
                    if out_of_range:
                        print(f"    WARNING: values outside documented 9000-21000 range: {out_of_range}")
                if lines_read >= 10:
                    break
            ser.close()
            if lines_read == 0:
                raise RuntimeError("No lines read from serial port")
            print(f"  Read {lines_read} valid 16-channel lines.")

        check(results, "homemade_serial", _homemade_serial)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{status}] {name}")
    print(
        "\nFix/investigate any FAIL above, and re-read the printed signatures/docstrings "
        "against what ft_ground_truth_check.py assumes, before running the full pose sweep."
    )


if __name__ == "__main__":
    sys.exit(main())
