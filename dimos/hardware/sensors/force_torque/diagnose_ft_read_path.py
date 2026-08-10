#!/usr/bin/env python3
"""Diagnostic: which FT read path actually has live data on this arm/firmware?

XArmFTSensor's telemetry loop reads arm.ft_ext_force / arm.ft_raw_force -- cached
values populated only by the periodic report stream. Its startup check instead
calls arm.get_ft_sensor_data() directly, a different code path. If the firmware's
periodic report doesn't include FT, the cached attributes can sit at zero forever
while the direct call keeps working -- this compares them side by side, live.

Usage: python3 dimos/hardware/sensors/force_torque/diagnose_ft_read_path.py <xarm_ip>
"""

import sys
import time

from xarm.wrapper import XArmAPI


def main():
    if len(sys.argv) != 2:
        print("Usage: diagnose_ft_read_path.py <xarm_ip>")
        return 1

    arm = XArmAPI(sys.argv[1])
    if not arm.connected:
        print(f"FAIL: could not connect to {sys.argv[1]}")
        return 1

    print(f"error_code={arm.error_code}  warn_code={arm.warn_code}  state={arm.state}")
    if arm.error_code != 0 or arm.warn_code != 0:
        print("Active error/warning present -- clearing before touching the FT sensor.")
        arm.clean_error()
        arm.clean_warn()
        print(f"after clean: error_code={arm.error_code}  warn_code={arm.warn_code}  state={arm.state}")

    code = arm.set_ft_sensor_enable(1)
    print(f"set_ft_sensor_enable(1) -> code={code}")
    mode_code = arm.set_ft_sensor_mode(0)
    print(f"set_ft_sensor_mode(0) -> code={mode_code}")

    print("\nPress the arm/sensor by hand while this runs to see if anything moves.\n")
    print(f"{'t':>5}  {'get_ft_sensor_data()':>45}  {'ft_ext_force (cached)':>35}  {'ft_raw_force (cached)':>35}")

    for i in range(30):
        code, data = arm.get_ft_sensor_data()
        cached_ext = getattr(arm, "ft_ext_force", None)
        cached_raw = getattr(arm, "ft_raw_force", None)
        print(f"{i:>5}  code={code} {data}  {cached_ext}  {cached_raw}")
        time.sleep(0.5)

    arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
