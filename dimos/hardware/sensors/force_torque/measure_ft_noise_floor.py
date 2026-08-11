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

"""Bench test, no door, no motion: measures the FT sensor's noise floor at rest
and suggests k_trans/k_rot ceilings for admittance_pull_law.AdmittanceConfig.

An admittance law commands velocity = k * measured_wrench. If k is bigger than
noise_std_velocity_budget / noise_std_wrench, sensor noise alone commands
visible jitter with nothing touching the sensor -- this measures the real
noise_std directly instead of guessing k. Run this once per sensor/mount, not
per door: it characterizes the robot+sensor, not what it's about to open.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from xarm.wrapper import XArmAPI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ip", help="xArm controller IP")
    p.add_argument("--duration", type=float, default=20.0, help="seconds to sample at rest")
    p.add_argument("--rate", type=float, default=50.0, help="Hz")
    p.add_argument("--trans-jitter-budget", type=float, default=0.002, help="m/s, acceptable idle jitter")
    p.add_argument("--rot-jitter-budget", type=float, default=0.02, help="rad/s, acceptable idle jitter")
    args = p.parse_args()

    arm = XArmAPI(args.ip)
    if not arm.connected:
        raise SystemExit(f"Could not connect to {args.ip}")
    code = arm.set_ft_sensor_enable(1)
    if code != 0:
        raise SystemExit(f"set_ft_sensor_enable failed, code={code}")

    print(f"Sampling {args.duration:.0f}s at {args.rate:.0f}Hz -- do NOT touch the sensor/gripper during this.")
    samples = []
    n = int(args.duration * args.rate)
    for i in range(n):
        code, wrench = arm.get_ft_sensor_data()
        if code == 0:
            samples.append(wrench)
        time.sleep(1.0 / args.rate)
        if i % int(args.rate * 2) == 0:
            print(f"  {i / args.rate:.0f}s / {args.duration:.0f}s ({len(samples)} good samples)")

    arm.set_ft_sensor_enable(0)
    arm.disconnect()

    if len(samples) < 10:
        raise SystemExit(f"Only {len(samples)} good samples -- can't compute meaningful stats.")

    data = np.array(samples)  # (N, 6): Fx,Fy,Fz,Mx,My,Mz
    force_std = np.linalg.norm(data[:, :3].std(axis=0))  # N, combined 3-axis noise magnitude
    torque_std = np.linalg.norm(data[:, 3:].std(axis=0))  # N*m

    print(f"\nForce noise (per-axis std, N):  {data[:, :3].std(axis=0)}")
    print(f"Torque noise (per-axis std, N*m): {data[:, 3:].std(axis=0)}")
    print(f"Combined force noise magnitude:  {force_std:.4f} N")
    print(f"Combined torque noise magnitude: {torque_std:.4f} N*m")

    k_trans_max = args.trans_jitter_budget / force_std if force_std > 1e-9 else float("inf")
    k_rot_max = args.rot_jitter_budget / torque_std if torque_std > 1e-9 else float("inf")
    print(
        f"\nSuggested ceilings (idle jitter <= {args.trans_jitter_budget}m/s / {args.rot_jitter_budget}rad/s):\n"
        f"  k_trans <= {k_trans_max:.5f}  (current default in admittance_pull_law.py: 0.003)\n"
        f"  k_rot   <= {k_rot_max:.5f}  (current default: 0.05)\n"
        "These are noise-floor ceilings, not the values to use directly -- pick something "
        "comfortably under them, then confirm by hand-pushing the sensor and watching the "
        "commanded twist for responsiveness vs. jitter."
    )


if __name__ == "__main__":
    main()
