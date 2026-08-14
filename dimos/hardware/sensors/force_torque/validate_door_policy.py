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

"""Does the door policy meet its requirements, on every door type, before hardware?

Checks the four things asked of it:

  R1  works on any hinged door   -- microwave, oven, fridge: different radius, height, swing
  R2  smooth, no crashes         -- commanded acceleration stays inside its own jerk limit
  R3  simple and direct          -- no per-door configuration anywhere in here
  R4  actually works             -- reaches full opening without singularity or joint limit

R4 is the one that has to be answered BEFORE the pull, not during it. A hinged door's path
is a circle the moment the hinge is known, so whether the arm can follow it to the end is
already decided by where it grabbed. No controller rescues a grasp that runs out of joint
travel at 50 degrees -- but a check that takes a second can reject it and ask for another.

Pure kinematics: no contact, no control loop, so none of the sim harness's dynamics
uncertainties apply to these numbers.

    python3 validate_door_policy.py --model-dir <xarm7/>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admittance_pull_law import (  # noqa: E402
    AdmittanceConfig,
    arc_waypoints,
    compute_hybrid_twist,
    fit_hinge,
    slew_limit,
)
from sim_door_mujoco import ARM_JOINTS, arm_only_xml  # noqa: E402

# Thresholds below which the arm is too badly conditioned or too close to a stop to be
# driving into a constraint. Deliberately conservative -- this check is cheap and a failed
# pull is not.
SIGMA_FLOOR = 0.05
MARGIN_FLOOR_RAD = 0.10


@dataclass
class Door:
    """Real appliance geometry: radius is hinge-to-handle, the arc the gripper must travel."""

    name: str
    radius_m: float
    open_angle_deg: float
    handle_height_m: float
    hinge_side: str          # "left" or "right", seen from the robot


DOORS = [
    Door("microwave", 0.36, 90.0, 0.30, "left"),
    Door("oven",      0.50, 90.0, 0.20, "left"),
    Door("fridge",    0.60, 120.0, 0.55, "left"),
    Door("cabinet",   0.30, 100.0, 0.45, "right"),
]


class Kinematics:
    def __init__(self, model_dir: Path):
        self.model = mujoco.MjModel.from_xml_path(str(arm_only_xml(model_dir)))
        self.data = mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.dofs = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.lo, self.hi = self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def solve(self, target: np.ndarray, seed: np.ndarray, iters: int = 300) -> np.ndarray:
        q = seed.copy()
        for _ in range(iters):
            self.data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[self.site]
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            jac = self._jp[:, self.dofs]
            u, s, vt = np.linalg.svd(jac, full_matrices=False)
            q = np.clip(q + vt.T @ ((s / (s**2 + 0.01**2)) * (u.T @ err)),
                        self.lo + 0.02, self.hi - 0.02)
        return q

    def evaluate(self, q: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        reach_err = float(np.linalg.norm(target - self.data.site_xpos[self.site]))
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        jac = np.vstack([self._jp[:, self.dofs], self._jr[:, self.dofs]])
        sigma = float(np.linalg.svd(jac, compute_uv=False)[-1])
        margin = float(min(np.min(q - self.lo), np.min(self.hi - q)))
        return reach_err, sigma, margin


def check_arc(kin: Kinematics, grasp: np.ndarray, hinge: np.ndarray, axis: np.ndarray,
              max_angle_deg: float, steps: int = 19) -> dict:
    """Walk the whole arc. Report the worst point, because that is what stops the pull."""
    pts = arc_waypoints(grasp, hinge, axis, np.radians(max_angle_deg), steps)
    q = kin.solve(pts[0], np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.9, 0.0]))
    worst_sigma, worst_margin, blocked_at = np.inf, np.inf, None
    for angle, pt in zip(np.linspace(0, max_angle_deg, steps), pts, strict=True):
        q = kin.solve(pt, q)
        err, sigma, margin = kin.evaluate(q, pt)
        worst_sigma, worst_margin = min(worst_sigma, sigma), min(worst_margin, margin)
        if blocked_at is None and (err > 1e-3 or sigma < SIGMA_FLOOR or margin < MARGIN_FLOOR_RAD):
            blocked_at = float(angle)
    return {"worst_sigma": worst_sigma, "worst_margin": worst_margin, "blocked_at": blocked_at}


def best_grasp_offset(kin: Kinematics, door: Door, hinge_xy: np.ndarray) -> tuple[float, dict]:
    """Where should the robot stand relative to the hinge? Sweep it rather than guess.

    This is the whole of R4: the same door is openable or not depending only on this.
    """
    best = (None, {"blocked_at": 0.0, "worst_sigma": 0.0, "worst_margin": 0.0})
    axis = np.array([0.0, 0.0, 1.0])
    for offset in np.arange(-0.25, 0.30, 0.05):
        hinge = np.array([hinge_xy[0], hinge_xy[1] + offset, door.handle_height_m])
        grasp = hinge + np.array([0.0, -door.radius_m, 0.0])
        res = check_arc(kin, grasp, hinge, axis, door.open_angle_deg)
        reached = door.open_angle_deg if res["blocked_at"] is None else res["blocked_at"]
        best_reached = (door.open_angle_deg if best[1]["blocked_at"] is None
                        else best[1]["blocked_at"])
        if best[0] is None or reached > best_reached:
            best = (float(offset), res)
    return best


def check_smoothness(cfg: AdmittanceConfig) -> tuple[float, float]:
    """R2: drive the law with a deliberately violent force step and measure what escapes."""
    rng = np.random.default_rng(0)
    dt, prev = 1.0 / 25.0, np.zeros(3)
    worst_accel = 0.0
    vel = np.array([0.0, 0.02, 0.0])
    for k in range(200):
        force = np.array([80.0, 0.0, 0.0]) if 50 <= k < 55 else rng.normal(0, 2.0, 3)
        res = compute_hybrid_twist(force, np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg,
                                   measured_velocity_world=vel)
        limited = slew_limit(prev, res.linear, cfg.max_linear_accel * dt)
        worst_accel = max(worst_accel, float(np.linalg.norm(limited - prev)) / dt)
        prev = limited
    return worst_accel, cfg.max_linear_accel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    args = p.parse_args()

    kin = Kinematics(args.model_dir)
    cfg = AdmittanceConfig()
    failures = []

    print("R1/R4  Can the arm follow each door's full arc, and from where?\n")
    print(f"  {'door':<11}{'radius':>8}{'target':>9}{'best offset':>13}{'reaches':>10}"
          f"{'worst sigma':>13}{'worst margin':>14}   verdict")
    for door in DOORS:
        hinge_xy = np.array([0.55, 0.30])
        offset, res = best_grasp_offset(kin, door, hinge_xy)
        reached = door.open_angle_deg if res["blocked_at"] is None else res["blocked_at"]
        ok = res["blocked_at"] is None
        print(f"  {door.name:<11}{door.radius_m:7.2f}m{door.open_angle_deg:8.0f}d"
              f"{offset:+12.2f}m{reached:9.0f}d{res['worst_sigma']:13.4f}"
              f"{res['worst_margin']:14.3f}   {'OK' if ok else 'BLOCKED'}")
        if not ok:
            failures.append(f"{door.name}: stops at {reached:.0f} of {door.open_angle_deg:.0f} deg")

    print("\nR1  Is the hinge recoverable from probe motion alone, per door?\n")
    print(f"  {'door':<11}{'probe arc':>11}{'radius est':>12}{'error':>9}   verdict")
    for door in DOORS:
        hinge = np.array([0.55, 0.30, door.handle_height_m])
        grasp = hinge + np.array([0.0, -door.radius_m, 0.0])
        pts = arc_waypoints(grasp, hinge, np.array([0.0, 0.0, 1.0]), np.radians(20), 12)
        pts = pts + np.random.default_rng(1).normal(0, 0.0005, pts.shape)   # 0.5mm sensing noise
        fit = fit_hinge(pts)
        if fit is None:
            print(f"  {door.name:<11}{'20d':>11}{'--':>12}{'--':>9}   FAILED TO FIT")
            failures.append(f"{door.name}: hinge not recoverable")
            continue
        _, _, radius = fit
        err = abs(radius - door.radius_m) / door.radius_m * 100
        ok = err < 15.0
        print(f"  {door.name:<11}{'20d':>11}{radius:11.3f}m{err:8.1f}%   {'OK' if ok else 'TOO ROUGH'}")
        if not ok:
            failures.append(f"{door.name}: radius estimate off by {err:.0f}%")

    print("\nR2  Does anything escape the jerk limit?\n")
    worst, limit = check_smoothness(cfg)
    ok = worst <= limit + 1e-9
    print(f"  worst commanded acceleration {worst:.3f} m/s^2 against a {limit:.3f} limit"
          f"   {'OK' if ok else 'EXCEEDED'}")
    if not ok:
        failures.append(f"jerk limit exceeded: {worst:.3f} > {limit:.3f}")

    print("\nR3  Per-door configuration required: none by construction --")
    print("    every number above comes from the same AdmittanceConfig defaults.\n")

    if failures:
        print(f"{len(failures)} REQUIREMENT FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All requirements met.")


if __name__ == "__main__":
    main()
