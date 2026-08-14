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

"""Arc feasibility against the real xArm7 kinematics.

This is the check that decides whether a pull is attempted, so a wrong answer either wastes a
hardware run or refuses a door the arm could open. Needs the unpacked xArm7 MJCF; skipped
when it is absent.

    tar xzf data/.lfs/xarm7.tar.gz -C /tmp/x7
    XARM7_MODEL_DIR=/tmp/x7/xarm7 python3 -m pytest test_arc_check.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dimos.hardware.sensors.force_torque.admittance_pull_law import arc_waypoints, fit_hinge

mujoco = pytest.importorskip("mujoco")

_MODEL_DIR = Path(os.environ.get("XARM7_MODEL_DIR", "/tmp/x7/xarm7"))
pytestmark = pytest.mark.skipif(
    not (_MODEL_DIR / "xarm7.xml").exists(),
    reason=f"xArm7 model not found at {_MODEL_DIR}; unpack data/.lfs/xarm7.tar.gz",
)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
SEED_Q = np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.9, 0.0])


class Arm:
    """Minimal FK/IK/Jacobian, mirroring what the module does with pinocchio."""

    def __init__(self, model_dir: Path):
        text = (model_dir / "xarm7.xml").read_text()
        while "<keyframe" in text:                      # keyframe is sized for scene.xml
            start = text.index("<keyframe")
            end = text.find("</keyframe>", start)
            text = text[:start] + (text[end + len("</keyframe>"):] if end != -1 else "")
        path = model_dir / "_test_arm.xml"
        path.write_text(text)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.dofs = np.array([self.model.jnt_dofadr[j] for j in jid])
        self.lo, self.hi = self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def ik(self, target: np.ndarray, seed: np.ndarray, iters: int = 300) -> np.ndarray:
        q = seed.copy()
        for _ in range(iters):
            self.data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[self.site]
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            u, s, vt = np.linalg.svd(self._jp[:, self.dofs], full_matrices=False)
            q = np.clip(q + vt.T @ ((s / (s**2 + 0.01**2)) * (u.T @ err)), self.lo + 0.02, self.hi - 0.02)
        return q

    def evaluate(self, q: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        reach = float(np.linalg.norm(target - self.data.site_xpos[self.site]))
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        jac = np.vstack([self._jp[:, self.dofs], self._jr[:, self.dofs]])
        return (reach, float(np.linalg.svd(jac, compute_uv=False)[-1]),
                float(min(np.min(q - self.lo), np.min(self.hi - q))))

    def walk_arc(self, grasp, hinge, axis, max_angle_deg, steps=19,
                 min_sigma=0.05, min_margin=0.10, tol=0.005):
        points = arc_waypoints(grasp, hinge, axis, np.radians(max_angle_deg), steps)
        angles = np.linspace(0, max_angle_deg, steps)
        q, blocked = SEED_Q.copy(), None
        worst_sigma = worst_margin = np.inf
        for angle, point in zip(angles, points, strict=True):
            q = self.ik(point, q)
            reach, sigma, margin = self.evaluate(q, point)
            worst_sigma, worst_margin = min(worst_sigma, sigma), min(worst_margin, margin)
            if blocked is None and (reach > tol or sigma < min_sigma or margin < min_margin):
                blocked = float(angle)
        return {"blocked_at": blocked, "worst_sigma": worst_sigma, "worst_margin": worst_margin}


@pytest.fixture(scope="module")
def arm() -> Arm:
    return Arm(_MODEL_DIR)


def test_ik_reaches_a_point_it_should(arm):
    target = np.array([0.45, 0.0, 0.35])
    reach, sigma, margin = arm.evaluate(arm.ik(target, SEED_Q), target)
    assert reach < 1e-3, f"comfortable point unreachable, error {reach * 1000:.1f}mm"
    assert sigma > 0.05 and margin > 0.1


def test_ik_does_not_pretend_to_reach_beyond_the_workspace(arm):
    """A silent false success here would let the check bless an impossible arc."""
    far = np.array([2.5, 0.0, 0.35])
    reach, _, _ = arm.evaluate(arm.ik(far, SEED_Q), far)
    assert reach > 0.5, f"claimed to reach 2.5m away, error only {reach:.3f}m"


def test_ik_respects_joint_limits(arm):
    for target in ([0.45, 0.0, 0.35], [0.2, 0.3, 0.6], [0.6, -0.2, 0.2]):
        q = arm.ik(np.array(target), SEED_Q)
        assert np.all(q >= arm.lo) and np.all(q <= arm.hi), f"IK left the joint range for {target}"


def test_small_door_arc_is_reachable(arm):
    hinge = np.array([0.55, 0.30, 0.36])
    grasp = hinge + np.array([0.0, -0.30, 0.0])
    result = arm.walk_arc(grasp, hinge, np.array([0.0, 0.0, 1.0]), 90.0)
    assert result["blocked_at"] is None, (
        f"a 0.30m-radius door should be openable, blocked at {result['blocked_at']}deg"
    )


def test_large_door_arc_is_correctly_refused(arm):
    """A fridge-sized swing is beyond a fixed base. The check must SAY so, not fail silently."""
    hinge = np.array([0.55, 0.30, 0.55])
    grasp = hinge + np.array([0.0, -0.60, 0.0])
    result = arm.walk_arc(grasp, hinge, np.array([0.0, 0.0, 1.0]), 120.0)
    assert result["blocked_at"] is not None, "an unreachable arc was reported as clear"
    assert result["blocked_at"] < 120.0


def test_blocked_angle_is_monotone_in_target(arm):
    """Asking for more swing can never report jamming later than asking for less."""
    hinge = np.array([0.55, 0.30, 0.36])
    grasp = hinge + np.array([0.0, -0.50, 0.0])
    reached = []
    for target in (30.0, 60.0, 90.0):
        r = arm.walk_arc(grasp, hinge, np.array([0.0, 0.0, 1.0]), target)
        reached.append(target if r["blocked_at"] is None else r["blocked_at"])
    assert reached[0] <= reached[1] + 1e-6 and reached[1] <= reached[2] + 1e-6, reached


def test_fitted_hinge_predicts_the_same_verdict_as_the_true_one(arm):
    """The module never has the true hinge -- only fit_hinge's estimate. They must agree."""
    hinge = np.array([0.55, 0.30, 0.36])
    axis = np.array([0.0, 0.0, 1.0])
    grasp = hinge + np.array([0.0, -0.36, 0.0])
    probe = arc_waypoints(grasp, hinge, axis, np.radians(20), 20)
    probe = probe + np.random.default_rng(0).normal(0, 0.0005, probe.shape)
    fitted_centre, fitted_axis, _ = fit_hinge(probe)

    truth = arm.walk_arc(grasp, hinge, axis, 90.0)
    estimate = arm.walk_arc(grasp, fitted_centre, fitted_axis, 90.0)
    assert (truth["blocked_at"] is None) == (estimate["blocked_at"] is None), (
        f"fitted hinge disagreed with truth: {estimate['blocked_at']} vs {truth['blocked_at']}"
    )
