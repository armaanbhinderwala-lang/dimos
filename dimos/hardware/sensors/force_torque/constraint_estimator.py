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

"""Online estimation of a door/drawer's kinematic constraint from EE pose history.

The gripper is rigidly attached to the handle (no slip), so the sequence of EE
positions recorded while pulling is a direct measurement of the appliance's own
constraint: a circular arc around some axis (a hinge -- vertical for a fridge/
microwave, horizontal for an oven) or a straight line (a drawer, the r=infinity
limit of the same arc). Fitting that trace, instead of assuming the axis and
pivot distance as fixed config, is what lets one control law handle any of them.

Math: center the points, take the SVD. The largest singular vector is the
direction points spread out the most; for a real 3D point cloud the pattern of
the three singular values tells you the shape:
  - s1, s2 << s0            -> collinear -> prismatic (line), direction = v0.
  - s2 << s0, s1 not tiny   -> coplanar  -> revolute (circle), axis = v2 (the
                                normal), fit the 2D circle in the (v0, v1) plane.
  - otherwise                -> not enough shape yet, keep collecting.
This is the standard way to fit a 3D circle without assuming its plane; the 2D
circle fit itself is the classic Kasa algebraic least-squares closed form.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Kind = Literal["revolute", "prismatic", "unknown"]


@dataclass
class ConstraintEstimate:
    kind: Kind
    axis: Optional[np.ndarray] = None  # unit vector; hinge axis (revolute) or slide direction (prismatic)
    pivot: Optional[np.ndarray] = None  # 3D point on the hinge axis (revolute only)
    radius: Optional[float] = None  # distance from pivot to the handle (revolute only)
    residual: float = float("inf")  # RMS geometric fit error, meters
    n_samples: int = 0

    def confident(self, max_residual: float = 0.01, min_samples: int = 8) -> bool:
        return self.kind != "unknown" and self.n_samples >= min_samples and self.residual <= max_residual


def fit_constraint(
    points: np.ndarray,
    *,
    line_ratio_thresh: float = 0.08,
    plane_ratio_thresh: float = 0.08,
) -> ConstraintEstimate:
    """Fit a line or circle through a (N, 3) array of EE positions. N must be >= 3."""
    n = len(points)
    if n < 3:
        return ConstraintEstimate(kind="unknown", n_samples=n)

    centroid = points.mean(axis=0)
    centered = points - centroid
    # Full SVD of the centered points directly gives the principal directions
    # (right singular vectors) and their spread (singular values), no separate
    # covariance/eigendecomposition step needed.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    s0 = s[0] if s[0] > 1e-12 else 1e-12
    v0, v1, v2 = vt[0], vt[1], vt[2]

    if s[1] / s0 < line_ratio_thresh:
        # Points collinear -> prismatic. Residual = RMS distance off the line.
        along = centered @ v0
        off_line = centered - np.outer(along, v0)
        residual = float(np.sqrt(np.mean(np.sum(off_line**2, axis=1))))
        return ConstraintEstimate(kind="prismatic", axis=v0, residual=residual, n_samples=n)

    if s[2] / s0 < plane_ratio_thresh:
        # Points coplanar -> revolute. Fit a 2D circle in the (v0, v1) plane.
        x = centered @ v0
        y = centered @ v1
        # Kasa fit: (x-a)^2+(y-b)^2=r^2 rearranged into a linear system for [a, b, c].
        A = np.column_stack([2 * x, 2 * y, np.ones(n)])
        rhs = x**2 + y**2
        (a, b, c), *_ = np.linalg.lstsq(A, rhs, rcond=None)
        radius = float(np.sqrt(c + a**2 + b**2))
        center_3d = centroid + a * v0 + b * v1
        geo_residual = float(np.sqrt(np.mean((np.sqrt((x - a) ** 2 + (y - b) ** 2) - radius) ** 2)))
        return ConstraintEstimate(
            kind="revolute", axis=v2, pivot=center_3d, radius=radius, residual=geo_residual, n_samples=n
        )

    return ConstraintEstimate(kind="unknown", n_samples=n)


def tangent_direction(estimate: ConstraintEstimate, position: np.ndarray, prev_tangent: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit direction of allowed motion at `position`, continuous with `prev_tangent` if given.

    Revolute: tangent = axis x (position - pivot), normalized -- the direction
    a point on a rotating rigid body instantaneously moves in.
    Prismatic: the slide direction itself.
    Sign is ambiguous from geometry alone (a circle fit doesn't know which way
    is "open"); `prev_tangent` picks the branch that continues the same way the
    handle has actually been moving, rather than flipping every tick.
    """
    if estimate.kind == "prismatic":
        t = estimate.axis
    elif estimate.kind == "revolute":
        r = position - estimate.pivot
        t = np.cross(estimate.axis, r)
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            return prev_tangent if prev_tangent is not None else np.zeros(3)
        t = t / norm
    else:
        return prev_tangent if prev_tangent is not None else np.zeros(3)

    if prev_tangent is not None and np.dot(t, prev_tangent) < 0:
        t = -t
    return t


class ConstraintEstimator:
    """Bounded position buffer + incremental re-fit, for use inside a control loop."""

    def __init__(self, max_samples: int = 60):
        self._positions: deque[np.ndarray] = deque(maxlen=max_samples)
        self._prev_tangent: Optional[np.ndarray] = None

    def add_sample(self, position: np.ndarray) -> None:
        self._positions.append(np.asarray(position, dtype=float))

    def estimate(self) -> ConstraintEstimate:
        if len(self._positions) < 3:
            return ConstraintEstimate(kind="unknown", n_samples=len(self._positions))
        return fit_constraint(np.array(self._positions))

    def tangent_at(self, position: np.ndarray, est: Optional[ConstraintEstimate] = None) -> np.ndarray:
        est = est if est is not None else self.estimate()
        t = tangent_direction(est, np.asarray(position, dtype=float), self._prev_tangent)
        if np.linalg.norm(t) > 1e-9:
            self._prev_tangent = t
        return t


if __name__ == "__main__":
    # Synthetic self-test -- no hardware needed. Three cases: a clean hinge, a
    # clean drawer, and too little data to say anything yet.
    rng = np.random.default_rng(0)

    def make_arc(pivot, axis, radius, start_deg, sweep_deg, n, noise=0.0005):
        axis = axis / np.linalg.norm(axis)
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - axis * np.dot(ref, axis)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angles = np.radians(np.linspace(start_deg, start_deg + sweep_deg, n))
        pts = pivot + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
        return pts + rng.normal(scale=noise, size=pts.shape)

    print("=== Case 1: vertical hinge (microwave/fridge-style), axis=z, radius=0.35m ===")
    true_pivot = np.array([0.5, 0.0, 0.3])
    true_axis = np.array([0.0, 0.0, 1.0])
    pts = make_arc(true_pivot, true_axis, radius=0.35, start_deg=0, sweep_deg=25, n=20)
    est = fit_constraint(pts)
    axis_err_deg = np.degrees(np.arccos(np.clip(abs(np.dot(est.axis, true_axis)), -1, 1)))
    print(f"kind={est.kind} axis_err={axis_err_deg:.3f}deg pivot_err={np.linalg.norm(est.pivot - true_pivot):.4f}m "
          f"radius_err={abs(est.radius - 0.35):.4f}m residual={est.residual:.5f}m confident={est.confident()}")
    assert est.kind == "revolute" and axis_err_deg < 1.0 and est.confident()

    print("\n=== Case 2: horizontal hinge (oven-style), axis=y, radius=0.25m ===")
    true_pivot2 = np.array([0.5, 0.2, 0.1])
    true_axis2 = np.array([0.0, 1.0, 0.0])
    pts2 = make_arc(true_pivot2, true_axis2, radius=0.25, start_deg=10, sweep_deg=-30, n=20)
    est2 = fit_constraint(pts2)
    axis_err2 = np.degrees(np.arccos(np.clip(abs(np.dot(est2.axis, true_axis2)), -1, 1)))
    print(f"kind={est2.kind} axis_err={axis_err2:.3f}deg residual={est2.residual:.5f}m confident={est2.confident()}")
    assert est2.kind == "revolute" and axis_err2 < 1.0

    print("\n=== Case 3: drawer (straight pull), direction=(1,0,0) ===")
    true_dir = np.array([1.0, 0.0, 0.0])
    t = np.linspace(0, 0.2, 20)
    pts3 = np.outer(t, true_dir) + np.array([0.4, 0.1, 0.2]) + rng.normal(scale=0.0005, size=(20, 3))
    est3 = fit_constraint(pts3)
    dir_err = np.degrees(np.arccos(np.clip(abs(np.dot(est3.axis, true_dir)), -1, 1)))
    print(f"kind={est3.kind} dir_err={dir_err:.3f}deg residual={est3.residual:.5f}m confident={est3.confident()}")
    assert est3.kind == "prismatic" and dir_err < 1.0

    print("\n=== Case 4: too little / ambiguous data (3 near-identical points) ===")
    pts4 = np.array([0.4, 0.1, 0.2]) + rng.normal(scale=0.0005, size=(3, 3))
    est4 = fit_constraint(pts4)
    print(f"kind={est4.kind} n_samples={est4.n_samples} confident={est4.confident()}")
    assert not est4.confident()

    print("\n=== Tangent-direction sign continuity (revolute) ===")
    est = fit_constraint(pts)
    p_mid = pts[10]
    t1 = tangent_direction(est, p_mid)
    t2 = tangent_direction(est, p_mid, prev_tangent=-t1)
    assert np.allclose(t2, t1), "sign continuity should flip toward prev_tangent"
    print("sign continuity OK")

    print("\nAll self-tests passed.")
