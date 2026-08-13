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

"""Checks on the uFactory<->DIY wrench transform.

A wrong frame transform corrupts TORQUE specifically while leaving force untouched,
which looks exactly like a sensor that cannot measure torque. These tests exist so that
symptom can never be blamed on the transform without evidence.

Run:  python3 test_frame_transform.py     (or under pytest)
"""

from __future__ import annotations

import numpy as np

from session_data import (
    AXES,
    DIY_ORIGIN_IN_UFACTORY_M as P,
    R_DIY_FROM_UFACTORY as R,
    R_UFACTORY_FROM_DIY,
    diy_to_ufactory_frame,
    ufactory_to_diy_frame,
)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def wrench_adjoint() -> np.ndarray:
    """The 6x6 wrench transform, wrench ordered [F; tau].

        F'   =  R F
        tau' =  R (tau - p x F)  =  -R[p]x F + R tau
    """
    return np.block([[R, np.zeros((3, 3))], [-R @ skew(P), R]])


def test_matches_wrench_adjoint():
    """Our two-step implementation must equal the assembled 6x6 adjoint exactly."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(500, 6)) * [50, 50, 50, 3, 3, 3]
    assert np.allclose(ufactory_to_diy_frame(w), w @ wrench_adjoint().T, atol=1e-12)


def test_is_not_the_twist_adjoint():
    """The velocity (twist) adjoint has the lever term in the OTHER block.

    Using it for a wrench is a classic mix-up and would silently corrupt torque.
    """
    twist_adjoint = np.block([[R, -R @ skew(P)], [np.zeros((3, 3)), R]])
    w = np.array([[10.0, 5.0, -3.0, 0.4, -0.2, 0.1]])
    assert not np.allclose(ufactory_to_diy_frame(w), w @ twist_adjoint.T)


def test_force_at_the_origin_makes_no_torque():
    """The decisive physical check: a force acting AT a point exerts no moment about it."""
    force = np.array([[30.0, -12.0, 45.0]])
    wrench = np.hstack([force, np.cross(P, force)])      # what the uFactory would read
    assert np.allclose(ufactory_to_diy_frame(wrench)[0, 3:], 0.0, atol=1e-12)


def test_known_lever_arm():
    """20 N applied 10 cm past the DIY origin must give exactly 2 N*m about it."""
    offset = 0.10
    force = np.array([[20.0, 0.0, 0.0]])
    position = P + np.array([0.0, 0.0, offset])
    wrench = np.hstack([force, np.cross(position, force)])
    torque = ufactory_to_diy_frame(wrench)[0, 3:]
    assert np.isclose(np.linalg.norm(torque), 20.0 * offset)


def test_round_trip():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(500, 6)) * [50, 50, 50, 3, 3, 3]
    assert np.allclose(diy_to_ufactory_frame(ufactory_to_diy_frame(w)), w, atol=1e-10)


def test_rotation_is_proper():
    """Orthogonal with det +1 -- a real rotation, not a reflection or a scaling."""
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R_UFACTORY_FROM_DIY, R.T)


def test_measured_axis_mapping():
    """The mapping confirmed both physically and by two independent fits."""
    assert np.allclose(R_UFACTORY_FROM_DIY @ [1, 0, 0], [0, -1, 0])   # DIY +X -> uF -Y
    assert np.allclose(R_UFACTORY_FROM_DIY @ [0, 1, 0], [1, 0, 0])    # DIY +Y -> uF +X
    assert np.allclose(R_UFACTORY_FROM_DIY @ [0, 0, 1], [0, 0, 1])    # Z unchanged


def test_force_magnitude_preserved():
    """A change of frame rotates force but cannot change how hard the push was."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=(500, 6)) * [50, 50, 50, 3, 3, 3]
    out = ufactory_to_diy_frame(w)
    assert np.allclose(np.linalg.norm(w[:, :3], axis=1), np.linalg.norm(out[:, :3], axis=1))


def test_adjoint_determinant():
    """A rigid change of frame preserves volume in wrench space."""
    assert np.isclose(np.linalg.det(wrench_adjoint()), 1.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} checks passed -- the frame transform is not a suspect.")
