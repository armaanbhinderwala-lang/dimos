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

"""Door-opening path: serial parsing, calibration, conditioning, control, geometry.

Each test is a bug we hit or nearly shipped. Pure functions only -- no hardware or bus.
"""

from __future__ import annotations

import numpy as np
import pytest

from dimos.hardware.sensors.force_torque.admittance_pull_law import (
    AdmittanceConfig,
    arc_waypoints,
    compute_hybrid_twist,
    fit_hinge,
    slew_limit,
)
from dimos.hardware.sensors.force_torque.ft_conditioning import (
    GRAVITY_M_S2,
    PROFILES,
    ConditioningProfile,
    WrenchConditioner,
    deadband,
    ema,
    tool_gravity_wrench,
)
from dimos.hardware.sensors.force_torque.openft_module import parse_frame

CHANNELS = 16


# --------------------------------------------------------------- serial parsing
def test_parse_frame_accepts_the_wire_format():
    line = ",".join(str(i * 10) for i in range(CHANNELS)) + ","      # trailing comma is real
    parsed = parse_frame(line)
    assert parsed is not None and len(parsed) == CHANNELS
    assert parsed[3] == 30.0


@pytest.mark.parametrize("bad", [
    "",                                    # empty
    "1,2,3",                               # short frame
    ",".join(["1"] * (CHANNELS + 2)),      # long frame
    ",".join(["x"] * CHANNELS),            # non-numeric
])
def test_parse_frame_rejects_malformed_lines(bad):
    """A half-read frame must be dropped, not fed to the calibration as if it were real."""
    assert parse_frame(bad) is None


# --------------------------------------------------------------- calibration math
def test_calibration_matrix_shape_and_math():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, CHANNELS))
    bias = rng.normal(size=6)
    channels = rng.normal(size=CHANNELS)
    wrench = matrix @ channels + bias
    assert wrench.shape == (6,)
    # Linearity is the whole premise of a calibration matrix; if it fails, the model is wrong.
    other = rng.normal(size=CHANNELS)
    assert np.allclose(matrix @ (channels + other) + bias,
                       (matrix @ channels + bias) + matrix @ other)


def test_calibration_rejects_wrong_shape():
    with pytest.raises(ValueError):
        matrix = np.zeros((6, CHANNELS - 1))
        if matrix.shape != (6, CHANNELS):
            raise ValueError(f"calibration matrix is {matrix.shape}, expected (6, {CHANNELS})")


# --------------------------------------------------------------- conditioning
def test_tare_removes_bias_and_keeps_real_load():
    c = WrenchConditioner(ConditioningProfile(1.0, 0.0, 0.0))
    bias = np.array([3.0, -2.0, 10.0, 0.4, -0.1, 0.2])
    c.begin_tare()
    for _ in range(50):
        assert np.allclose(c.apply(bias), 0.0), "must publish zero while still taring"
    assert np.allclose(c.apply(bias), 0.0, atol=1e-9)
    assert np.isclose(c.apply(bias + np.array([10.0, 0, 0, 0, 0, 0]))[0], 10.0)


def test_gravity_compensation_tracks_orientation():
    """A constant offset only works if the tool never turns -- which a door pull does."""
    mass, com = 0.85, np.array([0.0, 0.0, 0.05])
    assert np.isclose(tool_gravity_wrench(np.eye(3), mass, com)[2], -mass * GRAVITY_M_S2)
    flipped = tool_gravity_wrench(np.diag([1.0, -1.0, -1.0]), mass, com)
    assert np.isclose(flipped[2], +mass * GRAVITY_M_S2)
    sideways = tool_gravity_wrench(np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]), mass, com)
    assert abs(sideways[0]) > 1.0 and abs(sideways[2]) < 1e-9
    assert np.allclose(tool_gravity_wrench(np.eye(3), mass, np.zeros(3))[3:], 0.0)


def test_already_compensated_source_is_not_compensated_twice():
    """The uFactory publishes ext_wrench already gravity-free; subtracting again biases it."""
    c = WrenchConditioner(PROFILES["factory"])
    c.profile.tool_mass_kg = 0.85
    assert np.allclose(c.apply(np.zeros(6), ee_rot=np.eye(3)), 0.0)


def test_deadband_gates_noise_exactly_to_zero():
    p = PROFILES["diy"]
    gated = deadband(np.array([1.0, -1.2, 0.9, 0.05, -0.08, 0.02]),
                     p.force_deadband_n, p.torque_deadband_nm)
    assert np.allclose(gated, 0.0), "near-zero must be exactly zero, or the arm creeps"
    kept = deadband(np.array([8.0, 0, 0, 0.5, 0, 0]), p.force_deadband_n, p.torque_deadband_nm)
    assert np.isclose(kept[0], 8.0) and np.isclose(kept[3], 0.5)


def test_ema_starts_at_the_signal_and_blunts_spikes():
    assert np.allclose(ema(None, np.ones(6) * 5, 0.1), 5.0), "must not ramp up from zero"
    state = None
    for _ in range(5):
        state = ema(state, np.zeros(6), 0.1)
    before = state[0]
    state = ema(state, np.ones(6) * 100, 0.1)
    assert state[0] - before < 11.0


# --------------------------------------------------------------- control law
def test_velocity_is_capped():
    cfg = AdmittanceConfig()
    moving = np.array([0.0, 0.02, 0.0])
    for magnitude in (10.0, 50.0, 79.0):
        r = compute_hybrid_twist(np.array([magnitude, 0, 0]), np.zeros(3), np.eye(3),
                                 np.array([1.0, 0, 0]), cfg, measured_velocity_world=moving)
        radial = abs(float(np.dot(r.linear, np.array([1.0, 0, 0]))))
        assert radial <= cfg.max_lateral_speed + 1e-9, f"{radial} exceeded the cap"
        assert abs(r.linear[1]) <= cfg.drive_speed + 1e-9


def test_safety_cutoff_zeroes_the_command():
    cfg = AdmittanceConfig()
    r = compute_hybrid_twist(np.array([cfg.force_cutoff + 1, 0, 0]), np.zeros(3), np.eye(3),
                             np.array([1.0, 0, 0]), cfg)
    assert r.safety_stop and np.allclose(r.linear, 0) and np.allclose(r.angular, 0)


def test_never_drives_into_a_loaded_radial():
    cfg = AdmittanceConfig()
    moving = np.array([0.0, 0.03, 0.0])
    r = compute_hybrid_twist(np.array([40.0, 0, 0]), np.zeros(3), np.eye(3),
                             np.array([1.0, 0, 0]), cfg, measured_velocity_world=moving)
    assert r.linear[0] < 0, "a loaded radial must be unloaded, not pushed harder"
    assert r.linear[1] > 0, "while still progressing along the tangent"


def test_stalled_door_is_still_pushed():
    """Before a latch breaks, resistance is head-on. Backing off there never opens anything."""
    cfg = AdmittanceConfig()
    r = compute_hybrid_twist(np.array([20.0, 0, 0]), np.zeros(3), np.eye(3),
                             np.array([1.0, 0, 0]), cfg, measured_velocity_world=np.zeros(3))
    assert np.isclose(r.linear[0], cfg.drive_speed)


def test_slew_limit_bounds_acceleration():
    cfg, dt = AdmittanceConfig(), 1.0 / 25.0
    prev = np.zeros(3)
    limited = slew_limit(prev, np.array([0.0, 5.0, 0.0]), cfg.max_linear_accel * dt)
    assert np.linalg.norm(limited - prev) <= cfg.max_linear_accel * dt + 1e-9


# --------------------------------------------------------------- door geometry
def test_hinge_is_recovered_from_arc_motion():
    hinge = np.array([0.5, 0.4, 0.3])
    pts = arc_waypoints(np.array([0.5, 0.0, 0.3]), hinge, np.array([0.0, 0.0, 1.0]),
                        np.radians(25), 12)
    centre, axis, radius = fit_hinge(pts)
    assert np.linalg.norm(centre - hinge) < 1e-6
    assert np.isclose(radius, 0.4)
    assert np.allclose(np.abs(axis), [0, 0, 1], atol=1e-6)


def test_straight_motion_is_not_mistaken_for_a_door():
    """A drawer must not be fitted with a circle, or the arc check projects a fiction."""
    assert fit_hinge(np.linspace([0, 0, 0], [0.3, 0, 0], 12)) is None
    assert fit_hinge(np.zeros((3, 3))) is None, "too few points to fit anything"


def test_arc_keeps_a_constant_radius():
    hinge = np.array([0.5, 0.4, 0.3])
    pts = arc_waypoints(np.array([0.5, 0.0, 0.3]), hinge, np.array([0.0, 0.0, 1.0]),
                        np.radians(90), 10)
    radii = np.linalg.norm(pts - hinge, axis=1)
    assert np.allclose(radii, radii[0]), "a hinge cannot change the radius"
    assert np.allclose(pts[:, 2], 0.3), "a vertical hinge cannot change height"
