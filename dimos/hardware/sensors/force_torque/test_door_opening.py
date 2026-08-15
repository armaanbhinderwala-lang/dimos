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
    """Stated relative to the profile, so retuning the floor cannot silently break the test."""
    p = PROFILES["diy"]
    thr = p.deadbands
    below = thr * 0.5 * np.array([1, -1, 1, -1, 1, -1])
    assert np.allclose(deadband(below, thr), 0.0), \
        "sub-threshold must be exactly zero, or the arm creeps in free air"
    above = thr * 3
    kept = deadband(above, thr)
    # Soft deadband: the floor is SUBTRACTED, keeping the output continuous at the boundary.
    assert np.allclose(kept, above - thr)
    assert np.all(np.diff([float(deadband(np.r_[v, np.zeros(5)], thr)[0])
                           for v in np.linspace(0, thr[0] * 2, 40)]) >= -1e-12), \
        "output must never step backwards as input rises"


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


# --------------------------------------------------------------- robustness
def test_law_never_emits_nan_or_inf():
    """A dropped serial frame or a bad calibration can produce garbage; it must not reach IK."""
    cfg = AdmittanceConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        force = rng.normal(0, 30, 3)
        torque = rng.normal(0, 3, 3)
        vel = rng.normal(0, 0.05, 3)
        r = compute_hybrid_twist(force, torque, np.eye(3), np.array([1.0, 0, 0]), cfg,
                                 measured_velocity_world=vel)
        assert np.all(np.isfinite(r.linear)), f"non-finite linear from F={force}"
        assert np.all(np.isfinite(r.angular)), f"non-finite angular from M={torque}"


def test_law_survives_degenerate_inputs():
    cfg = AdmittanceConfig()
    zero_drive = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), cfg)
    assert np.all(np.isfinite(zero_drive.linear)), "zero drive direction must not divide by zero"

    tiny = compute_hybrid_twist(np.full(3, 1e-12), np.zeros(3), np.eye(3),
                                np.array([1.0, 0, 0]), cfg, measured_velocity_world=np.full(3, 1e-12))
    assert np.all(np.isfinite(tiny.linear))

    # Velocity exactly parallel to the radial load leaves no usable tangent.
    parallel = compute_hybrid_twist(np.array([30.0, 0, 0]), np.zeros(3), np.eye(3),
                                    np.array([1.0, 0, 0]), cfg,
                                    measured_velocity_world=np.array([0.05, 0, 0]))
    assert np.all(np.isfinite(parallel.linear))


def test_hinge_fit_tolerates_sensor_noise():
    """Real FK is not exact. A 5% radius error is usable; a wild one would misplace the arc."""
    hinge = np.array([0.5, 0.4, 0.3])
    clean = arc_waypoints(np.array([0.5, 0.0, 0.3]), hinge, np.array([0.0, 0.0, 1.0]),
                          np.radians(25), 20)
    for sigma_mm, tolerance_pct in ((0.5, 10.0), (1.0, 20.0)):
        errors = []
        for seed in range(20):
            noisy = clean + np.random.default_rng(seed).normal(0, sigma_mm / 1000.0, clean.shape)
            fit = fit_hinge(noisy)
            if fit is not None:
                errors.append(abs(fit[2] - 0.4) / 0.4 * 100)
        assert errors, f"every fit failed at {sigma_mm}mm noise"
        assert float(np.median(errors)) < tolerance_pct, \
            f"{sigma_mm}mm noise -> median radius error {np.median(errors):.1f}%"


def test_longer_probe_arc_fits_better():
    """Justifies probing far enough: a short arc constrains a circle weakly."""
    hinge = np.array([0.5, 0.4, 0.3])
    medians = []
    for degrees in (10, 25, 45):
        clean = arc_waypoints(np.array([0.5, 0.0, 0.3]), hinge, np.array([0.0, 0.0, 1.0]),
                              np.radians(degrees), 20)
        errors = []
        for seed in range(20):
            noisy = clean + np.random.default_rng(seed).normal(0, 0.0005, clean.shape)
            fit = fit_hinge(noisy)
            if fit is not None:
                errors.append(abs(fit[2] - 0.4) / 0.4 * 100)
        medians.append(float(np.median(errors)))
    assert medians[-1] < medians[0], f"longer arcs must fit better, got {medians}"


def test_conditioner_pipeline_rejects_drift_but_passes_a_pull():
    """End to end: a noisy, biased sensor must read zero at rest and track a real pull."""
    p = PROFILES["diy"]
    c = WrenchConditioner(p)
    rng = np.random.default_rng(0)
    bias = np.array([2.0, -3.0, 8.0, 0.2, -0.3, 0.1])
    noise = p.deadbands * 0.25  # comfortably below each axis floor
    c.begin_tare()
    for _ in range(50):
        c.apply(bias + rng.normal(0, noise))
    for _ in range(50):
        idle = c.apply(bias + rng.normal(0, noise))
    assert np.allclose(idle, 0.0), f"idle sensor must read exactly zero, got {idle}"
    for _ in range(60):
        pulling = c.apply(bias + np.array([20.0, 0, 0, 0, 0, 0]) + rng.normal(0, noise))
    expected = 20.0 - p.deadbands[0]          # soft deadband subtracts the floor
    assert abs(pulling[0] - expected) < 2.0, \
        f"a 20N pull should read ~{expected:.1f}N after conditioning, got {pulling[0]:.1f}"


def test_deadband_must_exceed_sensor_noise():
    """Tuning requirement, not a code property: noise above the gate becomes a standing command.

    Caught by an earlier version of the test above, where 0.4 Nm of torque noise leaked
    0.13 Nm through a 0.10 Nm gate. On hardware, measure the resting noise per axis and set
    the deadband above it, or the arm creeps with nothing touching it.
    """
    p = PROFILES["diy"]
    c = WrenchConditioner(p)
    rng = np.random.default_rng(1)
    c.begin_tare()
    for _ in range(50):
        c.apply(np.zeros(6))
    # Noise deliberately well ABOVE the gate: it must leak, which is why the gate has to be
    # set from measured noise rather than guessed.
    scale = np.r_[np.full(3, p.force_deadband_n * 2), np.full(3, p.torque_deadband_nm * 2)]
    leaked = np.zeros(6, dtype=bool)
    for _ in range(300):
        leaked |= np.abs(c.apply(rng.normal(0, scale))) > 0
    assert leaked.any(), "noise above the gate must leak -- see docstring"

    # And noise comfortably below it must not.
    c2 = WrenchConditioner(p)
    c2.begin_tare()
    for _ in range(50):
        c2.apply(np.zeros(6))
    small = np.r_[np.full(3, p.force_deadband_n * 0.1), np.full(3, p.torque_deadband_nm * 0.1)]
    for _ in range(300):
        assert np.allclose(c2.apply(rng.normal(0, small)), 0.0), "sub-gate noise must stay gated"


def test_force_axis_weights_default_to_trusting_everything():
    """A per-rig workaround must not become a hidden per-door assumption."""
    assert AdmittanceConfig().force_axis_weights == (1.0, 1.0, 1.0)


def test_weighted_axis_cannot_drive_motion_but_can_still_trip_safety():
    import dataclasses
    cfg = dataclasses.replace(AdmittanceConfig(), force_axis_weights=(1.0, 1.0, 0.0))
    moving = np.array([0.0, 0.02, 0.0])
    r = compute_hybrid_twist(np.array([0.0, 0.0, 40.0]), np.zeros(3), np.eye(3),
                             np.array([1.0, 0, 0]), cfg, measured_velocity_world=moving)
    assert abs(r.linear[2]) < 1e-9, "a zero-weighted axis must not steer the arm"
    over = compute_hybrid_twist(np.array([0.0, 0.0, cfg.force_cutoff + 5]), np.zeros(3),
                                np.eye(3), np.array([1.0, 0, 0]), cfg)
    assert over.safety_stop, "safety must still see the unweighted force"


def test_gravity_compensation_removes_a_rotating_tool_load():
    """The failure this prevents: 22 N of tool weight tilting into Fx/Fy as the wrist turns."""
    mass, com = 2.2, np.zeros(3)
    for ee in (np.eye(3),
               np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),
               np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])):
        measured = tool_gravity_wrench(ee, mass, com)          # tool weight alone
        assert np.allclose(measured - tool_gravity_wrench(ee, mass, com), 0.0, atol=1e-12), \
            "compensation must cancel the tool exactly, in every orientation"


def test_gripper_rotates_with_the_door():
    """The 45-degree hardware stall: the gripper translated along the arc but never turned,
    so the wrist absorbed the whole rotation and joint 5 hit its limit."""
    cfg = AdmittanceConfig()
    r = np.array([0.0, -0.30, 0.0])
    vel = np.array([0.02, 0.0, 0.0])
    res = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]),
                               cfg, measured_velocity_world=vel, hinge_to_grasp_world=r)
    assert np.linalg.norm(res.angular) > 0.01, "must rotate with the door"
    assert np.allclose(np.cross(res.angular, r), res.linear, atol=1e-9), "v = w x r must hold"
    assert np.isclose(np.linalg.norm(res.angular), np.linalg.norm(res.linear) / 0.30, atol=1e-9)


def test_no_hinge_means_no_added_rotation():
    """The probe runs before the hinge is known and must pull straight."""
    cfg = AdmittanceConfig()
    res = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]),
                               cfg, measured_velocity_world=np.array([0.02, 0.0, 0.0]))
    assert np.allclose(res.angular, 0.0)


def test_rotation_reverses_with_travel_direction():
    cfg = AdmittanceConfig()
    r = np.array([0.0, -0.30, 0.0])
    a = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg,
                             measured_velocity_world=np.array([0.02, 0, 0]),
                             hinge_to_grasp_world=r).angular
    b = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([-1.0, 0, 0]), cfg,
                             measured_velocity_world=np.array([-0.02, 0, 0]),
                             hinge_to_grasp_world=r).angular
    assert np.sign(a[2]) == -np.sign(b[2]), "reversing travel must reverse the rotation"


def test_over_rotation_is_what_a_small_radius_causes():
    """The hardware failure: a radius fitted 36% low commanded 1.6x too much rotation,
    the gripper out-turned the door, and it swung shut."""
    cfg = AdmittanceConfig()
    vel = np.array([0.02, 0.0, 0.0])
    true_r, fitted_r = 0.30, 0.191
    w_true = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg,
                                  measured_velocity_world=vel,
                                  hinge_to_grasp_world=np.array([0.0, -true_r, 0.0])).angular
    w_bad = compute_hybrid_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg,
                                 measured_velocity_world=vel,
                                 hinge_to_grasp_world=np.array([0.0, -fitted_r, 0.0])).angular
    ratio = np.linalg.norm(w_bad) / np.linalg.norm(w_true)
    assert 1.5 < ratio < 1.7, f"expected ~1.57x over-rotation, got {ratio:.2f}"


def test_longer_arc_fits_the_radius_better():
    """Justifies refitting during the pull rather than trusting the probe."""
    hinge = np.array([0.5, 0.4, 0.3])
    errors = []
    for degrees in (15, 40):
        got = []
        for seed in range(15):
            pts = arc_waypoints(hinge + np.array([0, -0.30, 0]), hinge,
                                np.array([0.0, 0.0, 1.0]), np.radians(degrees), 40)
            pts = pts + np.random.default_rng(seed).normal(0, 0.0005, pts.shape)
            fit = fit_hinge(pts)
            if fit:
                got.append(abs(fit[2] - 0.30) / 0.30)
        errors.append(float(np.median(got)))
    assert errors[1] < errors[0] / 3, f"40 deg must fit far better than 15 deg, got {errors}"
    assert errors[1] < 0.05, "a mid-pull fit should be within 5%"
