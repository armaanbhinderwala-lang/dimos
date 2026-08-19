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

"""General door/drawer-opening control law: one driven axis, five compliant.

Replaces the old fixed-pivot/fixed-axis law (assume rotation about local Z at
a fixed pivot_distance, decide direction from Fx alone). Instead: pick one
drive direction (the initial approach guess, e.g. straight back), scale its
speed by how much it's resisting -- reusing Yashas's force-adaptive bands,
just keyed on total force magnitude instead of lateral-only so a straight-pull
resistance (a fridge's magnetic seal) triggers it same as a sideways one. Every
other DOF is compliant: lateral force and ALL THREE torque axes each drive
their own velocity directly (v = -k*F, omega = -k*M), letting the door's own
resistance steer the gripper along whatever its true constraint turns out to
be (vertical hinge, horizontal hinge, or a drawer showing near-zero rotational
resistance) with no explicit axis or pivot assumption anywhere.

Why no pivot_distance: an admittance law doesn't need its open-loop command to
already be geometrically consistent with the door's true screw motion. If it
isn't, the mismatch shows up as extra constraint force next tick, which the
same compliance terms react to -- the closed loop converges on the true
constraint by itself. Manually converting a decided rotation into a
pivot-relative translation (the old approach) is an assumption that can be
wrong with nothing to correct it; not doing that removes a failure mode.

Frame note: the FT sensor reports in its own frame (see read_FTModule.py,
frame_id="ft_sensor_link"), but the twist command is WORLD-frame (confirmed
against eef_twist_task.py in ft_pull_module.py's own docstring) -- force and
torque are rotated by the current EE orientation before use here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# UFACTORY 6-axis FT sensor datasheet: rated 150N (Fx/Fy) / 200N (Fz), 4N*m (any torque axis);
# overload trips at 150% of rated. torque_cutoff/force_cutoff below must stay under these --
# error 53 ("sensor overloaded or reading exceeds limit") is the sensor's OWN hardware trip,
# faster and stricter than anything our software checks once per tick. A software cutoff set
# above these numbers can never actually protect anything -- the hardware faults first, always.
SENSOR_FORCE_OVERLOAD_N = 225.0  # 150N rated x 1.5, the smaller (more conservative) of Fx/Fy/Fz
SENSOR_TORQUE_OVERLOAD_NM = 6.0  # 4N*m rated x 1.5, same on all three torque axes


@dataclass
class AdmittanceConfig:
    drive_speed: float = 0.03  # m/s, base speed along the drive direction when unresisted
    k_trans: float = 0.003  # (m/s)/N, lateral compliance gain -- UNVERIFIED, retune after first real pull
    # 0.01, not 0.05: on the DIY sensor the smallest trustworthy torque change is ~0.7 N*m
    # (measured), so a high gain turns sensor noise into wrist motion. Torque steers coarsely
    # here; force does the work.
    # 0.03: with door-following supplying the bulk of the rotation, this term's job is to
    # correct the residual. Hardware shows 0.8-2.5 N*m during a pull -- well above the 0.35
    # N*m deadband -- so it has real signal to work with, and it opposes over-rotation.
    k_rot: float = 0.03  # (rad/s)/(N*m)
    max_lateral_speed: float = 0.03  # m/s -- half the drive speed, so compliance cannot dominate
    max_rotation_rate: float = 0.15  # rad/s (~9 deg/s) -- smoothness over responsiveness
    force_cutoff: float = 80.0  # N, total force magnitude -- comfortably under SENSOR_FORCE_OVERLOAD_N
    torque_cutoff: float = 4.5  # N*m -- was 15.0, ABOVE the sensor's real ~6N*m overload rating; grounded in the datasheet now
    # (resistance_force upper bound N, speed multiplier) -- max capped at 1.0, unlike Yashas's original
    # 1.5x-when-free band: under admittance a stuck door doesn't need extra commanded
    # speed to break free (steady moderate velocity still builds real reaction force
    # through the arm's own stiffness), and speeding up the instant resistance drops
    # is exactly what produced the "opens too fast right after the latch releases"
    # behavior on the microwave -- so low resistance now means "at most base speed,"
    # never faster.
    speed_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(15, 1.0), (25, 0.8), (40, 0.6), (60, 0.4), (float("inf"), 0.2)]
    )
    # Progress-based taper, independent of instantaneous resistance: once
    # total drive distance passes decel_start, ease the speed multiplier down
    # to decel_floor by decel_full. This is what actually produces "goes slower
    # once past the lock" -- resistance alone can't, since it drops to ~0 right
    # when the lock releases, which is the one moment you want to be cautious,
    # not fast. Defaults are a starting guess (roughly: cruise for the first
    # 5cm, be down to 35% speed by 25cm) -- retune once you see real pull-distance numbers.
    decel_start_m: float = 0.05
    decel_full_m: float = 0.25
    decel_floor_scale: float = 0.35

    # Jerk limit -- caps how much the PUBLISHED twist can change tick-to-tick,
    # independent of what compute_twist would otherwise jump straight to. Same
    # role as acceleration/jerk limiting on any industrial motion controller:
    # protects against a single discontinuous command reaching the joints
    # instantly, whatever caused the discontinuity (sensor glitch, a real force
    # snag, a singularity). Applied by the module (see slew_limit), not here --
    # needs cross-tick state this pure function doesn't keep.
    # Tightened for smoothness: with a 6 N noise floor the commanded twist jitters, and the
    # slew limit is what stops that reaching the joints as visible judder.
    max_linear_accel: float = 0.15  # m/s^2
    max_angular_accel: float = 0.5  # rad/s^2

    # Manipulability (smallest singular value of the tool Jacobian) below which speed
    # tapers off, reaching zero at singularity_sigma_stop -- see _singularity_speed_scale.
    # UNVERIFIED placeholders: motivated by a real hardware fault (a joint snap near a
    # singularity tripped the FT sensor's own overload protection, error 53) but these
    # exact thresholds have no calibration data yet. The module logs sigma_min every
    # tick summary specifically so real numbers can replace these.
    singularity_sigma_caution: float = 0.05
    singularity_sigma_stop: float = 0.01

    # How fast the drive direction chases measured velocity (see steer_drive_direction).
    # DEFAULT 0.0 -- OFF, because measurement says it does not help. The tool's -Z already
    # tracks the door's opening tangent at 0.99 alignment from the first tick, so there is
    # almost nothing for steering to correct; enabling it scored 21-27 deg against 27 deg
    # fixed. Kept because it is the right mechanism for a door whose arc the initial guess
    # does NOT already match (a side-hinged fridge approached head-on, say), but it is not
    # the fix for the stall we are chasing.
    drive_steer_blend: float = 0.0

    # Hybrid force/velocity control (compute_hybrid_twist).
    # World-frame weights on the measured force, [x y z]. Default trusts all three.
    # Set (1, 1, 0) only when the sensor's Fz is untrustworthy AND the door swings on a
    # VERTICAL hinge (microwave, fridge, cabinet), where vertical force carries no motion.
    # An oven or dishwasher hinges horizontally and moves through the vertical plane, so
    # zeroing Fz there deletes the axis the door actually travels along.
    force_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Radial load below this is ignored. Raised from 3.0: after the deadband the residual
    # noise is a few newtons, and with the compliance gain raised, chasing it would jitter.
    contact_force_n: float = 8.0
    min_motion_speed: float = 0.003     # m/s, below this motion is too slow to read a tangent from
    # 0.0, not 5.0: while the door is moving there is no reason to hold a sideways load
    # against it. Maintaining 5 N is 5 N the appliance has to resist.
    desired_radial_force: float = 0.0
    k_force: float = 0.004              # (m/s)/N, how hard radial force error is corrected
    # Radial correction is capped at this FRACTION of the tangential speed. Hardware ran at
    # 10-33 N against a 12 N saturation point, so the arm retreated radially as fast as it
    # advanced -- which walks the gripper toward the hinge and folds the door shut. Bounding
    # it relatively keeps forward progress guaranteed however wrong the radial estimate is.
    # 1.5, was 0.4. At 0.4 this clip -- not k_force -- was what made the arm stiff sideways:
    # a 20 N error asks for 60 mm/s of relief and was allowed 8. The arm behaved like
    # ~14,000 N/m where a human hand is ~100 N/m, so any error in the arc became force on
    # the appliance rather than motion.
    max_radial_fraction: float = 1.5


@dataclass
class TwistResult:
    linear: np.ndarray  # world frame, m/s
    angular: np.ndarray  # world frame, rad/s
    resistance_force: float  # N, total |F_world|, the signal driving the speed schedule
    torque_mag: float  # N*m, total |M_world|
    safety_stop: bool  # True if force_cutoff or torque_cutoff was exceeded this tick


def rotate_wrench_to_world(force_tool: np.ndarray, torque_tool: np.ndarray, ee_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ee_rot: 3x3 world_R_tool. Wrench axes rotate the same way any vector does, no r x F correction --
    that would only apply if changing which POINT the wrench is measured about, not just its frame."""
    return ee_rot @ force_tool, ee_rot @ torque_tool


def slew_limit(prev: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """Move from prev toward target by at most max_delta. Direction-preserving, not per-axis."""
    delta = target - prev
    norm = np.linalg.norm(delta)
    if norm <= max_delta or norm < 1e-12:
        return target
    return prev + delta * (max_delta / norm)


def fit_hinge(points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit the door's circle from where the gripper has been: (centre, axis, radius).

    None when the motion is too straight to call -- a drawer, or too little travel.
    """
    pts = np.asarray(points_world, float)
    if len(pts) < 5:
        return None
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    # Weakest singular direction is the plane normal, i.e. the hinge axis.
    _, sing, vt = np.linalg.svd(centred, full_matrices=False)
    if sing[1] < 1e-12:
        return None
    axis, u, v = vt[2], vt[0], vt[1]
    x, y = centred @ u, centred @ v
    # Algebraic circle fit: x^2 + y^2 = 2*cx*x + 2*cy*y + c
    sol, *_ = np.linalg.lstsq(np.column_stack([2 * x, 2 * y, np.ones(len(x))]),
                              x**2 + y**2, rcond=None)
    cx, cy, c = sol
    radius_sq = c + cx**2 + cy**2
    if radius_sq <= 0:
        return None
    radius = float(np.sqrt(radius_sq))
    # A near-straight sweep "fits" a huge circle with a meaningless centre; demand curvature.
    span = float(np.linalg.norm(pts[-1] - pts[0]))
    if radius > 5.0 or span < 1e-9 or radius / span > 50.0:
        return None
    return centroid + cx * u + cy * v, axis, radius


def arc_waypoints(
    grasp_world: np.ndarray,
    hinge_point_world: np.ndarray,
    hinge_axis_world: np.ndarray,
    max_angle_rad: float,
    count: int = 10,
) -> np.ndarray:
    """Handle positions along the door's arc. Known once the hinge is, so reachability can be
    checked before the pull rather than discovered halfway through it."""
    axis = np.asarray(hinge_axis_world, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    radial = np.asarray(grasp_world, float) - np.asarray(hinge_point_world, float)
    radial = radial - np.dot(radial, axis) * axis
    out = []
    for angle in np.linspace(0.0, max_angle_rad, count):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = radial * cos_a + np.cross(axis, radial) * sin_a
        out.append(np.asarray(hinge_point_world, float) + rotated)
    return np.array(out)


def compute_hybrid_twist(
    force_tool: np.ndarray,
    torque_tool: np.ndarray,
    ee_rot: np.ndarray,
    drive_direction_world: np.ndarray,
    cfg: AdmittanceConfig,
    measured_velocity_world: np.ndarray | None = None,
    hinge_to_grasp_world: np.ndarray | None = None,
    hinge_axis_world: np.ndarray | None = None,
    follow_scale: float = 1.0,
    progress_m: float = 0.0,
    singularity_scale: float = 1.0,
) -> TwistResult:
    """Velocity along the tangent, force regulated on the radial (Karayiannidis, IROS 2012).

    compute_twist commands velocity in every direction including the constrained one, so the
    unusable component becomes position error each tick and force climbs without bound. Here a
    wrong tangent costs a little force instead. Needs no hinge, radius or door model.
    """
    if measured_velocity_world is None:
        measured_velocity_world = np.zeros(3)
    f_world, m_world = rotate_wrench_to_world(force_tool, torque_tool, ee_rot)
    # Safety sees the FULL force; only the control terms use the weighted one, so a weighted
    # axis can still trip a cutoff even though it never drives motion.
    resistance_force = float(np.linalg.norm(f_world))
    torque_mag = float(np.linalg.norm(m_world))

    if resistance_force > cfg.force_cutoff or torque_mag > cfg.torque_cutoff:
        return TwistResult(np.zeros(3), np.zeros(3), resistance_force, torque_mag, True)

    f_world = f_world * np.asarray(cfg.force_axis_weights, float)

    speed = cfg.drive_speed * _progress_scale(
        progress_m, cfg.decel_start_m, cfg.decel_full_m, cfg.decel_floor_scale
    )

    drive_hat = drive_direction_world / max(float(np.linalg.norm(drive_direction_world)), 1e-12)
    # With a committed hinge the door allows exactly one direction: perpendicular to the
    # radius, in the hinge plane. Take it from geometry. Steering off measured velocity makes
    # the command follow whatever the arm last did, so a shove is adopted as the new heading
    # and the door can be walked back shut. Geometry cannot reverse.
    tangent = None
    if hinge_to_grasp_world is not None and hinge_axis_world is not None:
        axis = np.asarray(hinge_axis_world, float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        r = np.asarray(hinge_to_grasp_world, float)
        r = r - np.dot(r, axis) * axis
        if np.linalg.norm(r) > 1e-6:
            t = np.cross(axis, r / np.linalg.norm(r))
            if np.linalg.norm(t) > 1e-9:
                t = t / np.linalg.norm(t)
                # Opening sense, fixed by the tool's own pull axis, which turns with the door.
                tangent = t if float(np.dot(t, drive_hat)) >= 0.0 else -t
    if tangent is None:
        speed_now = float(np.linalg.norm(measured_velocity_world))
        tangent = (measured_velocity_world / speed_now
                   if speed_now > cfg.min_motion_speed else drive_hat)

    # Only the part of the force perpendicular to travel is fighting the constraint; the
    # tangential part is friction, the price of moving.
    radial_force = f_world - np.dot(f_world, tangent) * tangent
    radial_mag = float(np.linalg.norm(radial_force))
    if radial_mag > cfg.contact_force_n:
        radial = radial_force / radial_mag
        limit = min(cfg.max_lateral_speed, cfg.max_radial_fraction * speed)
        correction = float(np.clip(cfg.k_force * (cfg.desired_radial_force - radial_mag),
                                   -limit, limit))
        linear = speed * tangent + correction * radial
    else:
        linear = speed * tangent

    # Hand rotation over from torque compliance to geometry as following ramps in. Compliance
    # is a trim term -- on this sensor it reached 0.11 rad/s against the 0.054 the arc wanted,
    # and its mz row is uncalibrated, so letting it run alongside fights the arc.
    follow_now = float(np.clip(follow_scale, 0.0, 1.0)) if hinge_to_grasp_world is not None else 0.0
    omega = -(1.0 - follow_now) * cfg.k_rot * m_world
    # Turn the gripper WITH the door. A body rotating about a hinge satisfies v = w x r, so
    # w = (r x v) / |r|^2 -- exact, and the sign falls out of the geometry. A gripper that only
    # translates along the arc makes the wrist absorb the whole rotation, which is what drove
    # joint 5 into its limit at 45 degrees on hardware. Torque compliance cannot supply this:
    # the moment is small, mostly inside the deadband, so omega stayed at zero all run.
    if hinge_to_grasp_world is not None:
        r = np.asarray(hinge_to_grasp_world, float)
        r_sq = float(r @ r)
        if r_sq > 1e-6:
            # follow_scale ramps this in: switching full rotation on at a threshold is a step
            # the wrist feels as a jolt, and the radius estimate is still settling at that point.
            omega = omega + float(np.clip(follow_scale, 0.0, 1.0)) * np.cross(r, linear) / r_sq

    omega_norm = float(np.linalg.norm(omega))
    if omega_norm > cfg.max_rotation_rate:
        omega *= cfg.max_rotation_rate / omega_norm

    return TwistResult(linear * singularity_scale, omega * singularity_scale,
                       resistance_force, torque_mag, False)


def steer_drive_direction(
    drive_direction_world: np.ndarray,
    measured_velocity_world: np.ndarray,
    blend: float,
    min_speed: float = 0.002,
) -> np.ndarray:
    """Turn the drive direction toward the direction the tool is ACTUALLY travelling.

    A fixed drive direction is only correct for a straight pull. A hinged door moves its
    handle along an arc, so the direction that makes progress rotates continuously; holding
    the original one means an ever-growing share of the command pushes into the constraint
    rather than along it, and that shows up as rising force with the door barely moving.

    Steering toward measured velocity needs no hinge axis, no radius and no per-door config:
    the constraint itself decides which way the tool can go, and the command follows. A
    drawer moves in a straight line and this leaves the direction alone; a door curves and
    the direction curves with it. Below min_speed there is no reliable direction to read, so
    the current one is kept rather than chasing noise.
    """
    speed = float(np.linalg.norm(measured_velocity_world))
    if speed < min_speed:
        return drive_direction_world
    target = measured_velocity_world / speed
    blended = drive_direction_world + blend * (target - drive_direction_world)
    norm = float(np.linalg.norm(blended))
    if norm < 1e-9:
        return drive_direction_world
    return blended / norm


def _speed_scale(resistance_force: float, bands: list[tuple[float, float]]) -> float:
    for upper, scale in bands:
        if resistance_force < upper:
            return scale
    return bands[-1][1]


def _progress_scale(progress_m: float, decel_start_m: float, decel_full_m: float, floor: float) -> float:
    if progress_m <= decel_start_m:
        return 1.0
    if progress_m >= decel_full_m:
        return floor
    frac = (progress_m - decel_start_m) / (decel_full_m - decel_start_m)
    return 1.0 - frac * (1.0 - floor)


def singularity_speed_scale(sigma_min: float, sigma_caution: float, sigma_stop: float) -> float:
    """1.0 when well-conditioned (sigma_min >= sigma_caution), tapering linearly to 0.0 at
    sigma_min <= sigma_stop. sigma_min is the smallest singular value of the tool-frame
    Jacobian -- as it shrinks, a small commanded Cartesian twist demands a disproportionately
    large joint velocity to track, which is the actual mechanism (not resistance, not gain
    instability) behind a sudden joint "snap" near a singularity."""
    if sigma_min >= sigma_caution:
        return 1.0
    if sigma_min <= sigma_stop:
        return 0.0
    return (sigma_min - sigma_stop) / (sigma_caution - sigma_stop)


def compute_twist(
    force_tool: np.ndarray,
    torque_tool: np.ndarray,
    ee_rot: np.ndarray,
    drive_direction_world: np.ndarray,
    cfg: AdmittanceConfig,
    progress_m: float = 0.0,
    singularity_scale: float = 1.0,
) -> TwistResult:
    """progress_m: total drive distance covered so far this pull (module tracks and passes
    this in) -- resistance alone can't signal "slow down now," since it's lowest right when
    a latch just released, exactly the moment that needs care, not speed.

    singularity_scale: 0..1, computed by the module from the live Jacobian (see
    singularity_speed_scale) and passed in -- kept out of this function's own math since it
    needs the robot's kinematic model, which this pure law deliberately has no dependency on."""
    f_world, m_world = rotate_wrench_to_world(force_tool, torque_tool, ee_rot)
    resistance_force = float(np.linalg.norm(f_world))
    torque_mag = float(np.linalg.norm(m_world))

    safety_stop = resistance_force > cfg.force_cutoff or torque_mag > cfg.torque_cutoff
    if safety_stop:
        return TwistResult(np.zeros(3), np.zeros(3), resistance_force, torque_mag, True)

    f_along = float(np.dot(f_world, drive_direction_world))
    f_lateral = f_world - f_along * drive_direction_world

    speed_scale = _speed_scale(resistance_force, cfg.speed_bands) * _progress_scale(
        progress_m, cfg.decel_start_m, cfg.decel_full_m, cfg.decel_floor_scale
    )
    v_drive = drive_direction_world * cfg.drive_speed * speed_scale
    v_lateral = -cfg.k_trans * f_lateral
    lateral_norm = np.linalg.norm(v_lateral)
    if lateral_norm > cfg.max_lateral_speed:
        v_lateral *= cfg.max_lateral_speed / lateral_norm

    omega = -cfg.k_rot * m_world
    omega_norm = np.linalg.norm(omega)
    if omega_norm > cfg.max_rotation_rate:
        omega *= cfg.max_rotation_rate / omega_norm

    linear = (v_drive + v_lateral) * singularity_scale
    angular = omega * singularity_scale
    return TwistResult(linear, angular, resistance_force, torque_mag, False)


if __name__ == "__main__":
    # Synthetic self-test -- no hardware needed.
    cfg = AdmittanceConfig()

    print("=== Case 1: no resistance, no progress yet -- drive at exactly base speed, never boosted ===")
    r = compute_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg)
    print(f"linear={r.linear} angular={r.angular} resistance={r.resistance_force:.2f}N")
    assert np.allclose(r.linear, [cfg.drive_speed, 0, 0], atol=1e-9), "must never exceed base speed when free"
    assert np.allclose(r.angular, 0)

    print("\n=== Case 2: pure lateral resistance, identity orientation (tool frame == world frame) ===")
    r = compute_twist(np.array([0.0, 20.0, 0.0]), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg)
    print(f"linear={r.linear} resistance={r.resistance_force:.2f}N")
    assert r.linear[1] < 0, "should push back against +Y force with -Y compliant velocity"
    assert np.isclose(r.resistance_force, 20.0)

    print("\n=== Case 3: frame rotation -- 90deg about Z, tool-frame torque about tool's own Z ===")
    # ee_rot rotates tool +X to world +Y, tool +Y to world -X, tool Z stays Z.
    ee_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    torque_tool = np.array([0.0, 0.0, 4.0])
    r = compute_twist(np.zeros(3), torque_tool, ee_rot, np.array([1.0, 0, 0]), cfg)
    expected_m_world = ee_rot @ torque_tool  # Z stays Z under this rotation
    print(f"angular={r.angular} expected_direction={-cfg.k_rot * expected_m_world}")
    assert np.allclose(r.angular, -cfg.k_rot * expected_m_world)

    print("\n=== Case 4: frame rotation matters -- 90deg about X, torque about tool's Y should NOT show up as world Y ===")
    ee_rot2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # tool +Y -> world +Z
    torque_tool2 = np.array([0.0, 3.0, 0.0])
    r2 = compute_twist(np.zeros(3), torque_tool2, ee_rot2, np.array([1.0, 0, 0]), cfg)
    print(f"angular={r2.angular} (torque about tool-Y with this rotation should land on world Z, not Y)")
    assert abs(r2.angular[1]) < 1e-9 and abs(r2.angular[2]) > 1e-9

    print("\n=== Case 5: lateral velocity saturates at max_lateral_speed ===")
    r = compute_twist(np.array([0.0, 500.0, 0.0]), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg)
    print(f"|lateral|={np.linalg.norm(r.linear[1:]):.4f} cap={cfg.max_lateral_speed}")
    # 500N total force alone exceeds force_cutoff, so this should actually be a safety stop --
    # use a smaller value that saturates lateral without tripping the cutoff instead.
    r = compute_twist(np.array([0.0, 40.0, 0.0]), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg)
    assert np.isclose(np.linalg.norm(r.linear - r.linear[0] * np.array([1, 0, 0])), cfg.max_lateral_speed, atol=1e-6)
    print("lateral saturation OK")

    print("\n=== Case 6: force cutoff trips safety_stop, zero twist ===")
    r = compute_twist(np.array([90.0, 0.0, 0.0]), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg)
    print(f"safety_stop={r.safety_stop} linear={r.linear}")
    assert r.safety_stop and np.allclose(r.linear, 0) and np.allclose(r.angular, 0)

    print("\n=== Case 7: torque cutoff trips safety_stop too, independent of force ===")
    r = compute_twist(np.zeros(3), np.array([0.0, 0.0, 20.0]), np.eye(3), np.array([1.0, 0, 0]), cfg)
    print(f"safety_stop={r.safety_stop}")
    assert r.safety_stop

    print("\n=== Case 8: speed schedule is monotonically non-increasing in resistance, capped at 1.0 ===")
    forces = [0, 5, 12, 20, 30, 50, 70]
    speeds = [_speed_scale(f, cfg.speed_bands) for f in forces]
    print(list(zip(forces, speeds, strict=True)))
    assert all(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1))
    assert max(speeds) <= 1.0, "must never boost above base speed just because resistance is low"

    print("\n=== Case 9: progress-based deceleration -- zero resistance throughout, but speed still tapers with distance ===")
    progresses = [0.0, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40]
    prog_scales = [_progress_scale(p, cfg.decel_start_m, cfg.decel_full_m, cfg.decel_floor_scale) for p in progresses]
    print(list(zip(progresses, prog_scales, strict=True)))
    assert prog_scales[0] == 1.0 and prog_scales[1] == 1.0, "no taper before decel_start_m"
    assert all(prog_scales[i] >= prog_scales[i + 1] for i in range(len(prog_scales) - 1)), "must not increase"
    assert prog_scales[-1] == cfg.decel_floor_scale, "clamps at the floor past decel_full_m"
    # Direct check against compute_twist: same zero-resistance case as Case 1, but well past
    # decel_full_m -- speed should be at the floor, not base speed, even though resistance is 0.
    r_far = compute_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg, progress_m=0.40)
    print(f"far-progress linear={r_far.linear} (expect {cfg.drive_speed * cfg.decel_floor_scale:.4f} along x)")
    assert np.isclose(r_far.linear[0], cfg.drive_speed * cfg.decel_floor_scale)

    print("\n=== Case 10: singularity scaling -- well-conditioned unaffected, tapers to exactly zero at/below sigma_stop ===")
    sigmas = [0.20, 0.05, 0.03, 0.01, 0.005]
    sing_scales = [singularity_speed_scale(s, cfg.singularity_sigma_caution, cfg.singularity_sigma_stop) for s in sigmas]
    print(list(zip(sigmas, sing_scales, strict=True)))
    assert sing_scales[0] == 1.0, "well-conditioned (sigma_min >= caution) must be unaffected"
    assert sing_scales[-1] == 0.0 and sing_scales[-2] == 0.0, "at/below sigma_stop must be exactly zero, not just small"
    assert all(sing_scales[i] >= sing_scales[i + 1] for i in range(len(sing_scales) - 1)), "must not increase as sigma_min shrinks"
    # Direct check against compute_twist: same no-resistance case as Case 1, but near-singular --
    # output should scale down proportionally, not just the drive term.
    r_sing = compute_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg, singularity_scale=0.5)
    print(f"half-scale linear={r_sing.linear} (expect {cfg.drive_speed * 0.5:.4f} along x)")
    assert np.isclose(r_sing.linear[0], cfg.drive_speed * 0.5)
    r_sing_stop = compute_twist(np.zeros(3), np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), cfg, singularity_scale=0.0)
    assert np.allclose(r_sing_stop.linear, 0) and np.allclose(r_sing_stop.angular, 0)

    print("\n=== Case 11: slew limiting -- caps a discontinuous jump, passes through small changes unchanged ===")
    prev = np.array([0.02, 0.0, 0.0])
    big_jump = np.array([0.02, 0.5, 0.0])  # a 0.5 m/s sideways jump in one tick
    limited = slew_limit(prev, big_jump, max_delta=0.05)
    print(f"prev={prev} target={big_jump} limited={limited} |delta|={np.linalg.norm(limited - prev):.4f}")
    assert np.isclose(np.linalg.norm(limited - prev), 0.05), "must move exactly max_delta toward target, not jump"
    step, full = limited - prev, big_jump - prev
    assert np.allclose(step / np.linalg.norm(step), full / np.linalg.norm(full)), "step direction must match prev->target"
    small_change = prev + np.array([0.0, 0.01, 0.0])
    assert np.allclose(slew_limit(prev, small_change, max_delta=0.05), small_change), "small changes pass through untouched"

    print("\n=== Case 12: arc waypoints trace a circle about the hinge ===")
    hinge = np.array([0.5, 0.4, 0.3]); grasp = np.array([0.5, 0.0, 0.3])
    pts = arc_waypoints(grasp, hinge, np.array([0.0, 0.0, 1.0]), np.radians(90), count=10)
    radii = [np.linalg.norm(p - hinge) for p in pts]
    print(f"radius over the arc: min={min(radii):.4f} max={max(radii):.4f} (must be constant)")
    assert np.allclose(radii, radii[0], atol=1e-9), "a hinge cannot change the radius"
    assert np.allclose(pts[0], grasp), "the arc must start at the grasp"
    assert np.isclose(np.linalg.norm(pts[-1] - hinge), 0.4)
    assert np.allclose([p[2] for p in pts], 0.3), "a vertical hinge keeps height constant"
    # 90 degrees about +Z takes (0,-0.4) to (0.4, 0)
    assert np.allclose(pts[-1], hinge + np.array([0.4, 0.0, 0.0]), atol=1e-9), pts[-1]

    print("\n=== Case 13: with the door moving, never push into the RADIAL load ===")
    ee = np.eye(3)
    moving = np.array([0.0, 0.03, 0.0])          # travelling +Y, so +Y is the tangent
    for fx in (10.0, 30.0, 60.0):
        f = np.array([fx, 0.0, 0.0])             # constraint pushes back along +X
        r = compute_hybrid_twist(f, np.zeros(3), ee, np.array([1.0, 0, 0]), cfg,
                                 measured_velocity_world=moving)
        print(f"  radial |F|={fx:5.1f}N -> radial vel {r.linear[0]:+.4f}, tangential {r.linear[1]:+.4f}")
        assert r.linear[0] < 0, "must retreat along a radial that is already loaded"
        assert r.linear[1] > 0, "and must keep making progress along the tangent"

    print("\n=== Case 14: before it breaks free, push along the hint (a latch needs breaking) ===")
    r = compute_hybrid_twist(np.array([20.0, 0, 0]), np.zeros(3), ee, np.array([1.0, 0, 0]), cfg,
                             measured_velocity_world=np.zeros(3))
    print(f"  stationary, head-on resistance -> {r.linear[0]:+.4f} m/s along the drive")
    assert np.isclose(r.linear[0], cfg.drive_speed), "a stalled door must still be pushed"

    print("\n=== Case 15: friction along the tangent is NOT treated as radial load ===")
    # Force purely opposing motion is the cost of moving, not the arm fighting the constraint.
    r = compute_hybrid_twist(np.array([0.0, -40.0, 0.0]), np.zeros(3), ee, np.array([1.0, 0, 0]),
                             cfg, measured_velocity_world=moving)
    print(f"  40N of pure drag -> radial correction {np.linalg.norm(r.linear - r.linear[1] * np.array([0, 1.0, 0])):.5f} m/s")
    assert np.allclose(r.linear, [0, cfg.drive_speed, 0], atol=1e-9), "drag must not trigger retreat"

    print("\n=== Case 16: the tangent follows measured motion, so the arc is tracked ===")
    for vel, want in [(np.array([0.0, 0.03, 0.0]), 1), (np.array([0.0, -0.03, 0.0]), -1)]:
        r = compute_hybrid_twist(np.zeros(3), np.zeros(3), ee, np.array([1.0, 0, 0]), cfg,
                                 measured_velocity_world=vel)
        print(f"  moving {vel} -> commanded {np.round(r.linear, 4)}")
        assert np.sign(r.linear[1]) == want, "must keep going the way the door is actually opening"

    print("\nAll self-tests passed.")
