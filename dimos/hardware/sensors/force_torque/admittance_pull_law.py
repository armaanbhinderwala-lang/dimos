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


@dataclass
class AdmittanceConfig:
    drive_speed: float = 0.03  # m/s, base speed along the drive direction when unresisted
    k_trans: float = 0.003  # (m/s)/N, lateral compliance gain -- UNVERIFIED, retune after first real pull
    k_rot: float = 0.05  # (rad/s)/(N*m), rotational compliance gain -- UNVERIFIED, retune after first real pull
    max_lateral_speed: float = 0.06  # m/s cap on the compliant (non-drive) linear velocity
    max_rotation_rate: float = 0.35  # rad/s cap (~20 deg/s)
    force_cutoff: float = 80.0  # N, total force magnitude -- hard stop, same number as the old lateral-only cutoff
    torque_cutoff: float = 15.0  # N*m -- UNVERIFIED placeholder, no prior data point for this axis, tune down after first test
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

    # Manipulability (smallest singular value of the tool Jacobian) below which speed
    # tapers off, reaching zero at singularity_sigma_stop -- see _singularity_speed_scale.
    # UNVERIFIED placeholders: motivated by a real hardware fault (a joint snap near a
    # singularity tripped the FT sensor's own overload protection, error 53) but these
    # exact thresholds have no calibration data yet. The module logs sigma_min every
    # tick summary specifically so real numbers can replace these.
    singularity_sigma_caution: float = 0.05
    singularity_sigma_stop: float = 0.01


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

    print("\nAll self-tests passed.")
