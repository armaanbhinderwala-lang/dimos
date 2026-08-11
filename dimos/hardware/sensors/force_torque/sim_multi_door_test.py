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

"""No-hardware validation: does ONE admittance policy open microwave, oven, AND
fridge doors, with no per-door config?

Drives the real, unmodified admittance_pull_law.compute_twist against a
lightweight rigid-hinge simulator -- not MuJoCo (not installed here, and no
appliance assets exist in this repo yet), and not a re-implementation of the
control law, just the physical door replaced with a numpy model. This is
deliberately a QUASI-STATIC kinematic model, not full rigid-body dynamics with
inertia: the real system is itself twist/velocity-commanded (Pink IK tracks a
commanded velocity, not a commanded force), so "the door advances at whatever
rate the controller's tangent-projected command implies" is a reasonably
faithful stand-in for the real closed loop, not a hand-wavy shortcut.

Per tick:
  1. Project the commanded twist onto the door's one true DOF (rotation about
     its axis) to get how fast the door actually turns this tick.
  2. Reaction wrench reported back = friction/latch resistance opposing that
     motion (Newton-scale, per preset) + a moderate penalty on whatever part
     of the twist ISN'T along the true tangent (this is what gives the
     rotational-compliance term something real to react to, and is what lets
     us check whether it actually converges onto the true axis over the pull,
     not just assume it does).
  3. Feed that wrench through the SAME rotate-to-world step the real sensor
     integration does, then call compute_twist -- unmodified production code.

Three presets, same two-phase probe/execute orchestration as
FTAdaptivePullModule (reimplemented here in ~15 lines since it's just loop
orchestration -- the control math itself is 100% imported, not duplicated).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dimos.hardware.sensors.force_torque.admittance_pull_law import AdmittanceConfig, compute_twist


@dataclass
class DoorPreset:
    name: str
    axis: np.ndarray  # world frame, unit vector
    pivot: np.ndarray  # world frame point the door hinges about
    handle_offset: np.ndarray  # initial (handle - pivot), world frame -- sets the radius
    theta_max_deg: float
    latch_angle_deg: float  # theta below this = still fighting the seal/latch
    breakaway_force_n: float
    kinetic_force_n: float
    breakaway_torque_nm: float
    kinetic_torque_nm: float


MICROWAVE = DoorPreset(
    "microwave", axis=np.array([0.0, 0.0, 1.0]), pivot=np.array([0.5, 0.35, 0.3]),
    handle_offset=np.array([0.0, -0.35, 0.0]), theta_max_deg=100, latch_angle_deg=3,
    breakaway_force_n=8, kinetic_force_n=2, breakaway_torque_nm=1.0, kinetic_torque_nm=0.3,
)
OVEN = DoorPreset(
    "oven", axis=np.array([0.0, 1.0, 0.0]), pivot=np.array([0.5, 0.0, 0.05]),
    handle_offset=np.array([0.0, 0.0, 0.30]), theta_max_deg=90, latch_angle_deg=3,
    breakaway_force_n=6, kinetic_force_n=3, breakaway_torque_nm=1.5, kinetic_torque_nm=0.6,
)
FRIDGE = DoorPreset(
    "fridge", axis=np.array([0.0, 0.0, 1.0]), pivot=np.array([0.5, 0.5, 0.5]),
    handle_offset=np.array([0.0, -0.50, 0.0]), theta_max_deg=110, latch_angle_deg=6,
    breakaway_force_n=35, kinetic_force_n=4, breakaway_torque_nm=4.0, kinetic_torque_nm=0.5,
)

K_PEN_TRANS = 150.0  # N per (m/s) of off-tangent linear command -- numerical constraint stiffness, not a measured value
# Must satisfy k_rot * K_PEN_ROT < 1 (discrete feedback loop gain through compute_twist's
# own omega = -k_rot*M) or this penalty alone oscillates/saturates every tick regardless of
# the door preset -- caught by exactly that happening (torque pegged near the cap, identical
# across all three presets, independent of their actual breakaway_torque_nm) at K_PEN_ROT=30
# against k_rot=0.05. 5 keeps the loop gain at 0.25, comfortably damped.
K_PEN_ROT = 5.0  # N*m per (rad/s) of off-tangent angular command
MAX_OMEGA_RATE = 1.0  # rad/s, sanity cap on how fast the sim door can move in one tick


def _rot_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _rot_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R with R @ a = b (a, b unit vectors) -- Rodrigues' rotation-between-vectors formula."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V / (1 + c)


def initial_ee_rot(preset: DoorPreset, local_drive: np.ndarray) -> np.ndarray:
    """A grasp orientation where local_drive (usually local -Z) points along the door's
    ACTUAL initial tangent direction -- same as how a real grasp roughly starts "pulling
    perpendicular to the door face," which is the tangent direction at theta=0. Without
    this, the sim's initial pull direction is physically arbitrary and the door never
    starts opening at all -- caught by actually running this against all three presets."""
    tangent0 = np.cross(preset.axis, preset.handle_offset)
    tangent0 /= np.linalg.norm(tangent0)
    return _rot_aligning(local_drive, tangent0)


class HingeDoorSim:
    def __init__(self, preset: DoorPreset, ee_rot0: np.ndarray):
        self.preset = preset
        self.theta = 0.0
        self._ee_rot0 = ee_rot0

    @property
    def ee_rot(self) -> np.ndarray:
        return _rot_about_axis(self.preset.axis, self.theta) @ self._ee_rot0

    @property
    def open_fraction(self) -> float:
        return self.theta / np.radians(self.preset.theta_max_deg)

    def step(self, v_cmd: np.ndarray, omega_cmd: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the door by one tick given a commanded world twist; returns the
        reaction (force_tool, torque_tool) the simulated sensor would report next."""
        p = self.preset
        r = _rot_about_axis(p.axis, self.theta) @ p.handle_offset
        radius = float(np.linalg.norm(r))
        tangent = np.cross(p.axis, r)
        tangent_hat = tangent / np.linalg.norm(tangent)

        # True rate derives from the LINEAR command's tangential projection only, not an
        # average of two independently-normalized estimates -- that average was manufacturing
        # phantom "mismatch" (and a phantom resisting force) even for a perfectly valid pull,
        # since half of a purely-tangential v_cmd would get counted as unmatched. This way
        # v_true is exactly v_cmd's own tangential component, so a clean pull has zero linear
        # mismatch by construction, and omega_mismatch becomes a meaningful, honest signal for
        # whether the rotational-compliance term is tracking the door's true rotation or not.
        requested_omega = float(np.dot(v_cmd, tangent_hat)) / max(radius, 1e-6)
        omega_true = float(np.clip(requested_omega, -MAX_OMEGA_RATE, MAX_OMEGA_RATE))
        theta_max = np.radians(p.theta_max_deg)
        if (self.theta <= 0 and omega_true < 0) or (self.theta >= theta_max and omega_true > 0):
            omega_true = 0.0
        self.theta = float(np.clip(self.theta + omega_true * dt, 0.0, theta_max))

        v_true = tangent_hat * (omega_true * radius)
        omega_true_vec = p.axis * omega_true
        v_mismatch = v_cmd - v_true
        omega_mismatch = omega_cmd - omega_true_vec

        still_latched = self.theta < np.radians(p.latch_angle_deg)
        force_mag = p.breakaway_force_n if still_latched else p.kinetic_force_n
        torque_mag = p.breakaway_torque_nm if still_latched else p.kinetic_torque_nm
        drive_sign = np.sign(requested_omega) if abs(requested_omega) > 1e-6 else 0.0

        f_world = -force_mag * drive_sign * tangent_hat - K_PEN_TRANS * v_mismatch
        m_world = -torque_mag * drive_sign * p.axis - K_PEN_ROT * omega_mismatch
        f_world = np.clip(f_world, -150, 150)
        m_world = np.clip(m_world, -25, 25)

        ee_rot = self.ee_rot  # after the update, i.e. the pose the NEXT tick's compute_twist will see
        return ee_rot.T @ f_world, ee_rot.T @ m_world


def run_pull(preset: DoorPreset, control_hz: float = 25.0, trace: list[dict] | None = None) -> None:
    """trace, if given, gets one row per tick appended -- for sim_multi_door_visualize.py.
    Purely additive: run_pull's own printed behavior is unchanged when trace=None."""
    local_drive = np.array([0.0, 0.0, -1.0])
    ee_rot0 = initial_ee_rot(preset, local_drive)
    sim = HingeDoorSim(preset, ee_rot0)
    dt = 1.0 / control_hz
    base = AdmittanceConfig()
    force_tool, torque_tool = np.zeros(3), np.zeros(3)
    t = 0.0

    def run_phase(name: str, cfg: AdmittanceConfig, max_progress_m: float, max_ticks: int) -> tuple[float, float, str, int]:
        nonlocal force_tool, torque_tool, t
        peak_force, peak_torque, progress, ticks = 0.0, 0.0, 0.0, 0
        stop_reason = "reached target"
        while ticks < max_ticks:
            ee_rot = sim.ee_rot
            drive_dir = ee_rot @ local_drive
            result = compute_twist(force_tool, torque_tool, ee_rot, drive_dir, cfg, progress_m=progress)
            if result.safety_stop:
                stop_reason = "safety cutoff"
                break
            force_tool, torque_tool = sim.step(result.linear, result.angular, dt)
            progress += float(np.linalg.norm(result.linear) * dt)
            peak_force = max(peak_force, result.resistance_force)
            peak_torque = max(peak_torque, result.torque_mag)
            ticks += 1
            t += dt
            if trace is not None:
                trace.append({
                    "t": round(t, 3), "phase": name, "theta_deg": round(np.degrees(sim.theta), 2),
                    "open_pct": round(100 * sim.open_fraction, 1),
                    "force": round(result.resistance_force, 2), "torque": round(result.torque_mag, 3),
                    "speed": round(float(np.linalg.norm(result.linear)), 4),
                })
            if progress >= max_progress_m:
                break
        else:
            stop_reason = "max_duration"
        return peak_force, peak_torque, stop_reason, ticks

    print(f"\n=== {preset.name} ===")
    probe_cfg = replace(base, drive_speed=0.04, decel_start_m=0.02, decel_full_m=0.04)
    probe_peak_f, probe_peak_t, probe_reason, probe_ticks = run_phase("probe", probe_cfg, max_progress_m=0.04, max_ticks=int(5 * control_hz))
    print(f"probe:   peak_force={probe_peak_f:5.1f}N peak_torque={probe_peak_t:4.2f}Nm  ({probe_reason}, {probe_ticks} ticks)")

    if probe_reason == "safety cutoff":
        print("  -> probe itself hit a safety cutoff, would NOT proceed to execute on real hardware.")
        return

    force_cutoff = max(probe_peak_f * 2.5, 30.0)
    torque_cutoff = max(probe_peak_t * 2.5, 4.0)
    exec_cfg = replace(base, drive_speed=0.02, force_cutoff=force_cutoff, torque_cutoff=torque_cutoff,
                        decel_start_m=0.4, decel_full_m=0.8)
    exec_peak_f, exec_peak_t, exec_reason, exec_ticks = run_phase("execute", exec_cfg, max_progress_m=0.8, max_ticks=int(30 * control_hz))
    print(f"execute: peak_force={exec_peak_f:5.1f}N peak_torque={exec_peak_t:4.2f}Nm  ({exec_reason}, {exec_ticks} ticks, "
          f"calibrated cutoffs {force_cutoff:.1f}N/{torque_cutoff:.1f}Nm)")
    print(f"final door angle: {np.degrees(sim.theta):.1f}deg / {preset.theta_max_deg}deg "
          f"({100 * sim.open_fraction:.0f}% open)")


if __name__ == "__main__":
    for preset in (MICROWAVE, OVEN, FRIDGE):
        run_pull(preset)
