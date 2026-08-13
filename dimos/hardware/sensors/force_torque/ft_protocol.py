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

"""Static-weight calibration protocol: 10 runs of known, gravity-derived loads.

Why this matters beyond cleaner data: a hanging mass at a measured lever arm has a wrench
you can WRITE DOWN. That is a third reference, independent of both sensors, so for the first
time the uFactory becomes something we can test rather than something we assume.

Conventions, all in the DIY sensor frame, all SI:
    gravity      unit vector along which gravity pulls, expressed in the sensor frame
    r            attachment point relative to the sensor origin, metres
    F = m*g*ghat force a hanging mass applies
    tau = r x F  moment it makes about the sensor origin

A couple (Run 4) is the one load gravity cannot make on its own: two hanging masses both pull
down, so their moments cancel and you get 2mg of axial force instead of torsion. It needs
cables over pulleys pulling tangentially in opposite directions -- hence force=0, tau=2*L*T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665

# Sensor +Z up. Override per run when the sensor is mounted on its side.
GRAVITY_AXIS_UP = np.array([0.0, 0.0, -1.0])
GRAVITY_AXIS_HORIZONTAL = np.array([-1.0, 0.0, 0.0])


@dataclass
class Load:
    """One static load. `kind` selects how mass and geometry become a wrench."""

    kind: str                       # axial | bending | shear | couple | none
    mass_kg: float = 0.0
    lever_mm: float = 0.0
    angle_deg: float = 0.0          # position around the sensor axis, 0 = +X
    gravity: np.ndarray = field(default_factory=lambda: GRAVITY_AXIS_UP.copy())
    axis: str = "z"                 # couple only: which axis the torsion is about

    def wrench(self) -> np.ndarray:
        """Expected [Fx, Fy, Fz, Mx, My, Mz] at the sensor origin."""
        ghat = np.asarray(self.gravity, float)
        ghat = ghat / max(np.linalg.norm(ghat), 1e-12)
        weight = self.mass_kg * G
        lever = self.lever_mm / 1000.0
        theta = np.radians(self.angle_deg)

        if self.kind == "none":
            return np.zeros(6)

        if self.kind == "couple":
            # Pure torsion: two tangential tensions, equal and opposite. No net force.
            unit = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[self.axis]
            return np.hstack([np.zeros(3), np.array(unit, float) * 2.0 * lever * weight])

        if self.kind == "axial":
            # Hung on the centre eyebolt: force parallel to the axis, so no moment either way.
            return np.hstack([weight * ghat, np.zeros(3)])

        if self.kind == "bending":
            # Weight hangs vertically at a horizontal offset. Only the HORIZONTAL part of the
            # offset matters -- an axial component is parallel to the force and drops out.
            force = weight * ghat
            r = lever * np.array([np.cos(theta), np.sin(theta), 0.0])
            return np.hstack([force, np.cross(r, force)])

        if self.kind == "shear":
            # Cable over a pulley pulls in-plane at `angle`; `lever_mm` is how far the anchor
            # stands off along the axis. That standoff is the residual moment you cannot avoid,
            # which is why the anchor goes as close to the mounting plate as it will fit.
            force = weight * np.array([np.cos(theta), np.sin(theta), 0.0])
            r = np.array([0.0, 0.0, lever])
            return np.hstack([force, np.cross(r, force)])

        raise ValueError(f"unknown load kind {self.kind!r}")


@dataclass
class Step:
    label: str
    hint: str
    load: Load
    hold_s: float = 3.0


@dataclass
class Run:
    number: int
    name: str
    goal: str
    setup: str
    steps: list[Step]


def _ramp(masses, kind, lever=0.0, angle=0.0, gravity=None, prefix="") -> list[Step]:
    grav = GRAVITY_AXIS_UP if gravity is None else gravity
    out = []
    for m in masses:
        load = Load(kind, mass_kg=m, lever_mm=lever, angle_deg=angle, gravity=grav)
        w = load.wrench()
        out.append(Step(f"{prefix}{m:g}kg", f"hang {m:g} kg -- expect " + fmt(w), load))
    return out


def fmt(w: np.ndarray) -> str:
    parts = [f"{n}{v:+.2f}" for n, v in zip(("Fx", "Fy", "Fz"), w[:3]) if abs(v) > 0.05]
    parts += [f"{n}{v:+.3f}" for n, v in zip(("Mx", "My", "Mz"), w[3:]) if abs(v) > 0.005]
    return " ".join(parts) or "zero"


REST = Step("rest", "hands off, nothing hanging -- let it settle", Load("none"), hold_s=5.0)


def build_protocol(lever_mm: float = 100.0) -> list[Run]:
    """The 10-run master plan. `lever_mm` is the bending arm you actually bolted on."""
    masses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    light = [1, 2, 3, 4, 5]
    ring = [0, 45, 90, 135, 180, 225, 270, 315]

    return [
        Run(1, "Fz axial ramp", "Fz sensitivity and vertical symmetry",
            "sensor vertical, weights on a centre eyebolt (no lever)",
            [REST] + _ramp(masses, "axial") + [REST]
            + _ramp(masses[::-1], "axial", prefix="down ")),

        Run(2, "Fx / Fy shear", "pure shear with minimal lever-arm torque",
            "sensor horizontal, cable as close to the mounting plate as possible",
            [REST] + sum(([*_ramp(light, "shear", lever=5.0, angle=a,
                                  gravity=GRAVITY_AXIS_HORIZONTAL, prefix=f"{a}deg ")]
                          for a in (0, 90, 180, 270)), [])),

        Run(3, "Tx / Ty bending", "how shear converts into pure bending moment",
            f"{lever_mm:g} mm lever arm bolted to the top plate",
            [REST] + sum(([*_ramp(light, "bending", lever=lever_mm, angle=a, prefix=f"{a}deg ")]
                          for a in (0, 90, 180, 270)), [])),

        Run(4, "Tz torsion (doorknob)", "isolate Tz with no side load",
            f"symmetric T-bar, +/-{lever_mm:g} mm, cables over pulleys pulling tangentially",
            [REST] + [Step(f"cw {m:g}kg", f"tension {m:g} kg each side, clockwise -- expect "
                           + fmt(Load("couple", m, lever_mm).wrench()),
                           Load("couple", m, lever_mm)) for m in light]
            + [REST]
            + [Step(f"ccw {m:g}kg", f"tension {m:g} kg each side, anticlockwise -- expect "
                    + fmt(-Load("couple", m, lever_mm).wrench()),
                    Load("couple", -m, lever_mm)) for m in light]),

        Run(5, "360 moment ring", "sinusoidal phase across all 16 channels",
            f"lever arm relocated every 45 deg at {lever_mm:g} mm",
            [REST] + [Step(f"{a}deg", f"2 kg at {a} deg -- expect "
                           + fmt(Load("bending", 2.0, lever_mm, a).wrench()),
                           Load("bending", 2.0, lever_mm, a)) for a in ring]),

        Run(6, "Fz + shear", "does heavy compression shift the side-channel zero",
            "5 kg hanging axially throughout, then pull sideways",
            [REST, Step("preload", "hang 5 kg axially and leave it", Load("axial", 5.0))]
            + [Step(f"preload+{a}deg", f"keep 5 kg on, pull sideways at {a} deg",
                    Load("bending", 2.0, lever_mm, a)) for a in (0, 90, 180, 270)]),

        Run(7, "hysteresis loop", "mechanical memory in the flexure",
            f"{lever_mm:g} mm lever, step up then straight back down",
            [REST] + _ramp([1, 2, 3, 4, 5], "bending", lever_mm, prefix="up ")
            + _ramp([4, 3, 2, 1], "bending", lever_mm, prefix="down ") + [REST]),

        Run(8, "6-DOF sweep", "continuous inter-axis correlation for the fit",
            "by hand, no weights",
            [REST, Step("sweep", "smooth figure-8: roll, pitch, push, pull, twist for 2-3 min",
                        Load("none"), hold_s=150.0), REST]),

        Run(9, "thermal drift", "drift from cold power-on, zero load",
            "mounted rigidly in open air, nothing attached",
            [Step("cold soak", "leave it completely alone for 15 minutes", Load("none"),
                  hold_s=900.0)]),

        Run(10, "transient / resonance", "high-frequency response and settling",
            "calibrated weight on a quick-release hook",
            [REST] + sum(([Step(f"drop {m:g}kg", f"release {m:g} kg suddenly, then hold still",
                                Load("axial", m), hold_s=8.0), REST] for m in (2, 5)), [])),
    ]


# --------------------------------------------------------------------------- checks
def _test() -> None:
    w = Load("bending", 1.0, 100.0, 0.0).wrench()
    assert np.isclose(np.linalg.norm(w[3:]), 0.980665, atol=1e-6), w
    assert np.isclose(w[4], 0.980665, atol=1e-6), "1 kg at +X, 100 mm -> +My"
    assert np.allclose(w[:3], [0, 0, -9.80665]), w

    assert np.allclose(Load("axial", 5.0).wrench()[3:], 0.0), "centre load makes no moment"
    assert np.isclose(Load("axial", 5.0).wrench()[2], -49.03325, atol=1e-4)

    c = Load("couple", 1.0, 100.0).wrench()
    assert np.allclose(c[:3], 0.0), "a couple applies no net force"
    assert np.isclose(c[5], 2 * 0.1 * 9.80665, atol=1e-6), c

    at90 = Load("bending", 1.0, 100.0, 90.0).wrench()
    assert np.isclose(at90[3], -0.980665, atol=1e-6), "1 kg at +Y, 100 mm -> -Mx"

    # Attachment height must not change a bending moment: the force is parallel to the axis.
    high = Load("bending", 1.0, 100.0, 0.0)
    assert np.allclose(high.wrench(), Load("bending", 1.0, 100.0, 0.0).wrench())

    sh = Load("shear", 2.0, 5.0, 0.0).wrench()
    assert np.isclose(sh[0], 2 * G) and np.allclose(sh[1:3], 0.0), "shear pulls along +X"
    assert np.isclose(sh[4], 0.005 * 2 * G, atol=1e-9), "axial standoff x +X pull -> +My"
    assert abs(sh[4]) < 0.1, "shear residual moment must stay small"
    sh90 = Load("shear", 2.0, 5.0, 90.0).wrench()
    assert np.isclose(sh90[1], 2 * G) and abs(sh90[0]) < 1e-9, "90 deg rotates the PULL direction"

    ring = [Load("bending", 2.0, 100.0, a).wrench()[3:5] for a in range(0, 360, 45)]
    assert np.allclose([np.linalg.norm(v) for v in ring], np.linalg.norm(ring[0])), \
        "moment magnitude must be constant around the ring"

    runs = build_protocol()
    assert len(runs) == 10 and all(r.steps for r in runs)
    print(f"protocol OK: {len(runs)} runs, {sum(len(r.steps) for r in runs)} steps")
    for r in runs:
        print(f"  run {r.number:>2}  {r.name:<24} {len(r.steps):>3} steps   {r.goal}")


if __name__ == "__main__":
    _test()
