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

"""Wrench conditioning: tare, gravity removal, smoothing, deadband.

Order matters: gravity before filtering (it is real signal that moves with pose), deadband
last (applied earlier it lets spikes through and then freezes them).

The uFactory's ext_wrench is already gravity-compensated -- compensating it again biases
every reading by the tool's weight. Profiles declare this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY_M_S2 = 9.80665


@dataclass
class ConditioningProfile:
    ema_alpha: float                 # 1.0 = no smoothing
    force_deadband_n: float
    torque_deadband_nm: float
    tool_mass_kg: float = 0.0
    tool_com_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    already_gravity_compensated: bool = False


PROFILES: dict[str, ConditioningProfile] = {
    "diy": ConditioningProfile(ema_alpha=0.10, force_deadband_n=1.5, torque_deadband_nm=0.1),
    "factory": ConditioningProfile(ema_alpha=0.35, force_deadband_n=0.5, torque_deadband_nm=0.03,
                                   already_gravity_compensated=True),
}


def ema(previous: np.ndarray | None, sample: np.ndarray, alpha: float) -> np.ndarray:
    """First sample passes through, so the filter starts at the signal rather than at zero."""
    if previous is None:
        return np.asarray(sample, float).copy()
    return alpha * np.asarray(sample, float) + (1.0 - alpha) * previous


def tool_gravity_wrench(ee_rot: np.ndarray, mass_kg: float, com_m: np.ndarray) -> np.ndarray:
    """Tool weight in the sensor frame. Rotates with the wrist, so it is not a constant offset."""
    force_world = np.array([0.0, 0.0, -mass_kg * GRAVITY_M_S2])
    force_sensor = np.asarray(ee_rot, float).T @ force_world
    return np.hstack([force_sensor, np.cross(np.asarray(com_m, float), force_sensor)])


def deadband(wrench: np.ndarray, force_n: float, torque_nm: float) -> np.ndarray:
    """Zero sub-noise values, else drift becomes a standing velocity command in free air."""
    out = np.asarray(wrench, float).copy()
    out[:3] = np.where(np.abs(out[:3]) < force_n, 0.0, out[:3])
    out[3:] = np.where(np.abs(out[3:]) < torque_nm, 0.0, out[3:])
    return out


class WrenchConditioner:
    """Stateful pipeline: tare, gravity, EMA, deadband."""

    def __init__(self, profile: ConditioningProfile):
        self.profile = profile
        self._bias = np.zeros(6)
        self._filtered: np.ndarray | None = None
        self._tare_samples: list[np.ndarray] = []
        self._taring = False

    def begin_tare(self) -> None:
        self._tare_samples = []
        self._taring = True

    @property
    def taring(self) -> bool:
        return self._taring

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()

    def apply(self, raw: np.ndarray, ee_rot: np.ndarray | None = None,
              tare_target: int = 50) -> np.ndarray:
        raw = np.asarray(raw, float)
        if self._taring:
            self._tare_samples.append(raw.copy())
            if len(self._tare_samples) >= tare_target:
                self._bias = np.mean(self._tare_samples, axis=0)
                self._taring = False
            return np.zeros(6)  # a half-taken bias looks like a real load

        wrench = raw - self._bias
        if (not self.profile.already_gravity_compensated
                and self.profile.tool_mass_kg > 0.0 and ee_rot is not None):
            wrench = wrench - tool_gravity_wrench(ee_rot, self.profile.tool_mass_kg,
                                                  self.profile.tool_com_m)
        self._filtered = ema(self._filtered, wrench, self.profile.ema_alpha)
        return deadband(self._filtered, self.profile.force_deadband_n,
                        self.profile.torque_deadband_nm)
