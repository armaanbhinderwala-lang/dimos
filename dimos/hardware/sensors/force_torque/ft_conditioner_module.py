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

"""Raw wrench in, conditioned wrench out.

Sits between any FT driver and the pull policy so the policy need not know which sensor it
is reading. Maths lives in ft_conditioning.py; this is wiring, lifecycle and the tare RPC.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.force_torque.ft_conditioning import (
    PROFILES,
    ConditioningProfile,
    WrenchConditioner,
)
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class FTConditionerConfig(ModuleConfig):
    profile: str = Field(default="diy", description="'diy' or 'factory' -- see PROFILES")
    frame_id: str = "ft_sensor_link"
    tool_mass_kg: float = Field(default=0.0, ge=0.0)
    tool_com_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tare_samples: int = Field(default=50, gt=0)
    tare_on_start: bool = Field(default=True, description="Requires an unloaded tool at startup")


class FTConditionerModule(Module):
    config: FTConditionerConfig

    raw_wrench: In[WrenchStamped]
    clean_wrench: Out[WrenchStamped]

    _conditioner: WrenchConditioner | None = None

    @rpc
    def start(self) -> None:
        super().start()
        if self.config.profile not in PROFILES:
            raise ValueError(
                f"FTConditionerModule: unknown profile {self.config.profile!r}, "
                f"expected one of {sorted(PROFILES)}"
            )
        base = PROFILES[self.config.profile]
        profile = ConditioningProfile(
            ema_alpha=base.ema_alpha,
            force_deadband_n=base.force_deadband_n,
            torque_deadband_nm=base.torque_deadband_nm,
            tool_mass_kg=self.config.tool_mass_kg,
            tool_com_m=np.array(self.config.tool_com_m, dtype=float),
            already_gravity_compensated=base.already_gravity_compensated,
        )
        if profile.already_gravity_compensated and self.config.tool_mass_kg > 0.0:
            logger.info("Profile %r is gravity-compensated upstream; ignoring tool_mass_kg=%.2f.",
                        self.config.profile, self.config.tool_mass_kg)
        self._conditioner = WrenchConditioner(profile)
        self.raw_wrench.subscribe(self._on_raw)
        if self.config.tare_on_start:
            logger.info("Taring over %d samples -- keep the tool untouched.", self.config.tare_samples)
            self._conditioner.begin_tare()

    @rpc
    def tare(self) -> None:
        """Re-zero the sensor. Nothing may be touching the tool while this runs."""
        if self._conditioner is None:
            raise RuntimeError("FTConditionerModule: not started")
        logger.info("Taring over %d samples -- keep the tool untouched.", self.config.tare_samples)
        self._conditioner.begin_tare()

    def _on_raw(self, msg: WrenchStamped) -> None:
        if self._conditioner is None:
            return
        raw = np.array([msg.force.x, msg.force.y, msg.force.z,
                        msg.torque.x, msg.torque.y, msg.torque.z], dtype=float)
        was_taring = self._conditioner.taring
        # ee_rot=None: gravity comp needs tool orientation, which this module does not subscribe
        # to. Skipping beats using a stale rotation; wire a pose stream before enabling it.
        clean = self._conditioner.apply(raw, ee_rot=None, tare_target=self.config.tare_samples)
        if was_taring and not self._conditioner.taring:
            logger.info("Tare complete. Bias = %s", np.round(self._conditioner.bias, 3).tolist())
        # from_array validates the 6 elements; the bare constructor falls through to Wrench.
        self.clean_wrench.publish(
            WrenchStamped.from_array(clean, frame_id=self.config.frame_id, ts=msg.ts)
        )
