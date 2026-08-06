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

"""Record xArm FT sensor wrenches into a memory2 SQLite db.

A ``Recorder`` whose In ports are named for :class:`XArmFTSensor`'s outputs, so
``autoconnect`` wires them without remappings. Each observation is stored with
the sensor's own timestamp; the wrenches carry no pose because this stack
publishes no tf (see the config for why that is off).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from dimos.core.stream import In
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped


class FTRecorderConfig(RecorderConfig):
    db_path: str | Path = "ft_recording.db"

    # The FT stack publishes no tf, so there is no transform tree to record and
    # nothing to anchor a wrench to. Left on, the Recorder would open an empty
    # "tf" stream fed by a port nothing is connected to.
    record_tf: bool = False

    # Declaring both streams poseless is what silences the per-message "No pose
    # for time ..." warning. Without it that fires on every observation — 100
    # lines a second at the default 50 Hz across two streams.
    poseless_streams: list[str] = Field(
        default_factory=lambda: ["ext_wrench", "raw_wrench"],
    )


class FTRecorder(Recorder):
    """Persists both wrench streams to SQLite, one memory2 stream per port."""

    config: FTRecorderConfig

    ext_wrench: In[WrenchStamped]
    raw_wrench: In[WrenchStamped]
