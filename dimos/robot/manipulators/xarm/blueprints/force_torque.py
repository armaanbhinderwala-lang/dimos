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

"""Real Force-Torque Sensor Data Stream, recorded to a memory2 SQLite db.

The recorder's In ports are named for the sensor's outputs, so ``autoconnect``
wires them directly. Query the result with ``SqliteStore(path="ft_recording.db")``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.stream import In
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.read_FTModule import XArmFTSensor
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
    """Persists both wrench streams to SQLite, one memory2 stream per port.

    In ports are named for :class:`XArmFTSensor`'s outputs, so ``autoconnect``
    wires them without remappings. Each observation is stored with the sensor's
    own timestamp; the wrenches carry no pose because this stack publishes no
    tf (see the config for why that is off).
    """

    config: FTRecorderConfig

    ext_wrench: In[WrenchStamped]
    raw_wrench: In[WrenchStamped]


xarm_force_torque = autoconnect(
    XArmFTSensor.blueprint(),
    # Passed explicitly, not left to the default: pydantic skips validators on
    # defaults, so RecorderConfig._resolve_path only fires for an explicit value.
    # Without this the db lands relative to the worker's cwd instead of the repo.
    FTRecorder.blueprint(db_path="ft_recording.db"),
).transports(
    {
        ("ext_wrench", WrenchStamped): LCMTransport("/ft/ext_wrench", WrenchStamped),
        ("raw_wrench", WrenchStamped): LCMTransport("/ft/raw_wrench", WrenchStamped),
    }
)
