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

"""Homemade OpenFT sensor, graphed live in Rerun and recorded to SQLite.

Same shape as the xarm-force-torque blueprint, with the serial OpenFT driver
swapped in for the xArm one. Because both publish :class:`WrenchStamped` on
ports named ``ext_wrench``/``raw_wrench``, ``autoconnect`` wires the recorder
and plotter without remappings and the graphs behave identically.

Query a recording with ``SqliteStore(path="openft_recording.db")``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.stream import In
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.openft_module import OpenFTSensor
from dimos.memory2.module import Recorder, RecorderConfig
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.visualization.wrench_plotter import WrenchPlotter


class OpenFTRecorderConfig(RecorderConfig):
    # Distinct from the xArm stack's ft_recording.db so running one sensor
    # never overwrites the other's recordings.
    db_path: str | Path = "openft_recording.db"

    # No tf is published here, so there is no transform tree to record and
    # nothing to anchor a wrench to. Left on, the Recorder would open an empty
    # "tf" stream fed by a port nothing is connected to.
    record_tf: bool = False

    # Declaring both streams poseless silences the per-message "No pose for
    # time ..." warning, which would otherwise fire on every observation --
    # and the MCU sets the rate here, so that is as fast as it streams.
    poseless_streams: list[str] = Field(
        default_factory=lambda: ["ext_wrench", "raw_wrench"],
    )


class OpenFTRecorder(Recorder):
    """Persists both wrench streams to SQLite, one memory2 stream per port.

    Declared here rather than reusing the xArm blueprint's FTRecorder: that
    module imports XArmFTSensor, which imports the xArm SDK, and the homemade
    sensor has no business requiring it.
    """

    config: OpenFTRecorderConfig

    ext_wrench: In[WrenchStamped]
    raw_wrench: In[WrenchStamped]


openft = autoconnect(
    OpenFTSensor.blueprint(),
    # Passed explicitly, not left to the default: pydantic skips validators on
    # defaults, so RecorderConfig._resolve_path only fires for an explicit
    # value. Without this the db lands relative to the worker's cwd.
    OpenFTRecorder.blueprint(db_path="openft_recording.db"),
    WrenchPlotter.blueprint(),
).transports(
    {
        ("ext_wrench", WrenchStamped): LCMTransport("/ft/ext_wrench", WrenchStamped),
        ("raw_wrench", WrenchStamped): LCMTransport("/ft/raw_wrench", WrenchStamped),
    }
)
