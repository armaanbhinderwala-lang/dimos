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

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.force_torque.read_FTModule import XArmFTSensor
from dimos.hardware.sensors.force_torque.recorder import FTRecorder
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped

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

if __name__ == "__main__":
    xarm_force_torque.build().loop()
