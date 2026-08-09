#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
Force-Torque Sensor Logging Module for Dimos

Subscribes to force, torque and raw sensor streams and writes them to a
SQLite database file for offline analysis.
"""

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dimos.core import Module, In, rpc
from dimos.msgs.geometry_msgs import Vector3
from dimos.hardware.ft_driver_module import RawSensorData
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS force (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    rel_time REAL NOT NULL,
    x REAL, y REAL, z REAL,
    magnitude REAL
);

CREATE TABLE IF NOT EXISTS torque (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    rel_time REAL NOT NULL,
    x REAL, y REAL, z REAL,
    magnitude REAL
);

CREATE TABLE IF NOT EXISTS raw_sensors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    rel_time REAL NOT NULL,
    sensor_timestamp REAL,
    sensor_values TEXT
);

CREATE INDEX IF NOT EXISTS idx_force_timestamp ON force(timestamp);
CREATE INDEX IF NOT EXISTS idx_torque_timestamp ON torque(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_sensors(timestamp);

-- The driver publishes force and torque as a pair for every calibrated
-- sample, so row ids line up one-to-one between the two tables.
CREATE VIEW IF NOT EXISTS force_torque AS
SELECT f.id          AS id,
       f.timestamp   AS timestamp,
       f.rel_time    AS rel_time,
       f.x           AS fx,
       f.y           AS fy,
       f.z           AS fz,
       f.magnitude   AS force_magnitude,
       t.x           AS tx,
       t.y           AS ty,
       t.z           AS tz,
       t.magnitude   AS torque_magnitude
FROM force f JOIN torque t ON t.id = f.id;
"""


class FTLoggerModule(Module):
    """Logs force-torque sensor streams to a SQLite database file."""

    # Input ports - mirror the driver's outputs
    force: In[Vector3] = None  # Force vector in Newtons
    torque: In[Vector3] = None  # Torque vector in Newton-meters
    raw_sensor_data: In[RawSensorData] = None  # Raw sensor values (optional)

    def __init__(
        self,
        db_path: str = "ft_log.db",
        flush_interval: float = 1.0,
        log_raw: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the FT logger module.

        Args:
            db_path: Path of the SQLite database file to write
            flush_interval: Seconds between commits of buffered samples
            log_raw: Also log raw (moving-averaged) sensor values
            metadata: Arbitrary key/value pairs stored in the session table
            verbose: Enable verbose output
        """
        super().__init__()

        self.db_path = str(db_path)
        self.flush_interval = flush_interval
        self.log_raw = log_raw
        self.metadata = metadata or {}
        self.verbose = verbose

        # SQLite connection is created in start(), inside the worker process
        self.conn = None

        # Buffers holding rows not yet written to disk
        self._force_rows = []
        self._torque_rows = []
        self._raw_rows = []
        self._buffer_lock = threading.Lock()

        # Unsubscribe callbacks returned by stream.subscribe()
        self._unsubscribers = []

        # Statistics
        self.force_count = 0
        self.torque_count = 0
        self.raw_count = 0
        self.written_count = 0

        self.start_time = None
        self.running = False
        self._writer_thread = None

    def _rel_time(self, timestamp: float) -> float:
        """Seconds since the logger started."""
        return timestamp - self.start_time if self.start_time else 0.0

    def handle_force(self, msg: Vector3):
        """Buffer an incoming force Vector3."""
        timestamp = time.time()
        magnitude = math.sqrt(msg.x**2 + msg.y**2 + msg.z**2)

        with self._buffer_lock:
            self._force_rows.append(
                (timestamp, self._rel_time(timestamp), msg.x, msg.y, msg.z, magnitude)
            )
            self.force_count += 1

        if self.verbose:
            logger.debug(f"Logged force: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}) N")

    def handle_torque(self, msg: Vector3):
        """Buffer an incoming torque Vector3."""
        timestamp = time.time()
        magnitude = math.sqrt(msg.x**2 + msg.y**2 + msg.z**2)

        with self._buffer_lock:
            self._torque_rows.append(
                (timestamp, self._rel_time(timestamp), msg.x, msg.y, msg.z, magnitude)
            )
            self.torque_count += 1

        if self.verbose:
            logger.debug(f"Logged torque: ({msg.x:.4f}, {msg.y:.4f}, {msg.z:.4f}) N⋅m")

    def handle_raw(self, msg: RawSensorData):
        """Buffer an incoming raw sensor sample."""
        timestamp = time.time()

        with self._buffer_lock:
            self._raw_rows.append(
                (
                    timestamp,
                    self._rel_time(timestamp),
                    msg.timestamp,
                    json.dumps(list(msg.sensor_values)),
                )
            )
            self.raw_count += 1

    def _flush(self):
        """Write buffered rows to the database."""
        with self._buffer_lock:
            force_rows, self._force_rows = self._force_rows, []
            torque_rows, self._torque_rows = self._torque_rows, []
            raw_rows, self._raw_rows = self._raw_rows, []

        if not (force_rows or torque_rows or raw_rows):
            return

        try:
            if force_rows:
                self.conn.executemany(
                    "INSERT INTO force (timestamp, rel_time, x, y, z, magnitude) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    force_rows,
                )
            if torque_rows:
                self.conn.executemany(
                    "INSERT INTO torque (timestamp, rel_time, x, y, z, magnitude) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    torque_rows,
                )
            if raw_rows:
                self.conn.executemany(
                    "INSERT INTO raw_sensors (timestamp, rel_time, sensor_timestamp, sensor_values) "
                    "VALUES (?, ?, ?, ?)",
                    raw_rows,
                )
            self.conn.commit()
            self.written_count += len(force_rows) + len(torque_rows) + len(raw_rows)
        except Exception as e:
            logger.error(f"Error writing to {self.db_path}: {e}")

    def _writer_loop(self):
        """Periodically commit buffered samples to disk."""
        logger.info(f"FT logger writer thread started for {self.db_path}")
        while self.running:
            time.sleep(self.flush_interval)
            self._flush()
        logger.info("FT logger writer thread stopping")

    def _subscribe(self, stream, handler, label: str):
        """Subscribe to a stream, tolerating one that was never connected."""
        if stream is None:
            logger.warning(f"Logger has no {label} stream - not recording it")
            return

        try:
            self._unsubscribers.append(stream.subscribe(handler))
            logger.info(f"Logger subscribed to {label} data")
        except Exception as e:
            logger.warning(f"Logger could not subscribe to {label} stream: {e}")

    def _write_metadata(self, extra: Optional[Dict[str, Any]] = None):
        """Upsert session metadata rows."""
        rows = dict(self.metadata)
        if extra:
            rows.update(extra)

        self.conn.executemany(
            "INSERT INTO session (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(str(k), str(v)) for k, v in rows.items()],
        )
        self.conn.commit()

    @rpc
    def start(self):
        """Open the database and start logging."""
        if self.running:
            logger.warning("FT logger already running")
            return True

        db_path = Path(self.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # check_same_thread=False: subscriber callbacks and the writer
            # thread both touch the connection, serialised by _buffer_lock
            # for the buffers and by the writer thread for the inserts.
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to open database {db_path}: {e}")
            return False

        self.start_time = time.time()
        self._write_metadata(
            {
                "db_path": str(db_path.resolve()),
                "start_time": self.start_time,
                "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "log_raw": self.log_raw,
            }
        )

        self.running = True

        self._subscribe(self.force, self.handle_force, "force")
        self._subscribe(self.torque, self.handle_torque, "torque")
        if self.log_raw:
            self._subscribe(self.raw_sensor_data, self.handle_raw, "raw sensor")

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        logger.info(f"FT logger started - writing to {db_path.resolve()}")
        return True

    @rpc
    def stop(self):
        """Flush remaining samples and close the database."""
        if not self.running:
            return

        logger.info("Stopping FT logger...")
        self.running = False

        for unsubscribe in self._unsubscribers:
            try:
                if callable(unsubscribe):
                    unsubscribe()
            except Exception as e:
                logger.warning(f"Error unsubscribing logger stream: {e}")
        self._unsubscribers = []

        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=self.flush_interval + 2.0)

        # Final flush of anything buffered after the last writer pass
        self._flush()

        if self.conn:
            try:
                self._write_metadata(
                    {
                        "end_time": time.time(),
                        "duration_s": time.time() - (self.start_time or time.time()),
                        "force_samples": self.force_count,
                        "torque_samples": self.torque_count,
                        "raw_samples": self.raw_count,
                    }
                )
                self.conn.close()
            except Exception as e:
                logger.error(f"Error closing database: {e}")
            self.conn = None

        logger.info(
            f"FT logger stopped. Wrote {self.written_count} rows to {self.db_path} "
            f"(force={self.force_count}, torque={self.torque_count}, raw={self.raw_count})"
        )

    @rpc
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        with self._buffer_lock:
            pending = len(self._force_rows) + len(self._torque_rows) + len(self._raw_rows)

        return {
            "db_path": self.db_path,
            "running": self.running,
            "force_count": self.force_count,
            "torque_count": self.torque_count,
            "raw_count": self.raw_count,
            "written_count": self.written_count,
            "pending_rows": pending,
        }


if __name__ == "__main__":
    # For testing standalone
    import argparse

    parser = argparse.ArgumentParser(description="FT Logger Module")
    parser.add_argument("--db", default="ft_log.db", help="SQLite database path")
    parser.add_argument("--flush-interval", type=float, default=1.0, help="Commit interval (s)")
    parser.add_argument("--raw", action="store_true", help="Also log raw sensor values")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    from dimos.core import start

    dimos = start(1)
    ft_logger = dimos.deploy(
        FTLoggerModule,
        db_path=args.db,
        flush_interval=args.flush_interval,
        log_raw=args.raw,
        verbose=args.verbose,
    )

    ft_logger.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ft_logger.stop()
