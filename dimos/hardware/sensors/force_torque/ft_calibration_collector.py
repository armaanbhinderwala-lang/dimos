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

"""Interactive dual-sensor FT calibration data collector.

Standalone, no dimos Module/framework dependency -- this branch (off main) has
no FT sensor code at all, and this tool is meant to run during a hands-on
session, not as a deployed blueprint. Records BOTH sensors simultaneously
(uFactory's built-in FT sensor is assumed physically upstream of the homemade
sensor + grip/weight holder, per the "post payload identification" setup)
into 4 SEPARATE tables/files, never mixed together, so raw vs. calibrated and
sensor vs. sensor are never ambiguous:

  - ufactory_raw / ufactory_calibrated (arm.ft_raw_force / arm.ft_ext_force).
  - homemade_raw (16 channels) / homemade_calibrated (matrix @ channels + bias,
    same math as openft_module.py, reimplemented here since that module isn't
    part of this branch -- kept intentionally tiny so it can't drift).

export_csv() (run automatically at the end of a session, or anytime after via
--export-only) dumps each of those 4 tables to its own CSV, plus experiments.csv
and run_metadata.csv (arm IP, serial port, calibration file, session type --
provenance for exactly what config produced this data) -- 6 files total,
label/session/experiment-start-time already joined into every sensor row.

Live display: uFactory raw and calibrated as two 6-bar banks (uFactory's own
RATED range -- not overload -- 150N Fx/Fy, 200N Fz, 4N*m any torque axis, same
datasheet numbers used for the door-pull cutoffs; this is what "stop at 90%%"
means), then homemade calibrated as a third bank on the same scale, then
homemade's 16 raw channels as plain numbers with no assumed scale.

Labeling keys mirror KeyboardTeleopModule's jog layout exactly (push = the
translation keys, twist = the rotation keys) -- same layout, no new scheme to
learn:
    W/S : push +X / -X      R/F : twist +X (roll)  / -X
    A/D : push +Y / -Y      T/G : twist +Y (pitch) / -Y
    Q/E : push +Z / -Z      Y/H : twist +Z (yaw)   / -Z
    SPACE or ENTER : done with this action, back to resting
    ESC : quit

session type (fast/slow) is a --session-type flag, set once per run -- like a
new pose, a new session type is just a new run of this script, not an
in-app toggle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

CHANNELS = 16
AXIS_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
# UFACTORY 6-axis FT sensor datasheet: rated range per axis, used as the 100% reference
# for the live bars. Same source as SENSOR_FORCE_OVERLOAD_N/SENSOR_TORQUE_OVERLOAD_NM in
# admittance_pull_law.py on the microwave-door-opener branch.
RATED_RANGE = (150.0, 150.0, 200.0, 4.0, 4.0, 4.0)

LABEL_KEYS = {
    "w": "push +X", "s": "push -X",
    "a": "push +Y", "d": "push -Y",
    "q": "push +Z", "e": "push -Z",
    "r": "twist +X", "f": "twist -X",
    "t": "twist +Y", "g": "twist -Y",
    "y": "twist +Z", "h": "twist -Z",
    "space": "resting",
}

# Shown on screen for whichever label is currently active. Exact physical +/- direction
# depends on how the sensor is mounted -- watch the matching bar (named in each line) to
# confirm you're actually exciting the axis you think you are, not guessing from the label.
LABEL_INSTRUCTIONS = {
    "push +X": "Push the grip straight along +X. Watch the Fx bar.",
    "push -X": "Push the grip straight along -X. Watch the Fx bar.",
    "push +Y": "Push the grip straight along +Y. Watch the Fy bar.",
    "push -Y": "Push the grip straight along -Y. Watch the Fy bar.",
    "push +Z": "Push the grip straight along +Z. Watch the Fz bar.",
    "push -Z": "Push the grip straight along -Z. Watch the Fz bar.",
    "twist +X": "Twist the grip about X (roll). Watch the Mx bar.",
    "twist -X": "Twist the grip the other way about X. Watch the Mx bar.",
    "twist +Y": "Twist the grip about Y (pitch). Watch the My bar.",
    "twist -Y": "Twist the grip the other way about Y. Watch the My bar.",
    "twist +Z": "Twist the grip about Z (yaw). Watch the Mz bar.",
    "twist -Z": "Twist the grip the other way about Z. Watch the Mz bar.",
    "resting": "Let go completely. Ignore the first second after releasing.",
}


def parse_frame(line: str) -> list[float] | None:
    """Same wire format as openft_module.py -- 16 comma-separated channels, trailing comma."""
    values = [v for v in line.strip().rstrip(",").split(",") if v]
    if len(values) != CHANNELS:
        return None
    try:
        return [float(v) for v in values]
    except ValueError:
        return None


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    matrix = np.asarray(payload["calibration_matrix"], dtype=float)
    bias = np.asarray(payload.get("bias_vector", np.zeros(6)), dtype=float)
    if matrix.shape != (6, CHANNELS):
        raise ValueError(f"calibration matrix in {path} is {matrix.shape}, expected (6, {CHANNELS})")
    return matrix, bias


class ExperimentLog:
    """One sqlite3 connection per writer thread -- each reader logs its own samples
    independently, at its own natural rate, tagged with whatever experiment is current.

    session_type is fixed for the whole run (fast/slow -- see --session-type), not an
    in-app toggle: same reasoning as a new pose being a new run, not an in-app marker."""

    def __init__(self, db_path: Path, session_type: str = "fast"):
        self.db_path = db_path
        self._state_lock = threading.Lock()
        self._experiment_id = 0
        self._label = "resting"
        self._session_type = session_type

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY, label TEXT, session_type TEXT,
                start_ts REAL, end_ts REAL
            );
            CREATE TABLE IF NOT EXISTS ufactory_raw (
                ts REAL, experiment_id INTEGER,
                fx REAL, fy REAL, fz REAL, mx REAL, my REAL, mz REAL
            );
            CREATE TABLE IF NOT EXISTS ufactory_calibrated (
                ts REAL, experiment_id INTEGER,
                fx REAL, fy REAL, fz REAL, mx REAL, my REAL, mz REAL
            );
            CREATE TABLE IF NOT EXISTS homemade_raw (
                ts REAL, experiment_id INTEGER,
                ch1 REAL, ch2 REAL, ch3 REAL, ch4 REAL, ch5 REAL, ch6 REAL, ch7 REAL, ch8 REAL,
                ch9 REAL, ch10 REAL, ch11 REAL, ch12 REAL, ch13 REAL, ch14 REAL, ch15 REAL, ch16 REAL
            );
            CREATE TABLE IF NOT EXISTS homemade_calibrated (
                ts REAL, experiment_id INTEGER,
                fx REAL, fy REAL, fz REAL, mx REAL, my REAL, mz REAL
            );
            """
        )
        conn.commit()
        conn.close()
        self._start_experiment("resting")

    def set_metadata(self, **fields: object) -> None:
        """Provenance for this session -- what config actually produced this data.
        Free-form key/value, not a fixed schema, so it's cheap to add more later."""
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?,?)",
            [(k, str(v)) for k, v in fields.items()],
        )
        conn.commit()
        conn.close()

    def _start_experiment(self, label: str) -> None:
        conn = sqlite3.connect(self.db_path)
        now = time.time()
        with self._state_lock:
            if self._experiment_id:
                conn.execute("UPDATE experiments SET end_ts=? WHERE id=?", (now, self._experiment_id))
            cur = conn.execute(
                "INSERT INTO experiments (label, session_type, start_ts) VALUES (?,?,?)",
                (label, self._session_type, now),
            )
            self._experiment_id = cur.lastrowid
            self._label = label
        conn.commit()
        conn.close()

    def set_label(self, label: str) -> None:
        self._start_experiment(label)

    def close(self) -> None:
        conn = sqlite3.connect(self.db_path)
        with self._state_lock:
            conn.execute("UPDATE experiments SET end_ts=? WHERE id=?", (time.time(), self._experiment_id))
        conn.commit()
        conn.close()

    @property
    def status(self) -> tuple[str, str]:
        with self._state_lock:
            return self._label, self._session_type

    def writer(self) -> "_LogWriter":
        return _LogWriter(self)


class _LogWriter:
    """Per-thread sqlite3 connection + a bound reference to the shared experiment_id."""

    def __init__(self, log: ExperimentLog):
        self._log = log
        self._conn = sqlite3.connect(log.db_path)

    def log_ufactory(self, raw: np.ndarray, ext: np.ndarray) -> None:
        with self._log._state_lock:
            experiment_id = self._log._experiment_id
        now = time.time()
        self._conn.execute(
            "INSERT INTO ufactory_raw VALUES (?,?,?,?,?,?,?,?)", (now, experiment_id, *raw.tolist())
        )
        self._conn.execute(
            "INSERT INTO ufactory_calibrated VALUES (?,?,?,?,?,?,?,?)", (now, experiment_id, *ext.tolist())
        )
        self._conn.commit()

    def log_homemade(self, channels: np.ndarray, cal: np.ndarray) -> None:
        with self._log._state_lock:
            experiment_id = self._log._experiment_id
        now = time.time()
        self._conn.execute(
            "INSERT INTO homemade_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, experiment_id, *channels.tolist()),
        )
        self._conn.execute(
            "INSERT INTO homemade_calibrated VALUES (?,?,?,?,?,?,?,?)", (now, experiment_id, *cal.tolist())
        )
        self._conn.commit()


class UFactoryReader:
    def __init__(self, ip: str, log: ExperimentLog, rate_hz: float = 100.0):
        self.ip = ip
        self._log = log
        self._dt = 1.0 / rate_hz
        self._running = False
        self.latest_raw = np.zeros(6)
        self.latest_ext = np.zeros(6)

    def start(self) -> None:
        from xarm.wrapper import XArmAPI

        arm = XArmAPI(self.ip)
        if not arm.connected:
            raise RuntimeError(f"Could not connect to xArm at {self.ip}")
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_state(0)
        code = arm.set_ft_sensor_enable(1)
        if code != 0:
            raise RuntimeError(f"set_ft_sensor_enable failed, code={code}")
        self._arm = arm
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        writer = self._log.writer()
        while self._running:
            raw = np.array(self._arm.ft_raw_force, dtype=float)
            ext = np.array(self._arm.ft_ext_force, dtype=float)
            self.latest_raw = raw
            self.latest_ext = ext
            writer.log_ufactory(raw, ext)
            time.sleep(self._dt)

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_arm"):
            self._arm.set_ft_sensor_enable(0)
            self._arm.disconnect()


class HomemadeReader:
    def __init__(self, port: str, baud: int, log: ExperimentLog, calibration: Path | None, window: int = 3):
        self.port = port
        self.baud = baud
        self._log = log
        self._running = False
        self._buffers = [deque(maxlen=window) for _ in range(CHANNELS)]
        self._matrix, self._bias = (load_calibration(calibration) if calibration else (None, None))
        self.latest_channels = np.zeros(CHANNELS)
        self.latest_cal = np.zeros(6)

    def start(self) -> None:
        import serial

        link = serial.Serial(self.port, self.baud, timeout=1.0)
        link.reset_input_buffer()
        self._serial = link
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        writer = self._log.writer()
        while self._running:
            try:
                line = self._serial.readline().decode("utf-8").strip()
            except Exception:
                continue
            parsed = parse_frame(line) if line else None
            if parsed is None:
                continue
            for buf, value in zip(self._buffers, parsed, strict=True):
                buf.append(value)
            channels = np.array([float(np.mean(b)) for b in self._buffers])
            cal = self._matrix @ channels + self._bias if self._matrix is not None else np.zeros(6)
            self.latest_channels = channels
            self.latest_cal = cal
            writer.log_homemade(channels, cal)

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_serial"):
            self._serial.close()


SAMPLE_TABLES = ("ufactory_raw", "ufactory_calibrated", "homemade_raw", "homemade_calibrated")


def export_csv(db_path: Path) -> list[Path]:
    """Dump each table to its own CSV -- 4 separate sensor files (raw/calibrated x
    ufactory/homemade, never mixed together) plus experiments.csv and
    run_metadata.csv, so nothing needs cross-referencing by hand. Label, session
    type, and the labeled segment's own start time are joined into every sensor
    row directly from the experiments table, using each table's own real columns
    (via PRAGMA table_info) so this can't drift from the schema. Rerunnable
    anytime against any session's DB, not just right after collecting it."""
    stem = db_path.with_suffix("")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    written = []

    def dump(name: str, query: str) -> None:
        rows = conn.execute(query).fetchall()
        out_path = Path(f"{stem}_{name}.csv")
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows(rows)
        written.append(out_path)

    dump("run_metadata", "SELECT * FROM run_metadata ORDER BY key")
    dump("experiments", "SELECT * FROM experiments ORDER BY start_ts")
    for table in SAMPLE_TABLES:
        value_cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})") if row[1] not in ("ts", "experiment_id")]
        cols = ", ".join(f"s.{c}" for c in value_cols)
        dump(
            table,
            f"""
            SELECT s.ts, e.label, e.session_type, e.start_ts AS experiment_start_ts, {cols}
            FROM {table} s JOIN experiments e ON s.experiment_id = e.id
            ORDER BY s.ts
            """,
        )
    conn.close()
    return written


def _bar_color(pct: float) -> tuple[int, int, int]:
    if pct >= 90:
        return (220, 70, 70)
    if pct >= 70:
        return (220, 190, 70)
    return (90, 200, 120)


def _selftest() -> None:
    """No hardware needed -- parsing, calibration math, and the DB pipeline."""
    import tempfile

    print("=== parse_frame ===")
    assert parse_frame("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,") == [float(i) for i in range(1, 17)]
    assert parse_frame("1,2,3") is None
    assert parse_frame("garbage,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16") is None
    print("OK")

    print("=== load_calibration ===")
    with tempfile.TemporaryDirectory() as d:
        calib_path = Path(d) / "cal.json"
        calib_path.write_text(json.dumps({
            "calibration_matrix": np.random.randn(6, CHANNELS).tolist(),
            "bias_vector": [0.1] * 6,
        }))
        m, b = load_calibration(calib_path)
        assert m.shape == (6, CHANNELS)
        assert np.allclose(b, [0.1] * 6)
    print("OK")

    print("=== _bar_color ===")
    assert _bar_color(50) == (90, 200, 120)
    assert _bar_color(75) == (220, 190, 70)
    assert _bar_color(95) == (220, 70, 70)
    print("OK")

    print("=== ExperimentLog: labels, session_type, writer round-trip ===")
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        log = ExperimentLog(db, session_type="slow")
        assert log.status == ("resting", "slow")
        log.set_label("push +X")
        assert log.status == ("push +X", "slow")

        writer = log.writer()
        raw_uf = np.arange(6, dtype=float)
        cal_uf = np.arange(6, dtype=float) + 0.5
        raw_hm = np.arange(CHANNELS, dtype=float)
        cal_hm = np.arange(6, dtype=float) + 1.5
        writer.log_ufactory(raw_uf, cal_uf)
        writer.log_homemade(raw_hm, cal_hm)
        log.close()

        conn = sqlite3.connect(db)
        experiments = conn.execute("SELECT label, end_ts FROM experiments").fetchall()
        assert len(experiments) == 2, experiments
        assert all(row[1] is not None for row in experiments), "every segment must get an end_ts, including the last"
        # 4 separate tables, never mixed -- and each row really is what was logged, not
        # e.g. raw and calibrated swapped or written to the wrong table.
        for table in SAMPLE_TABLES:
            assert len(conn.execute(f"SELECT * FROM {table}").fetchall()) == 1, table
        assert conn.execute("SELECT fx,fy,fz,mx,my,mz FROM ufactory_raw").fetchone() == tuple(raw_uf)
        assert conn.execute("SELECT fx,fy,fz,mx,my,mz FROM ufactory_calibrated").fetchone() == tuple(cal_uf)
        assert conn.execute("SELECT ch1,ch16 FROM homemade_raw").fetchone() == (raw_hm[0], raw_hm[15])
        assert conn.execute("SELECT fx,fy,fz,mx,my,mz FROM homemade_calibrated").fetchone() == tuple(cal_hm)
    print("OK")

    print("=== export_csv: 6 files, metadata + joined columns, row counts match the DB ===")
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        log = ExperimentLog(db, session_type="slow")
        log.set_metadata(xarm_ip="1.2.3.4", homemade_port="/dev/ttyACM0")
        writer = log.writer()
        log.set_label("push +X")
        writer.log_ufactory(np.arange(6, dtype=float), np.arange(6, dtype=float))
        writer.log_homemade(np.arange(CHANNELS, dtype=float), np.arange(6, dtype=float))
        log.close()

        paths = export_csv(db)
        assert len(paths) == 6, paths  # run_metadata + experiments + 4 sensor tables
        for path in paths:
            assert path.exists(), path
        import csv as _csv

        with (Path(d) / "test_run_metadata.csv").open() as fh:
            meta = {row["key"]: row["value"] for row in _csv.DictReader(fh)}
        assert meta["xarm_ip"] == "1.2.3.4" and meta["homemade_port"] == "/dev/ttyACM0"

        with (Path(d) / "test_ufactory_raw.csv").open() as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["label"] == "push +X" and rows[0]["session_type"] == "slow"
        assert set(rows[0]) == {"ts", "label", "session_type", "experiment_start_ts", "fx", "fy", "fz", "mx", "my", "mz"}
    print("OK")

    print("\nAll self-tests passed.")


BAR_X = 210
BAR_W = 520
BAR_H = 30
ROW_STEP = 40


def _draw_bars(screen: "pygame.Surface", small: "pygame.font.Font", y: int, values: np.ndarray) -> int:
    """Draw one 6-bar bank at y; returns the y position just below it."""
    for i, name in enumerate(AXIS_NAMES):
        pct = 100.0 * abs(values[i]) / RATED_RANGE[i]
        bar_w = int(min(pct, 100) / 100 * BAR_W)
        color = _bar_color(pct)
        pygame.draw.rect(screen, (60, 60, 65), (BAR_X, y, BAR_W, BAR_H))
        pygame.draw.rect(screen, color, (BAR_X, y, bar_w, BAR_H))
        pygame.draw.line(screen, (255, 255, 255), (BAR_X + int(BAR_W * 0.9), y), (BAR_X + int(BAR_W * 0.9), y + BAR_H), 3)  # 90% line
        unit = "N" if i < 3 else "N*m"
        screen.blit(small.render(name, True, (230, 230, 230)), (20, y + 4))
        screen.blit(small.render(f"{values[i]:.1f}{unit} ({pct:.0f}%)", True, (230, 230, 230)), (BAR_X + BAR_W + 16, y + 4))
        y += ROW_STEP
    return y


def run_ui(ufactory: UFactoryReader, homemade: HomemadeReader, log: ExperimentLog) -> None:
    if pygame is None:
        raise ImportError("pygame is required. Install it with: pip install pygame")

    pygame.init()
    screen = pygame.display.set_mode((980, 1470))
    pygame.display.set_caption("FT Calibration Collector")
    font = pygame.font.Font(None, 40)
    small = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    # ENTER: "I'm done with this action" -- same as SPACE, back to resting.
                    log.set_label(LABEL_KEYS["space"])
                else:
                    key_name = pygame.key.name(event.key)
                    if key_name in LABEL_KEYS:
                        log.set_label(LABEL_KEYS[key_name])

        screen.fill((25, 25, 30))
        y = 20
        screen.blit(font.render("FT Calibration Collector", True, (255, 255, 255)), (20, y))
        y += 50

        screen.blit(small.render(
            "The arm does NOT move from this window -- put it in position-hold and push/twist the sensor by hand.",
            True, (255, 190, 90),
        ), (20, y))
        y += 44

        label, session_type = log.status
        status = f"[{label}]  session={session_type}"
        screen.blit(font.render(status, True, (120, 220, 255)), (20, y))
        y += 44
        screen.blit(small.render(LABEL_INSTRUCTIONS.get(label, ""), True, (200, 200, 200)), (20, y))
        y += 40

        screen.blit(font.render("uFactory -- raw", True, (210, 210, 210)), (20, y))
        y += 44
        y = _draw_bars(screen, small, y, ufactory.latest_raw)

        y += 26
        screen.blit(font.render("uFactory -- compensated (trusted reference)", True, (210, 210, 210)), (20, y))
        y += 44
        y = _draw_bars(screen, small, y, ufactory.latest_ext)

        y += 26
        screen.blit(font.render("Homemade -- calibrated (not fully trusted yet)", True, (210, 210, 210)), (20, y))
        y += 44
        y = _draw_bars(screen, small, y, homemade.latest_cal)

        y += 26
        screen.blit(font.render("Homemade -- raw (16 channels, plain values, no assumed scale)", True, (210, 210, 210)), (20, y))
        y += 44
        channels = homemade.latest_channels
        for row in range(6):
            for col in range(3):
                idx = row * 3 + col
                if idx >= CHANNELS:
                    continue
                text = f"ch{idx + 1:>2}: {channels[idx]:9.1f}"
                screen.blit(small.render(text, True, (210, 210, 210)), (20 + col * 300, y + row * 34))
        y += 6 * 34 + 20

        legend = [
            "W/S A/D Q/E : push +/-X +/-Y +/-Z      R/F T/G Y/H : twist +/-X +/-Y +/-Z",
            "SPACE or ENTER: done with this action, back to resting      ESC: quit",
        ]
        for line in legend:
            screen.blit(small.render(line, True, (160, 160, 160)), (20, y))
            y += 32

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xarm-ip")
    p.add_argument("--homemade-port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--calibration", type=Path, default=None, help="existing ft_calibration.json for the live homemade preview (optional)")
    p.add_argument("--db", type=Path, default=None, help="defaults to ft_calibration_session_<timestamp>.db")
    p.add_argument(
        "--session-type", choices=["fast", "slow"], default="fast",
        help="fast: 2-3s cycles, lots of variety. slow: 15-20s ramps. Run each as a separate session -- don't mix.",
    )
    p.add_argument("--selftest", action="store_true", help="run no-hardware self-tests and exit")
    p.add_argument("--export-only", type=Path, default=None, help="re-export CSVs from an existing session DB, no hardware, and exit")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.export_only:
        for path in export_csv(args.export_only):
            print(f"wrote {path}")
        return
    if not args.xarm_ip:
        p.error("--xarm-ip is required (or pass --selftest / --export-only)")

    db_path = args.db or Path(f"ft_calibration_session_{int(time.time())}.db")
    log = ExperimentLog(db_path, session_type=args.session_type)
    log.set_metadata(
        xarm_ip=args.xarm_ip,
        homemade_port=args.homemade_port,
        baud=args.baud,
        calibration_file=str(args.calibration) if args.calibration else "",
        session_type=args.session_type,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        created_ts=time.time(),
    )
    print(f"Logging to {db_path} (session_type={args.session_type})")

    ufactory = UFactoryReader(args.xarm_ip, log)
    homemade = HomemadeReader(args.homemade_port, args.baud, log, args.calibration)

    ufactory.start()
    homemade.start()
    try:
        run_ui(ufactory, homemade, log)
    finally:
        ufactory.stop()
        homemade.stop()
        log.close()
        print(f"Done. {db_path}")
        for path in export_csv(db_path):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
