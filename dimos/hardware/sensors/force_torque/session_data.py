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

"""Shared loading and preprocessing for FT calibration sessions.

Single source of truth for turning collector CSVs into arrays ready to fit, so the
inspection notebook and the fitting script can never disagree about what the data is.

Written by ft_calibration_collector.py, one set per session:
    ft_calibration_session_<id>_ufactory_raw.csv          reference wrench, no gravity comp
    ft_calibration_session_<id>_ufactory_calibrated.csv   reference wrench, gravity removed
    ft_calibration_session_<id>_homemade_raw.csv          16 Hall channels, ADC counts
    ft_calibration_session_<id>_homemade_calibrated.csv   old matrix applied (never train on this)
    ft_calibration_session_<id>_experiments.csv           labelled segments
    ft_calibration_session_<id>_run_metadata.csv          arm ip, port, session type, ...
"""

from __future__ import annotations

import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AXES = ("fx", "fy", "fz", "mx", "my", "mz")
CHANNELS = 16
CHAN_COLS = tuple(f"ch{i}" for i in range(1, CHANNELS + 1))

# UFACTORY 6-axis FT sensor datasheet: rated range per axis. Samples beyond this are
# outside the REFERENCE's own spec, so their labels are untrustworthy as ground truth.
UF_RATED = {"fx": 150.0, "fy": 150.0, "fz": 200.0, "mx": 4.0, "my": 4.0, "mz": 4.0}

# openFT firmware treats channel readings outside this band as invalid.
OPENFT_VALID = (9000.0, 21000.0)

# Distance between the two sensors' origins along the tool axis, from CAD:
# uFactory mounting flange -> its sensor origin is 49.2 mm; DIY CNC backplate -> its
# sensor origin is 7.25 mm, on the far side. Independently, sweeping the data for the
# offset that best decouples force from torque lands at 52-55 mm (see the notebook).
SENSOR_ORIGIN_OFFSET_M = 0.05645


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def find_sessions(data_dir: Path | str = ".") -> list[str]:
    """Session ids (unix timestamps of each run start), oldest first."""
    pattern = str(Path(data_dir) / "ft_calibration_session_*_run_metadata.csv")
    return sorted(re.search(r"session_(\d+)_run_metadata", p).group(1) for p in glob.glob(pattern))


def find_data_dir(start: Path | str = ".") -> Path:
    """Nearest directory at or above `start` that holds session CSVs."""
    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        if list(candidate.glob("ft_calibration_session_*_run_metadata.csv")):
            return candidate
    return Path(start)


def stream(session: str, name: str, data_dir: Path | str = ".") -> list[dict[str, str]]:
    """One CSV for one session. `name` is e.g. 'homemade_raw' or 'ufactory_calibrated'."""
    return read_csv(Path(data_dir) / f"ft_calibration_session_{session}_{name}.csv")


def columns(rows: list[dict[str, str]], cols) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows])


def timestamps(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([float(r["ts"]) for r in rows])


def shift_wrench_frame(y: np.ndarray, offset_m: float) -> np.ndarray:
    """Re-express a wrench about a point `offset_m` further along +Z.

    Standard transform M' = M - r x F with r = (0, 0, d), which reduces to
        mx' = mx + d*fy,   my' = my - d*fx,   mz' unchanged.
    Force is unchanged by a pure translation of the reference point.

    Use this to express the uFactory's wrench about the HOMEMADE sensor's origin --
    the physically honest target, since that is the wrench the homemade sensor
    actually experiences.
    """
    if not offset_m:
        return y
    out = y.copy()
    out[:, 3] = y[:, 3] + offset_m * y[:, 1]
    out[:, 4] = y[:, 4] - offset_m * y[:, 0]
    return out


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Filter each column. ~1s of samples measurably improves the fit (see notebook)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(x[:, j], kernel, mode="same") for j in range(x.shape[1])])


def within_reference_spec(y: np.ndarray) -> np.ndarray:
    """Mask of samples the reference sensor could actually measure trustworthily."""
    return (np.linalg.norm(y[:, :3], axis=1) <= UF_RATED["fx"]) & (
        np.linalg.norm(y[:, 3:], axis=1) <= UF_RATED["mx"]
    )


@dataclass
class Session:
    """One run, aligned and zeroed, ready to fit.

    channels : (N, 16) homemade ADC counts, baseline removed
    wrench   : (N, 6)  reference wrench on the same timestamps, baseline removed
    labels   : (N,)    what the operator was doing at each sample
    raw_wrench : (N, 6) before baseline removal -- for spec filtering and range checks
    """

    session_id: str
    channels: np.ndarray
    wrench: np.ndarray
    labels: np.ndarray
    raw_wrench: np.ndarray
    times: np.ndarray

    @property
    def resting(self) -> np.ndarray:
        return self.labels == "resting"

    def __len__(self) -> int:
        return len(self.channels)


def load_session(
    session: str,
    data_dir: Path | str = ".",
    frame_offset_m: float = 0.0,
    filter_window: int = 0,
) -> Session:
    """Load one session: align the two sensors, then remove a time-varying baseline.

    Alignment: the reference streams faster (~89 Hz) than the homemade sensor (~35 Hz),
    so the reference is interpolated ONTO the homemade timestamps. Never the other way --
    upsampling the target invents data you then try to predict.

    Baseline: the collector records a `resting` segment every couple of minutes. Zero
    points are taken from each and interpolated between, because the sensor's zero drifts
    measurably more than the load signal itself over a run.

    frame_offset_m: see shift_wrench_frame. 0.0 keeps the reference's own frame
    (a drop-in replacement that reports what the uFactory reports); SENSOR_ORIGIN_OFFSET_M
    targets the wrench about the homemade sensor's own origin.
    """
    uf = stream(session, "ufactory_calibrated", data_dir)
    hm = stream(session, "homemade_raw", data_dir)

    uf_times, uf_wrench = timestamps(uf), columns(uf, AXES)
    times, channels = timestamps(hm), columns(hm, CHAN_COLS)
    labels = np.array([r["label"] for r in hm])
    segment_start = np.array([float(r["experiment_start_ts"]) for r in hm])

    wrench = np.column_stack([np.interp(times, uf_times, uf_wrench[:, i]) for i in range(len(AXES))])
    wrench = shift_wrench_frame(wrench, frame_offset_m)
    raw_wrench = wrench.copy()

    resting = labels == "resting"
    if resting.sum() < 10:
        raise ValueError(f"session {session}: only {resting.sum()} resting samples, need >= 10")

    groups: dict[float, list[int]] = {}
    for i in np.where(resting)[0]:
        groups.setdefault(segment_start[i], []).append(i)
    keys = sorted(groups)
    centres = np.array([times[groups[k]].mean() for k in keys])

    def debias(values: np.ndarray) -> np.ndarray:
        zeros = np.array([values[groups[k]].mean(axis=0) for k in keys])
        baseline = np.column_stack(
            [np.interp(times, centres, zeros[:, j]) for j in range(values.shape[1])]
        )
        return values - baseline

    channels, wrench = debias(channels), debias(wrench)
    if filter_window > 1:
        channels = moving_average(channels, filter_window)

    return Session(session, channels, wrench, labels, raw_wrench, times)


def load_all(
    data_dir: Path | str = ".",
    sessions: list[str] | None = None,
    frame_offset_m: float = 0.0,
    filter_window: int = 0,
) -> list[Session]:
    ids = sessions or find_sessions(data_dir)
    return [load_session(s, data_dir, frame_offset_m, filter_window) for s in ids]
