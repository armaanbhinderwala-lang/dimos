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

# ---------------------------------------------------------------------------
# Geometry between the two sensors. Both quantities were established three ways
# (physical measurement, CAD, and fitting the data) -- see notebook steps 10 and 12.
# ---------------------------------------------------------------------------

# Position of the DIY sensor's origin relative to the uFactory's, expressed in
# uFACTORY coordinates. Physically measured at 63.3 mm; CAD (49.2 + 7.25) gives
# 56.5 mm and a data sweep prefers 52.5 mm. They differ because the PCB faces and
# the sensors' internal measurement origins are not the same points; any value in
# this range removes ~85% of the force/torque coupling.
DIY_ORIGIN_IN_UFACTORY_M = np.array([0.0, 0.0, 0.0633])
SENSOR_ORIGIN_OFFSET_M = float(DIY_ORIGIN_IN_UFACTORY_M[2])  # kept for callers that want the scalar

# Rotation taking a vector FROM the DIY frame TO the uFactory frame: v_U = R @ v_D.
# A rotation matrix's columns are the images of the source frame's basis vectors, so
# this encodes the measured mapping  DIY +X -> uF -Y,  DIY +Y -> uF +X,  DIY +Z -> uF +Z
# (a -90 deg rotation about Z). Confirmed physically and by two independent fits.
R_UFACTORY_FROM_DIY = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
R_DIY_FROM_UFACTORY = R_UFACTORY_FROM_DIY.T  # inverse of a rotation is its transpose


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
    """Move a wrench's reference POINT by `offset_m` along +Z, without rotating it.

    Moment about a new point B, given the moment about A:  tau_B = tau_A - p x F,
    where p is the vector from A to B. With p = (0, 0, d) this reduces to
        mx' = mx + d*fy,   my' = my - d*fx,   mz' unchanged.
    Force is a free vector and so is unaffected by a pure translation.

    This is the translation half of the transform only; use ufactory_to_diy_frame for
    the full change of frame. Kept separate because the notebook sweeps `offset_m` to
    estimate the sensor separation, which needs translation without rotation.
    """
    if not offset_m:
        return y
    out = y.copy()
    out[:, 3] = y[:, 3] + offset_m * y[:, 1]
    out[:, 4] = y[:, 4] - offset_m * y[:, 0]
    return out


def ufactory_to_diy_frame(wrench: np.ndarray) -> np.ndarray:
    """Express a uFactory wrench about the DIY sensor's origin, in DIY axes.

    Two steps, and the order matters in general:
      1. TRANSLATE, still in uFactory axes -- tau_at_D = tau_U - p x F_U
      2. ROTATE both force and torque into DIY axes with R_DIY_FROM_UFACTORY

    Doing it the other way round is also valid provided `p` is first rotated into DIY
    axes too. Here p lies along Z and the rotation is about Z, so p is identical in both
    frames and the two orders coincide -- but only by coincidence of this geometry, so
    the explicit order above is the one to rely on.

    A common mistake is `tau - p x F_D`, mixing a uFactory-frame p with a DIY-frame F.
    A cross product between vectors in different frames is not defined.
    """
    force, torque = wrench[:, :3], wrench[:, 3:]
    torque_at_diy = torque - np.cross(DIY_ORIGIN_IN_UFACTORY_M, force)
    return np.hstack([force @ R_DIY_FROM_UFACTORY.T, torque_at_diy @ R_DIY_FROM_UFACTORY.T])


def diy_to_ufactory_frame(wrench: np.ndarray) -> np.ndarray:
    """Inverse of ufactory_to_diy_frame: a DIY-frame wrench expressed at the uFactory origin.

    Undo the two steps in reverse order -- rotate back into uFactory axes, then translate
    the reference point back by +p (hence the sign flip against the forward transform).

    Needed at deployment: the calibration predicts a wrench in the DIY frame, but anything
    comparing it against the uFactory, or feeding a controller that expects the tool frame,
    needs it moved back.
    """
    force, torque = wrench[:, :3], wrench[:, 3:]
    force_u = force @ R_UFACTORY_FROM_DIY.T
    torque_at_diy_u = torque @ R_UFACTORY_FROM_DIY.T
    return np.hstack([force_u, torque_at_diy_u + np.cross(DIY_ORIGIN_IN_UFACTORY_M, force_u)])


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
    frame: str = "ufactory",
    filter_window: int = 0,
) -> Session:
    """Load one session: align the two sensors, then remove a time-varying baseline.

    Alignment: the reference streams faster (~89 Hz) than the homemade sensor (~35 Hz),
    so the reference is interpolated ONTO the homemade timestamps. Never the other way --
    upsampling the target invents data you then try to predict.

    Baseline: the collector records a `resting` segment every couple of minutes. Zero
    points are taken from each and interpolated between, because the sensor's zero drifts
    measurably more than the load signal itself over a run.

    frame:
      "ufactory" -- leave the reference wrench as measured. The calibration then reports
                    what the uFactory would report: a true drop-in replacement, but the
                    fit can absorb the fixed p x F term instead of learning real torque.
      "diy"      -- express the wrench about the DIY sensor's own origin and axes. The
                    physically honest target, and the one that transfers to loads applied
                    somewhere other than where the training loads were.
    """
    if frame not in ("ufactory", "diy"):
        raise ValueError(f"frame must be 'ufactory' or 'diy', got {frame!r}")

    uf = stream(session, "ufactory_calibrated", data_dir)
    hm = stream(session, "homemade_raw", data_dir)

    uf_times, uf_wrench = timestamps(uf), columns(uf, AXES)
    times, channels = timestamps(hm), columns(hm, CHAN_COLS)
    labels = np.array([r["label"] for r in hm])
    segment_start = np.array([float(r["experiment_start_ts"]) for r in hm])

    wrench = np.column_stack([np.interp(times, uf_times, uf_wrench[:, i]) for i in range(len(AXES))])
    if frame == "diy":
        wrench = ufactory_to_diy_frame(wrench)
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
    frame: str = "ufactory",
    filter_window: int = 0,
) -> list[Session]:
    ids = sessions or find_sessions(data_dir)
    out = []
    for s in ids:
        try:
            out.append(load_session(s, data_dir, frame, filter_window))
        except ValueError as exc:
            # uFactory-only runs (collector --no-homemade) have no channels to calibrate
            # from. Skip them rather than failing every caller that scans the directory.
            print(f"  skipping {s}: {exc}")
    return out
