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

"""Step-by-step data inspection for FT calibration sessions. Read before fitting anything.

Nine independent checks, run in order, each printing what it found and (where relevant)
whether it PASSES or needs attention. Run one at a time with --step N while you work
through them, or --step all for the full report.

  1  inventory          what sessions exist, how big, recorded with what config
  2  columns            every column in every file, units, and what it means
  3  sync               clock alignment between the two sensors, measured not assumed
  4  rates              sampling rate stability per sensor
  5  quality            missing values, NaNs, duplicate timestamps, stuck channels
  6  unloaded           where the genuinely-unloaded periods are
  7  offsets            zero offset per channel/axis, and how much it drifts
  8  corrected          what the data looks like after offset removal
  9  ranges             excitation coverage vs each sensor's rated range

Nothing here modifies data or fits a model; it only reports.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np

AXES = ("fx", "fy", "fz", "mx", "my", "mz")
CHANNELS = 16
CHAN_COLS = tuple(f"ch{i}" for i in range(1, CHANNELS + 1))

# UFACTORY 6-axis FT sensor datasheet.
UF_RATED = {"fx": 150.0, "fy": 150.0, "fz": 200.0, "mx": 4.0, "my": 4.0, "mz": 4.0}
# openFT firmware treats channel readings outside this band as invalid.
OPENFT_VALID = (9000.0, 21000.0)

STREAMS = ("ufactory_raw", "ufactory_calibrated", "homemade_raw", "homemade_calibrated")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def find_sessions(d: Path) -> list[str]:
    return sorted(
        re.search(r"session_(\d+)_run_metadata", p).group(1)
        for p in glob.glob(str(d / "ft_calibration_session_*_run_metadata.csv"))
    )


def arr(rows, cols) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows])


def ts_of(rows) -> np.ndarray:
    return np.array([float(r["ts"]) for r in rows])


def hdr(n: int, title: str) -> None:
    print(f"\n{'='*78}\nSTEP {n}  {title}\n{'='*78}")


# --------------------------------------------------------------------------- 1
def step1(d: Path, sessions: list[str]) -> None:
    hdr(1, "INVENTORY -- what data exists")
    print(f"{'session':>12} {'started':>20} {'dur_s':>7} {'type':>6} {'uf_rows':>9} {'hm_rows':>9}")
    total = 0
    for s in sessions:
        meta = {r["key"]: r["value"] for r in read_csv(d / f"ft_calibration_session_{s}_run_metadata.csv")}
        uf = read_csv(d / f"ft_calibration_session_{s}_ufactory_calibrated.csv")
        hm = read_csv(d / f"ft_calibration_session_{s}_homemade_raw.csv")
        t = ts_of(uf)
        total += len(uf) + len(hm)
        print(f"{s:>12} {meta.get('created_at','?'):>20} {t[-1]-t[0]:7.0f} "
              f"{meta.get('session_type','?'):>6} {len(uf):9d} {len(hm):9d}")
    print(f"\n{len(sessions)} sessions, {total:,} total rows across both sensors.")
    print("Each session = one arm pose. Session ids are unix timestamps of the run start.")


# --------------------------------------------------------------------------- 2
def step2(d: Path, sessions: list[str]) -> None:
    hdr(2, "COLUMNS -- what every field means")
    s = sessions[0]
    meaning = {
        "ts": "unix time of THIS sample (seconds, float)",
        "label": "what the operator was doing (push +X, twist -Z, resting, ...)",
        "session_type": "fast (2-3s cycles) or slow (15-20s ramps)",
        "experiment_start_ts": "start time of the labelled segment this sample belongs to",
        **{a: f"{'force, N' if a[0]=='f' else 'torque, N*m'} -- uFactory sensor frame" for a in AXES},
        **{c: "raw ADC counts from one Hall-effect channel (no units)" for c in CHAN_COLS},
    }
    for stream in STREAMS:
        p = d / f"ft_calibration_session_{s}_{stream}.csv"
        cols = list(read_csv(p)[0].keys())
        print(f"\n  {stream}.csv  ({len(cols)} columns)")
        for c in cols:
            print(f"      {c:<22} {meaning.get(c, '?')}")
    print("\n  ufactory_raw vs ufactory_calibrated: differ ONLY by gravity/payload compensation")
    print("  (a near-constant offset). Both are already in N / N*m -- the ADC->Newton step")
    print("  happens inside the uFactory sensor and is not exposed. Use 'calibrated' as truth.")
    print("  homemade_calibrated: the OLD shipped matrix applied to the raw channels. Do NOT")
    print("  train on it -- it is an output of a previous fit, not an independent measurement.")


# --------------------------------------------------------------------------- 3
def step3(d: Path, sessions: list[str]) -> None:
    hdr(3, "SYNCHRONISATION -- are the two sensors on the same clock?")
    print("Both streams are timestamped by the same collector process, so they SHOULD align.")
    print("Verified by sweeping an artificial lag and finding where a linear fit is best.\n")
    print(f"{'session':>12} {'best lag (s)':>13} {'verdict':>10}")
    for s in sessions:
        uf = read_csv(d / f"ft_calibration_session_{s}_ufactory_calibrated.csv")
        hm = read_csv(d / f"ft_calibration_session_{s}_homemade_raw.csv")
        ut, uy = ts_of(uf), arr(uf, AXES)
        ht, X = ts_of(hm), arr(hm, CHAN_COLS)
        Xc = X - X.mean(axis=0)
        best = (None, -np.inf)
        for lag in np.arange(-0.4, 0.41, 0.05):
            y = np.column_stack([np.interp(ht + lag, ut, uy[:, i]) for i in range(6)])
            y = y - y.mean(axis=0)
            A = np.hstack([Xc, np.ones((len(Xc), 1))])
            W, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = (1 - (y - A @ W).var(axis=0) / np.maximum(y.var(axis=0), 1e-12)).mean()
            if r2 > best[1]:
                best = (lag, r2)
        ok = "PASS" if abs(best[0]) <= 0.05 else "CHECK"
        print(f"{s:>12} {best[0]:13.2f} {ok:>10}")
    print("\nPASS = optimum within one sample period of zero, i.e. no correction needed.")


# --------------------------------------------------------------------------- 4
def step4(d: Path, sessions: list[str]) -> None:
    hdr(4, "SAMPLING RATES -- steady, or dropping frames?")
    print(f"{'session':>12} {'sensor':>10} {'mean Hz':>9} {'median dt':>10} {'p99 dt':>9} {'gaps>3x':>8}")
    for s in sessions:
        for name, f in (("uFactory", "ufactory_calibrated"), ("homemade", "homemade_raw")):
            t = ts_of(read_csv(d / f"ft_calibration_session_{s}_{f}.csv"))
            dt = np.diff(t)
            med = np.median(dt)
            print(f"{s:>12} {name:>10} {len(t)/(t[-1]-t[0]):9.1f} {med*1000:9.1f}ms "
                  f"{np.percentile(dt,99)*1000:8.1f}ms {(dt > 3*med).sum():8d}")
    print("\nThe homemade rate is set by the MCU firmware; uFactory by our polling loop.")
    print("Because they differ, ALWAYS interpolate the faster stream onto the slower one's")
    print("timestamps -- never upsample the target you are trying to predict.")


# --------------------------------------------------------------------------- 5
def step5(d: Path, sessions: list[str]) -> None:
    hdr(5, "DATA QUALITY -- missing, duplicate, or stuck values")
    print(f"{'session':>12} {'stream':>21} {'rows':>8} {'NaN':>5} {'dup_ts':>7} {'non-mono':>9} {'stuck':>6}")
    for s in sessions:
        for stream in STREAMS:
            rows = read_csv(d / f"ft_calibration_session_{s}_{stream}.csv")
            cols = CHAN_COLS if "homemade_raw" in stream else AXES
            v = arr(rows, cols)
            t = ts_of(rows)
            stuck = int(sum(1 for j in range(v.shape[1]) if np.std(v[:, j]) == 0))
            print(f"{s:>12} {stream:>21} {len(rows):8d} {int(np.isnan(v).sum()):5d} "
                  f"{len(t)-len(np.unique(t)):7d} {int((np.diff(t)<0).sum()):9d} {stuck:6d}")
    print("\nstuck = channels with exactly zero variance over the whole session (dead sensor).")


# --------------------------------------------------------------------------- 6
def step6(d: Path, sessions: list[str]) -> None:
    hdr(6, "UNLOADED PERIODS -- where is the true zero?")
    print("The operator marks these with SPACE/ENTER; they are labelled 'resting'.")
    print("These are what every offset correction depends on, so check they are real.\n")
    print(f"{'session':>12} {'segments':>9} {'samples':>8} {'% of run':>9} {'|F| during rest':>17}")
    for s in sessions:
        uf = read_csv(d / f"ft_calibration_session_{s}_ufactory_calibrated.csv")
        y = arr(uf, AXES)
        lab = np.array([r["label"] for r in uf])
        seg = np.array([float(r["experiment_start_ts"]) for r in uf])
        rest = lab == "resting"
        f = np.linalg.norm(y[rest][:, :3] - y[rest][:, :3].mean(axis=0), axis=1)
        print(f"{s:>12} {len(np.unique(seg[rest])):9d} {rest.sum():8d} "
              f"{100*rest.mean():8.0f}% {np.percentile(f,95):12.2f} N (p95)")
    print("\n'|F| during rest' is the spread AROUND the resting mean -- if this is large the")
    print("operator was still touching the sensor and that segment is not a true zero.")


# --------------------------------------------------------------------------- 7
def step7(d: Path, sessions: list[str]) -> None:
    hdr(7, "OFFSETS -- the zero point, and how far it wanders")
    print("Computed per resting segment, so drift within a run is visible.\n")
    for s in sessions:
        hm = read_csv(d / f"ft_calibration_session_{s}_homemade_raw.csv")
        X, t = arr(hm, CHAN_COLS), ts_of(hm)
        lab = np.array([r["label"] for r in hm])
        seg = np.array([float(r["experiment_start_ts"]) for r in hm])
        rest = lab == "resting"
        groups: dict[float, list[int]] = {}
        for i in np.where(rest)[0]:
            groups.setdefault(seg[i], []).append(i)
        keys = sorted(groups)
        base = np.array([X[groups[k]].mean(axis=0) for k in keys])
        noise = X[rest].std(axis=0).mean()
        drift = np.abs(base - base[0]).max()
        print(f"{s}: {len(keys):2d} zero points | within-run noise {noise:5.2f} counts | "
              f"max drift {drift:5.2f} counts ({drift/max(noise,1e-9):4.1f}x noise)")
    print("\nDrift larger than the noise means a single constant zero is NOT enough --")
    print("the fitting pipeline interpolates between these points for that reason.")


# --------------------------------------------------------------------------- 8
def step8(d: Path, sessions: list[str]) -> None:
    hdr(8, "AFTER OFFSET REMOVAL -- does zero load now read zero?")
    print(f"{'session':>12} {'homemade resid @rest':>22} {'uFactory resid @rest':>22}")
    for s in sessions:
        hm = read_csv(d / f"ft_calibration_session_{s}_homemade_raw.csv")
        uf = read_csv(d / f"ft_calibration_session_{s}_ufactory_calibrated.csv")
        X, ht = arr(hm, CHAN_COLS), ts_of(hm)
        uy, ut = arr(uf, AXES), ts_of(uf)
        lab = np.array([r["label"] for r in hm])
        seg = np.array([float(r["experiment_start_ts"]) for r in hm])
        y = np.column_stack([np.interp(ht, ut, uy[:, i]) for i in range(6)])
        rest = lab == "resting"
        groups: dict[float, list[int]] = {}
        for i in np.where(rest)[0]:
            groups.setdefault(seg[i], []).append(i)
        keys = sorted(groups)
        centres = np.array([ht[groups[k]].mean() for k in keys])
        bx = np.array([X[groups[k]].mean(axis=0) for k in keys])
        by = np.array([y[groups[k]].mean(axis=0) for k in keys])
        Xc = X - np.column_stack([np.interp(ht, centres, bx[:, j]) for j in range(CHANNELS)])
        yc = y - np.column_stack([np.interp(ht, centres, by[:, j]) for j in range(6)])
        print(f"{s:>12} {np.abs(Xc[rest]).mean():17.3f} counts {np.linalg.norm(yc[rest][:,:3],axis=1).mean():15.3f} N")
    print("\nBoth should be near zero by construction; large values mean the resting")
    print("segments disagree with each other, i.e. the zero is not stable.")


# --------------------------------------------------------------------------- 9
def step9(d: Path, sessions: list[str]) -> None:
    hdr(9, "RANGES -- how much of each sensor's capability was used")
    hmx = []
    Y = []
    for s in sessions:
        hm = read_csv(d / f"ft_calibration_session_{s}_homemade_raw.csv")
        uf = read_csv(d / f"ft_calibration_session_{s}_ufactory_calibrated.csv")
        hmx.append(arr(hm, CHAN_COLS))
        Y.append(arr(uf, AXES))
    X = np.vstack(hmx)
    Y = np.vstack(Y)
    print("uFactory (the reference) -- excitation vs its own rated range:")
    for i, a in enumerate(AXES):
        mx = np.abs(Y[:, i]).max()
        pct = 100 * mx / UF_RATED[a]
        flag = "  OVER RATED RANGE" if pct > 100 else ""
        print(f"  {a}: max {mx:7.2f}  of {UF_RATED[a]:6.1f} rated = {pct:5.1f}%{flag}")
    print(f"\nhomemade raw channels -- firmware valid band is {OPENFT_VALID[0]:.0f}..{OPENFT_VALID[1]:.0f}:")
    print(f"  observed overall min {X.min():.0f}, max {X.max():.0f}")
    print(f"  per-channel peak-to-peak: mean {np.mean(X.max(axis=0)-X.min(axis=0)):.1f} counts, "
          f"max {np.max(X.max(axis=0)-X.min(axis=0)):.1f}")
    span = OPENFT_VALID[1] - OPENFT_VALID[0]
    print(f"  that is {100*np.mean(X.max(axis=0)-X.min(axis=0))/span:.2f}% of the firmware's valid band")
    print("\nA small percentage here is the core hardware limitation: the sensor's usable")
    print("signal is a tiny fraction of its ADC range, so its resolution in N is coarse.")


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8, 9: step9}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--step", default="all", help="1-9, or 'all'")
    p.add_argument("--sessions", nargs="*")
    args = p.parse_args()

    sessions = args.sessions or find_sessions(args.data_dir)
    if not sessions:
        raise SystemExit(f"no sessions found in {args.data_dir}")

    if args.step == "all":
        for n in sorted(STEPS):
            STEPS[n](args.data_dir, sessions)
    else:
        STEPS[int(args.step)](args.data_dir, sessions)


if __name__ == "__main__":
    main()
