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

"""Fit a 6x16 calibration matrix for the homemade FT sensor from collector sessions.

Trains homemade raw channels -> uFactory wrench (the trusted reference), and writes
an ft_calibration.json that drops straight into openft_module.py.

Three preprocessing steps carry most of the accuracy, all validated on real sessions:

  1. Time-varying baseline. The collector records a `resting` segment every couple of
     minutes; baselines are interpolated between them and subtracted from BOTH sensors.
     This cancels thermal drift, which measurably exceeds the load signal itself between
     sessions. Measured: torque R2 0.37 -> 0.50 vs a single constant per session.
  2. Moving-average filter on the sensor input (~1s). Measured: mean R2 0.33 -> 0.40.
     Evaluated honestly -- input filtered, unsmoothed wrench predicted.
  3. Ridge, not plain least-squares. The 16 channels come from 4 physical magnet
     clusters and are strongly collinear; unregularized lstsq is what produced the
     all-zero Mz row in the original shipped calibration.

Model choice is empirical, not assumed: Ridge beat Poly2+Ridge, RandomForest and an
MLP under held-out-session CV on this data. Nonlinear models tie or overfit, which
says the bottleneck is sensor SNR, not model capacity.

Validation is ALWAYS held-out-session (leave-one-session-out). A random train/test
split leaks badly here -- samples arrive at ~35Hz and neighbours are near-duplicates.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

AXES = ("fx", "fy", "fz", "mx", "my", "mz")
CHANNELS = 16
CHAN_COLS = tuple(f"ch{i}" for i in range(1, CHANNELS + 1))

# UFACTORY 6-axis FT sensor rated range; samples beyond this are outside the REFERENCE
# sensor's spec, so their labels are untrustworthy regardless of the homemade sensor.
REF_FORCE_LIMIT_N = 150.0
REF_TORQUE_LIMIT_NM = 4.0


def _read(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def find_sessions(directory: Path) -> list[str]:
    pat = str(directory / "ft_calibration_session_*_run_metadata.csv")
    return sorted(re.search(r"session_(\d+)_run_metadata", p).group(1) for p in glob.glob(pat))


def load_session(directory: Path, session: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, y_absolute): baseline-corrected channels, baseline-corrected wrench,
    and the raw uncorrected wrench (used only for out-of-spec filtering)."""
    uf = _read(directory / f"ft_calibration_session_{session}_ufactory_calibrated.csv")
    hm = _read(directory / f"ft_calibration_session_{session}_homemade_raw.csv")

    uf_ts = np.array([float(r["ts"]) for r in uf])
    uf_y = np.array([[float(r[a]) for a in AXES] for r in uf])
    hm_ts = np.array([float(r["ts"]) for r in hm])
    X = np.array([[float(r[c]) for c in CHAN_COLS] for r in hm])
    labels = np.array([r["label"] for r in hm])
    seg_start = np.array([float(r["experiment_start_ts"]) for r in hm])

    # Interpolate the FASTER uFactory stream (~89Hz) onto the SLOWER homemade
    # timestamps (~35Hz) -- never upsample the target. A +/-0.5s lag sweep on real
    # data peaks at exactly 0.00s, so the collector's timestamps need no correction.
    y = np.column_stack([np.interp(hm_ts, uf_ts, uf_y[:, i]) for i in range(len(AXES))])

    rest = labels == "resting"
    if rest.sum() < 10:
        raise ValueError(f"session {session}: only {rest.sum()} resting samples, need >=10")

    segments: dict[float, list[int]] = {}
    for i in np.where(rest)[0]:
        segments.setdefault(seg_start[i], []).append(i)
    keys = sorted(segments)
    centres = np.array([hm_ts[segments[k]].mean() for k in keys])
    base_x = np.array([X[segments[k]].mean(axis=0) for k in keys])
    base_y = np.array([y[segments[k]].mean(axis=0) for k in keys])

    Xb = np.column_stack([np.interp(hm_ts, centres, base_x[:, j]) for j in range(CHANNELS)])
    yb = np.column_stack([np.interp(hm_ts, centres, base_y[:, j]) for j in range(len(AXES))])
    return X - Xb, y - yb, y


def moving_average(X: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return X
    kernel = np.ones(window) / window
    return np.column_stack([np.convolve(X[:, j], kernel, mode="same") for j in range(X.shape[1])])


def in_spec(y_absolute: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(y_absolute[:, :3], axis=1) <= REF_FORCE_LIMIT_N) & (
        np.linalg.norm(y_absolute[:, 3:], axis=1) <= REF_TORQUE_LIMIT_NM
    )


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Standardised ridge, returned in RAW channel units as (6x16 matrix, 6 bias) so the
    output is directly usable without shipping the scaler alongside it."""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Z = (X - mu) / sigma
    Za = np.hstack([Z, np.ones((len(Z), 1))])
    A = Za.T @ Za + alpha * np.eye(Za.shape[1])
    A[-1, -1] -= alpha  # never penalise the intercept
    W = np.linalg.solve(A, Za.T @ y)  # (17, 6)

    matrix = (W[:CHANNELS] / sigma[:, None]).T  # (6, 16), raw units
    bias = W[CHANNELS] - matrix @ mu
    return matrix, bias


def evaluate(matrix, bias, X, y) -> tuple[np.ndarray, np.ndarray]:
    resid = y - (X @ matrix.T + bias)
    r2 = 1 - resid.var(axis=0) / np.maximum(y.var(axis=0), 1e-12)
    return r2, np.sqrt((resid**2).mean(axis=0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."), help="directory holding the session CSVs")
    p.add_argument("--out", type=Path, default=Path("ft_calibration_fitted.json"))
    p.add_argument("--alpha", type=float, default=1000.0, help="ridge regularisation (default tuned on 10 sessions)")
    p.add_argument("--filter-window", type=int, default=35, help="input moving-average samples (~35 = 1s at 35Hz)")
    p.add_argument("--no-spec-filter", action="store_true", help="keep samples beyond the reference sensor's rated range")
    p.add_argument("--sessions", nargs="*", help="explicit session ids (default: all found)")
    args = p.parse_args()

    sessions = args.sessions or find_sessions(args.data_dir)
    if len(sessions) < 2:
        raise SystemExit(f"need >=2 sessions for held-out-session validation, found {len(sessions)}")
    print(f"Found {len(sessions)} sessions in {args.data_dir}")

    data = {}
    for s in sessions:
        Xc, yc, y_abs = load_session(args.data_dir, s)
        keep = np.ones(len(Xc), bool) if args.no_spec_filter else in_spec(y_abs)
        data[s] = (moving_average(Xc, args.filter_window)[keep], yc[keep])
        dropped = len(Xc) - keep.sum()
        print(f"  {s}: {keep.sum():6d} samples" + (f"  ({dropped} dropped as out-of-spec)" if dropped else ""))

    print(f"\n=== Leave-one-session-out validation (alpha={args.alpha}, filter={args.filter_window}) ===")
    print(f"{'held-out':>14}" + "".join(f"{a:>8}" for a in AXES))
    all_r2 = []
    for held in sessions:
        Xtr = np.vstack([data[s][0] for s in sessions if s != held])
        ytr = np.vstack([data[s][1] for s in sessions if s != held])
        m, b = ridge_fit(Xtr, ytr, args.alpha)
        r2, _ = evaluate(m, b, *data[held])
        all_r2.append(r2)
        print(f"{held:>14}" + "".join(f"{v:8.2f}" for v in r2))
    mean_r2 = np.mean(all_r2, axis=0)
    print(f"{'MEAN R2':>14}" + "".join(f"{v:8.2f}" for v in mean_r2) + f"   overall {mean_r2.mean():.2f}")

    X = np.vstack([data[s][0] for s in sessions])
    y = np.vstack([data[s][1] for s in sessions])
    matrix, bias = ridge_fit(X, y, args.alpha)
    r2_in, rmse_in = evaluate(matrix, bias, X, y)
    print(f"\n=== Final fit on all {len(sessions)} sessions ({len(X)} samples) ===")
    print(f"{'':>14}" + "".join(f"{a:>8}" for a in AXES))
    print(f"{'in-sample R2':>14}" + "".join(f"{v:8.2f}" for v in r2_in))
    print(f"{'RMSE':>14}" + "".join(f"{v:8.2f}" for v in rmse_in) + "   (N / N*m)")

    # A vanishing row means that axis outputs a constant no matter what the sensor does --
    # the exact failure mode in the originally shipped calibration.
    norms = np.linalg.norm(matrix, axis=1)
    dead = [AXES[i] for i in range(len(AXES)) if norms[i] <= 1e-12 * norms.max()]
    if dead:
        print(f"\nWARNING: degenerate (all-zero) row(s) for {', '.join(dead)} -- that axis will read a constant.")

    args.out.write_text(json.dumps({
        "calibration_matrix": matrix.tolist(),
        "bias_vector": bias.tolist(),
        "sensor_channels": CHANNELS,
        "output_channels": len(AXES),
        "timestamp": time.time(),
        "metadata": {
            "fit_by": "fit_calibration.py",
            "sessions": sessions,
            "num_samples": int(len(X)),
            "alpha": args.alpha,
            "filter_window_samples": args.filter_window,
            "spec_filtered": not args.no_spec_filter,
            "holdout_mean_r2": {a: float(mean_r2[i]) for i, a in enumerate(AXES)},
            "in_sample_rmse": {a: float(rmse_in[i]) for i, a in enumerate(AXES)},
            "note": (
                "Apply the SAME preprocessing at inference: subtract a current zero "
                f"(re-tare when unloaded) and moving-average the channels over ~{args.filter_window} samples."
            ),
        },
    }, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Deploy: point OpenFTSensorConfig.calibration_file at it. Re-tare when unloaded; "
          f"apply the same ~{args.filter_window}-sample input filter.")


if __name__ == "__main__":
    main()
