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

"""Baseline calibration: fit W = A x + b mapping DIY sensor channels to a physical wrench.

    x  in R^16   raw Hall channels, ADC counts, zero already removed
    W  in R^6    wrench (Fx Fy Fz Mx My Mz) expressed in the DIY sensor's own frame
    A  in R^6x16 calibration matrix        b in R^6  bias

Physically this is the inverse of the sensor's own sensitivity: if x = S*W, then W = S^-1 x,
so A estimates S^-1. We learn it rather than derive it because S depends on magnet placement,
cross-axis coupling and manufacturing variation we cannot measure directly.

A is NOT expected to be diagonal-ish. Cross-axis coupling is real and the matrix is supposed
to capture it -- every channel may contribute to every output.

Deliberately the simplest physically justified model: ordinary least squares, no filtering,
no nonlinearity. This is the number every later improvement has to beat.

Loading, alignment, zero removal and the frame transform live in session_data.py.
Run 01_inspect_data.ipynb first; this script assumes the data has already been checked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from session_data import (
    AXES,
    CHANNELS,
    DIY_ORIGIN_IN_UFACTORY_M,
    R_UFACTORY_FROM_DIY,
    Session,
    find_sessions,
    load_all,
)


# --------------------------------------------------------------------------- fit
def fit_ols(x: np.ndarray, y: np.ndarray, with_bias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Least squares fit of y = A x + b. Returns (A [6xN], b [6]).

    Solved with lstsq (which uses the pseudoinverse internally) rather than forming
    (X^T X)^-1 explicitly -- the normal equations square the condition number and lose
    precision when channels are near-duplicates, which they are here.
    """
    design = np.hstack([x, np.ones((len(x), 1))]) if with_bias else x
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)   # (N+1, 6) or (N, 6)
    if with_bias:
        return solution[:-1].T, solution[-1]
    return solution.T, np.zeros(y.shape[1])


def predict(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ A.T + b


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-axis R2, MAE and RMSE.

    R2 = 1 - var(error)/var(truth):  1.0 perfect, 0.0 no better than guessing the mean,
    negative worse than guessing the mean.
    """
    err = truth - pred
    return {
        "r2": 1 - err.var(axis=0) / np.maximum(truth.var(axis=0), 1e-12),
        "mae": np.abs(err).mean(axis=0),
        "rmse": np.sqrt((err**2).mean(axis=0)),
    }


def stack(sessions: list[Session]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([s.channels for s in sessions]), np.vstack([s.wrench for s in sessions])


# --------------------------------------------------------------- validation
def leave_one_session_out(sessions: list[Session], with_bias: bool) -> dict:
    """Train on every session but one, test on the held-out one, rotate.

    The only honest score here: samples arrive at ~35 Hz so neighbouring rows are
    near-duplicates, and a random split would put near-copies of the test data into
    training. This measures whether the calibration transfers to a NEW physical session.
    """
    per_session = []
    for held in sessions:
        train = [s for s in sessions if s is not held]
        A, b = fit_ols(*stack(train), with_bias)
        m = metrics(held.wrench, predict(A, b, held.channels))
        per_session.append({"session": held.session_id, "n": len(held), **m})
    mean = {k: np.mean([p[k] for p in per_session], axis=0) for k in ("r2", "mae", "rmse")}
    return {"per_session": per_session, "mean": mean}


# --------------------------------------------------------------- reporting
def print_axis_table(title: str, m: dict[str, np.ndarray]) -> None:
    print(f"\n{title}")
    print(f"  {'axis':<5} {'R2':>8} {'MAE':>10} {'RMSE':>10}   unit")
    for i, a in enumerate(AXES):
        unit = "N" if i < 3 else "N*m"
        print(f"  {a:<5} {m['r2'][i]:8.3f} {m['mae'][i]:10.3f} {m['rmse'][i]:10.3f}   {unit}")


def print_per_session(result: dict) -> None:
    print(f"\n  per-session held-out R2")
    print(f"  {'session':>12} {'n':>7}  " + "".join(f"{a:>7}" for a in AXES))
    for p in result["per_session"]:
        print(f"  {p['session']:>12} {p['n']:7d}  " + "".join(f"{v:7.2f}" for v in p["r2"]))


# --------------------------------------------------------------- entry point
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("calibration_baseline.npz"))
    p.add_argument("--json-out", type=Path, default=Path("ft_calibration_baseline.json"),
                   help="same matrix in the format openft_module.py already reads")
    p.add_argument("--frame", choices=["diy", "ufactory"], default="diy",
                   help="frame the target wrench is expressed in (default: the DIY sensor's own)")
    p.add_argument("--filter-window", type=int, default=0,
                   help="input moving-average samples; 0 = none, which is the true baseline")
    p.add_argument("--sessions", nargs="*")
    args = p.parse_args()

    ids = args.sessions or find_sessions(args.data_dir)
    if len(ids) < 2:
        raise SystemExit(f"need >= 2 sessions for leave-one-session-out, found {len(ids)}")

    sessions = load_all(args.data_dir, ids, frame=args.frame, filter_window=args.filter_window)
    x_all, y_all = stack(sessions)
    print(f"{len(ids)} sessions | {len(x_all):,} samples | {CHANNELS} channels -> {len(AXES)} outputs")
    print(f"target frame: {args.frame}   input filter: {args.filter_window or 'none'}")

    # --- the two models the baseline compares ------------------------------
    results = {}
    for name, with_bias in (("Model 1:  W = A x + b", True), ("Model 2:  W = A x", False)):
        r = leave_one_session_out(sessions, with_bias)
        results[with_bias] = r
        print_axis_table(f"{name}   [leave-one-session-out]", r["mean"])

    d = results[True]["mean"]["r2"] - results[False]["mean"]["r2"]
    print(f"\n  bias term changes R2 by: " + " ".join(f"{a}:{v:+.3f}" for a, v in zip(AXES, d, strict=True)))
    print("  (near zero means the measured zero-removal already handles the offset,")
    print("   so the model does not need to learn one)")

    print_per_session(results[True])

    # --- final fit on everything -------------------------------------------
    A, b = fit_ols(x_all, y_all, with_bias=True)
    print_axis_table("Final fit on ALL sessions   [in-sample, optimistic by definition]",
                     metrics(y_all, predict(A, b, x_all)))

    print(f"\n  bias magnitude |b| = {np.abs(b).max():.4f} (max element)")
    norms = np.linalg.norm(A, axis=1)
    dead = [AXES[i] for i in range(len(AXES)) if norms[i] <= 1e-12 * norms.max()]
    if dead:
        print(f"  WARNING: all-zero row(s) for {', '.join(dead)} -- that axis reads a constant.")

    # --- save --------------------------------------------------------------
    # Everything needed to reproduce a prediction travels with the matrix. A calibration
    # without its preprocessing is unusable, so the frame and the filter are saved too.
    holdout = results[True]["mean"]
    np.savez(
        args.out,
        A=A, b=b,
        channels=CHANNELS, axes=np.array(AXES),
        sessions=np.array(ids), num_samples=len(x_all),
        frame=args.frame,
        filter_window=args.filter_window,
        zero_removal="time-varying baseline interpolated between resting segments",
        R_ufactory_from_diy=R_UFACTORY_FROM_DIY,
        diy_origin_in_ufactory_m=DIY_ORIGIN_IN_UFACTORY_M,
        holdout_r2=holdout["r2"], holdout_mae=holdout["mae"], holdout_rmse=holdout["rmse"],
    )

    args.json_out.write_text(json.dumps({
        "calibration_matrix": A.tolist(),
        "bias_vector": b.tolist(),
        "sensor_channels": CHANNELS,
        "output_channels": len(AXES),
        "timestamp": time.time(),
        "metadata": {
            "fit_by": "fit_calibration.py (OLS baseline)",
            "sessions": ids,
            "num_samples": int(len(x_all)),
            "target_frame": args.frame,
            "filter_window_samples": args.filter_window,
            "holdout_r2": dict(zip(AXES, holdout["r2"].round(4).tolist(), strict=True)),
            "holdout_rmse": dict(zip(AXES, holdout["rmse"].round(4).tolist(), strict=True)),
            "inference_note": (
                "Re-zero the channels while unloaded before applying this matrix. Output is "
                f"the wrench in the {args.frame} frame; use R_ufactory_from_diy and "
                "diy_origin_in_ufactory_m from the npz to move it to another frame."
            ),
        },
    }, indent=2))

    print(f"\nwrote {args.out}  (A, b, frame definition, preprocessing, metrics)")
    print(f"wrote {args.json_out}  (driver-compatible)")


if __name__ == "__main__":
    main()
