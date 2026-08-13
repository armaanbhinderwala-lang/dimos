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

"""Check that a collected dataset actually covers the space before fitting anything.

Six panels. The first four describe the data; the last two are the ones that decide whether
the fit can succeed:

    5  identifiability -- VIF per axis. If two axes always move together no fit can separate
       them, and the calibration will silently attribute one to the other.
    6  gravity check   -- only when a protocol run supplies known masses and lever arms.
       Compares the uFactory against a wrench computed from m*g*L, which depends on neither
       sensor. This is the only panel that can tell you the REFERENCE is wrong.

    python3 visualize_dataset.py --data-dir . --out coverage.png
    python3 visualize_dataset.py --data-dir . --steps session_protocol_steps.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from session_data import AXES, load_all  # noqa: E402


def identifiability(wrench: np.ndarray) -> list[tuple[str, float, float]]:
    z = (wrench - wrench.mean(0)) / np.maximum(wrench.std(0), 1e-12)
    out = []
    for i, axis in enumerate(AXES):
        other = [j for j in range(len(AXES)) if j != i]
        resid = np.linalg.lstsq(z[:, other], z[:, i], rcond=None)[1]
        r2 = 1 - (resid[0] / len(z) if len(resid) else 0.0)
        out.append((axis, r2, 1 / max(1 - r2, 1e-9)))
    return out


def gravity_truth(steps_csv: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(steps_csv.open()))
    return {r["label"]: np.array([float(r[f"exp_{a}"]) for a in AXES]) for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--steps", type=Path, default=None, help="protocol steps CSV for the gravity check")
    p.add_argument("--frame", choices=["diy", "ufactory"], default="diy")
    p.add_argument("--out", type=Path, default=Path("coverage.png"))
    args = p.parse_args()

    sessions = load_all(args.data_dir, frame=args.frame)
    W = np.vstack([s.wrench for s in sessions])
    S = np.vstack([s.channels for s in sessions])
    labels = np.concatenate([s.labels for s in sessions])
    print(f"{len(sessions)} sessions | {len(W):,} samples | frame={args.frame}")

    fig = plt.figure(figsize=(17, 10.5))
    sub = W[:: max(1, len(W) // 6000)]

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap="viridis", s=2, alpha=.5)
    ax.set_title("force coverage"); ax.set_xlabel("Fx (N)"); ax.set_ylabel("Fy (N)"); ax.set_zlabel("Fz (N)")

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter(sub[:, 3], sub[:, 4], sub[:, 5], c=sub[:, 5], cmap="plasma", s=2, alpha=.5)
    ax.set_title("torque coverage"); ax.set_xlabel("Mx"); ax.set_ylabel("My"); ax.set_zlabel("Mz")

    ax = fig.add_subplot(2, 3, 3)
    im = ax.matshow(np.corrcoef(S.T), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("16-channel correlation", pad=16); fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_xticks(range(0, 16, 3), [str(i + 1) for i in range(0, 16, 3)])
    ax.set_yticks(range(0, 16, 3), [str(i + 1) for i in range(0, 16, 3)])

    ax = fig.add_subplot(2, 3, 4)
    step = max(1, len(W) // 4000)
    for i, a in enumerate(AXES[:3]):
        ax.plot(W[::step, i], lw=.7, label=a)
    ax.set_title("force over the whole dataset"); ax.set_xlabel("sample"); ax.set_ylabel("N")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 3, 5)
    rows = identifiability(W)
    colours = ["#2f7d5c" if v < 2 else "#d08a2a" if v < 5 else "#a8332f" for _, _, v in rows]
    ax.barh([r[0] for r in rows], [r[2] for r in rows], color=colours)
    ax.axvline(2, ls="--", c="#666", lw=1); ax.axvline(5, ls="--", c="#a8332f", lw=1)
    ax.set_title("identifiability (VIF)\n<2 separable, >5 entangled"); ax.set_xlabel("VIF")
    ax.grid(alpha=.3, axis="x")
    for i, (_, _, v) in enumerate(rows):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)

    ax = fig.add_subplot(2, 3, 6)
    if args.steps and args.steps.exists():
        truth = gravity_truth(args.steps)
        xs, ys = [], []
        for label in {lb for lb in labels if lb in truth}:
            mask = labels == label
            if mask.sum() < 20:
                continue
            xs.append(truth[label]); ys.append(W[mask].mean(0))
        if xs:
            xs, ys = np.array(xs), np.array(ys)
            for i, a in enumerate(AXES):
                ax.scatter(xs[:, i], ys[:, i], s=18, label=a, alpha=.75)
            lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
            ax.plot(lim, lim, "k--", lw=1)
            err = np.abs(xs - ys)
            ax.set_title(f"uFactory vs gravity truth\nmedian |error| {np.median(err):.2f}")
            ax.set_xlabel("expected from m·g·L"); ax.set_ylabel("uFactory reading")
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
            print("\ngravity check (uFactory vs m*g*L):")
            for i, a in enumerate(AXES):
                print(f"  {a}: median |err| {np.median(np.abs(xs[:, i] - ys[:, i])):.3f}"
                      f"   slope {np.polyfit(xs[:, i], ys[:, i], 1)[0]:+.3f}")
        else:
            ax.text(.5, .5, "no protocol labels matched", ha="center", transform=ax.transAxes)
    else:
        ax.text(.5, .5, "no --steps given\n\nrun the protocol collector to enable\nthe gravity check",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")

    print("\nidentifiability:")
    for axis, r2, vif in rows:
        print(f"  {axis}: R2_other {r2:.3f}  VIF {vif:5.2f}"
              f"   {'separable' if vif < 2 else 'entangled' if vif < 5 else 'NOT SEPARABLE'}")
    print(f"  condition number of the load set: {np.linalg.cond(np.corrcoef(W.T)):.1f}")


if __name__ == "__main__":
    main()
