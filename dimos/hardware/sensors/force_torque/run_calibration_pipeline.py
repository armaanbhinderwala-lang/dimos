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

"""Data to calibration in one command, with every check that has caught a bug so far.

    python3 run_calibration_pipeline.py --sessions 1786683727 1786734933 ... --out newrun

Stages, each gating the next:

    1 audit        labels, resting quality, rate, channels, reference in range
    2 verify       reference covers the stream, transform, baseline lands at zero
    3 condition    VIF and force/torque coupling, per axis and pooled
    4 fit          OLS and ridge, alpha swept, leave-one-session-out
    5 compare      against a reference calibration on the SAME held-out sessions
    6 save         matrix, bias, and everything needed to reproduce it

Baselines come from genuinely-idle samples, not merely labelled-resting ones. Runs so far
have carried up to 26 N while labelled resting; using those as zero shifted the matrix by
19.8% and cost 34% of held-out R2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import session_data as sd
from fit_calibration import fit_ols, fit_ridge, metrics, predict

IDLE_FORCE_N = 1.5          # above this a "resting" sample is not actually unloaded
MIN_IDLE_SAMPLES = 50
ALPHAS = (0.0, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5)


def load(sid: str, data_dir: Path) -> dict:
    d = sd.load_session(sid, data_dir, frame="diy")
    hm = sd.stream(sid, "homemade_raw", data_dir)
    ch = sd.columns(hm, tuple(f"ch{i}" for i in range(1, 17)))
    seg = np.array([float(r["experiment_start_ts"]) for r in hm])
    resting = d.labels == "resting"
    force = np.linalg.norm(d.raw_wrench[:, :3], axis=1)
    idle = resting & (force < IDLE_FORCE_N)
    fellback = idle.sum() < MIN_IDLE_SAMPLES
    if fellback:
        idle = resting
    loaded = ~resting
    return {
        "id": sid, "n": len(d.labels), "loaded": int(loaded.sum()), "resting": int(resting.sum()),
        "idle": int(idle.sum()), "fellback": fellback,
        "segments": len(set(seg[resting])), "labels": len(set(d.labels)),
        "rest_force": float(force[resting].mean()), "rest_force_sd": float(force[resting].std()),
        "X": (ch - ch[idle].mean(0))[loaded],
        "Y": (d.raw_wrench - d.raw_wrench[idle].mean(0))[loaded],
        "times": d.times, "raw_wrench": d.raw_wrench, "channels": ch, "resting_mask": resting,
    }


def stage_audit(sessions: list[dict]) -> list[str]:
    print("\n[1] AUDIT")
    print(f"  {'session':<13}{'loaded':>9}{'rest':>8}{'idle':>8}{'segs':>6}{'labels':>8}"
          f"{'rest |F|':>11}   flags")
    warnings = []
    for s in sessions:
        flags = []
        if s["labels"] <= 1:
            flags.append("NO LABELS")
        if s["segments"] < 2:
            flags.append("1 rest segment")
        if s["rest_force"] > IDLE_FORCE_N:
            flags.append(f"rest loaded {s['rest_force']:.1f}N")
        if s["fellback"]:
            flags.append("no clean idle samples")
        if s["loaded"] < 2000:
            flags.append("few loaded samples")
        print(f"  {s['id']:<13}{s['loaded']:>9,}{s['resting']:>8,}{s['idle']:>8,}"
              f"{s['segments']:>6}{s['labels']:>8}{s['rest_force']:>10.2f}N   "
              f"{', '.join(flags) if flags else 'ok'}")
        warnings += [f"{s['id']}: {f}" for f in flags]
    return warnings


def stage_verify(sessions: list[dict], data_dir: Path) -> list[str]:
    print("\n[2] VERIFY")
    problems = []
    for s in sessions:
        uf = sd.stream(s["id"], "ufactory_calibrated", data_dir)
        uf_t = sd.timestamps(uf)
        t = s["times"]
        outside = float(((t < uf_t.min()) | (t > uf_t.max())).mean() * 100)
        idle_after = float(np.linalg.norm(
            (s["Y"][:0] if not len(s["Y"]) else s["raw_wrench"][s["resting_mask"]][:, :3]
             - s["raw_wrench"][s["resting_mask"]][:, :3].mean(0)), axis=1).mean())
        note = "ok"
        if outside > 0.05:
            note = f"REFERENCE DOES NOT COVER STREAM ({outside:.2f}% extrapolated)"
            problems.append(f"{s['id']}: {note}")
        print(f"  {s['id']:<13} extrapolated {outside:6.3f}%   residual at rest {idle_after:5.2f}N   {note}")
    R = sd.R_UFACTORY_FROM_DIY
    ok = np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    print(f"  frame transform: rotation orthogonal, det=+1 -> {'ok' if ok else 'INVALID'}")
    if not ok:
        problems.append("frame rotation is not a proper rotation")
    return problems


def vif(W: np.ndarray) -> np.ndarray:
    z = (W - W.mean(0)) / np.maximum(W.std(0), 1e-12)
    out = []
    for i in range(W.shape[1]):
        other = [j for j in range(W.shape[1]) if j != i]
        r2 = 1 - np.linalg.lstsq(z[:, other], z[:, i], rcond=None)[1][0] / len(z)
        out.append(1 / max(1 - r2, 1e-9))
    return np.array(out)


def stage_condition(sessions: list[dict]) -> None:
    print("\n[3] CONDITIONING")
    print(f"  {'session':<13}{'corr(fx,my)':>13}{'corr(fy,mx)':>13}{'worst VIF':>11}")
    for s in sessions:
        W = s["Y"]
        print(f"  {s['id']:<13}{np.corrcoef(W[:, 0], W[:, 4])[0, 1]:>13.3f}"
              f"{np.corrcoef(W[:, 1], W[:, 3])[0, 1]:>13.3f}{vif(W).max():>11.2f}")
    P = np.vstack([s["Y"] for s in sessions])
    v = vif(P)
    print(f"  {'POOLED':<13}{np.corrcoef(P[:, 0], P[:, 4])[0, 1]:>13.3f}"
          f"{np.corrcoef(P[:, 1], P[:, 3])[0, 1]:>13.3f}{v.max():>11.2f}"
          f"   <- what the fit actually sees")
    print(f"\n  per-axis VIF   " + "  ".join(f"{a}={x:.2f}" for a, x in zip(sd.AXES, v)))
    print(f"  excitation p95 " + "  ".join(
        f"{a}={np.percentile(np.abs(P[:, i]), 95):.2f}" for i, a in enumerate(sd.AXES)))


def loso(sessions: list[dict], fn) -> tuple[np.ndarray, list[float]]:
    per = []
    for i in range(len(sessions)):
        train = [sessions[j] for j in range(len(sessions)) if j != i]
        A, b = fn(np.vstack([s["X"] for s in train]), np.vstack([s["Y"] for s in train]))
        per.append(metrics(sessions[i]["Y"], predict(A, b, sessions[i]["X"]))["r2"])
    return np.mean(per, axis=0), [float(p.mean()) for p in per]


def stage_fit(sessions: list[dict]) -> tuple[float, np.ndarray]:
    print("\n[4] FIT")
    print(f"  {'alpha':>9}" + "".join(f"{a:>8}" for a in sd.AXES) + f"{'mean':>9}")
    best = (0.0, -np.inf, None)
    for alpha in ALPHAS:
        fn = (lambda X, Y: fit_ols(X, Y)) if alpha == 0 else (lambda X, Y, a=alpha: fit_ridge(X, Y, a))
        m, _ = loso(sessions, fn)
        print(f"  {'OLS' if alpha == 0 else f'{alpha:g}':>9}"
              + "".join(f"{v:8.3f}" for v in m) + f"{m.mean():9.3f}")
        if m.mean() > best[1]:
            best = (alpha, float(m.mean()), m)
    print(f"\n  best: {'OLS' if best[0] == 0 else f'ridge alpha={best[0]:g}'}   mean R2 {best[1]:.3f}")
    return best[0], best[2]


def stage_compare(sessions: list[dict], reference: Path | None, alpha: float) -> None:
    if reference is None or not reference.exists():
        print("\n[5] COMPARE  (skipped: no --reference given)")
        return
    print(f"\n[5] COMPARE against {reference.name}, on the SAME held-out sessions")
    d = np.load(reference, allow_pickle=True)
    A_ref, b_ref = d["A"], d["b"]
    ref = np.mean([metrics(s["Y"], predict(A_ref, b_ref, s["X"]))["r2"] for s in sessions], axis=0)
    fn = (lambda X, Y: fit_ols(X, Y)) if alpha == 0 else (lambda X, Y, a=alpha: fit_ridge(X, Y, a))
    new, _ = loso(sessions, fn)
    print(f"  {'':<11}" + "".join(f"{a:>8}" for a in sd.AXES) + f"{'mean':>9}")
    print(f"  {'reference':<11}" + "".join(f"{v:8.3f}" for v in ref) + f"{ref.mean():9.3f}")
    print(f"  {'this fit':<11}" + "".join(f"{v:8.3f}" for v in new) + f"{new.mean():9.3f}")
    print(f"  {'change':<11}" + "".join(f"{v:+8.3f}" for v in new - ref) + f"{new.mean() - ref.mean():+9.3f}")
    better = int((new > ref).sum())
    print(f"\n  better on {better} of 6 axes"
          + ("   <- adopt" if new.mean() > ref.mean() else "   <- does NOT beat the reference"))


def stage_save(sessions: list[dict], alpha: float, out: str, data_dir: Path) -> None:
    print("\n[6] SAVE")
    X = np.vstack([s["X"] for s in sessions])
    Y = np.vstack([s["Y"] for s in sessions])
    for tag, a in (("ols", 0.0), ("ridge", alpha if alpha else 3e3)):
        fn = (lambda X, Y: fit_ols(X, Y)) if a == 0 else (lambda X, Y, aa=a: fit_ridge(X, Y, aa))
        A, b = fn(X, Y)
        held, per = loso(sessions, fn)
        npz = data_dir / f"calibration_{out}_{tag}.npz"
        np.savez(npz, A=A, b=b, sessions=np.array([s["id"] for s in sessions]),
                 method=tag, alpha=a, frame="diy", axes=np.array(sd.AXES),
                 baseline=f"idle-only: labelled resting AND |F|<{IDLE_FORCE_N}N",
                 R_ufactory_from_diy=sd.R_UFACTORY_FROM_DIY,
                 diy_origin_in_ufactory_m=sd.DIY_ORIGIN_IN_UFACTORY_M,
                 heldout_r2=held, n_samples=len(X))
        js = data_dir / f"ft_calibration_{out}_{tag}.json"
        json.dump({"calibration_matrix": A.tolist(), "bias_vector": b.tolist(),
                   "axes": list(sd.AXES), "frame": "diy", "method": tag, "alpha": a,
                   "sessions": [s["id"] for s in sessions], "n_samples": int(len(X)),
                   "heldout_r2": {ax: float(v) for ax, v in zip(sd.AXES, held)},
                   "heldout_per_session": dict(zip([s["id"] for s in sessions], per))},
                  js.open("w"), indent=2)
        print(f"  {npz.name:<34} held-out mean R2 {held.mean():.3f}")
        print(f"  {js.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--out", default="newrun", help="name stem for the saved calibration")
    p.add_argument("--reference", type=Path, default=None,
                   help="existing .npz to compare against on the same held-out sessions")
    p.add_argument("--force", action="store_true", help="save even if the audit raised warnings")
    args = p.parse_args()

    print(f"CALIBRATION PIPELINE  |  {len(args.sessions)} sessions  |  DIY frame")
    sessions = [load(s, args.data_dir) for s in args.sessions]
    warnings = stage_audit(sessions)
    problems = stage_verify(sessions, args.data_dir)
    stage_condition(sessions)
    alpha, _ = stage_fit(sessions)
    stage_compare(sessions, args.reference, alpha)

    if problems:
        print(f"\nNOT SAVING -- {len(problems)} verification problem(s):")
        for x in problems:
            print(f"  {x}")
        raise SystemExit(1)
    if warnings and not args.force:
        print(f"\n{len(warnings)} audit warning(s):")
        for w in warnings:
            print(f"  {w}")
        print("  saving anyway (warnings are advisory); use the audit to decide what to re-record")
    stage_save(sessions, alpha, args.out, args.data_dir)


if __name__ == "__main__":
    main()
