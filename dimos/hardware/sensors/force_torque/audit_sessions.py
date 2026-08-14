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

"""Decide which sessions are fit to train on, before any of them reach the fit.

A calibration is only as honest as the worst session inside it, and a bad session does not
announce itself -- it quietly shifts the matrix and shows up later as a sensor that reads
wrong on the robot. Two we already collected are unusable (one unbroken `resting` block, so
the baseline step subtracts the signal itself), and nothing in the pipeline would have said
so. This says so, per session, with the number that justifies the verdict.

Checks, in the order a problem would actually bite:

    labels     did the operator label anything, and is any single label swamping the run
    baseline   are there enough resting segments, spread through the run, to track drift
    timing     sample rate steady, no dropouts, and the two streams actually aligned
    channels   dead, stuck, saturated or unusually noisy ADC channels
    reference  uFactory readings inside its own rated range, so ground truth is trustworthy
    coverage   is each wrench axis actually excited, or is the session blind to some of them
    geometry   does the grip lever arm vary, without which torque cannot be separated from force

Exit code is nonzero if any audited session fails, so this can gate a fit.

    python3 audit_sessions.py --data-dir .
    python3 audit_sessions.py --data-dir . --sessions 1786683727 1786690000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from session_data import AXES, CHANNELS, UF_RATED, columns, find_sessions, stream, timestamps

# A session failing any of these is excluded from a fit rather than quietly weakening it.
MIN_LOADED_SAMPLES = 2000
MIN_RESTING_SAMPLES = 200
MIN_RESTING_SEGMENTS = 2
MAX_SINGLE_LABEL_SHARE = 0.80      # one action dominating means the run is not varied
MAX_SYNC_LAG_S = 0.05
MAX_RATE_JITTER = 0.35             # std/mean of sample intervals
MIN_LEVER_SPREAD_MM = 3.0          # a perfectly rigid grip cannot separate force from torque


def audit(session: str, data_dir: Path) -> dict:
    problems: list[str] = []
    notes: list[str] = []

    uf = stream(session, "ufactory_calibrated", data_dir)
    hm = stream(session, "homemade_raw", data_dir)
    uf_t, uf_w = timestamps(uf), columns(uf, AXES)
    t, ch = timestamps(hm), columns(hm, CHAN_COLS := tuple(f"ch{i}" for i in range(1, CHANNELS + 1)))
    labels = np.array([r["label"] for r in hm])
    seg = np.array([float(r["experiment_start_ts"]) for r in hm])

    # --- labels -----------------------------------------------------------
    distinct = sorted(set(labels))
    resting = labels == "resting"
    loaded = ~resting
    if len(distinct) <= 1:
        problems.append(f"only one label ({distinct[0]!r}) for the entire run -- nothing to train on")
    if loaded.sum() < MIN_LOADED_SAMPLES:
        problems.append(f"only {loaded.sum():,} loaded samples, need >= {MIN_LOADED_SAMPLES:,}")
    if loaded.sum():
        counts = {lb: int((labels == lb).sum()) for lb in distinct if lb != "resting"}
        if counts:
            top, n = max(counts.items(), key=lambda kv: kv[1])
            share = n / max(loaded.sum(), 1)
            if share > MAX_SINGLE_LABEL_SHARE:
                problems.append(f"label {top!r} is {share:.0%} of all loaded samples -- run is not varied")

    # --- baseline ---------------------------------------------------------
    segments = len(set(seg[resting])) if resting.any() else 0
    if resting.sum() < MIN_RESTING_SAMPLES:
        problems.append(f"only {resting.sum():,} resting samples, need >= {MIN_RESTING_SAMPLES:,}")
    if segments < MIN_RESTING_SEGMENTS:
        problems.append(f"only {segments} resting segment(s); drift cannot be tracked across the run")

    # --- timing -----------------------------------------------------------
    dt = np.diff(t)
    rate = 1.0 / np.median(dt) if len(dt) else 0.0
    jitter = float(np.std(dt) / max(np.mean(dt), 1e-12)) if len(dt) else 0.0
    gaps = int((dt > 5 * np.median(dt)).sum()) if len(dt) else 0
    if jitter > MAX_RATE_JITTER:
        problems.append(f"sample interval jitter {jitter:.2f} (std/mean) -- logging was not keeping up")
    if gaps:
        notes.append(f"{gaps} timing gap(s) over 5x the median interval")

    wrench = np.column_stack([np.interp(t, uf_t, uf_w[:, i]) for i in range(len(AXES))])
    a = np.linalg.norm(ch - ch.mean(0), axis=1)
    b = np.linalg.norm(wrench[:, :3] - wrench[:, :3].mean(0), axis=1)
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(-30, 31)
    score = [float(np.corrcoef(np.roll(a, k), b)[0, 1]) for k in lags]
    best_lag = float(lags[int(np.argmax(score))] / max(rate, 1e-9))
    peak = max(score)
    if peak < 0.3:
        notes.append(f"streams correlate only {peak:.2f}; sync lag estimate is unreliable")
    elif abs(best_lag) > MAX_SYNC_LAG_S:
        problems.append(f"streams misaligned by {best_lag * 1000:+.0f} ms")

    # --- channels ---------------------------------------------------------
    noise = ch[resting].std(axis=0) if resting.sum() > 30 else ch.std(axis=0)
    span = ch.max(axis=0) - ch.min(axis=0)
    dead = [i + 1 for i in range(CHANNELS) if span[i] < 1e-6]
    stuck = [i + 1 for i in range(CHANNELS) if noise[i] < 1e-9]
    med = float(np.median(noise))
    noisy = [i + 1 for i in range(CHANNELS) if med > 0 and noise[i] > 5 * med]
    if dead:
        problems.append(f"channel(s) {dead} never move")
    if stuck:
        problems.append(f"channel(s) {stuck} have zero noise -- likely not being read")
    if noisy:
        notes.append(f"channel(s) {noisy} noisier than 5x the median")

    # --- reference --------------------------------------------------------
    over = {a_: int((np.abs(wrench[:, i]) > UF_RATED[a_]).sum()) for i, a_ in enumerate(AXES)}
    beyond = {k: v for k, v in over.items() if v}
    if beyond:
        worst = max(beyond.values()) / len(wrench)
        msg = f"reference beyond its rated range on {list(beyond)} ({worst:.1%} of samples)"
        (problems if worst > 0.01 else notes).append(msg)

    # --- coverage ---------------------------------------------------------
    W = wrench[loaded] if loaded.any() else wrench
    p95 = {a_: float(np.percentile(np.abs(W[:, i]), 95)) for i, a_ in enumerate(AXES)}
    blind = [a_ for a_ in AXES if p95[a_] < (3.0 if a_[0] == "f" else 0.15)]
    if blind:
        notes.append(f"barely excited: {blind} -- this run carries no information about them")

    # --- geometry ---------------------------------------------------------
    lever_spread = float("nan")
    if loaded.sum() > 500:
        F, T = W[:, :3], W[:, 3:]
        keep = np.linalg.norm(F, axis=1) > 5
        if keep.sum() > 200:
            C = np.linalg.lstsq(np.hstack([F[keep], np.ones((keep.sum(), 1))]), T[keep],
                                rcond=None)[0][:3].T
            lever_spread = abs(-C[0, 1] * 1000 - C[1, 0] * 1000)
            if lever_spread < MIN_LEVER_SPREAD_MM:
                notes.append(f"grip lever arm barely varies ({lever_spread:.1f} mm) -- "
                             "torque is nearly a fixed multiple of force in this run")

    return {
        "session": session, "n": len(t), "loaded": int(loaded.sum()), "resting": int(resting.sum()),
        "segments": segments, "labels": len(distinct), "rate": rate, "lag_ms": best_lag * 1000,
        "sync_corr": peak, "lever_spread": lever_spread, "p95": p95,
        "problems": problems, "notes": notes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--sessions", nargs="*", help="audit only these (default: all found)")
    args = p.parse_args()

    ids = args.sessions or find_sessions(args.data_dir)
    results = [audit(s, args.data_dir) for s in ids]

    print(f"\n{'session':<14}{'samples':>9}{'loaded':>9}{'rest':>7}{'segs':>6}{'labels':>8}"
          f"{'Hz':>6}{'lag ms':>8}{'lever mm':>10}   verdict")
    for r in results:
        verdict = "FAIL" if r["problems"] else ("ok" if not r["notes"] else "ok (see notes)")
        lever = "--" if np.isnan(r["lever_spread"]) else f"{r['lever_spread']:.1f}"
        print(f"{r['session']:<14}{r['n']:>9,}{r['loaded']:>9,}{r['resting']:>7,}{r['segments']:>6}"
              f"{r['labels']:>8}{r['rate']:>6.0f}{r['lag_ms']:>8.0f}{lever:>10}   {verdict}")

    for r in results:
        if r["problems"] or r["notes"]:
            print(f"\n  {r['session']}")
            for problem in r["problems"]:
                print(f"    FAIL  {problem}")
            for note in r["notes"]:
                print(f"    note  {note}")

    print("\nexcitation per axis (p95 of |value| over loaded samples)")
    print(f"  {'session':<14}" + "".join(f"{a:>9}" for a in AXES))
    for r in results:
        print(f"  {r['session']:<14}" + "".join(f"{r['p95'][a]:9.2f}" for a in AXES))

    good = [r["session"] for r in results if not r["problems"]]
    bad = [r["session"] for r in results if r["problems"]]
    print(f"\n{len(good)} of {len(results)} sessions usable")
    if bad:
        print(f"excluded: {' '.join(bad)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
