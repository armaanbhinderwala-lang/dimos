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

"""Record a labelled run and debug the calibration pipeline one level at a time.

Every score we have so far (R2, RMSE, cross-talk) is computed on the FITTED relationship,
so a failure anywhere upstream shows up as "the fit is bad" with no way to localise it.
This walks the stages independently and prints a number per stage.

    L0  channel health        dead, saturated or drifting channels
    L1  raw response          do the channels move under load, and HOW
    L2  sync                  lag between the two streams at a step input
    L3  zero                  residual offset after baseline removal
    L4  transform             force/torque coupling before vs after the adjoint
    L5  fit                   end-to-end A*S gain and cross-talk

L1 is the one that has never been run. A moment applied to the plate should push opposing
magnet clusters in OPPOSITE directions (differential); a pure force pushes them the SAME way
(common mode). If a pure-couple trial produces a common-mode response, the geometry never
encoded torque and nothing downstream can recover it.

    python3 ft_diagnostic.py record --ip 192.168.1.226 --port /dev/tty.usbmodem1101
    python3 ft_diagnostic.py report --run diagnostic_<id>.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ft_calibration_collector import HomemadeReader, UFactoryReader
from session_data import AXES, CHANNELS

# Held on the plate long enough to average; keys mirror the collector's layout.
TRIALS = {
    "1": ("rest", "hands off, let it settle"),
    "2": ("push_x", "pure push along +X, through the sensor centre"),
    "3": ("push_y", "pure push along +Y, through the sensor centre"),
    "4": ("push_z", "pure push along +Z (straight down the axis)"),
    "5": ("couple_x", "TWIST about X only -- two hands, equal and opposite, no net push"),
    "6": ("couple_y", "TWIST about Y only -- two hands, equal and opposite, no net push"),
    "7": ("couple_z", "TWIST about Z only -- two hands, equal and opposite, no net push"),
    "8": ("loop", "run the normal door-pull loop"),
}
COLUMNS = (["ts", "label"] + [f"ch{i}" for i in range(1, CHANNELS + 1)]
           + [f"uf_{a}" for a in AXES] + [f"ufraw_{a}" for a in AXES])


# --------------------------------------------------------------------------- record
def record(args: argparse.Namespace) -> None:
    import pygame

    uf = UFactoryReader(args.ip, rate_hz=args.rate)
    hm = HomemadeReader(args.port, args.baud, window=args.filter_window)
    uf.start()
    hm.start()

    out = args.out or Path(f"diagnostic_{int(time.time())}.csv")
    fh = out.open("w", newline="")
    writer = csv.writer(fh)
    writer.writerow(COLUMNS)

    pygame.init()
    screen = pygame.display.set_mode((980, 620))
    pygame.display.set_caption("FT diagnostic recorder")
    big = pygame.font.SysFont("menlo", 26)
    mid = pygame.font.SysFont("menlo", 19)
    small = pygame.font.SysFont("menlo", 15)

    label, n, running = "rest", 0, True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    key = event.unicode
                    if key in TRIALS:
                        label = TRIALS[key][0]

            channels, ext, raw = hm.latest_channels, uf.latest_ext, uf.latest_raw
            writer.writerow([f"{time.time():.6f}", label, *channels, *ext, *raw])
            n += 1

            screen.fill((16, 18, 22))
            screen.blit(big.render(f"recording: {label}", True, (255, 190, 90)), (18, 16))
            screen.blit(mid.render(f"{n} samples -> {out.name}", True, (150, 150, 150)), (18, 52))
            y = 92
            for key, (name, hint) in TRIALS.items():
                on = name == label
                screen.blit(mid.render(f"[{key}] {name:<10} {hint}", True,
                                       (255, 210, 120) if on else (135, 140, 148)), (18, y))
                y += 26
            y += 12
            screen.blit(small.render("uFactory  " + "  ".join(f"{a}={v:8.2f}" for a, v in zip(AXES, ext)),
                                     True, (120, 190, 240)), (18, y))
            y += 26
            for row in range(0, CHANNELS, 8):
                screen.blit(small.render("ch " + " ".join(f"{v:7.1f}" for v in channels[row:row + 8]),
                                         True, (230, 150, 100)), (18, y))
                y += 24
            screen.blit(small.render("ESC to stop.  Hold each trial ~10 s with a clear rest between.",
                                     True, (120, 124, 130)), (18, 580))
            pygame.display.flip()
            time.sleep(1.0 / args.rate)
    finally:
        hm.stop()
        uf.stop()
        fh.close()
        pygame.quit()
        print(f"wrote {out}  ({n} samples)")


# --------------------------------------------------------------------------- report
def load_run(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path} is empty")
    get = lambda keys: np.array([[float(r[k]) for k in keys] for r in rows])
    return {
        "t": np.array([float(r["ts"]) for r in rows]),
        "label": np.array([r["label"] for r in rows]),
        "ch": get([f"ch{i}" for i in range(1, CHANNELS + 1)]),
        "uf": get([f"uf_{a}" for a in AXES]),
    }


def segments(run: dict) -> dict[str, np.ndarray]:
    return {name: run["label"] == name for name in dict.fromkeys(run["label"])}


def level0(run: dict, rest: np.ndarray) -> None:
    print("\nL0  channel health")
    ch = run["ch"]
    noise = ch[rest].std(axis=0) if rest.sum() > 30 else ch.std(axis=0)
    span = ch.max(axis=0) - ch.min(axis=0)
    drift = np.abs(ch[rest][-1] - ch[rest][0]) if rest.sum() > 30 else np.zeros(CHANNELS)
    print(f"  {'ch':>4}{'noise':>9}{'span':>9}{'drift':>9}   status")
    for i in range(CHANNELS):
        flags = []
        if span[i] < 1e-6:
            flags.append("DEAD")
        if noise[i] > 5 * np.median(noise):
            flags.append("NOISY")
        if drift[i] > 5 * max(noise[i], 1e-9):
            flags.append("DRIFTING")
        print(f"  {i + 1:>4}{noise[i]:9.2f}{span[i]:9.1f}{drift[i]:9.2f}   {' '.join(flags) or 'ok'}")
    print(f"  median noise {np.median(noise):.2f} counts")


def level1(run: dict, segs: dict, rest: np.ndarray) -> None:
    """The unrun test: is a moment encoded differentially, or not at all?"""
    print("\nL1  raw channel response per trial")
    if rest.sum() < 30:
        print("  no 'rest' segment -- cannot measure a baseline. Re-record with rest periods.")
        return
    base = run["ch"][rest].mean(axis=0)
    noise = run["ch"][rest].std(axis=0)
    floor = float(np.median(noise))
    ones = np.ones(CHANNELS) / np.sqrt(CHANNELS)

    print(f"  {'trial':<11}{'|dch|':>8}{'SNR':>7}{'common':>9}{'diff':>7}   {'load applied (uFactory)':<26} verdict")
    for name, mask in segs.items():
        if name == "rest" or mask.sum() < 30:
            continue
        delta = run["ch"][mask].mean(axis=0) - base
        mag = float(np.linalg.norm(delta))
        snr = mag / (floor * np.sqrt(CHANNELS)) if floor else 0.0
        common = float((delta @ ones) ** 2 / max(delta @ delta, 1e-12))
        load = run["uf"][mask].mean(axis=0) - run["uf"][rest].mean(axis=0)
        desc = " ".join(f"{a}{v:+.1f}" for a, v in zip(AXES, load) if abs(v) > (2.0 if a[0] == "f" else 0.15))
        if snr < 2:
            verdict = "NO RESPONSE"
        elif name.startswith("couple"):
            verdict = "differential -- torque IS encoded" if common < 0.4 else "COMMON MODE -- torque not encoded"
        else:
            verdict = "responds"
        print(f"  {name:<11}{mag:8.1f}{snr:7.1f}{common:9.2f}{1 - common:7.2f}   {desc[:26]:<26} {verdict}")

    print("\n  per-cluster mean deflection (channels grouped 1-4, 5-8, 9-12, 13-16)")
    print(f"  {'trial':<11}" + "".join(f"{f'grp{g}':>9}" for g in range(4)) + "   opposing pairs should invert under a couple")
    for name, mask in segs.items():
        if name == "rest" or mask.sum() < 30:
            continue
        delta = run["ch"][mask].mean(axis=0) - base
        groups = [float(delta[g * 4:(g + 1) * 4].mean()) for g in range(4)]
        print(f"  {name:<11}" + "".join(f"{v:9.2f}" for v in groups))


def level2(run: dict, segs: dict) -> None:
    print("\nL2  sync")
    ch, uf, t = run["ch"], run["uf"], run["t"]
    a = np.linalg.norm(ch - ch.mean(axis=0), axis=1)
    b = np.linalg.norm(uf[:, :3] - uf[:, :3].mean(axis=0), axis=1)
    a, b = a - a.mean(), b - b.mean()
    dt = float(np.median(np.diff(t)))
    lags = np.arange(-30, 31)
    score = [np.corrcoef(np.roll(a, k), b)[0, 1] for k in lags]
    best, peak = lags[int(np.argmax(score))], max(score)
    verdict = ("UNRELIABLE -- streams barely correlate, lag estimate is meaningless" if peak < 0.3
               else "ok" if abs(best * dt) < 0.05 else "MISALIGNED")
    print(f"  rate {1 / dt:.1f} Hz   best lag {best * dt * 1000:+.0f} ms   corr {peak:.3f}   {verdict}")


def level3(run: dict, rest: np.ndarray) -> None:
    print("\nL3  zero")
    if rest.sum() < 60:
        print("  not enough rest samples")
        return
    half = rest.sum() // 2
    idx = np.flatnonzero(rest)
    first, second = run["ch"][idx[:half]].mean(axis=0), run["ch"][idx[half:]].mean(axis=0)
    shift = np.abs(second - first)
    noise = run["ch"][rest].std(axis=0)
    bad = [i + 1 for i in range(CHANNELS) if shift[i] > 3 * max(noise[i], 1e-9)]
    print(f"  max baseline shift between first and second half: {shift.max():.2f} counts")
    print(f"  channels drifting past 3 sigma: {bad or 'none'}")


def level4(run: dict) -> None:
    from session_data import ufactory_to_diy_frame

    print("\nL4  transform")
    uf = run["uf"]
    for name, W in (("at uFactory origin", uf), ("at DIY origin (after adjoint)", ufactory_to_diy_frame(uf))):
        F, T = W[:, :3], W[:, 3:]
        keep = np.linalg.norm(F, axis=1) > 2.0
        if keep.sum() < 50:
            print(f"  {name}: too little loading to measure")
            continue
        C = np.linalg.lstsq(np.hstack([F[keep], np.ones((keep.sum(), 1))]), T[keep], rcond=None)[0][:3].T
        anti, sym = (C - C.T) / 2, (C + C.T) / 2
        print(f"  {name:<32} coupling |C| {np.linalg.norm(C):.4f}"
              f"   lever {np.linalg.norm(anti) * 1000:5.1f} mm"
              f"   non-geometric {np.linalg.norm(sym) * 1000:5.1f} mm")


def level5(run: dict, matrix: Path | None) -> None:
    print("\nL5  fit")
    if matrix is None or not matrix.exists():
        print("  no calibration given (--matrix), skipping")
        return
    from fit_calibration import end_to_end, print_cross_talk
    from session_data import ufactory_to_diy_frame

    d = np.load(matrix, allow_pickle=True)
    print_cross_talk(end_to_end(d["A"], run["ch"], ufactory_to_diy_frame(run["uf"])))


def report(args: argparse.Namespace) -> None:
    run = load_run(args.run)
    segs = segments(run)
    rest = segs.get("rest", np.zeros(len(run["t"]), bool))
    print(f"{args.run.name}: {len(run['t'])} samples, {run['t'][-1] - run['t'][0]:.1f} s")
    print("  trials: " + ", ".join(f"{k}({v.sum()})" for k, v in segs.items()))
    level0(run, rest)
    level1(run, segs, rest)
    level2(run, segs)
    level3(run, rest)
    level4(run)
    level5(run, args.matrix)


# --------------------------------------------------------------------------- entry
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("record", help="record a labelled diagnostic run")
    r.add_argument("--ip", default="192.168.1.226")
    r.add_argument("--port", required=True)
    r.add_argument("--baud", type=int, default=115200)
    r.add_argument("--rate", type=float, default=50.0)
    r.add_argument("--filter-window", type=int, default=3)
    r.add_argument("--out", type=Path, default=None)
    r.set_defaults(func=record)

    a = sub.add_parser("report", help="walk the pipeline level by level")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--matrix", type=Path, default=Path("calibration_ridge.npz"))
    a.set_defaults(func=report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
