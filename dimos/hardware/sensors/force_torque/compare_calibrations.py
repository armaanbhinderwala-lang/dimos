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

"""Live side-by-side of every wrench estimate, against the uFactory as ground truth.

Five streams on one screen while you push the sensor by hand:

    1  uFactory raw            reference, before gravity compensation
    2  uFactory calibrated     reference, gravity removed -- THE GROUND TRUTH
    3  homemade raw            16 Hall channels, ADC counts
    4  homemade OLD cal        the originally shipped ft_calibration.json
    5  homemade NEW cal        the baseline we fitted (calibration_baseline.npz)

Rows 4 and 5 are scored live against row 2, so you can see directly whether the new
calibration is an improvement rather than inferring it from offline numbers.

Frames: the new calibration predicts a wrench in the DIY sensor's own frame, so it is
transformed back to the uFactory frame before display -- otherwise the comparison would
be measuring our own coordinate change rather than calibration quality. The old
calibration's frame convention is undocumented, so it is shown as-is and flagged.

The arm does NOT move from this window. Put it in position-hold and push by hand,
exactly as during collection.

Keys:  SPACE/ENTER re-zero the homemade sensor   R reset the error statistics   ESC quit
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from ft_calibration_collector import CHANNELS, HomemadeReader, UFactoryReader
from session_data import AXES, diy_to_ufactory_frame, ufactory_to_diy_frame

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

UF_RATED = (150.0, 150.0, 200.0, 4.0, 4.0, 4.0)


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a calibration from either .npz (our baseline) or .json (driver format)."""
    if not path.exists():
        raise SystemExit(
            f"\ncalibration file not found: {path.resolve()}\n\n"
            "If this is the baseline, generate it first from the repo root:\n"
            "    python3 dimos/hardware/sensors/force_torque/fit_calibration.py --data-dir .\n"
        )
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        return np.asarray(d["A"], float), np.asarray(d["b"], float)
    payload = json.loads(path.read_text())
    matrix = np.asarray(payload["calibration_matrix"], float)
    bias = np.asarray(payload.get("bias_vector") or np.zeros(6), float)
    return matrix, bias


class ErrorTracker:
    """Running error of an estimate against the reference, over a sliding window."""

    def __init__(self, window: int = 400):
        self.pred = deque(maxlen=window)
        self.truth = deque(maxlen=window)

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        self.pred.append(pred.copy())
        self.truth.append(truth.copy())

    def reset(self) -> None:
        self.pred.clear()
        self.truth.clear()

    def rmse(self) -> np.ndarray:
        if len(self.pred) < 10:
            return np.full(6, np.nan)
        return np.sqrt(((np.array(self.pred) - np.array(self.truth)) ** 2).mean(axis=0))


def draw_bar_row(screen, font, y, label, values, colour):
    """One 6-axis row, each axis as a percentage of the reference's rated range."""
    screen.blit(font.render(label, True, (215, 215, 215)), (16, y))
    x0, width = 250, 62
    for i, v in enumerate(values):
        pct = min(abs(v) / UF_RATED[i], 1.0)
        x = x0 + i * (width + 8)
        pygame.draw.rect(screen, (58, 58, 64), (x, y, width, 20))
        pygame.draw.rect(screen, colour, (x, y, int(width * pct), 20))
        screen.blit(font.render(f"{v:6.1f}", True, (205, 205, 205)), (x + 2, y + 22))
    return y + 52


def run(uf: UFactoryReader, hm: HomemadeReader, cals: list[dict],
        log_path: Path | None, report_every: float = 60.0) -> None:
    """cals: [{name, A, b, frame, colour}] -- one bar row and one error row each."""
    if pygame is None:
        raise ImportError("pygame is required: pip install pygame")

    pygame.init()
    height = 420 + 34 * len(cals) * 2
    screen = pygame.display.set_mode((1060, height))
    pygame.display.set_caption("Calibration comparison")
    font = pygame.font.Font(None, 26)
    small = pygame.font.Font(None, 21)
    clock = pygame.time.Clock()

    zero = np.zeros(CHANNELS)
    zeroed = False
    for c in cals:
        c["err"] = ErrorTracker()          # scored in the uFactory frame, against truth
        c["err_diy"] = ErrorTracker()      # the honest torque number
    # Stream to disk rather than buffering: a killed window or a crash used to lose the
    # whole recording, which is exactly when you most want it.
    writer = fh = None
    if log_path is not None:
        import csv as _csv
        fh = log_path.open("w", newline="")
        writer = _csv.writer(fh)
        writer.writerow(["ts"] + [f"uf_raw_{a}" for a in AXES] + [f"uf_cal_{a}" for a in AXES]
                        + [f"ch{i}" for i in range(1, CHANNELS + 1)]
                        + [f"{c['name'].replace(' ', '_')}_{a}" for c in cals for a in AXES])
        print(f"recording to {log_path}")
    n_logged = 0
    running = True
    last_report = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    zero = hm.latest_channels.copy(); zeroed = True
                    print(f"[zeroed] baseline = {np.round(zero[:4], 1)} ...")
                    for c in cals: c["err"].reset(); c["err_diy"].reset()
                elif event.key == pygame.K_r:
                    for c in cals: c["err"].reset(); c["err_diy"].reset()

        if not zeroed and np.any(hm.latest_channels):
            zero = hm.latest_channels.copy(); zeroed = True
            print(f"[auto-zeroed on first reading] baseline = {np.round(zero[:4], 1)} ...")
            print("  press SPACE while unloaded to re-zero if the sensor was loaded at startup")

        channels = hm.latest_channels - zero
        truth = uf.latest_ext
        truth_diy = ufactory_to_diy_frame(truth[None, :])[0]

        for c in cals:
            pred = channels @ c["A"].T + c["b"]
            # A calibration fitted in the DIY frame predicts there; rotate it into the
            # uFactory frame so the bars compare like with like.
            if c["frame"] == "diy":
                c["pred_diy"] = pred
                c["pred"] = diy_to_ufactory_frame(pred[None, :])[0]
            else:
                c["pred"] = pred
                c["pred_diy"] = ufactory_to_diy_frame(pred[None, :])[0]
            c["err"].add(c["pred"], truth)
            c["err_diy"].add(truth_diy, c["pred_diy"])

        if writer is not None:
            writer.writerow([time.time(), *uf.latest_raw, *truth, *hm.latest_channels,
                             *[v for c in cals for v in c["pred"]]])
            n_logged += 1
            if n_logged % 200 == 0:
                fh.flush()

        screen.fill((24, 24, 28))
        y = 16
        screen.blit(font.render("Calibration comparison", True, (255, 255, 255)), (16, y)); y += 30
        screen.blit(small.render(
            "Arm does NOT move here -- position-hold and push by hand.  "
            "SPACE re-zero   R reset stats   ESC quit", True, (255, 190, 90)), (16, y)); y += 24
        if not zeroed:
            screen.blit(font.render("NOT ZEROED -- readings meaningless. Press SPACE unloaded.",
                                    True, (235, 90, 90)), (16, y))
        else:
            screen.blit(small.render("zeroed OK", True, (120, 200, 140)), (16, y))
        if writer is not None:
            screen.blit(small.render(f"REC  {n_logged:,} frames -> {log_path.name}",
                                     True, (235, 120, 120)), (330, y))
        y += 30

        screen.blit(small.render("        " + "".join(f"{a:>10}" for a in AXES),
                                 True, (150, 150, 150)), (300, y)); y += 22
        y = draw_bar_row(screen, small, y, "uFactory raw", uf.latest_raw, (110, 110, 190))
        y = draw_bar_row(screen, small, y, "uFactory calibrated  <- truth", truth, (90, 200, 120))
        for c in cals:
            y = draw_bar_row(screen, small, y, c["name"], c["pred"], c["colour"])

        y += 6
        screen.blit(font.render("Live error vs truth (RMSE, sliding window)", True, (230, 230, 230)), (16, y)); y += 28
        best = None
        for c in cals:
            e = c["err"].rmse()
            screen.blit(small.render(c["name"], True, c["colour"]), (16, y))
            for i in range(6):
                txt = "--" if np.isnan(e[i]) else f"{e[i]:6.2f}"
                screen.blit(small.render(txt, True, c["colour"]), (300 + i * 62, y))
            if not np.isnan(e).any():
                score = float(np.nanmean(e))
                if best is None or score < best[0]:
                    best = (score, c["name"])
            y += 24
        if best is not None:
            screen.blit(font.render(f"lowest mean RMSE: {best[1]}", True, (90, 200, 120)), (16, y))
        y += 30

        screen.blit(small.render("DIY frame -- the honest torque number", True, (170, 170, 170)), (16, y)); y += 24
        for c in cals:
            e = c["err_diy"].rmse()
            screen.blit(small.render(c["name"], True, c["colour"]), (16, y))
            for i in range(6):
                txt = "--" if np.isnan(e[i]) else f"{e[i]:6.2f}"
                screen.blit(small.render(txt, True, c["colour"]), (300 + i * 62, y))
            y += 24

        pygame.display.flip()
        clock.tick(30)

        now = time.time()
        if report_every > 0 and now - last_report >= report_every:
            last_report = now
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            print(f"  {'':<22}" + "".join(f"{a:>9}" for a in AXES))
            print(f"  {'uFactory truth':<22}" + "".join(f"{v:9.2f}" for v in truth))
            for c in cals:
                print(f"  {c['name']:<22}" + "".join(f"{v:9.2f}" for v in c["pred"]))
            print("  RMSE (uFactory frame)")
            for c in cals:
                e = c["err"].rmse()
                if not np.isnan(e).any():
                    print(f"    {c['name']:<20}" + "".join(f"{v:9.2f}" for v in e))

    pygame.quit()
    if fh is not None:
        fh.close()
        print(f"wrote {log_path} ({n_logged:,} rows)")


def compare_offline(data_dir: Path, cals: list[dict]) -> None:
    """Score every calibration against the uFactory on already-recorded sessions.

    Needs no hardware, and uses far more data than you could push by hand -- so this is
    the definitive answer to "is the new calibration better", with the live view being
    the sanity check that it also behaves correctly in real time.
    """
    import session_data as sd

    sessions = sd.load_all(data_dir, frame="diy")
    channels = np.vstack([s.channels for s in sessions])
    truth = sd.diy_to_ufactory_frame(np.vstack([s.wrench for s in sessions]))
    print(f"\n{len(channels):,} samples from {len(sessions)} sessions, scored against the uFactory\n")
    print(f"  {'calibration':<24}" + "".join(f"{a:>9}" for a in AXES) + f"{'mean':>9}")
    best = None
    for c in cals:
        pred = channels @ c["A"].T + c["b"]
        if c["frame"] == "diy":
            pred = sd.diy_to_ufactory_frame(pred)
        r = np.sqrt(((pred - truth) ** 2).mean(axis=0))
        print(f"  {c['name']:<24}" + "".join(f"{v:9.2f}" for v in r) + f"{r.mean():9.2f}")
        if best is None or r.mean() < best[0]:
            best = (r.mean(), c["name"])
    print(f"\n  lowest mean RMSE: {best[1]}")
    print("  (N for force rows, N*m for torque -- the mean mixes units, use it only to rank)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xarm-ip", help="required unless --offline")
    p.add_argument("--offline", action="store_true",
                   help="score both calibrations on recorded sessions instead of live hardware")
    p.add_argument("--data-dir", type=Path, default=Path("."), help="where the session CSVs live (offline mode)")
    p.add_argument("--homemade-port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--cal", action="append", default=[], metavar="LABEL=PATH",
                   help="a calibration to show, repeatable. Frame is read from the file "
                        "(defaults to diy for ours). e.g. --cal 'new ridge=calibration_newrun_ridge.npz'")
    p.add_argument("--log", type=Path, default=None, help="also record every frame to CSV")
    p.add_argument("--report-every", type=float, default=60.0,
                   help="seconds between terminal summaries (0 disables)")
    args = p.parse_args()

    if not args.cal:
        raise SystemExit("pass at least one --cal 'LABEL=path.npz'")
    palette = [(200, 150, 80), (110, 190, 220), (220, 130, 190), (150, 200, 120),
               (230, 200, 110), (170, 150, 230)]
    cals = []
    for i, spec in enumerate(args.cal):
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--cal needs LABEL=PATH, got {spec!r}")
        A, b = load_matrix(Path(path))
        frame = "diy"
        if Path(path).suffix == ".npz":
            d = np.load(path, allow_pickle=True)
            if "frame" in d:
                frame = str(d["frame"])
        cals.append({"name": label, "A": A, "b": b, "frame": frame,
                     "colour": palette[i % len(palette)]})
        print(f"  {label:<24} {path}   A{A.shape}  frame={frame}")

    if args.offline:
        compare_offline(args.data_dir, cals)
        return
    if not args.xarm_ip:
        raise SystemExit("--xarm-ip is required for the live view (or pass --offline)")

    print("\n" + "=" * 62)
    print("  LIVE CALIBRATION COMPARISON  (not the data collector)")
    print(f"  {2 + len(cals)} bar rows: uFactory raw, uFactory calibrated, then:")
    for c in cals:
        print(f"    - {c['name']}")
    print("=" * 62)

    uf = UFactoryReader(args.xarm_ip)
    hm = HomemadeReader(args.homemade_port, args.baud)
    uf.start(); hm.start()
    print("Press SPACE once, unloaded, to zero the homemade sensor before comparing.")
    try:
        run(uf, hm, cals, args.log, args.report_every)
    finally:
        uf.stop(); hm.stop()


if __name__ == "__main__":
    main()
