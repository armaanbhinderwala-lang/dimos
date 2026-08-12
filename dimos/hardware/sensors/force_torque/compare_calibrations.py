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


def run(uf: UFactoryReader, hm: HomemadeReader, new_cal, old_cal,
        log_path: Path | None, report_every: float = 60.0) -> None:
    if pygame is None:
        raise ImportError("pygame is required: pip install pygame")

    A_new, b_new = new_cal
    A_old, b_old = old_cal if old_cal else (None, None)

    pygame.init()
    screen = pygame.display.set_mode((980, 700))
    pygame.display.set_caption("Calibration comparison")
    font = pygame.font.Font(None, 26)
    small = pygame.font.Font(None, 21)
    clock = pygame.time.Clock()

    zero = np.zeros(CHANNELS)
    err_new, err_old = ErrorTracker(), ErrorTracker()
    err_true = ErrorTracker()   # same prediction, scored in the DIY frame (the honest one)
    rows: list[list[float]] = []
    running = True
    last_report = 0.0          # periodic terminal summary, for debugging without watching the window

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    zero = hm.latest_channels.copy()
                    err_new.reset(); err_old.reset(); err_true.reset()
                elif event.key == pygame.K_r:
                    err_new.reset(); err_old.reset(); err_true.reset()

        channels = hm.latest_channels - zero
        truth = uf.latest_ext

        # New calibration predicts in the DIY frame -> bring it back to the uFactory
        # frame so the comparison against the reference is like-for-like.
        pred_new_diy = (channels @ A_new.T + b_new)[None, :]
        pred_new = diy_to_ufactory_frame(pred_new_diy)[0]
        pred_old = channels @ A_old.T + b_old if A_old is not None else np.zeros(6)

        err_new.add(pred_new, truth)
        if A_old is not None:
            err_old.add(pred_old, truth)
        # Also track error in the DIY frame. Scoring in the uFactory frame FLATTERS the
        # torque axes, because moving the frame re-introduces the d x F term -- which the
        # model reproduces well simply by predicting force. Measured offline: mean R2 0.44
        # in the DIY frame vs 0.55 in the uFactory frame on the same fit. The DIY-frame
        # number is the sensor's true torque ability.
        err_true.add(ufactory_to_diy_frame(truth[None, :])[0], pred_new_diy[0])
        if log_path is not None:
            rows.append([time.time(), *uf.latest_raw, *truth, *hm.latest_channels, *pred_old, *pred_new])

        screen.fill((24, 24, 28))
        y = 16
        screen.blit(font.render("Calibration comparison", True, (255, 255, 255)), (16, y)); y += 30
        screen.blit(small.render(
            "Arm does NOT move here -- position-hold and push by hand.  "
            "SPACE re-zero   R reset stats   ESC quit", True, (255, 190, 90)), (16, y)); y += 34

        screen.blit(small.render("        " + "".join(f"{a:>10}" for a in AXES), True, (150, 150, 150)), (238, y)); y += 22
        y = draw_bar_row(screen, small, y, "uFactory raw", uf.latest_raw, (110, 110, 190))
        y = draw_bar_row(screen, small, y, "uFactory calibrated  <- truth", truth, (90, 200, 120))
        y = draw_bar_row(screen, small, y, "homemade OLD cal", pred_old, (200, 150, 80))
        y = draw_bar_row(screen, small, y, "homemade NEW cal", pred_new, (110, 190, 220))

        y += 6
        rn, ro = err_new.rmse(), err_old.rmse()
        screen.blit(font.render("Live error vs truth (RMSE, sliding window)", True, (230, 230, 230)), (16, y)); y += 30
        for label, e, colour in (("OLD cal", ro, (200, 150, 80)), ("NEW cal", rn, (110, 190, 220))):
            screen.blit(small.render(label, True, colour), (16, y))
            for i in range(6):
                txt = "--" if np.isnan(e[i]) else f"{e[i]:6.2f}"
                screen.blit(small.render(txt, True, colour), (250 + i * 70, y))
            y += 24
        if not np.isnan(rn).any() and not np.isnan(ro).any():
            better = int((rn < ro).sum())
            msg = f"NEW better on {better}/6 axes"
            screen.blit(font.render(msg, True, (90, 200, 120) if better >= 4 else (200, 150, 80)), (16, y))
        y += 30
        rt = err_true.rmse()
        screen.blit(small.render("NEW (DIY frame)", True, (170, 170, 170)), (16, y))
        for i in range(6):
            txt = "--" if np.isnan(rt[i]) else f"{rt[i]:6.2f}"
            screen.blit(small.render(txt, True, (170, 170, 170)), (250 + i * 70, y))
        y += 22
        screen.blit(small.render(
            "^ the honest torque number -- the uFactory-frame rows above flatter torque",
            True, (130, 130, 130)), (16, y))
        y += 30

        screen.blit(small.render("homemade raw channels (zeroed)", True, (170, 170, 170)), (16, y)); y += 22
        for r in range(4):
            line = "  ".join(f"{i+1:2d}:{channels[i]:7.1f}" for i in range(r * 4, r * 4 + 4))
            screen.blit(small.render(line, True, (185, 185, 185)), (16, y)); y += 21

        # Print a summary every `report_every` seconds so a run leaves a readable trace
        # in the terminal -- useful when the window is on another machine, or afterwards.
        now = time.time()
        if report_every and now - last_report >= report_every:
            last_report = now
            rn_, ro_, rt_ = err_new.rmse(), err_old.rmse(), err_true.rmse()
            print(f"\n[{time.strftime('%H:%M:%S')}]  |F| applied = {np.linalg.norm(truth[:3]):.1f} N")
            print("            " + "".join(f"{a:>9}" for a in AXES))
            print("  truth     " + "".join(f"{v:9.2f}" for v in truth))
            print("  OLD cal   " + "".join(f"{v:9.2f}" for v in pred_old))
            print("  NEW cal   " + "".join(f"{v:9.2f}" for v in pred_new))
            if not np.isnan(rn_).any():
                print("  RMSE OLD  " + "".join(f"{v:9.2f}" for v in ro_))
                print("  RMSE NEW  " + "".join(f"{v:9.2f}" for v in rn_))
                print(f"  -> NEW better on {int((rn_ < ro_).sum())}/6 axes")
                print("  RMSE NEW (DIY frame, honest torque)" + "".join(f"{v:8.2f}" for v in rt_))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    if log_path is not None and rows:
        import csv
        cols = (["ts"] + [f"uf_raw_{a}" for a in AXES] + [f"uf_cal_{a}" for a in AXES]
                + [f"ch{i+1}" for i in range(CHANNELS)]
                + [f"old_{a}" for a in AXES] + [f"new_{a}" for a in AXES])
        with log_path.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
        print(f"wrote {log_path} ({len(rows)} rows)")


def compare_offline(data_dir: Path, new_cal, old_cal) -> None:
    """Score both calibrations against the uFactory on already-recorded sessions.

    Needs no hardware, and uses far more data than you could push by hand -- so this is
    the definitive answer to "is the new calibration better", with the live view being
    the sanity check that it also behaves correctly in real time.
    """
    import session_data as sd

    A_new, b_new = new_cal
    sessions = sd.load_all(data_dir, frame="diy")
    channels = np.vstack([s.channels for s in sessions])
    truth = sd.diy_to_ufactory_frame(np.vstack([s.wrench for s in sessions]))
    pred_new = sd.diy_to_ufactory_frame(channels @ A_new.T + b_new)

    def rmse(pred):
        return np.sqrt(((pred - truth) ** 2).mean(axis=0))

    r_new = rmse(pred_new)
    print(f"\n{len(channels):,} samples from {len(sessions)} sessions, scored against the uFactory\n")
    if old_cal is None:
        print(f"{'axis':>5} {'NEW RMSE':>12}")
        for i, a in enumerate(AXES):
            print(f"{a:>5} {r_new[i]:10.2f} {'N' if i < 3 else 'N*m'}")
        return

    A_old, b_old = old_cal
    r_old = rmse(channels @ A_old.T + b_old)
    print(f"{'axis':>5} {'OLD RMSE':>12} {'NEW RMSE':>12} {'improvement':>13}")
    for i, a in enumerate(AXES):
        unit = "N" if i < 3 else "N*m"
        gain = (r_old[i] - r_new[i]) / r_old[i] * 100
        print(f"{a:>5} {r_old[i]:10.2f} {unit:<3} {r_new[i]:10.2f} {unit:<3} {gain:+11.0f}%")
    print(f"\nNEW better on {int((r_new < r_old).sum())}/6 axes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xarm-ip", help="required unless --offline")
    p.add_argument("--offline", action="store_true",
                   help="score both calibrations on recorded sessions instead of live hardware")
    p.add_argument("--data-dir", type=Path, default=Path("."), help="where the session CSVs live (offline mode)")
    p.add_argument("--homemade-port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--new-cal", type=Path, default=Path("calibration_baseline.npz"))
    p.add_argument("--old-cal", type=Path, default=None,
                   help="previous ft_calibration.json, for comparison (optional)")
    p.add_argument("--log", type=Path, default=None, help="also record every frame to CSV")
    p.add_argument("--report-every", type=float, default=60.0,
                   help="seconds between terminal summaries (0 disables)")
    args = p.parse_args()

    new_cal = load_matrix(args.new_cal)
    print(f"new calibration: {args.new_cal}  A{new_cal[0].shape}")
    old_cal = None
    if args.old_cal:
        old_cal = load_matrix(args.old_cal)
        print(f"old calibration: {args.old_cal}  A{old_cal[0].shape}")
        print("NOTE: the old calibration's frame convention is undocumented; it is shown")
        print("      as-is and may differ from the uFactory frame by a rotation.")

    if args.offline:
        compare_offline(args.data_dir, new_cal, old_cal)
        return
    if not args.xarm_ip:
        raise SystemExit("--xarm-ip is required for the live view (or pass --offline)")

    print("\n" + "=" * 62)
    print("  LIVE CALIBRATION COMPARISON  (not the data collector)")
    print("  You should see FOUR bar rows: uFactory raw, uFactory calibrated,")
    print("  homemade OLD cal, homemade NEW cal.")
    print("=" * 62)

    uf = UFactoryReader(args.xarm_ip)
    hm = HomemadeReader(args.homemade_port, args.baud)
    uf.start(); hm.start()
    print("Press SPACE once, unloaded, to zero the homemade sensor before comparing.")
    try:
        run(uf, hm, new_cal, old_cal, args.log, args.report_every)
    finally:
        uf.stop(); hm.stop()


if __name__ == "__main__":
    main()
