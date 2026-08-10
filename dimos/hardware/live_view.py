#!/usr/bin/env python3
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

"""Plain live view of a camera image topic.

Subscribes to an Image topic on LCM and shows the frames in an OpenCV window -
no 3D scene, no entity tree, just the picture. Channels default to the ones
handle_grab_test.py publishes on. Start a camera first::

    python handle_grab_test.py --xarm 192.168.1.210   # then, in another terminal:
    python live_view.py

Press q or Esc to quit.
"""

import argparse
import time

import cv2
import lcm
import numpy as np

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def to_bgr(img: Image) -> np.ndarray:
    """Convert a dimos Image to something cv2 can display."""
    data = np.asarray(img.data)

    if img.format in (ImageFormat.DEPTH, ImageFormat.DEPTH16):
        # Depth arrives as uint16 millimetres - colourise it so it's readable
        depth = data.astype(np.float32)
        valid = depth[depth > 0]
        if valid.size:
            lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
            depth = np.clip((depth - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/camera/color_image#sensor_msgs.Image",
        help="LCM topic to view (default: /camera/color_image#sensor_msgs.Image)",
    )
    parser.add_argument("--depth", action="store_true", help="Shorthand for the depth topic")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Write frames to this path instead of opening a window (headless check)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the first frame (default: 15)",
    )
    args = parser.parse_args()

    topic = "/camera/depth_image#sensor_msgs.Image" if args.depth else args.topic

    latest: dict[str, Image] = {}
    count = [0]

    def on_msg(channel: str, data: bytes) -> None:
        latest["img"] = Image.lcm_decode(data)
        count[0] += 1

    lc = lcm.LCM()
    lc.subscribe(topic, on_msg)
    logger.info(f"Waiting for frames on {topic} ...")

    window = f"dimos live view - {topic.split('#')[0]}"
    deadline = time.time() + args.timeout
    last_report = time.time()
    reported = 0

    while True:
        lc.handle_timeout(200)

        img = latest.pop("img", None)
        if img is None:
            if count[0] == 0 and time.time() > deadline:
                logger.error(f"No frames on {topic} after {args.timeout:.0f}s")
                logger.error("Is a camera module running? Try: python handle_grab_test.py --xarm <ip>")
                return
            continue

        frame = to_bgr(img)

        if args.save:
            cv2.imwrite(args.save, frame)
            logger.info(f"Wrote {args.save} ({frame.shape[1]}x{frame.shape[0]})")
            return

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

        now = time.time()
        if now - last_report >= 5.0:
            logger.info(f"{(count[0] - reported) / (now - last_report):.1f} fps")
            reported, last_report = count[0], now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
