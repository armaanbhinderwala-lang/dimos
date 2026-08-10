#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
RealSense Camera Module for Dimos, replacing ZEDModule (dimos/hardware/zed_camera.py)
as the camera source for HandleGrabModule.

Ported from dimensionalOS/dimos main's dimos/hardware/sensors/camera/realsense/camera.py,
which doesn't import as-is on this branch (it depends on dimos.core.transport_factory,
dimos.hardware.sensors.*, dimos.spec.perception -- none of which exist here). Rebuilt
against this branch's actual Module/Out/rpc API instead, keeping only what
HandleGrabModule actually consumes (color_image, depth_image, camera_info) -- no
pose/tf/pointcloud, since RealSense has no equivalent to the ZED's onboard VIO tracking
and HandleGrabModule never subscribed to those anyway.

One contract detail that matters and isn't obvious from main's version: HandleGrabModule
expects depth_image as float32 METERS (ImageFormat.DEPTH), matching what ZEDModule
publishes -- not raw uint16 (ImageFormat.DEPTH16) the way main's module does for its own
(different) consumers. Publishing DEPTH16 here would silently make HandleGrabModule
misread every depth value by ~1000x. Scaled to meters before publishing for that reason.
"""

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from dimos.core import Module, Out, rpc
from dimos.msgs.sensor_msgs import CameraInfo, Image, ImageFormat
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class RealsenseModule(Module):
    """Publishes RealSense color/depth/camera_info on the same LCM contract as ZEDModule."""

    color_image: Out[Image] = None
    depth_image: Out[Image] = None
    camera_info: Out[CameraInfo] = None

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        frame_id: str = "camera_color_optical_frame",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_id = frame_id
        self.verbose = verbose

        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self._camera_info: Optional[CameraInfo] = None

        self.running = False
        self._thread = None
        self.frame_count = 0
        self.error_count = 0

    def _build_camera_info(self, profile):
        import pyrealsense2 as rs

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        distortion_model = {
            rs.distortion.none: "",
            rs.distortion.modified_brown_conrady: "plumb_bob",
            rs.distortion.inverse_brown_conrady: "plumb_bob",
            rs.distortion.ftheta: "equidistant",
            rs.distortion.brown_conrady: "plumb_bob",
            rs.distortion.kannala_brandt4: "equidistant",
        }.get(intr.model, "")

        self._camera_info = CameraInfo(
            height=intr.height,
            width=intr.width,
            distortion_model=distortion_model,
            D=list(intr.coeffs) if intr.coeffs else [],
            K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            frame_id=self.frame_id,
        )

    def _capture_loop(self):
        logger.info("RealSense capture loop started")
        while self.running:
            try:
                frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                ts = time.time()

                color_data = np.asanyarray(color_frame.get_data())  # requested as rgb8 in start()
                self.color_image.publish(
                    Image(data=color_data, format=ImageFormat.RGB, frame_id=self.frame_id, ts=ts)
                )

                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                self.depth_image.publish(
                    Image(data=depth_m, format=ImageFormat.DEPTH, frame_id=self.frame_id, ts=ts)
                )

                if self._camera_info is not None:
                    self._camera_info.ts = ts
                    self.camera_info.publish(self._camera_info)

                self.frame_count += 1
            except Exception as e:
                # Always surface the first one - a silent error count reads as
                # "camera alive, no frames" and hides missing transports entirely.
                if self.verbose or self.error_count == 0:
                    logger.warning(f"Capture error: {e}")
                self.error_count += 1
        logger.info(f"RealSense capture loop stopped after {self.frame_count} frames")

    @rpc
    def start(self) -> bool:
        if self.running:
            logger.warning("RealSense module already running")
            return True

        try:
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            profile = self.pipeline.start(config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self._build_camera_info(profile)
        except Exception as e:
            logger.error(f"Failed to start RealSense pipeline: {e}")
            return False

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        time.sleep(0.1)
        if not self._thread.is_alive():
            logger.error("RealSense capture thread failed to start")
            self.running = False
            return False

        logger.info(f"RealSense module started ({self.width}x{self.height} @ {self.fps}fps)")
        return True

    @rpc
    def stop(self):
        if not self.running:
            return
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        logger.info(f"RealSense module stopped. Frames={self.frame_count}, Errors={self.error_count}")

    @rpc
    def get_stats(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "running": self.running,
            "depth_scale": self.depth_scale,
        }


if __name__ == "__main__":
    import argparse

    from dimos.core import LCMTransport, start

    parser = argparse.ArgumentParser(description="RealSense Camera Module")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--lcm-color-channel", default="/camera/color_image")
    parser.add_argument("--lcm-depth-channel", default="/camera/depth_image")
    parser.add_argument("--lcm-info-channel", default="/camera/camera_info")
    args = parser.parse_args()

    dimos = start(1)
    cam = dimos.deploy(
        RealsenseModule, width=args.width, height=args.height, fps=args.fps, verbose=args.verbose
    )

    # Without transports every publish raises and the capture loop just counts errors,
    # so the module looks alive while emitting nothing. Same channels handle_grab_test uses.
    cam.color_image.transport = LCMTransport(args.lcm_color_channel, Image)
    cam.depth_image.transport = LCMTransport(args.lcm_depth_channel, Image)
    cam.camera_info.transport = LCMTransport(args.lcm_info_channel, CameraInfo)

    if not cam.start():
        dimos.shutdown()
        raise SystemExit(1)

    logger.info(f"Publishing on {args.lcm_color_channel} / {args.lcm_depth_channel}")
    logger.info("Press Ctrl+C to stop...")

    # The dask cluster dies with this process, so hold it open until interrupted.
    try:
        while True:
            time.sleep(5)
            stats = cam.get_stats()
            logger.info(f"Frames={stats['frame_count']}, Errors={stats['error_count']}")
    except KeyboardInterrupt:
        cam.stop()
        time.sleep(0.5)
        dimos.shutdown()
