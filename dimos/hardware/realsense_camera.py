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
Lean Intel RealSense wrapper, replacing the direct pyzed.sl calls in
normal_move_test.py. Depth and points are in millimeters, RGB order --
matches the ZED convention the rest of that file expects, so no downstream
unit changes were needed.
"""

import numpy as np


class RealsenseCamera:
    """Color+depth capture, aligned, with per-pixel XYZ back-projection."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.depth_scale = None  # meters per raw depth unit
        self.intrinsics = None  # rs.intrinsics of the color stream

    def open(self):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()

    def capture(self, warmup_frames: int = 5):
        """Returns (rgb_image HxWx3 uint8, depth_mm HxW float, points_mm HxWx3 float, camera frame)."""
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to grab frame from RealSense")

        rgb_image = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale * 1000.0

        points_mm = self._backproject(depth_mm)
        return rgb_image, depth_mm, points_mm

    def _backproject(self, depth_mm: np.ndarray) -> np.ndarray:
        """Pinhole back-projection: depth_mm[v,u] -> (X,Y,Z) mm in camera frame."""
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
        h, w = depth_mm.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_mm
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x, y, z], axis=-1)
        points[z <= 0] = np.nan  # no depth return -> invalid, matches ZED's non-finite convention
        return points

    def close(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
