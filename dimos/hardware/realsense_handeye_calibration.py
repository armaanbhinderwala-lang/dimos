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
Step 4: eye-in-hand hand-eye calibration for the RealSense mounted on the arm.

Solves for camera_extrinsic (tool frame -> camera optical frame), the
placeholder currently sitting in normal_move_test.py and handle_grab_skill.py.

Setup: a checkerboard fixed in the WORLD (taped to a table, not on the robot).
The camera moves with the tool; at each of N poses you jog it to, this script
reads the arm's tool pose (base->gripper, from get_position -- same rotation
convention assumed by FT_CALIBRATION_MATH.md's expected_wrench, still
unverified independent of this script) and the checkerboard's pose in the
camera image (target->camera, via solvePnP). cv2.calibrateHandEye solves the
classic AX=XB problem for the one unknown constant transform across all
poses: gripper->camera.

Usage:
  python3 dimos/hardware/realsense_handeye_calibration.py \\
      --xarm 192.168.1.210 --checkerboard-cols 9 --checkerboard-rows 6 \\
      --square-size 0.025 --out camera_extrinsic.json

Jog the arm by hand between captures (teach pendant / hand-guide / xArm
Studio) -- this script never commands motion, same as the FT ground-truth
tools. Vary orientation, not just position, across captures -- translation-
only poses make the AX=XB system poorly conditioned and calibrateHandEye's
resulting rotation frequently comes out wrong even when the fit reports no
error, so 15-20 poses with real orientation spread beats more poses in a
narrow orientation range.
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from realsense_camera import RealsenseCamera


def get_tool_pose(arm):
    """(R_gripper2base 3x3, t_gripper2base 3,) from the arm's current TCP pose."""
    code, pose = arm.get_position(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_position() failed with code {code}")
    x, y, z, roll, pitch, yaw = pose
    R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    t = np.array([x, y, z]) / 1000.0
    return R, t


def find_target_pose(gray, cols, rows, square_size, K, dist):
    """(R_target2cam 3x3, t_target2cam 3,) or None if the board wasn't found."""
    found, corners = cv2.findChessboardCorners(gray, (cols, rows))
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def check_residual(R_g2b, t_g2b, R_t2c, t_t2c, R_cam2gripper, t_cam2gripper):
    """Base->target should be ~constant across poses if the calibration is good."""
    targets_in_base = []
    for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        R_base_target = Rg @ R_cam2gripper @ Rt
        t_base_target = Rg @ (R_cam2gripper @ tt + t_cam2gripper) + tg
        targets_in_base.append(t_base_target)
    targets_in_base = np.array(targets_in_base)
    return targets_in_base.mean(axis=0), targets_in_base.std(axis=0)


def main():
    parser = argparse.ArgumentParser(description="RealSense eye-in-hand calibration")
    parser.add_argument("--xarm", required=True)
    parser.add_argument("--checkerboard-cols", type=int, default=9, help="Internal corners, columns")
    parser.add_argument("--checkerboard-rows", type=int, default=6, help="Internal corners, rows")
    parser.add_argument("--square-size", type=float, required=True, help="Checkerboard square size, meters")
    parser.add_argument("--min-poses", type=int, default=15)
    parser.add_argument("--out", default="camera_extrinsic.json")
    args = parser.parse_args()

    from xarm.wrapper import XArmAPI

    print(f"Connecting to xArm at {args.xarm}...")
    arm = XArmAPI(args.xarm, do_not_open=False, is_radian=True)
    arm.clean_error()
    arm.clean_warn()

    print("Opening RealSense...")
    camera = RealsenseCamera()
    camera.open()
    K = np.array(
        [
            [camera.intrinsics.fx, 0, camera.intrinsics.ppx],
            [0, camera.intrinsics.fy, camera.intrinsics.ppy],
            [0, 0, 1],
        ]
    )
    dist = np.array(camera.intrinsics.coeffs)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []

    print("\nJog the arm so the checkerboard is fully visible, then press Enter to capture.")
    print(f"Need at least {args.min_poses} poses, with real orientation variety. 'q' to finish.\n")

    pose_id = 0
    try:
        while True:
            user_in = input(f"[Pose {pose_id + 1}] Enter to capture, 'q' to finish: ")
            if user_in.strip().lower() == "q":
                break

            rgb, _, _ = camera.capture(warmup_frames=3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            target_pose = find_target_pose(
                gray, args.checkerboard_cols, args.checkerboard_rows, args.square_size, K, dist
            )
            if target_pose is None:
                print("  Checkerboard not found in frame, skipping. Adjust pose and retry.")
                continue

            Rt, tt = target_pose
            Rg, tg = get_tool_pose(arm)

            R_g2b.append(Rg)
            t_g2b.append(tg)
            R_t2c.append(Rt)
            t_t2c.append(tt)
            print(f"  Captured. Target distance from camera: {np.linalg.norm(tt):.3f} m")
            pose_id += 1
    finally:
        camera.close()

    if len(R_g2b) < args.min_poses:
        print(f"\nOnly {len(R_g2b)} poses captured, wanted >= {args.min_poses}. Not solving.")
        return 1

    print(f"\nSolving hand-eye calibration from {len(R_g2b)} poses...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    t_cam2gripper = t_cam2gripper.flatten()

    mean_target, std_target = check_residual(R_g2b, t_g2b, R_t2c, t_t2c, R_cam2gripper, t_cam2gripper)
    print(f"\nResidual check: base->target position should be constant across poses.")
    print(f"  mean: {mean_target}")
    print(f"  std:  {std_target}  (millimeters: {std_target * 1000})")
    print("  Large std relative to your setup's expected precision means bad poses (not enough")
    print("  orientation variety, motion blur, or a loose checkerboard) -- consider recapturing.")

    quat = Rotation.from_matrix(R_cam2gripper).as_quat()  # [x, y, z, w]
    rpy = Rotation.from_matrix(R_cam2gripper).as_euler("xyz")

    result = {
        "transform_id": "TOOL_TO_COLOR_OPT",
        "parent_frame": "link_openft (or link_eef)",
        "child_frame": "camera_color_optical_frame",
        "direction": "T_parent_child",
        "units": {"translation": "meters", "angles": "radians"},
        "translation_m": {"x": t_cam2gripper[0], "y": t_cam2gripper[1], "z": t_cam2gripper[2]},
        "rotation_quat_xyzw": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
        "rotation_rpy_rad": {
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
            "convention": "R = Rz(yaw) * Ry(pitch) * Rx(roll)",
        },
        "residual_std_m": list(std_target),
        "num_poses": len(R_g2b),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")

    print("\nPaste this into camera_extrinsic in normal_move_test.py / handle_grab_skill.py:")
    print(
        f"    self.camera_extrinsic = RigidTransform(\n"
        f"        RotationMatrix(RollPitchYaw({rpy[0]!r}, {rpy[1]!r}, {rpy[2]!r})),\n"
        f"        [{t_cam2gripper[0]!r}, {t_cam2gripper[1]!r}, {t_cam2gripper[2]!r}],\n"
        f"    )"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
