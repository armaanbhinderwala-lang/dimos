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
Step 0: sim-only dry run. No xArm, no RealSense, no LCM -- just Drake + the
URDF + Meshcat. Confirms the model loads, meshes resolve, and you can command
joints, before any hardware is involved.

Usage:
  python3 dimos/hardware/sim_dry_run.py --xarm7
  python3 dimos/hardware/sim_dry_run.py            # xArm6, for comparison
"""

import argparse
import os
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    StartMeshcat,
)


def main():
    parser = argparse.ArgumentParser(description="Sim-only Drake/URDF/Meshcat dry run")
    parser.add_argument("--xarm7", action="store_true", help="Load the xArm7 URDF instead of xArm6")
    parser.add_argument("--hold", type=float, default=30.0, help="Seconds to keep meshcat open")
    args = parser.parse_args()

    urdf_filename = "xarm7_openft_gripper.urdf" if args.xarm7 else "xarm6_openft_gripper.urdf"
    num_arm_joints = 7 if args.xarm7 else 6

    package_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(package_path, urdf_filename)
    if not os.path.exists(urdf_path):
        print(f"FAIL: {urdf_path} not found")
        return 1
    if not os.path.isdir(os.path.join(package_path, "dim_cpp")):
        print(f"FAIL: dim_cpp/ not found next to {urdf_filename} -- meshes won't resolve")
        return 1

    meshcat = StartMeshcat()
    print(f"Meshcat URL: {meshcat.web_url()}  (open this in a browser)")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser_ = Parser(plant)
    parser_.package_map().Add("dim_cpp", os.path.join(package_path, "dim_cpp"))
    parser_.AddModels(urdf_path)
    plant.Finalize()

    checks = {
        "num_positions": plant.num_positions(),
        f"has joint{num_arm_joints}": plant.HasJointNamed(f"joint{num_arm_joints}"),
        "has drive_joint": plant.HasJointNamed("drive_joint"),
        "has link_openft": plant.HasFrameNamed("link_openft"),
        "has link_eef": plant.HasFrameNamed("link_eef"),
    }
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # Sweep each arm joint a little so you can see it move in the browser
    q0 = np.zeros(plant.num_positions())
    plant.SetPositions(plant_context, q0)
    diagram.ForcedPublish(diagram_context)

    print("\nModel checks:")
    for name, val in checks.items():
        print(f"  {name}: {val}")

    print(f"\nSweeping joints for {args.hold:.0f}s -- watch the browser...")
    t0 = time.time()
    while time.time() - t0 < args.hold:
        t = time.time() - t0
        q = q0.copy()
        for i in range(num_arm_joints):
            joint = plant.GetJointByName(f"joint{i + 1}")
            q[joint.position_start()] = 0.3 * np.sin(t + i)
        plant.SetPositions(plant_context, q)
        diagram.ForcedPublish(diagram_context)
        time.sleep(0.05)

    print("Done. If the arm moved smoothly in the browser with no missing meshes, sim checks out.")
    return 0


if __name__ == "__main__":
    exit(main())
