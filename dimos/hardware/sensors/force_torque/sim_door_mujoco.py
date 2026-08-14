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

"""MuJoCo door-opening sim: xArm7 + a hinged microwave, driven by the real pull law.

sim_multi_door_test.py models the door but has no arm in it, so it cannot reproduce the
failure we actually hit on hardware -- the pull dies after 5-10 s at a kinematic
singularity. Singularity is a property of the ARM, so the arm has to be in the loop.

What is real here vs. modelled:
  real      admittance_pull_law.compute_twist, imported unmodified
  real      xArm7 kinematics, joint limits, and Jacobian (MuJoCo, from the shipped MJCF)
  real      the UFACTORY xArm gripper, including its finger linkage
  real      contact and constraint forces, read at the tool flange
  modelled  the grasp, as a point coupling -- the policy starts AFTER you have the handle
  modelled  the latch, as hinge friction; a real latch releases, this just resists

WHAT THIS SIM FOUND, which is not what we went in expecting:

  With the shipped config the pull stops at ~9 deg on the TORQUE cutoff, and sigma_min never
  drops below 0.18 -- nowhere near singular. Relax the cutoffs and the same policy takes the
  door to 89.5 deg, still without ever approaching a singularity.

  The cause is a lever arm nobody had costed. The FT sensor sits at the flange; the grasp is
  172 mm further out. A grasp force F therefore appears at the sensor as 0.172*F of torque,
  so torque_cutoff=4.5 N*m caps the pull at roughly 26 N at the gripper -- no matter how much
  headroom force_cutoff=80 N looks like it allows. The torque limit is really a hidden, and
  far tighter, force limit.

  --free-roll (release rotation about the grasp axis, turning the 6-DOF task into 5-DOF on a
  7-joint arm) does NOT help: 38 deg against 89 deg once cutoffs are relaxed. Kept as a
  comparison, but it is not the fix.

    python3 sim_door_mujoco.py --model-dir <xarm7/> -v
    python3 sim_door_mujoco.py --model-dir <xarm7/> --viewer
    python3 sim_door_mujoco.py --model-dir <xarm7/> --compare --torque-cutoff 40 --force-cutoff 150
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admittance_pull_law import (  # noqa: E402
    AdmittanceConfig, compute_twist, slew_limit, steer_drive_direction)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Where the gripper should be when the pull begins, and how it should be oriented.
# The orientation is not cosmetic: the policy drives along the tool's local -Z
# (local_drive_direction in FTAdaptivePullModule), so the tool's +Z has to point INTO the
# door for -Z to be "away from the door". Grasp the handle from above instead and the policy
# faithfully pulls straight up, which a hinged door cannot follow -- force climbs to the
# cutoff having barely moved the door, and no singularity is ever reached.
TOOL_Z_WORLD = np.array([1.0, 0.0, 0.0])    # gripper points forward, at the microwave
TOOL_X_WORLD = np.array([0.0, 0.0, 1.0])    # finger axis vertical, straddling a vertical bar
SEED_Q = np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.9, 0.0])

# One shared table: the arm bolts to it and the microwave stands on it, which is the bench
# layout. TABLE_TOP is the arm's own mounting height -- the shipped MJCF puts link_base at
# z=0.12, so the surface has to meet it exactly there or the robot floats above it.
TABLE_TOP = 0.12
MW_HALF = np.array([0.14, 0.20, 0.15])      # microwave half-extents: 280 x 400 x 300 mm
# Set back along +X so the swinging door has clear air and never reaches the arm's own
# column, and offset in +Y so the hinge side sits away from the robot.
# 0.78 m back, chosen by sweeping placement against conditioning rather than by eye: closer
# in (0.55-0.62) the arm is cramped at its own base height, sigma 0.04-0.11 and joint margin
# under 0.35 rad. Here it is sigma 0.174, margin 0.864 -- and still on the table, no riser.
MW_CENTRE = np.array([0.78, 0.12, TABLE_TOP + MW_HALF[2]])
HANDLE_PROUD = 0.04                          # handle stands this far off the door face
HANDLE_FROM_EDGE = 0.04                      # and this far in from the door's free edge
# Grasp compliance. A perfectly rigid coupling is not physical -- a real gripper on a real
# handle has finger pads, a wrist and skin between the load and the sensor.
GRASP_SOLREF = "0.02 1"



def microwave_geometry() -> dict:
    """Face, hinge and handle all derived from one placement, so they cannot disagree."""
    face_x = MW_CENTRE[0] - MW_HALF[0]
    hinge_y = MW_CENTRE[1] + MW_HALF[1]
    handle_y = MW_CENTRE[1] - MW_HALF[1] + HANDLE_FROM_EDGE
    return {
        "face_x": face_x,
        "hinge_y": hinge_y,
        "hinge_offset_y": hinge_y - handle_y,     # radius the handle swings on
        "handle": np.array([face_x - HANDLE_PROUD, handle_y, MW_CENTRE[2]]),
    }


def solve_start_pose(model_dir: Path) -> np.ndarray:
    """Damped least squares IK for the grasp pose. Solved rather than hand-picked, so the
    orientation the drive direction depends on is actually the one we intend."""
    model = mujoco.MjModel.from_xml_path(str(arm_only_xml(model_dir)))
    data = mujoco.MjData(model)
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    dofs = np.array([model.jnt_dofadr[j] for j in jid])
    lo, hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")

    z = TOOL_Z_WORLD / np.linalg.norm(TOOL_Z_WORLD)
    x = TOOL_X_WORLD - np.dot(TOOL_X_WORLD, z) * z
    x /= np.linalg.norm(x)
    target = np.column_stack([x, np.cross(z, x), z])

    q = SEED_Q.copy()
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(400):
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        rot = data.site_xmat[site].reshape(3, 3)
        err_pos = microwave_geometry()['handle'] - data.site_xpos[site]
        skew = target @ rot.T - (target @ rot.T).T
        err_rot = 0.5 * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        err = np.hstack([err_pos, err_rot])
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack([jp[:, dofs], jr[:, dofs]])
        u, s, vt = np.linalg.svd(jac, full_matrices=False)
        q = np.clip(q + vt.T @ ((s / (s**2 + 0.02**2)) * (u.T @ err)), lo + 0.05, hi - 0.05)
    return q


def arm_only_xml(model_dir: Path) -> Path:
    """The shipped xarm7.xml carries a `home` keyframe sized for scene.xml's loose props,
    so it will not load on its own or under any scene that lacks them. Strip it."""
    text = (model_dir / "xarm7.xml").read_text()
    start = text.find("<keyframe")
    while start != -1:
        end = text.find("</keyframe>", start)
        text = text[:start] + (text[end + len("</keyframe>"):] if end != -1 else "")
        start = text.find("<keyframe")

    # FT site goes on link7, the tool flange -- where the real sensor bolts on, and crucially
    # PROXIMAL to the gripper. Sited inside the gripper instead, the sensor sits downstream of
    # the finger-linkage equality constraints (solref 0.005, very stiff) and reads hundreds of
    # newtons of the gripper's own internal linkage load with nothing touching the robot.
    anchor = '<joint name="joint7" class="size3"/>'
    text = text.replace(anchor, anchor + '\n                    <site name="ft_sensor" size="0.001" rgba="0 1 0 1"/>', 1)
    out = model_dir / "_xarm7_nokeyframe.xml"
    out.write_text(text)
    return out


def build_scene(model_dir: Path) -> str:
    """Arm and microwave on one table, with the door geometry taken from microwave_geometry()."""
    g = microwave_geometry()
    mx, my, mz = MW_CENTRE
    hx, hy, hz = g["handle"]
    leg = TABLE_TOP / 2
    return f"""
<mujoco model="xarm7 microwave door">
  <include file="{arm_only_xml(model_dir).name}"/>
  <!-- No gravity, deliberately. The real module subscribes to the arm's GRAVITY-COMPENSATED
       wrench (ft_ext_force, not ft_raw_force), so a gravity-free sim reproduces what the
       policy actually consumes. With gravity on, the position servos fight arm droop through
       the grasp constraint and that shows up as a standing ~250 N unrelated to the door. -->
  <option gravity="0 0 0"/>
  <statistic center="0.40 0.02 0.30" extent="0.85"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="148" elevation="-14"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75" markrgb="0.9 0.9 0.9"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
    <material name="mw_body" rgba="0.30 0.30 0.33 1"/>
    <material name="mw_door" rgba="0.62 0.72 0.82 1"/>
    <material name="mw_handle" rgba="0.90 0.25 0.15 1"/>
    <material name="mount" rgba="0.22 0.24 0.28 1"/>
    <material name="table_top" rgba="0.58 0.47 0.36 1"/>
    <material name="table_leg" rgba="0.40 0.35 0.30 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- One table carrying BOTH the robot and the microwave. Its top is at {TABLE_TOP:.2f} m,
         which is where the arm's link_base sits, so the robot is bolted to the surface. -->
    <geom name="table_top" type="box" size="0.62 0.52 0.012"
          pos="0.34 0.06 {TABLE_TOP - 0.012:.4f}" material="table_top"/>
    <geom name="table_leg1" type="cylinder" size="0.025 {leg:.4f}" pos="-0.22 -0.42 {leg:.4f}" material="table_leg"/>
    <geom name="table_leg2" type="cylinder" size="0.025 {leg:.4f}" pos="0.90 -0.42 {leg:.4f}" material="table_leg"/>
    <geom name="table_leg3" type="cylinder" size="0.025 {leg:.4f}" pos="-0.22 0.54 {leg:.4f}" material="table_leg"/>
    <geom name="table_leg4" type="cylinder" size="0.025 {leg:.4f}" pos="0.90 0.54 {leg:.4f}" material="table_leg"/>

    <!-- Robot mounting plate, flush on the table top. -->
    <geom name="arm_mount" type="cylinder" size="0.085 0.006" pos="0 0 {TABLE_TOP - 0.006:.4f}" material="mount"/>

    <geom name="mw_body" type="box" size="{MW_HALF[0]:.3f} {MW_HALF[1]:.3f} {MW_HALF[2]:.3f}"
          pos="{mx:.4f} {my:.4f} {mz:.4f}" material="mw_body"/>

    <body name="mw_door" pos="{g['face_x']:.4f} {g['hinge_y']:.4f} {mz:.4f}">
      <joint name="door_hinge" type="hinge" axis="0 0 1" range="-1.75 0"
             damping="0.8" frictionloss="1.2"/>
      <geom name="door_panel" type="box" size="0.012 {MW_HALF[1]:.3f} {MW_HALF[2]:.3f}"
            pos="0 {-MW_HALF[1]:.3f} 0" material="mw_door" mass="1.6"/>
      <body name="mw_handle" pos="{-HANDLE_PROUD:.3f} {hy - g['hinge_y']:.4f} 0">
        <geom name="handle_bar" type="cylinder" size="0.011 0.055" material="mw_handle" mass="0.1"/>
        <site name="handle_site" size="0.004" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <equality>
    <!-- The grasp. The policy under test begins once the handle is already held.
         A point coupling, not a weld: the door has exactly one DOF, so holding the handle
         in position already drives it completely, and a weld would additionally demand an
         orientation match that is unsatisfiable at t=0 (and that a real gripper on a round
         handle bar does not enforce either -- it can roll about the bar). -->
    <connect name="grasp" site1="link_tcp" site2="handle_site" solref="{GRASP_SOLREF}"/>
  </equality>

  <sensor>
    <force name="ft_force" site="ft_sensor"/>
    <torque name="ft_torque" site="ft_sensor"/>
  </sensor>
</mujoco>
"""


class ArmSim:
    def __init__(self, model_dir: Path, free_roll: bool, damping: float = 0.06,
                 start_q: np.ndarray | None = None):
        self.free_roll = free_roll
        self.damping = damping
        # Given joints win; otherwise solve for the grasp pose. The microwave is placed from
        # whichever pose this is, so an arbitrary pose moves the door with it.
        self.start_q = np.asarray(start_q, float) if start_q is not None else solve_start_pose(model_dir)
        scene = model_dir / "_generated_door_scene.xml"
        scene.write_text(build_scene(model_dir))
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jid])
        self.dofs = np.array([self.model.jnt_dofadr[j] for j in self.jid])
        self.act = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i}")
                             for i in range(1, 8)])
        self.tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self.hinge = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        self.hinge_q = self.model.jnt_qposadr[self.hinge]
        self.f_adr = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")]
        self.t_adr = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")]
        self.q_lo = self.model.jnt_range[self.jid, 0]
        self.q_hi = self.model.jnt_range[self.jid, 1]

        self.data.qpos[self.qadr] = self.start_q
        mujoco.mj_forward(self.model, self.data)
        self.q_cmd = self.start_q.copy()
        self.data.ctrl[self.act] = self.q_cmd
        # Settle before any control runs. The gripper's finger linkage is a closed chain held
        # by stiff equality constraints and starts strained, reading ~26 N at the flange and
        # decaying to zero over about a second. Starting the policy inside that transient makes
        # it trip its own force cutoff before the door is ever touched.
        for _ in range(int(2.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.data.ctrl[self.act] = self.data.qpos[self.qadr]   # follow, do not fight
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Latch the command to wherever settling actually left the arm. Holding it at start_q
        # instead leaves a standing pose error the stiff servo turns into ~190 N of preload
        # before the policy has issued a single command.
        self.q_cmd = self.data.qpos[self.qadr].copy()
        self.data.ctrl[self.act] = self.q_cmd

    def jacobian(self) -> np.ndarray:
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tcp)
        return np.vstack([jp[:, self.dofs], jr[:, self.dofs]])

    def task_matrix(self, ee_rot: np.ndarray) -> np.ndarray:
        """6x6 identity, or 5x6 with the grasp-axis rotation row removed.

        The gripper grips along its own local Z, so rotation about world `axis` is the DOF
        the handle does not constrain. Two unit vectors perpendicular to it span the
        rotations that still matter, and dropping the third is what frees a joint.
        """
        if not self.free_roll:
            return np.eye(6)
        axis = ee_rot @ np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, seed)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        sel = np.zeros((5, 6))
        sel[:3, :3] = np.eye(3)
        sel[3, 3:] = u
        sel[4, 3:] = v
        return sel

    def step(self, twist: np.ndarray, ee_rot: np.ndarray, dt: float) -> float:
        """Damped least squares, so conditioning degrades gracefully instead of exploding."""
        jac = self.jacobian()
        sel = self.task_matrix(ee_rot)
        task_jac, task_twist = sel @ jac, sel @ twist
        u, s, vt = np.linalg.svd(task_jac, full_matrices=False)
        qdot = vt.T @ ((s / (s**2 + self.damping**2)) * (u.T @ task_twist))
        # Integrate the COMMANDED position, never the measured one. Re-basing on measured qpos
        # each tick gives the servo zero stiffness: the door pushes the arm, the controller
        # adopts wherever it was pushed to, and the drift compounds -- a zero twist command
        # then still walks the door open and builds 40+ N out of nothing.
        self.q_cmd = np.clip(self.q_cmd + qdot * dt, self.q_lo, self.q_hi)
        self.data.ctrl[self.act] = self.q_cmd
        mujoco.mj_step(self.model, self.data)
        return float(s[-1])

    @property
    def wrench(self) -> tuple[np.ndarray, np.ndarray]:
        # Negated deliberately. MuJoCo reports the load ON the tool, but compute_twist's
        # compliance is v = -k*F, which only relieves load when F is the force the tool
        # EXERTS. Hand it the on-tool force and every compliance term inverts into positive
        # feedback: measured 1.7 deg of door travel against 24.4 deg with the sign correct.
        return (-self.data.sensordata[self.f_adr:self.f_adr + 3].copy(),
                -self.data.sensordata[self.t_adr:self.t_adr + 3].copy())

    @property
    def ee_rot(self) -> np.ndarray:
        return self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    @property
    def door_deg(self) -> float:
        return float(np.degrees(abs(self.data.qpos[self.hinge_q])))

    def joint_margin(self) -> float:
        q = self.data.qpos[self.qadr]
        return float(min(np.min(q - self.q_lo), np.min(self.q_hi - q)))


def _camera(model) -> "mujoco.MjvCamera":
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.42, 0.02, 0.30]
    cam.distance, cam.azimuth, cam.elevation = 1.25, 148.0, -14.0
    return cam


def run(model_dir: Path, free_roll: bool, seconds: float, use_viewer: bool,
        sigma_stop: float, verbose: bool, cfg: AdmittanceConfig | None = None,
        record: Path | None = None, fps: int = 12,
        start_q: np.ndarray | None = None) -> dict:
    sim = ArmSim(model_dir, free_roll, start_q=start_q)
    cfg = cfg or AdmittanceConfig()

    renderer = camera = None
    frames: list = []
    if record is not None:
        renderer = mujoco.Renderer(sim.model, 480, 640)
        camera = _camera(sim.model)
    rate, dt = 25.0, 1.0 / 25.0
    sub = max(1, int(round(dt / sim.model.opt.timestep)))

    # SPACE / ENTER start and pause the pull; the viewer stays live either way, so the scene
    # can be orbited and inspected before anything moves. Mirrors the real rig, where you set
    # the grasp up by hand and only then hand over to the policy.
    KEY_P, KEY_ENTER, KEY_KP_ENTER = 80, 257, 335   # NOT space: the viewer keeps that for itself
    gate = {"paused": use_viewer}

    def on_key(keycode: int) -> None:
        if keycode in (KEY_P, KEY_ENTER, KEY_KP_ENTER):
            gate["paused"] = not gate["paused"]
            print("\n>>> POLICY OFF -- arm holding\n" if gate["paused"]
                  else "\n>>> POLICY ON -- pulling\n", flush=True)

    viewer = None
    if use_viewer:
        from mujoco import viewer as mj_viewer   # not `import mujoco.viewer`: that binds the
        try:                                       # name `mujoco` locally and shadows the module
            viewer = mj_viewer.launch_passive(sim.model, sim.data, key_callback=on_key)
        except RuntimeError as exc:
            if "mjpython" not in str(exc):
                raise
            raise SystemExit(
                "--viewer needs mjpython on macOS (a GUI has to own the main thread).\n"
                "Re-run the same command with mjpython instead of python:\n"
                f"  {Path(sys.executable).parent / 'mjpython'} {' '.join(sys.argv)}\n"
                "Headless runs (-v, --compare) work under plain python."
            ) from exc
        print("\n" + "=" * 62)
        print("  POLICY IS OFF. The arm is holding the handle, sim is live.")
        print("  press  P  (or ENTER)  ->  toggle the pull policy ON / OFF")
        print("  drag = orbit    scroll = zoom    right-drag = pan")
        print("=" * 62 + "\n", flush=True)

    prev_lin, prev_ang = np.zeros(3), np.zeros(3)
    drive_world = None
    prev_tcp = sim.data.site_xpos[sim.tcp].copy()
    progress, min_sigma, reason = 0.0, float("inf"), "completed"
    start_pos = sim.data.site_xpos[sim.tcp].copy()
    ticks = int(seconds * rate)

    for tick in range(ticks):
        if viewer is not None and not viewer.is_running():
            reason = "viewer closed"
            break
        tick_start = time.time()

        ee_rot = sim.ee_rot
        force_tool, torque_tool = sim.wrench
        sigma = np.linalg.svd(sim.task_matrix(ee_rot) @ sim.jacobian(), compute_uv=False)[-1]

        # Policy OFF: physics keeps running and the arm holds its grasp, exactly as it does on
        # the real rig between grasping the handle and handing over. Nothing below this point
        # is evaluated, so a stalled policy cannot trip a cutoff while it is not even driving.
        if gate["paused"]:
            for _ in range(sub):
                sim.step(np.zeros(6), ee_rot, sim.model.opt.timestep)
            prev_lin, prev_ang = np.zeros(3), np.zeros(3)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - tick_start)))
            continue

        min_sigma = min(min_sigma, float(sigma))
        if sigma <= sigma_stop:
            reason = f"singularity (sigma={sigma:.4f})"
            break
        if sim.joint_margin() <= 0.02:
            reason = "joint limit"
            break

        # Seed from the tool's -Z (what the module does), then let it follow real motion.
        if drive_world is None:
            drive_world = ee_rot @ np.array([0.0, 0.0, -1.0])
        tcp_now = sim.data.site_xpos[sim.tcp].copy()
        drive_world = steer_drive_direction(drive_world, (tcp_now - prev_tcp) / dt,
                                            cfg.drive_steer_blend)
        prev_tcp = tcp_now
        res = compute_twist(force_tool, torque_tool, ee_rot, drive_world, cfg, progress_m=progress)
        if res.safety_stop:
            reason = f"safety cutoff (F={res.resistance_force:.0f}N M={res.torque_mag:.1f}Nm)"
            break

        lin = slew_limit(prev_lin, res.linear, cfg.max_linear_accel * dt)
        ang = slew_limit(prev_ang, res.angular, cfg.max_angular_accel * dt)
        prev_lin, prev_ang = lin, ang

        for _ in range(sub):
            sim.step(np.hstack([lin, ang]), ee_rot, sim.model.opt.timestep)
        progress = float(np.linalg.norm(sim.data.site_xpos[sim.tcp] - start_pos))

        if viewer is not None:
            viewer.sync()
            # Pace to wall clock so the motion reads at true speed, not as fast as it solves.
            time.sleep(max(0.0, dt - (time.time() - tick_start)))
        if renderer is not None and tick % max(1, int(rate / fps)) == 0:
            renderer.update_scene(sim.data, camera)
            frames.append(renderer.render().copy())
        if verbose and tick % 25 == 0:
            print(f"  t={tick / rate:5.1f}s  door={sim.door_deg:5.1f}deg  sigma={sigma:.4f}  "
                  f"F={res.resistance_force:5.1f}N  M={res.torque_mag:4.2f}Nm  "
                  f"margin={sim.joint_margin():.3f}rad")

    if viewer is not None:
        viewer.close()
    if frames:
        from PIL import Image

        record.parent.mkdir(parents=True, exist_ok=True)
        keep = frames if len(frames) <= 160 else [frames[i] for i in
                np.linspace(0, len(frames) - 1, 160).astype(int)]
        images = [Image.fromarray(f) for f in keep]
        images[0].save(record, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        strip = Image.new("RGB", (640 * 4, 480 * 2), "white")
        picks = np.linspace(0, len(images) - 1, 8).astype(int)
        for i, idx in enumerate(picks):
            strip.paste(images[int(idx)], ((i % 4) * 640, (i // 4) * 480))
        strip.save(record.with_name(record.stem + "_filmstrip.png"))
        print(f"wrote {record} ({len(frames)} frames) and {record.stem}_filmstrip.png")
    return {"door_deg": sim.door_deg, "reason": reason, "min_sigma": min_sigma,
            "seconds": min(ticks, tick + 1) / rate, "progress_m": progress}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True, help="unpacked xarm7/ (contains xarm7.xml)")
    p.add_argument("--free-roll", action="store_true", help="release rotation about the grasp axis")
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--sigma-stop", type=float, default=0.01)
    p.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    p.add_argument("--compare", action="store_true", help="run both modes and print the difference")
    p.add_argument("--force-cutoff", type=float, default=None, help="override AdmittanceConfig")
    p.add_argument("--torque-cutoff", type=float, default=None, help="override AdmittanceConfig")
    p.add_argument("--start-joints", type=str, default=None,
                   help="7 comma-separated joint angles in DEGREES, e.g. '0,0,0,0,0,0,-90'. "
                        "Overrides the solved grasp pose; the microwave follows the resulting TCP.")
    p.add_argument("--record", type=Path, default=None, help="write an animated GIF + filmstrip")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    start_q = None
    if args.start_joints:
        vals = [float(v) for v in args.start_joints.replace(" ", "").split(",")]
        if len(vals) != 7:
            raise SystemExit(f"--start-joints needs 7 values, got {len(vals)}")
        start_q = np.radians(vals)

    over = {k: v for k, v in (("force_cutoff", args.force_cutoff),
                              ("torque_cutoff", args.torque_cutoff)) if v is not None}
    cfg = dataclasses.replace(AdmittanceConfig(), **over) if over else None

    if args.compare:
        rows = {mode: run(args.model_dir, mode == "free-roll", args.seconds, False,
                          args.sigma_stop, args.verbose, cfg, start_q=start_q)
                for mode in ("baseline", "free-roll")}
        print(f"\n{'mode':<12}{'door opened':>13}{'stopped after':>15}{'min sigma':>12}   why it stopped")
        for mode, r in rows.items():
            print(f"{mode:<12}{r['door_deg']:>11.1f}deg{r['seconds']:>13.1f}s"
                  f"{r['min_sigma']:>12.4f}   {r['reason']}")
        a, b = rows["baseline"]["door_deg"], rows["free-roll"]["door_deg"]
        print(f"\nfree-roll opened the door {b - a:+.1f} deg further "
              f"({b / max(a, 1e-6):.1f}x)" if a > 0.1 else
              f"\nbaseline never moved the door; free-roll reached {b:.1f} deg")
        return

    r = run(args.model_dir, args.free_roll, args.seconds, args.viewer, args.sigma_stop,
            args.verbose, cfg, args.record, args.fps, start_q)
    print(f"\nmode={'free-roll' if args.free_roll else 'baseline'}  door={r['door_deg']:.1f}deg  "
          f"after {r['seconds']:.1f}s  min_sigma={r['min_sigma']:.4f}  -> {r['reason']}")


if __name__ == "__main__":
    main()
