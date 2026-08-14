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
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admittance_pull_law import AdmittanceConfig, compute_twist, slew_limit  # noqa: E402

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Where the gripper should be when the pull begins, and how it should be oriented.
# The orientation is not cosmetic: the policy drives along the tool's local -Z
# (local_drive_direction in FTAdaptivePullModule), so the tool's +Z has to point INTO the
# door for -Z to be "away from the door". Grasp the handle from above instead and the policy
# faithfully pulls straight up, which a hinged door cannot follow -- force climbs to the
# cutoff having barely moved the door, and no singularity is ever reached.
GRASP_XYZ = np.array([0.46, 0.0, 0.36])
TOOL_Z_WORLD = np.array([1.0, 0.0, 0.0])    # gripper points forward, at the microwave
TOOL_X_WORLD = np.array([0.0, 0.0, 1.0])    # finger axis vertical, straddling a vertical bar
SEED_Q = np.array([0.0, -0.3, 0.0, 0.6, 0.0, 0.9, 0.0])


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
        err_pos = GRASP_XYZ - data.site_xpos[site]
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


def build_scene(model_dir: Path, handle_xyz: np.ndarray, hinge_offset_y: float = 0.36) -> str:
    """Microwave placed so its handle sits exactly at the gripper's start pose.

    Deriving the placement from FK instead of hand-tuning coordinates means the grasp is
    always consistent: change START_Q and the microwave follows.
    """
    hx, hy, hz = handle_xyz
    hinge_y = hy + hinge_offset_y                 # hinge is inboard of the handle, on +Y
    body_cx = hx + 0.04 + 0.14                    # door face sits 4 cm ahead of the handle
    mw_cy = hinge_y - 0.20                        # microwave centre, door spans the full width
    table_top = hz - 0.15                         # the microwave rests on it, so it follows the FK
    leg_h = max(0.02, table_top / 2)
    return f"""
<mujoco model="xarm7 microwave door">
  <include file="{arm_only_xml(model_dir).name}"/>
  <!-- No gravity, deliberately. The real module subscribes to the arm's GRAVITY-COMPENSATED
       wrench (ft_ext_force, not ft_raw_force), so a gravity-free sim reproduces what the
       policy actually consumes. Leaving gravity on would instead have the position servos
       fight arm droop through the grasp constraint, which shows up as a standing ~250 N
       that has nothing to do with the door. -->
  <option gravity="0 0 0"/>
  <statistic center="0.35 0 0.4" extent="1.0"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75" markrgb="0.9 0.9 0.9"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
    <material name="mw_body" rgba="0.30 0.30 0.33 1"/>
    <material name="mw_door" rgba="0.20 0.22 0.26 1"/>
    <material name="mw_handle" rgba="0.85 0.85 0.88 1"/>
    <material name="mount" rgba="0.22 0.24 0.28 1"/>
    <material name="table_top" rgba="0.55 0.45 0.35 1"/>
    <material name="table_leg" rgba="0.40 0.35 0.30 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Arm mount: the xarm7 model puts link_base at z=0.12, so the pedestal fills to there. -->
    <geom name="arm_mount" type="cylinder" size="0.09 0.06" pos="0 0 0.06" material="mount"/>
    <geom name="arm_mount_plate" type="box" size="0.13 0.13 0.008" pos="0 0 0.008" material="mount"/>

    <!-- Table the microwave stands on. Top height follows from FK so the handle is always
         exactly at the gripper, whatever START_Q is. -->
    <geom name="table_top" type="box" size="0.22 0.34 0.01"
          pos="{body_cx:.4f} {mw_cy:.4f} {table_top - 0.01:.4f}" material="table_top"/>
    <geom name="table_leg1" type="cylinder" size="0.022 {leg_h:.4f}"
          pos="{body_cx - 0.18:.4f} {mw_cy - 0.29:.4f} {leg_h:.4f}" material="table_leg"/>
    <geom name="table_leg2" type="cylinder" size="0.022 {leg_h:.4f}"
          pos="{body_cx + 0.18:.4f} {mw_cy - 0.29:.4f} {leg_h:.4f}" material="table_leg"/>
    <geom name="table_leg3" type="cylinder" size="0.022 {leg_h:.4f}"
          pos="{body_cx - 0.18:.4f} {mw_cy + 0.29:.4f} {leg_h:.4f}" material="table_leg"/>
    <geom name="table_leg4" type="cylinder" size="0.022 {leg_h:.4f}"
          pos="{body_cx + 0.18:.4f} {mw_cy + 0.29:.4f} {leg_h:.4f}" material="table_leg"/>

    <geom name="mw_body" type="box" size="0.14 0.20 0.15"
          pos="{body_cx:.4f} {mw_cy:.4f} {hz:.4f}" material="mw_body"/>

    <body name="mw_door" pos="{hx + 0.04:.4f} {hinge_y:.4f} {hz:.4f}">
      <joint name="door_hinge" type="hinge" axis="0 0 1" range="-1.75 0"
             damping="0.8" frictionloss="1.2"/>
      <geom name="door_panel" type="box" size="0.012 0.20 0.15"
            pos="0 -0.20 0" material="mw_door" mass="1.6"/>
      <body name="mw_handle" pos="-0.04 {-hinge_offset_y:.4f} 0">
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
    <connect name="grasp" site1="link_tcp" site2="handle_site" solref="0.02 1"/>
  </equality>

  <sensor>
    <force name="ft_force" site="ft_sensor"/>
    <torque name="ft_torque" site="ft_sensor"/>
  </sensor>
</mujoco>
"""


class ArmSim:
    def __init__(self, model_dir: Path, free_roll: bool, damping: float = 0.06):
        self.free_roll = free_roll
        self.damping = damping
        self.start_q = solve_start_pose(model_dir)
        base = mujoco.MjModel.from_xml_path(str(arm_only_xml(model_dir)))
        base_data = mujoco.MjData(base)
        self.jid = [mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.dofs = np.array([base.jnt_dofadr[j] for j in self.jid])
        qadr = [base.jnt_qposadr[j] for j in self.jid]
        base_data.qpos[qadr] = self.start_q
        mujoco.mj_forward(base, base_data)
        tcp = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        handle = base_data.site_xpos[tcp].copy()

        scene = model_dir / "_generated_door_scene.xml"
        scene.write_text(build_scene(model_dir, handle))
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
        self.data.ctrl[self.act] = self.start_q
        # Settle before any control runs. The gripper's finger linkage is a closed chain held
        # by stiff equality constraints and starts strained, reading ~26 N at the flange and
        # decaying to zero over about a second. Starting the policy inside that transient makes
        # it trip its own force cutoff before the door is ever touched.
        for _ in range(int(2.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.data.ctrl[self.act] = self.data.qpos[self.qadr]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

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
        q = self.data.qpos[self.qadr] + qdot * dt
        self.data.ctrl[self.act] = np.clip(q, self.q_lo, self.q_hi)
        mujoco.mj_step(self.model, self.data)
        return float(s[-1])

    @property
    def wrench(self) -> tuple[np.ndarray, np.ndarray]:
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


def run(model_dir: Path, free_roll: bool, seconds: float, use_viewer: bool,
        sigma_stop: float, verbose: bool, cfg: AdmittanceConfig | None = None) -> dict:
    sim = ArmSim(model_dir, free_roll)
    cfg = cfg or AdmittanceConfig()
    rate, dt = 25.0, 1.0 / 25.0
    sub = max(1, int(round(dt / sim.model.opt.timestep)))

    viewer = None
    if use_viewer:
        import mujoco.viewer
        try:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        except RuntimeError as exc:
            if "mjpython" not in str(exc):
                raise
            raise SystemExit(
                "--viewer needs mjpython on macOS (a GUI has to own the main thread).\n"
                "Re-run the same command with mjpython instead of python:\n"
                f"  {Path(sys.executable).parent / 'mjpython'} {' '.join(sys.argv)}\n"
                "Headless runs (-v, --compare) work under plain python."
            ) from exc

    prev_lin, prev_ang = np.zeros(3), np.zeros(3)
    progress, min_sigma, reason = 0.0, float("inf"), "completed"
    start_pos = sim.data.site_xpos[sim.tcp].copy()
    ticks = int(seconds * rate)

    for tick in range(ticks):
        ee_rot = sim.ee_rot
        force_tool, torque_tool = sim.wrench
        sigma = np.linalg.svd(sim.task_matrix(ee_rot) @ sim.jacobian(), compute_uv=False)[-1]
        min_sigma = min(min_sigma, float(sigma))

        if sigma <= sigma_stop:
            reason = f"singularity (sigma={sigma:.4f})"
            break
        if sim.joint_margin() <= 0.02:
            reason = "joint limit"
            break

        drive = ee_rot @ np.array([0.0, 0.0, -1.0])
        res = compute_twist(force_tool, torque_tool, ee_rot, drive, cfg, progress_m=progress)
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
        if verbose and tick % 25 == 0:
            print(f"  t={tick / rate:5.1f}s  door={sim.door_deg:5.1f}deg  sigma={sigma:.4f}  "
                  f"F={res.resistance_force:5.1f}N  M={res.torque_mag:4.2f}Nm  "
                  f"margin={sim.joint_margin():.3f}rad")

    if viewer is not None:
        viewer.close()
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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    over = {k: v for k, v in (("force_cutoff", args.force_cutoff),
                              ("torque_cutoff", args.torque_cutoff)) if v is not None}
    cfg = dataclasses.replace(AdmittanceConfig(), **over) if over else None

    if args.compare:
        rows = {mode: run(args.model_dir, mode == "free-roll", args.seconds, False,
                          args.sigma_stop, args.verbose, cfg)
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
            args.verbose, cfg)
    print(f"\nmode={'free-roll' if args.free_roll else 'baseline'}  door={r['door_deg']:.1f}deg  "
          f"after {r['seconds']:.1f}s  min_sigma={r['min_sigma']:.4f}  -> {r['reason']}")


if __name__ == "__main__":
    main()
