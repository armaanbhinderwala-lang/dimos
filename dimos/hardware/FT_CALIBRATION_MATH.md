# FT Calibration — Math & Conventions

This is the reference derivation everything else (`ft_ground_truth_check.py`,
`ft_ground_truth_smoke_test.py`, `calc_calibration_matrix.py`, and whatever
regression code comes next) is checked against. If code and this document
disagree, that's a bug in one of them — find out which before trusting either.

## 1. Notation

- Frames are written `{A}`, `{B}`, ... A vector `v` expressed in frame `{A}`'s
  axes is written `v_A`.
- A rigid transform from `{A}` to `{B}` is the pair `(R, d)` where:
  - `R` = rotation of `{B}` relative to `{A}` — its columns are `{B}`'s axes
    expressed in `{A}`. A vector fixed in `{B}`'s axes converts to `{A}`'s
    axes as `v_A = R v_B`; the inverse is `v_B = R^T v_A` (R is orthonormal,
    `R^{-1} = R^T`).
  - `d_A` = position of `{B}`'s origin, expressed in `{A}`'s axes.
- A **wrench** is the pair `(F, τ)` — a 3D force and the 3D moment it
  produces about a specific reference point, always stated together with
  *which point and which axes* it's expressed in. A wrench without a stated
  reference point/frame is not fully specified.

## 2. The wrench frame-transform law (derived, not quoted)

This is the one piece of math everything else leans on, so it's derived here
instead of pasted from memory.

**Change of reference point only** (frames `{A}`, `{B}` share the same axes,
origins differ by `r` = vector from `A`'s origin to `B`'s origin, both
expressed in those shared axes). By definition, moment about a point is
`τ = Σ (xᵢ − point) × fᵢ` for point forces `fᵢ` at positions `xᵢ` (or the
continuum/single-resultant equivalent). Then:

```
τ_B = Σ (xᵢ − B) × fᵢ = Σ [(xᵢ − A) + (A − B)] × fᵢ
    = τ_A + (A − B) × Σfᵢ = τ_A − r × F         (since A − B = −r)
F_B = F_A
```

**Change of axes only** (frames share an origin, `{B}` is `{A}` rotated by
`R`). Force and moment are ordinary vectors, so they rotate like any vector:

```
F_B = R^T F_A
τ_B = R^T τ_A
```

**General transform** `(R, d)` = rotate then transport. First re-express the
wrench in `{B}`'s axes (still about `A`'s origin): `F' = R^T F_A`,
`τ' = R^T τ_A`. Then transport the reference point, now working in `{B}`'s
axes throughout, using `r = R^T d_A` (the origin offset, re-expressed in
`{B}`'s axes):

```
F_B = R^T F_A
τ_B = R^T τ_A − (R^T d_A) × (R^T F_A)
```

Since rotation distributes over the cross product for proper rotations
(`R(a×b) = (Ra)×(Rb)`, so `R^T(a×b) = (R^T a)×(R^T b)`), this simplifies to
the closed form:

```
F_B = R^T F_A
τ_B = R^T ( τ_A − d_A × F_A )
```

This matches the standard "tool transform" equation used by commercial F/T
sensors (e.g. ATI's manuals state the identical structure) — a useful
external check that the derivation above didn't go sideways.

### Corollary used by `ft_ground_truth_check.py::expected_wrench()`

A rigidly-attached point mass under gravity applies a **pure force** through
its center of gravity — zero intrinsic moment there. So take `{A}` = a frame
at the mass's CG with `τ_A = 0`, `F_A` = the weight; take `{B}` = the sensor
frame, `d_A` = vector from the mass's CG to the sensor origin. Plugging
`τ_A = 0` into the general law:

```
F_B = R^T F_A
τ_B = R^T ( 0 − d_A × F_A ) = R^T ( (−d_A) × F_A )
```

Define `r := −d_A`, the vector **from the sensor origin to the mass's CG**
(both still expressed in `{A}`'s axes, before rotating into `{B}`). Then:

```
F_B = R^T F_A
τ_B = R^T ( r × F_A )
```

i.e. exactly `τ_sensor = lever_arm × F_sensor` once everything is expressed in
the sensor's own axes. **This is what the current code computes** —
confirmed consistent with the general law, not just asserted. Good to know
before trusting it against real hardware, not after.

### Where the general (non-corollary) form is actually needed

The point-mass shortcut above only works because `τ_A = 0` at the CG. It does
**not** apply when relating the uFactory sensor's reading to the homemade
sensor's reading directly (as opposed to each one separately vs. the known
weight) — that requires the full `(R, d)` transform between the two sensor
origins, with `d` measured from CAD (see §4). Don't reuse the point-mass
shortcut there; use the general closed form.

## 3. Frames in this system

| Frame | Meaning | How we get it |
|---|---|---|
| `{W}` | Arm base/world frame | Fixed, `+Z` up (xArm convention — **confirm with smoke test**, don't assume) |
| `{T}` | Tool flange | `arm.get_position()` reports this frame's pose in `{W}` |
| `{U}` | uFactory FT sensor origin | Assumed to coincide with `{T}` unless the smoke test / CAD says otherwise — **unverified assumption, not a fact** |
| `{H}` | Homemade sensor origin (`link_openft` in the URDF) | Fixed offset from `{T}`; **not currently defined anywhere in `xarm6_openft_gripper.urdf`** — needs a CAD measurement (§4) |

`{U}` and `{H}` are never populated at the same time (swap-only mounting),
which is exactly why §2's corollary — comparing each sensor independently
against a physics-computed reference — is the calibration strategy, not
"transform sensor A's reading into sensor B's frame and diff them."

## 4. The still-open transform (§5 of the roadmap: "transform and differences
between the sensors")

To ever answer "what would the homemade sensor have read, given what the
uFactory sensor just measured" (useful for the non-simultaneous door-opening
comparison, and for keeping `link_openft` consistent with whichever sensor is
mounted), we need `(R_H_from_U, d_U)`: the fixed rotation and translation from
`{U}` to `{H}`. This is a **mechanical measurement**, not something
regression can recover from swap-only data (there's no simultaneous
observation pair to fit it from). Source it from the CAD/bracket dimensions,
then apply §2's general closed form — not the point-mass shortcut — to move
a wrench between the two sensor frames.

## 5. The calibration regression, stated precisely

The model fit by `calc_calibration_matrix.py` is affine:

```
y = C x + b,   y ∈ ℝ⁶ (wrench),  x ∈ ℝ¹⁶ (raw channels),  C ∈ ℝ^{6×16},  b ∈ ℝ⁶
```

solved by ordinary least squares over `N` samples: `Y = X_aug Cᵀ_aug`,
`X_aug ∈ ℝ^{N×17}`. Two structural risks this framing makes explicit,
worth checking as a **design-of-experiment** property, not just a post-hoc
residual:

- **Observability**: each output axis (Fx..Mz) needs the *label* matrix
  `Y ∈ ℝ^{N×6}` to actually vary independently along that axis across the
  collected poses. This is exactly what broke the current
  `ft_calibration.json` — its Mz column was apparently ~constant across the
  160 training samples, so least squares had nothing to fit and returned a
  ~zero row. This should be checked on the *planned* pose/load set before
  physically running it (e.g. rank/condition number of `Y` restricted to
  each axis, or simpler: confirm the collection protocol includes deliberate
  off-axis/twisting loads), not discovered afterward in a shipped file.
- **Collinearity between inputs**: the 16 channels come from 4 physical
  magnet clusters, not 16 independent transducers — several channels move
  together under most loads. `X_aug` can be poorly conditioned even when `Y`
  is well-observed, which is a argument for L2-regularized (ridge) fitting
  over plain `lstsq`, not just a nice-to-have.

## 6. Temperature — where it enters the model, structurally

No temperature channel currently exists in the firmware's 16-value stream
(confirmed against the firmware repo — not just undocumented, actually
absent from the serial output). Until/unless that changes, temperature can
only enter as an *empirically characterized drift term*, not a regressed
input feature:

```
y_true = C x + b(T) ,   b(T) ≈ b₀ + κ (T − T_ref)     [first-order, unloaded-drift model]
```

`κ` (drift per °C, per axis) has to come from a dedicated static test (zero
load, temperature swept or allowed to drift, log raw channels vs. an external
thermometer reading over time) — it cannot come out of the pose-sweep
collection, which conflates orientation-driven and temperature-driven
changes if run over a long enough session to also drift thermally. Keep
those two experiments separate for exactly this reason.

## 7. Uncertainty — expected wrench is not exact ground truth, it has error bars

`expected_wrench()` inherits uncertainty from three inputs: mass (`Δm`,
scale/reference-weight tolerance), lever arm (`Δr`, caliper/CAD measurement
error), and orientation (`Δrpy`, joint-encoder resolution propagated through
FK). First-order propagation through `F = R^T(−m g ẑ)`, `τ = r × F`:

```
ΔF ≈ |g| ( |Δm| + m |Δrpy| )      [orientation error rotates the gravity vector]
Δτ ≈ |Δr| |F| + |r| |ΔF|          [product-rule bound on r × F]
```

Practical use: when `ft_ground_truth_check.py`'s residual table (measured −
expected) is printed, a residual smaller than these bounds is **noise in the
ground truth, not sensor error** — don't chase it. A residual larger than
these bounds is a real signal (wrong rotation convention, wrong lever arm, or
genuine sensor inaccuracy) worth chasing. Get real numbers for `Δm`, `Δr`,
`Δrpy` before the first real data collection run so this check means
something rather than being hand-waved.

## 8. Retargeting the door-opening skill (`ft_pull_skill.py`) to the uFactory sensor

Before writing any code: an inventory of exactly what in the control loop is
DIY-sensor-specific vs. genuinely sensor-agnostic, using the frame vocabulary
from §1-§3. Two different kinds of dependency show up, and they need two
different fixes.

### 8a. Already sensor-agnostic (no change needed)

`FTPullModule.force` / `.torque` are generic `In[Vector3]` ports fed over LCM
channels `/ft/force`, `/ft/torque` — the control code never touches serial
parsing or the calibration matrix directly. **Whatever publishes a `Vector3`
on those channels satisfies the contract.** This is the seam to exploit: a
new driver module that polls the uFactory sensor and republishes the same
two channels requires **zero changes** to `FTPullModule`, the SQLite logger,
or the visualizer — they all consume the LCM contract, not the sensor.

The diff-IK/joint-command path (`execute_motion` → `command_xarm`) is also
frame-agnostic in itself — it takes whatever target pose it's given and
solves for joints. It only becomes sensor-specific through the frame it's
told to control (next section).

### 8b. Baked to the DIY sensor's specific frame/geometry (must be re-derived, not copy-pasted)

| Code location | What it assumes | Why it's DIY-specific | uFactory equivalent |
|---|---|---|---|
| `setup_drake_simulation()`: `GetFrameByName("link_openft")` | Control frame `{E}` for diff IK is the DIY sensor's URDF frame | `link_openft` is defined in `xarm6_openft_gripper.urdf` at a specific offset from `joint6`/flange, sized for the DIY board's stack-up | Needs a **new URDF frame** at the uFactory sensor's actual origin `{U}` — doesn't exist in the URDF today (per §3). Until CAD gives the real offset, the only defensible placeholder is `{U} = {T}` (flange), which is itself an unconfirmed assumption per §3's status table, not a fact to build on silently. |
| `compute_combined_motion()` / `compute_target_pose()`: `EvalBodyPoseInWorld(..., self.openft_body)` | Current orientation for the pull-direction rotation (`openft_rot @ pull_local`) and the pivot-point construction is read from the DIY sensor's body | Same root cause as above — hardcoded to one specific body | Generalize to `self.control_body`, set once from config, not hardcoded to `openft_body` |
| `pivot_distance` (default `0.2`) | Distance from the DIY sensor's origin to the assumed hinge pivot, along local `-Z` | Tuned for the DIY board + gripper's specific stack length | Re-measurement, not reuse — the uFactory sensor sits at a different point in the tool stack, so this distance is physically different even against the same hinge |
| `pull_local = [0, 0, -pull_distance]` | The sensor frame's `-Z` axis points in the door-pull direction | True only if `link_openft`'s axes were defined (by the previous engineer) to point that way | **Unverified for the uFactory frame.** Manufacturer axis convention has no obligation to match a hobby board's silkscreen convention just because the physical mounting orientation is superficially "the same" |
| `force_state.x_force()`, `.lateral_force()` (`force[:2]`) | The DIY sensor's local `+X` is the primary "which side of target is the arm pulling from" signal axis, and the door-relevant force lives in the local XY plane | Same as above — an axis-labeling convention, not a physical law | **Unverified for the uFactory frame** — needs an empirical check (§8c), not an assumption |
| `xarm6_openft_gripper.urdf` overall kinematic chain | FK/handle-position estimates assume the DIY board's added length between wrist and gripper | Mechanical stack-up specific to that hardware | A URDF variant (or parametrized link) reflecting the uFactory sensor's actual stack-up — likely shorter, since it doesn't add the DIY board's housing |

Notably **absent from this list**: sensor calibration quality. `compute_combined_motion` never reads torque — only `Fx` and `‖(Fx,Fy)‖`. So the DIY sensor's degenerate Mz (§5) was never actually gating this skill, and the uFactory sensor's presumably-solid factory calibration on all 6 axes isn't required to fix anything here either — it's a nice bonus, not a blocker.

One structural parallel worth knowing: the uFactory SDK's `iden_ft_sensor_load_offset` / `set_ft_sensor_load_offset` calls (found in the SDK, signature unconfirmed — see the smoke test) almost certainly implement exactly the §2 point-mass corollary, applied to the gripper's own weight instead of a calibration mass, to zero out tool weight from the raw reading. Same math, different mass.

### 8c. The one unknown that gates everything else: axis convention

Everything in 8b's last three rows collapses to a single open question: **does the uFactory sensor's reported `(Fx,Fy,Fz)` correspond to the same physical directions as `link_openft`'s convention, once mounted "the same way"?** This cannot be assumed from mounting position alone — it must be checked empirically, the same way `ft_ground_truth_smoke_test.py` resolves other unknowns by reading real output instead of guessing:

1. Mount the uFactory sensor as the DIY sensor was mounted (per the engineer's orientation note in `FT_ENGINEER_NOTES.md`).
2. Zero the sensor unloaded (`set_ft_sensor_zero`).
3. Push by hand along each of the three physical axes that matter to the skill — the intended pull direction, and the two lateral directions — one at a time, and read `get_ft_sensor_data()` for each.
4. Record which reported axis moves, and its sign, for each physical push direction.

This is a poke test, not a derivation — cheap, five minutes, and it's the only way to fill in `pull_local`'s axis and sign and confirm whether `x_force()`/`lateral_force()` need their component indices remapped for the new sensor. Everything else in 8b (pivot_distance, the URDF frame) can be measured from CAD without the arm powered on; this one needs the live sensor.

## Status: what's verified vs. assumed

| Claim | Status |
|---|---|
| General wrench transform law (§2) | Derived here from the definition of moment + rotation properties. Solid. |
| Point-mass corollary matches `expected_wrench()` | Verified by direct substitution (§2 corollary). |
| xArm reports `{W}` as Z-up, `get_position()` = `[x,y,z,roll,pitch,yaw]`, extrinsic-XYZ | **Assumed, not confirmed** — this is exactly what `ft_ground_truth_smoke_test.py` exists to check. Do not trust §2/§3 numerically until that's run. |
| `{U}` coincides with `{T}` | **Assumed, not confirmed.** Needs CAD or an empirical check. |
| `{H}` offset from `{T}` | **Unmeasured** — not in the URDF. |
| Observability/collinearity risk in the regression (§5) | Diagnosed from the existing `ft_calibration.json` (degenerate Mz row) — real, not hypothetical. |
| Temperature drift coefficient `κ` (§6) | **Unmeasured** — needs its own bench test. |
| Uncertainty bounds (§7) | Structurally correct (first-order propagation); numerically empty until `Δm, Δr, Δrpy` are supplied. |
| `{U}` has a `link_openft`-equivalent URDF frame | **Does not exist yet** (§8b) — needed before Drake-based control can target the uFactory sensor's location. |
| uFactory axis convention matches `link_openft`'s (pull axis, lateral plane) | **Unverified** (§8c) — resolve with the poke test before touching `pivot_distance` or `pull_local`. |
| `pivot_distance` for the uFactory-sensor tool stack | **Unmeasured** — the DIY sensor's tuned value (`0.2`) has no reason to transfer. |
