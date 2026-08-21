# ADR-005 — Rotation Convention and Kinematic Retargeting Architecture

**Status:** Accepted  
**Phase:** 2.4D  
**Date:** 2026-08-20

---

## Context

Phase 2.4D introduces the first kinematic retargeting engine: performer joint
motion is extracted from `CanonicalMotionState` and applied to a target
`ActorSkeleton` with different body proportions.  This decision record
documents the rotation representation, the retargeting math, and the
architectural boundaries between the stages of the pipeline.

---

## Decision 1 — Why `_bone_rotation()` is not used for retargeting

`_bone_rotation()` in `mediapipe_adapter.py` produces a rotation by aligning
one vector to another using Rodrigues rotation.  This constrains only one DOF
(the bone direction) and leaves one DOF unconstrained: **twist around the bone
axis**.

Consequently, the parent-relative rotation derived as:

```
R_local = R_parent^T @ R_child
```

from two such rotations does not capture a physically meaningful joint rotation.
It conflates anatomical motion with arbitrary twist.

**Decision:** `_bone_rotation()` is not used downstream of the adapter. It remains in the adapter only because `JointState.rotation` stores a world-space alignment rotation, which is used only by the existing renderer.

> **Explicit Authority Rule:**  
> `CanonicalMotionState.JointState.rotation` is **not authoritative** for Phase 2.4D retargeting.  
> **Reason:** The retargeting engine requires complete anatomical SO(3) frames with resolved twist. These are reconstructed from stabilized landmark geometry using `AnatomicalFrameBuilder`.  
> The legacy `JointState.rotation` remains available for existing motion/debug consumers but **MUST NOT** be used as the source of retargeting rotations.

---

## Decision 2 — Anatomical frames from multiple landmarks

Each joint's SO(3) frame is built from **two** anatomically-meaningful
directions:

```
e1 = normalize(primary_direction)          # bone axis (proximal → distal)
e3 = normalize(cross(e1, secondary))       # perpendicular to plane of motion
e2 = cross(e3, e1)                         # right-hand basis completion

F_world(j) = column_stack(e1, e2, e3)     # (3,3) SO(3), no twist ambiguity
```

The secondary direction is the arm/leg plane normal, computed from:

```
arm:  cross(upper_arm, forearm)
leg:  cross(thigh, shin)
```

This completely constrains all three DOF of the joint frame.

**Producer:** `AnatomicalFrameBuilder`  
**Consumer:** `LocalRotationExtractor`  
**Not used for:** rendering (`BoneDirection` is a separate concept computed
by the renderer from joint positions)

---

## Decision 3 — Degenerate frame policy (fully extended limbs)

When `norm(cross(primary, secondary)) < DEGENERATE_THRESHOLD = 1e-4`, the
anatomical frame cannot be reliably determined (arm nearly straight).

The degeneracy condition is propagated from the **limb plane normal** to all
joints of the limb (`_arm_plane_normal()` / `_leg_plane_normal()` return a
`(normal, is_degenerate)` tuple).  When `is_degenerate=True`, the joint
immediately enters the temporal hold path — it does NOT build a frame from
an artificial canonical secondary, which would produce a discontinuous result.

Policy (in order of priority):

1. **Temporal hold:** Return `previous_valid_frame[joint]`.  Updated only when
   the cross-product norm ≥ DEGENERATE_THRESHOLD.
2. **Canonical fallback:** Used only when no previous frame exists for the joint
   (first frame of the sequence in a degenerate configuration).  Emits a
   `FrameDegenerateEvent` with `reason="canonical_fallback"`.

---

## Decision 4 — R_rest / R_motion / R_current composition contract

The retargeting math uses three distinct rotation representations:

| Symbol | Meaning | Stored in |
|--------|---------|-----------|
| `R_rest_local_performer` | Performer's neutral pose, parent-relative | Initialized by `LocalRotationExtractor` from first valid frame set |
| `R_motion_local` | Performer's deviation from its own neutral pose | `RetargetedActorState.motion_deltas` |
| `R_rest_local_actor` | Actor's neutral T-pose, parent-relative | `ActorSkeleton.rest_local_rotations` (identity for canonical T-pose) |
| `R_current_local_actor` | Actor's current orientation, parent-relative | `RetargetedActorState.local_rotations` |
| `R_world_actor` | Actor's world-space orientation | `RetargetedActorState.world_rotations` |

**Critical composition equation:**

```
R_current_local_actor(j) = R_rest_local_actor(j) @ R_motion_local(j)
```

`R_rest_local_actor` appears **exactly once** — in `KinematicRetargeter._compose()`.
It does NOT reappear in the FK position step.

**What this is NOT:**

```
R_current_local_actor = R_rest @ R_motion @ R_rest   # WRONG — double application
```

---

## Decision 5 — FK position equation

```
R_world_actor(j)  = R_world_actor(parent) @ R_current_local_actor(j)
P_actor(j)        = P_actor(parent) + R_world_actor(parent) @ v_rest(j)
```

where:

```
v_rest(j) = rest_primary_direction[j] * bone_lengths[j]
```

`v_rest(j)` is a **fixed rest-pose bone offset** in the parent-local frame.
It is derived purely from the actor's anatomy.  `R_rest_local_actor` does NOT
appear here — the rest rotation is already encoded in `R_current_local_actor`
from the composition step.

**Root joint:** `P_actor(pelvis) = (0, 0, 0)` (pose-only milestone, Phase 2.4D).

---

## Decision 6 — No root translation in Phase 2.4D

`CanonicalMotionState` normalizes every frame to `pelvis = (0, 0, 0)`.  There
is no global root trajectory stored in the canonical state.  Fabricating root
motion from normalized pelvis coordinates would produce values near zero in
every frame and test nothing meaningful.

Phase 2.4D is therefore a **pose-only milestone**.  Root motion across the
world (walking, stepping, etc.) is deferred to a future milestone that will
introduce an explicit `RootTrajectory` representation outside the normalized
joint coordinates.

---

## Decision 7 — Anatomical constraint enforcement

Constraints are applied as a diagnostic approximation, not full biomechanical
IK.  The procedure per joint:

1. Decompose `R_motion_local` via axis-angle.
2. Project axis onto `(e_flexion, e_abduction, e_twist)` derived from the
   actor's rest primary direction.
3. Clamp each component independently.
4. Reconstruct `R_motion_local_clamped = R_flex @ R_abd @ R_twist`.
5. Recompose `R_current_local = R_rest_actor @ R_motion_clamped`.
6. Emit `ConstraintViolation` if any component was clamped > 0.5°.

**Why not SLERP-based clamping:** SLERP between two complete rotations does not
enforce per-axis limits.  A SLERP cannot distinguish between flexion and
abduction — it blends the full quaternion.  The implementation must explicitly
decompose into anatomical components before clamping.

---

## Decision 8 — Three distinct representations

| Representation | Producer | Consumer | Definition |
|----------------|----------|----------|-----------|
| `BoneDirection` | renderer | renderer | Vector from parent to child joint position |
| `AnatomicalFrame` | `AnatomicalFrameBuilder` | `LocalRotationExtractor` | Complete SO(3) frame per joint |
| `JointMotionDelta` (`LocalJointRotations`) | `LocalRotationExtractor` | `KinematicRetargeter` | Parent-relative motion deviation |

These are **not interchangeable**.  In particular, `BoneDirection` must not be
mixed into the retargeting mathematics.

---

## Decision 9 — Gate G3 and geodesic invariance

Gate G3 proves the core retargeting contract:

> The retargeter changes bone lengths and rest orientations, but does not alter
> the actual joint motion.

Mathematical basis:

```
R_current_local_actor = R_rest_actor @ R_motion_local
geodesic_angle(R_rest @ A, R_rest @ B) = geodesic_angle(A, B)
  for any fixed R_rest ∈ SO(3)  (invariance under fixed left multiplication)
```

Therefore:

```
angle(R_current_local_actor(t), R_current_local_actor(t-1))
  == angle(R_motion_local_performer(t), R_motion_local_performer(t-1))
```

Tolerance: `< 0.5°` (numerical precision only — no FK-accumulation slack,
because this comparison is in local rotation space untouched by the parent chain).

---

## Verification

```
pytest services/inference/tests/test_motion_retargeting.py -v
# 30 test cases across 10 gates: A B C D E F G1 G2 G3 H
# All 30 passed on initial implementation (2026-08-20)

pytest services/inference/tests/
# 133 passed, 0 regressions (previously 103 tests)
```
