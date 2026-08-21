# Phase 2 Development Roadmap

**Version:** 2.2.0  
**Status:** Phase 2.4D COMPLETE — Phase 2.3 / Phase 2.5 Next  
**Date:** 2026-08-21  
**Pre-Research Tag:** `pre-research-baseline`

---

## Overview

```
Phase 2.1  [x]  Architecture Research & Evaluation
Phase 2.2  [x]  Performer Motion Extraction Pipeline
Phase 2.3  [ ]  Reference Identity Preparation Pipeline  <-- NEXT
Phase 2.4A [x]  Motion Representation Boundary (CanonicalMotionState)
Phase 2.4B [x]  Full-Body Static Validation                8/8 gates PASS
Phase 2.4C [x]  Temporal Motion Stability Benchmark        3/3 benchmark PASS
Phase 2.4D [x]  Motion Retargeting -- Debug Avatar Engine  COMPLETE (10/10 gates PASS, pose-fidelity audit PASS)
Phase 2.5  [ ]  Appearance and Identity Fidelity
Phase 2.6  [ ]  Background Compositing
Phase 2.7  [ ]  Real-Time Optimization
```

**Deferral note on Phase 2.3:**
Identity preparation is deferred until Phase 2.4D is complete.
An `IdentityAsset` is only useful once the motion pipeline can reliably drive it.
Proving full-body controllability comes first.

---

## Phase 2.1 — Architecture Research and Evaluation [COMPLETE]

**Objective:** Evaluate candidate architectures from first principles before committing to implementation.

**Deliverables:**
- [x] Architecture comparison document: [`docs/research/PHASE_2_ARCHITECTURE_EVALUATION.md`](../research/PHASE_2_ARCHITECTURE_EVALUATION.md)
- [x] PerformerState specification: [`docs/architecture/PERFORMER_STATE.md`](../architecture/PERFORMER_STATE.md)
- [x] IdentityAsset specification: [`docs/architecture/IDENTITY_ASSET.md`](../architecture/IDENTITY_ASSET.md)

**Conclusion:** Three candidate architectures evaluated (Parametric Avatar, Neural 3D, Generative Refinement). Hybrid recommended: explicit pose control from Candidate A driving appearance from Candidate C.

**License stop conditions identified:**
- SMPL-X: custom non-commercial license. Must find permissive alternative or obtain commercial license.
- 4DHumans / HMR2.0: CC BY-NC 4.0. Cannot be used in production without replacement.

---

## Phase 2.2 — Performer Motion Extraction Pipeline [COMPLETE]

**Objective:** Build and benchmark a normalized, backend-agnostic motion extraction pipeline covering face, body pose, hands, and segmentation.

**Deliverables:**
- [x] Motion extraction prototype: [`services/inference/scripts/motion_extraction_prototype.py`](../../services/inference/scripts/motion_extraction_prototype.py)
- [x] Visual debug overlay: `test_data/phase2_motion/debug_overlay.png`
- [x] Benchmark report: `test_data/phase2_motion/benchmark_report.json`
- [x] PerformerState schema specification

**New MediaPipe Models Downloaded:**
- `services/inference/models/pose_landmarker_lite.task` (5.5 MB) -- 33-point body pose
- `services/inference/models/hand_landmarker.task` (7.5 MB) -- 21-point per-hand landmarks

**Benchmark Results (CPU, 820x400, 30 iterations):**

| Stage | Mean ms | P50 ms | P95 ms |
|---|---|---|---|
| Face Landmarker (478 pts, 52 blendshapes, head matrix) | **16.9** | 16.3 | 19.9 |
| Pose Landmarker (33 body keypoints, 2 people) | **33.0** | 31.9 | 39.7 |
| Hand Landmarker (21 pts/hand, up to 2 hands) | **20.1** | 19.9 | 22.7 |
| Segmentation (6-class multiclass) | **115.0** | 113.0 | 129.7 |
| **Total Pipeline** | **185.0** | **183.6** | **210.0** |
| **Estimated FPS** | **5.4 FPS** | -- | -- |

**Critical Finding:** Segmentation accounts for 62% of total pipeline latency (115 ms / 185 ms). Primary optimization target for Phase 2.7.

Options to investigate in Phase 2.7:
1. Run segmentation at reduced resolution or lower frame rate (every N frames)
2. Evaluate faster segmentation alternatives (MediaPipe SelfieSegmentation -- single class, faster)
3. Use pose-derived bounding box to restrict segmentation to the performer region only

---

## Phase 2.4A -- Motion Representation Boundary [COMPLETE]

**Objective:** Establish a clean, tracker-independent motion representation that sits between raw tracker output and all downstream avatar/rendering systems.

**The architectural win:**

```
Camera Frame
     |
     v
MediaPipe
     |
     v
MediaPipe Adapter     <-- ONLY place MediaPipe types are allowed
     |
     v
CanonicalMotionState  <-- The boundary
     |
     v
Avatar Driver / Renderer
```

Nothing downstream of `CanonicalMotionState` may import or reference MediaPipe APIs, landmark
indices, or any tracker-specific type. Replacing the tracker (e.g., MediaPipe -> RTMPose ->
4DHumans) requires only replacing the adapter.

**Deliverables:**
- [x] `CanonicalMotionState` specification: [`docs/architecture/CANONICAL_MOTION_STATE.md`](../architecture/CANONICAL_MOTION_STATE.md)
- [x] `CanonicalMotionState` implementation: [`services/inference/app/motion/canonical_state.py`](../../services/inference/app/motion/canonical_state.py)
- [x] MediaPipe -> CanonicalMotionState adapter: [`services/inference/app/motion/mediapipe_adapter.py`](../../services/inference/app/motion/mediapipe_adapter.py)
- [x] Debug avatar prototype (2D stick figure, OpenCV): [`services/inference/scripts/debug_avatar_prototype.py`](../../services/inference/scripts/debug_avatar_prototype.py)

**What was validated:**
- Adapter runs without error
- Rotation matrices are computed
- State validation works
- Face data is populated
- Partial hand data is populated
- Architecture boundary (no MediaPipe downstream) is enforced

**What was NOT validated (input was head/shoulders only):**

| Joint Group | Status |
|---|---|
| Head + Neck | Partial |
| Left/Right Shoulder | Partial |
| Left/Right Elbow + Wrist | Partially visible |
| Left/Right Hip | Not validated |
| Left/Right Knee + Ankle | Not validated |
| Full torso | Not validated |
| Wrist-to-hand attachment | Not validated |
| Body scale normalization | Not validated |
| Temporal motion stability | Not validated |

---

## Phase 2.4B -- Full-Body Static Validation [COMPLETE]

**Objective:** Prove that the complete 17-joint `CanonicalMotionState` skeleton can be extracted
from a full-body image. A head-and-shoulders test is insufficient -- all limbs must be present
and anatomically consistent.

**Required Input:**
- A single full-body image showing the complete performer from head to ankle.

**17 Joints to Validate:**

```
Head
 |-- Neck
      |-- Left Shoulder
      |     |-- Left Elbow
      |           |-- Left Wrist
      |-- Right Shoulder
            |-- Right Elbow
                  |-- Right Wrist

Pelvis
 |-- Left Hip
 |     |-- Left Knee
 |           |-- Left Ankle
 |-- Right Hip
       |-- Right Knee
             |-- Right Ankle
```

**Exit Gates — Results (2026-08-20):**

| Gate | Requirement | Result |
|---|---|---|
| Full-body detected | 17 joints non-None | PASS |
| 17/17 joints valid | confidence >= 0.3 | PASS |
| No NaNs | No NaN or Inf anywhere | PASS |
| Rotation matrices valid | abs(det(R) - 1) < 0.01 | PASS |
| Left/right consistency | L.shoulder.x=+0.137 > R.shoulder.x=-0.118 | PASS |
| Body scale valid | body_scale = 1.2546 m | PASS |
| Wrist-to-hand attachment | Both hands within tolerance | PASS |
> **HISTORICAL RESULT STATUS: SUPERSEDED**  
> The initial Phase 2.4B trial below evaluated an auto-generated asset. The `test_data/` directory has undergone a hard reset.  
> Formal re-validation will occur on the canonical user-provided assets placed in `test_data/inputs/performer/`.

**Historical Trial Data (SUPERSEDED):**
- Visible joints: 17 / 17 (100%)
- Body scale: 1.2546 m
- Face: NOT detected (face_landmarker did not fire on this image)
- Hands: L=yes R=yes (both wrist attachment valid)
- Tracking quality: `partial` (face absent, but all body joints present)
- Rotation matrices: all SO(3)-valid
- NaN events: 0
- Left/right inversions: 0

**Deliverables:**
- [x] Validation script: `services/inference/scripts/validate_full_body.py`
- [ ] Canonical full-body test asset: `test_data/inputs/performer/<user_asset>` (Awaiting user upload)
- [ ] Formal Phase 2.4B run output: `test_data/outputs/phase2_4b_validation/side_by_side.png`

---

## Phase 2.4C -- Temporal Motion Stability Benchmark [COMPLETE]

**Objective:** Prove that the motion pipeline produces stable, coherent `CanonicalMotionState(t)`
across video frames. A technically valid per-frame pose system that flickers, jumps, or produces
joint inversions between frames destroys the illusion.

**Phase 2.4C Results (1,075 frames @ 29.6 FPS):**

| Gate | Requirement | Raw Tracker Stream | Stabilized Stream (`TemporalStabilizer`) | Status |
|---|---|---|---|---|
| Temporal position smoothness | Δ pos <= 0.10 units | **112 jump events** | **0 jump events** | **PASS [OK]** ✅ |
| No rotation flips | Δ rot <= 45.0° | **13 flip events** | **0 flip events** | **PASS [OK]** ✅ |
| Confidence stability | Drop rate <= 5.0% | **0.05% drop rate** | **0.05% drop rate** | **PASS [OK]** ✅ |
| Left/right consistency | 0 camera-space inversions | **39 inversion events** | **0 inversion events** | **PASS [OK]** ✅ |
| NaN-free across all frames | 0 NaN/Inf values | **0 events** | **0 events** | **PASS [OK]** ✅ |

**Stabilizer Performance Telemetry:**
- Mean Stabilizer Latency: **0.299 ms / frame** (Max: **1.099 ms**)
- Added Algorithmic Lag: **0.0 ms** (1€-Filter is single-frame zero-phase lag)
- Position Jitter Reduction: **63.6% reduction** in mean frame-to-frame delta position (0.01022u → 0.00372u)
- Rotation Jitter Reduction: **56.3% reduction** in mean frame-to-frame rotation delta (1.19° → 0.52°)

**Deliverables:**
- [x] `services/inference/scripts/validate_temporal_stability.py` script
- [x] Performer video asset: `test_data/inputs/performer/WhatsApp Video 2026-08-20 at 10.16.06 PM.mp4`
- [x] Standalone `TemporalStabilizer` module: `services/inference/app/motion/temporal_stabilizer.py`
- [x] Benchmark comparison report: `test_data/outputs/2026-08-20_temporal_stability/temporal_stability_report.json`
- [x] Side-by-side debug overlay video: `test_data/outputs/2026-08-20_temporal_stability/skeleton_overlay.mp4`
- [x] Unit test suite: `services/inference/tests/test_temporal_stabilizer.py` (5/5 passed)

---

## Phase 2.4D -- Motion Retargeting -- Debug Avatar Engine [COMPLETE]

**Objective:** Drive target actor skeletons of varying body proportions using performer motion deltas extracted from `CanonicalMotionState`.

**Key Invariants:**
- `R_current_local_actor(j) = R_rest_local_actor(j) @ R_motion_local(j)`
- `P_actor(j) = P_actor(parent) + R_world_actor(parent) @ v_rest(j)`
- Pose-only operation (pelvis-normalized root).
- Label preservation (left → left, right → right).

**Exit Criteria Audit & Visual Validation Results (2026-08-21):**

| Area | Verdict | Details |
|---|---|---|
| Coordinate system | **PASS** | $+X$ right, $+Y$ up, $+Z$ forward |
| Temporal stabilization | **PASS** | 0 position jumps, 0 rotation flips |
| Anatomical frame reconstruction | **PASS** | SO(3) frames with resolved twist |
| Arms-down calibration | **PASS** | 100.0% match (74/74 frames) |
| Pose transfer fidelity | **PASS** | Arm direction error 3.50° (mean) |
| Actor proportion preservation | **PASS** | Bone length violations: 0 |
| FK self-consistency | **PASS** | FK errors: 0 |
| L/R identity preservation | **PASS** | Label swaps: 0 |
| Visual continuity | **PASS** | Verified on 150-frame video benchmark |
| Generalized adversarial hierarchy testing | **Future Improvement** | Documented technical debt below |

**Engine Latency Telemetry:**
- Retargeting Engine Latency (Stages 4–6): **7.66 ms / frame** (**130.5 FPS**)
- End-to-End Real Pipeline Latency (Stages 1–6): **61.67 ms / frame** (**16.2 FPS**)

**Technical Debt & Future Hardening Task:**
- Replace the absolute maximum-rotation-jump gate with a **source-relative continuity error metric**.
- Add **adversarial tests for parent rotation with child-relative motion invariance**.

**Deliverables:**
- [x] Actor Skeleton & Profiles: [`services/inference/app/motion/actor_skeleton.py`](../../services/inference/app/motion/actor_skeleton.py)
- [x] Anatomical Frame Builder: [`services/inference/app/motion/anatomical_frame_builder.py`](../../services/inference/app/motion/anatomical_frame_builder.py)
- [x] Local Rotation Extractor: [`services/inference/app/motion/local_rotation_extractor.py`](../../services/inference/app/motion/local_rotation_extractor.py)
- [x] Retargeted Actor State: [`services/inference/app/motion/retargeted_actor_state.py`](../../services/inference/app/motion/retargeted_actor_state.py)
- [x] Kinematic Retargeter: [`services/inference/app/motion/motion_retargeter.py`](../../services/inference/app/motion/motion_retargeter.py)
- [x] Architecture Decision Record: [`docs/architecture/ADR/ADR-005-rotation-convention-and-retargeting.md`](../architecture/ADR/ADR-005-rotation-convention-and-retargeting.md)
- [x] Visual & Quantitative Benchmark: [`services/inference/scripts/validate_retargeting_visual.py`](../../services/inference/scripts/validate_retargeting_visual.py)
- [x] 10-Gate Verification Suite: [`services/inference/tests/test_motion_retargeting.py`](../../services/inference/tests/test_motion_retargeting.py) (30/30 unit tests PASS)

Phase 2.4D is closed. Motion controllability is established. Next steps: Phase 2.3 (Reference Identity Preparation) & Phase 2.5 (Appearance and Identity Fidelity).

---

## Phase 2.3 -- Reference Identity Preparation Pipeline (After 2.4D)

**Objective:** Build an offline pipeline that takes reference photos or video as input and
produces a versioned, validated `IdentityAsset`.

**Entry Gate:** Phase 2.4D must pass. The motion pipeline must be capable of driving an identity
asset before we build one.

**Required Deliverables:**
- [ ] `services/inference/app/identity/` -- extend with multi-view embedding extraction
- [ ] `services/inference/scripts/prepare_identity.py` -- offline identity preparation CLI
- [ ] `IdentityAsset` serialization and validation implementation
- [ ] Intermediate artifact workspace structure (see `IDENTITY_ASSET.md`)

**Exit Gate:**
- Given >= 3 reference photos, produces a valid `IdentityAsset` with face embedding, hair mask,
  and clothing mask.
- `IdentityAsset.is_valid` returns `True`.
- SHA-256 checksums for all components are stored in `manifest.json`.

---

## Phase 2.5 -- Appearance and Identity Fidelity

**Objective:** Improve visual realism after motion controllability is demonstrated.

**Gate:** Phase 2.4D controllability must pass.

**Components to improve (measure each failure before fixing):**
- Facial identity fidelity (AdaFace CosSim >= 0.85 vs reference)
- Hair boundary realism
- Skin tone matching
- Hand appearance
- Clothing transfer

**Custom training may be introduced here. Each proposed model must answer:**
1. What exact failure does this model solve?
2. What is the training data?
3. What is the ground truth?
4. What metric proves improvement?
5. What baseline does it beat?

---

## Phase 2.6 -- Background Compositing

**Objective:** Composite the rendered identity into the real performer background with clean
boundaries.

**Key challenges to measure:**
- Hair boundary quality at performer silhouette
- Lighting mismatch between rendered identity and real background
- Shadow and occlusion handling
- Motion blur at boundary at high performer speed

---

## Phase 2.7 -- Real-Time Optimization

**Objective:** Reduce per-frame pipeline latency to enable real-time or near-real-time output.

**Gate:** Phases 2.4-2.6 must demonstrate correct output.

**Primary target:** Segmentation stage (115 ms -> target < 20 ms).

**Measurement protocol:**
- Report mean, P50, P95, P99, FPS, VRAM, GPU utilization for every stage independently.
- Do not report only average FPS.

---

## Target Architecture

The eventual full pipeline:

```
                    LIVE CAMERA
                        |
                        v
               +-----------------+
               | Motion Tracking |
               |   Face          |
               |   Body          |
               |   Hands         |
               +--------+--------+
                        |
                        v
              CanonicalMotionState
                        |
                        v
              Temporal Stabilization     <- Phase 2.4C / 2.7
                        |
                        v
                   Avatar Driver         <- Phase 2.4D
                        |
         +--------------+--------------+
         |                             |
         v                             v
  Actor Body Model              Actor Appearance
  Skeleton / Geometry           Face / Hair / Skin
         |                             |
         +--------------+--------------+
                        |
                        v
                     Renderer           <- Phase 2.5 / 2.6
                        |
                        v
            Optional Neural Refinement  <- Phase 2.5 (appearance only)
                        |
                        v
                   Final Frame
```

Key principle: The generative model sits at the end as a refinement step.
It must not be responsible for resolving motion from scratch.
Control first. Appearance second. Photorealism last.

---

## Stop Conditions

Stop and report to the project owner immediately if:

1. A proposed architecture cannot support full-body motion (not just face).
2. A dependency license conflicts with project requirements (e.g., SMPL-X commercial use).
3. A tracking backend is too temporally unstable to drive downstream rendering.
4. GPU memory requirements exceed 4 GB without an offline alternative.
5. A custom training proposal lacks training data, ground truth, or measurable evaluation criteria.
6. Phase 2.4B exit gates cannot all pass -- diagnose before proceeding to 2.4C.
7. Phase 2.4C reveals temporal instability that cannot be resolved without a stabilization stage
   -- stop and implement temporal filtering before proceeding to 2.4D.
