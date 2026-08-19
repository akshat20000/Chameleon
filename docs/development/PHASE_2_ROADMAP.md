# Phase 2 Development Roadmap

**Version:** 1.0.0  
**Status:** IN PROGRESS — Phase 2.2 Motion Extraction Prototype COMPLETE  
**Date:** 2026-08-19  
**Pre-Research Tag:** `pre-research-baseline`

---

## Phase 2.1 — Architecture Research and Validation ✅

**Objective:** Evaluate candidate architectures from first principles before committing to implementation.

**Deliverables:**
- [x] Architecture comparison document: [`docs/research/PHASE_2_ARCHITECTURE_EVALUATION.md`](../research/PHASE_2_ARCHITECTURE_EVALUATION.md)
- [x] PerformerState specification: [`docs/architecture/PERFORMER_STATE.md`](PERFORMER_STATE.md)
- [x] IdentityAsset specification: [`docs/architecture/IDENTITY_ASSET.md`](IDENTITY_ASSET.md)

**Conclusion:** Three candidate architectures evaluated (Parametric Avatar, Neural 3D, Generative Refinement). Hybrid recommended: explicit pose control from Candidate A driving appearance from Candidate C. Full evaluation in [`PHASE_2_ARCHITECTURE_EVALUATION.md`](../research/PHASE_2_ARCHITECTURE_EVALUATION.md).

**License stop conditions identified:**
- SMPL-X: custom non-commercial license. Must find permissive alternative or obtain commercial license.
- 4DHumans / HMR2.0: CC BY-NC 4.0. Cannot be used in production without replacement.

---

## Phase 2.2 — Performer Motion Extraction Pipeline ✅

**Objective:** Build and benchmark a normalized, backend-agnostic motion extraction pipeline covering face, body pose, hands, and segmentation.

**Deliverables:**
- [x] Motion extraction prototype: [`services/inference/scripts/motion_extraction_prototype.py`](../../services/inference/scripts/motion_extraction_prototype.py)
- [x] Visual debug overlay: `test_data/phase2_motion/debug_overlay.png`
- [x] Benchmark report: `test_data/phase2_motion/benchmark_report.json`
- [x] PerformerState schema specification

**New MediaPipe Models Downloaded:**
- `services/inference/models/pose_landmarker_lite.task` (5.5 MB) — 33-point body pose
- `services/inference/models/hand_landmarker.task` (7.5 MB) — 21-point per-hand landmarks

**Benchmark Results (CPU, 820×400, 30 iterations):**

| Stage | Mean ms | P50 ms | P95 ms |
|---|---|---|---|
| Face Landmarker (478 pts, 52 blendshapes, head matrix) | **16.9** | 16.3 | 19.9 |
| Pose Landmarker (33 body keypoints, 2 people) | **33.0** | 31.9 | 39.7 |
| Hand Landmarker (21 pts/hand, up to 2 hands) | **20.1** | 19.9 | 22.7 |
| Segmentation (6-class multiclass) | **115.0** | 113.0 | 129.7 |
| **Total Pipeline** | **185.0** | **183.6** | **210.0** |
| **Estimated FPS** | **5.4 FPS** | — | — |

**Extraction Validation Results (test_data/2face_validation.png):**
- Faces detected: 2 / 2 ✅
- Body poses detected: 2 / 2 ✅
- Hands detected: 1 ✅
- Blendshapes per face: 52 ✅
- Head rotation: pitch=3.43°, yaw=4.18°, roll=5.68° ✅

### Critical Finding: Segmentation is the Bottleneck

The `selfie_multiclass_256x256.tflite` model accounts for **62% of total pipeline latency (115 ms / 185 ms)**. This is the primary optimization target before any real-time work.

Options to investigate in Phase 2.7:
1. Run segmentation at reduced resolution or lower frame rate (every N frames)
2. Evaluate faster segmentation alternatives (MediaPipe SelfieSegmentation — single class, faster)
3. Use pose-derived bounding box to restrict segmentation to the performer region only

---

## Phase 2.3 — Reference Identity Preparation Pipeline 🔲

**Objective:** Build an offline pipeline that takes reference photos or video as input and produces a versioned, validated `IdentityAsset`.

**Required Deliverables:**
- [ ] `services/inference/app/identity/` — extend with multi-view embedding extraction
- [ ] `services/inference/scripts/prepare_identity.py` — offline identity preparation CLI
- [ ] `IdentityAsset` serialization and validation implementation
- [ ] Intermediate artifact workspace structure (see `IDENTITY_ASSET.md`)

**Entry Gate:** Phase 2.2 benchmark must pass. ✅

**Exit Gate:**
- Given ≥ 3 reference photos, produces a valid `IdentityAsset` with face embedding, hair mask, and clothing mask.
- `IdentityAsset.is_valid` returns `True`.
- SHA-256 checksums for all components are stored in `manifest.json`.

---

## Phase 2.4 — Minimal Avatar Driving Prototype 🔲

**Objective:** Drive a simplified representation of the reference identity using live performer pose. Photorealism is NOT required at this stage. Controllability IS required.

**Success Criteria (must all pass before proceeding to Phase 2.5):**
- Head follows performer head direction
- Body skeleton orientation matches performer
- Arms track performer arms
- Output changes detectably when performer changes pose
- No crash on extreme angles or fast motion

**Architecture Decision Pending:**
- Body pose regressor choice (SMPL-X license vs. alternatives)
- Rendering backend (PyTorch3D / NVDiffRast / rasterization)

---

## Phase 2.5 — Appearance and Identity Fidelity 🔲

**Objective:** Improve visual realism after motion controllability is demonstrated.

**Gate:** Phase 2.4 controllability must pass.

**Components to improve (measure each failure before fixing):**
- Facial identity fidelity (AdaFace CosSim ≥ 0.85 vs reference)
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

## Phase 2.6 — Background Compositing 🔲

**Objective:** Composite the rendered identity into the real performer background with clean boundaries.

**Key challenges to measure:**
- Hair boundary quality at performer silhouette
- Lighting mismatch between rendered identity and real background
- Shadow and occlusion handling
- Motion blur at boundary at high performer speed

---

## Phase 2.7 — Real-Time Optimization 🔲

**Objective:** Reduce per-frame pipeline latency to enable real-time or near-real-time output.

**Gate:** Phases 2.4–2.6 must demonstrate correct output.

**Primary target:** Segmentation stage (115 ms → target < 20 ms).

**Measurement protocol:**
- Report mean, P50, P95, P99, FPS, VRAM, GPU utilization for every stage independently.
- Do not report only average FPS.

---

## Stop Conditions

Stop and report to the project owner immediately if:

1. A proposed architecture cannot support full-body motion (not just face).
2. A dependency license conflicts with project requirements (e.g., SMPL-X commercial use).
3. A tracking backend is too temporally unstable to drive downstream rendering.
4. GPU memory requirements exceed 4 GB without an offline alternative.
5. A custom training proposal lacks training data, ground truth, or measurable evaluation criteria.
