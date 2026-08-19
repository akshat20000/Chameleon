# ADR-002: Phase 2.2 Motion Extraction Backend — MediaPipe Tasks API

**Status:** ACCEPTED  
**Date:** 2026-08-19  
**Deciders:** Chameleon Engineering  

---

## Context

The Phase 2.2 performer motion pipeline requires a tracking backend that can simultaneously extract face landmarks (with blendshapes), full-body pose, hand landmarks, and person segmentation from a single camera frame. The backend must be permissively licensed for use in a production system.

## Decision

Use MediaPipe Tasks API (v1.0.0) as the Phase 2.2 motion extraction backend with four models:

| Model | File | License |
|---|---|---|
| Face Landmarker | `face_landmarker.task` | Apache 2.0 |
| Pose Landmarker (Lite) | `pose_landmarker_lite.task` | Apache 2.0 |
| Hand Landmarker | `hand_landmarker.task` | Apache 2.0 |
| Selfie Segmentation | `selfie_multiclass_256x256.tflite` | Apache 2.0 |

All models are available from Google Cloud Storage under Apache 2.0 license and are appropriate for commercial use.

## Alternatives Considered

### Alternative 1: OpenPose
- Not real-time on CPU for full-body + hands
- GPL license — incompatible with production

### Alternative 2: ViTPose / RTMPose
- Strong body pose quality
- Does not natively provide face blendshapes or hand tracking in a unified API
- Would require composing multiple separate models

### Alternative 3: MediaPipe Legacy Python (non-Tasks API)
- Deprecated in favor of Tasks API
- Less structured output types

## Trade-offs

| Factor | MediaPipe Tasks | OpenPose | ViTPose |
|---|---|---|---|
| License | ✅ Apache 2.0 | ❌ GPL | ✅ Apache |
| Unified API | ✅ | ❌ | ❌ |
| Blendshapes | ✅ 52 ARKit | ❌ | ❌ |
| Hand landmarks | ✅ 21 pts | ❌ | ❌ |
| Body pose | ✅ 33 pts | ✅ 25 pts | ✅ 17+ pts |
| CPU performance | ⚠️ Moderate | ❌ Slow | ✅ Fast |

## Measured Performance (Phase 2.2 Benchmark)

Hardware: CPU only (Intel), image 820×400, 30 iterations.

| Stage | Mean ms | P50 ms | P95 ms |
|---|---|---|---|
| Face Landmarker | 16.9 | 16.3 | 19.9 |
| Pose Landmarker | 33.0 | 31.9 | 39.7 |
| Hand Landmarker | 20.1 | 19.9 | 22.7 |
| Segmentation | 115.0 | 113.0 | 129.7 |
| **Total** | **185.0** | **183.6** | **210.0** |

**FPS: 5.4 FPS on CPU.** Segmentation is the dominant bottleneck (62%).

## Consequences

1. The `PerformerState` schema is initially tied to MediaPipe output field shapes (478 face points, 33 body points, 21 hand points). The adapter layer isolates downstream modules from these specifics.
2. The segmentation model (`selfie_multiclass_256x256.tflite`) must be optimized or replaced in Phase 2.7.
3. SMPL-X body parameters are NOT produced by MediaPipe alone. A separate body parameter regressor will be required in Phase 2.4.
