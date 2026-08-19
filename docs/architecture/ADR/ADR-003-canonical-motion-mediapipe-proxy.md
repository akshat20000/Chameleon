# ADR-003: Motion Representation Strategy — MediaPipe Keypoints Without SMPL-X

**Status:** ACCEPTED  
**Date:** 2026-08-19  
**Deciders:** Chameleon Engineering  

---

## Context

Phase 2.4 requires a body motion representation that:
1. Can drive a downstream avatar/renderer
2. Is decoupled from any specific tracker implementation
3. Does not introduce unresolved non-commercial licensing

The most physically accurate option — SMPL-X + 4DHumans as the body estimator — carries two blocking license issues:
- SMPL-X: custom non-commercial license (MPI-IS)
- 4DHumans / HMR2.0: CC BY-NC 4.0

## Decision

**Proceed with MediaPipe Pose (33 world landmarks) as the Phase 2.4 motion proxy, expressed through a `CanonicalMotionState` abstraction layer.**

SMPL-X and any CC BY-NC-licensed body estimator will NOT be integrated into the production architecture until their licensing is resolved or a permissive alternative is identified.

## Key Design Requirement

Raw MediaPipe landmark indices must NOT appear in any downstream module. The adapter (`mediapipe_adapter.py`) is the sole module allowed to reference MediaPipe-specific data structures. All avatar and rendering code consumes `CanonicalMotionState` exclusively.

```
Motion Backend (MediaPipe)
        ↓
    mediapipe_adapter.py  ← ONLY boundary where MediaPipe is referenced
        ↓
CanonicalMotionState      ← Everything downstream sees only this
        ↓
Avatar Driver / Renderer
```

This design means:

- Replacing MediaPipe with SMPL-X, RTMPose, or any other estimator requires writing a new adapter only.
- The avatar driver, renderer, compositor, and evaluation code require zero changes.

## What CanonicalMotionState Provides

| Feature | Available in v1.0.0 |
|---|---|
| Pelvis origin (root transform) | ✅ |
| Chest / spine mid | ✅ |
| Head position + rotation (SO3) | ✅ (from FaceLandmarker) |
| Shoulder, elbow, wrist positions | ✅ |
| Shoulder, elbow bone rotations | ✅ (from bone direction estimation) |
| Hip, knee, ankle positions | ✅ |
| 52 ARKit blendshapes | ✅ |
| 21-landmark hand pose (each hand) | ✅ |
| SMPL-X body shape parameters (beta) | ❌ Deferred |
| Metric-scale joint positions | ⚠️ Approximate (normalized to body height) |
| Finger joint rotations | ❌ Position only for now |

## Trade-Offs

| Factor | This Decision | SMPL-X + 4DHumans |
|---|---|---|
| License | ✅ Apache 2.0 | ❌ Blocked |
| Geometric fidelity | ⚠️ 33 keypoints, no mesh | ✅ Full mesh + shape |
| Development unblocked | ✅ Now | ❌ Until license resolved |
| Avatar driver rewrite on upgrade | ❌ Not needed (adapter only) | — |
| Hand tracking | ✅ 21 pts/hand | ⚠️ MANO but no live tracker |
| Face blendshapes | ✅ 52 ARKit | ⚠️ FLAME only |

## Future Path

When SMPL-X licensing is resolved or a permissive parametric model is identified:

1. Write `smplx_adapter.py` implementing the same `adapt_performer_state(state) -> CanonicalMotionState` interface.
2. Optionally extend `CanonicalMotionState` with a `smplx_params` field in `BodyPose`.
3. Avatar driver requires zero changes — it consumes `CanonicalMotionState` regardless of which adapter produced it.

## Validation

Phase 2.4A prototype demonstrates the pipeline end-to-end:
- MediaPipe → PerformerState → CanonicalMotionState → 2D Debug Avatar
- Validation confirms no left/right inversion, no NaN positions, orthogonal rotation matrices
- Tracking quality for test frame: confirmed in metrics.json
