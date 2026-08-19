# ADR-001: Phase 2 Core Architecture Direction

**Status:** ACCEPTED  
**Date:** 2026-08-19  
**Deciders:** Chameleon Engineering  

---

## Context

Following the conclusion of Phase 1.7 (GhostV2 evaluation), the project must select an architectural direction for Phase 2 — a real-time full-body identity replacement system. Three fundamentally different approaches were evaluated.

The project invariants require:
- Full-body coverage (not just face)
- Explicit motion control (performer controls all motion)
- Real-time or near-real-time output
- Background preservation (real environment retained)
- Photorealistic appearance of reference identity

## Decision

**Adopt a hybrid pipeline where motion and appearance are decoupled.**

Motion is controlled by explicit tracked pose (MediaPipe Face Landmarker + Pose Landmarker + Hand Landmarker), expressed as a normalized `PerformerState`. Motion is never delegated to a generative model.

Appearance is initially driven by a parametric avatar prototype (Candidate A) to validate controllability. Generative refinement (Candidate C) is deferred to Phase 2.5 and only introduced if and when the parametric prototype demonstrates the specific visual quality failure that refinement would address.

## Alternatives Considered

### Alternative 1: Pure Generative Video Refinement (Candidate C)
Rejected at this stage because:
- No explicit motion control guarantee — generator may hallucinate pose
- Not real-time capable on available hardware (RTX 3050 / GTX 1650)
- Temporal flickering without dedicated video training
- Would immediately require training data, model selection, and GPU budget before any motion correctness is validated

### Alternative 2: Pure Neural 3D Representation (Candidate B)
Deferred because:
- Requires 300+ frames of reference video per identity (single-pass photos insufficient)
- Per-identity training takes 30–120 minutes (acceptable for offline, but requires infrastructure)
- SMPL-X dependency carries license risk
- Higher-value once motion controllability is proven

## Trade-offs

| Factor | Chosen Hybrid | Pure Generative | Pure Neural 3D |
|---|---|---|---|
| Motion guarantee | ✅ Explicit | ❌ None | ⚠️ Skeleton-driven |
| Appearance quality | ⚠️ Lower initially | ✅ Highest | ✅ High |
| Real-time potential | ✅ Good | ❌ Not yet | ✅ With 3DGS |
| Development risk | Low | High | Medium |
| Validates before refining | ✅ | ❌ | ⚠️ |

## Evidence

- Phase 1.7 demonstrated that delegating identity to a generator (GhostV2 AEI-Net) without explicit motion control produces visually unacceptable results even when the pipeline is technically correct.
- Phase 2.2 benchmark confirms MediaPipe-based motion extraction is feasible (face + pose + hands + segmentation).

## Consequences

1. SMPL-X (or equivalent) body model license must be resolved before Phase 2.4.
2. 4DHumans (CC BY-NC 4.0) cannot be used in production without replacement.
3. Phase 2.5 generative refinement introduction requires a specific, measured visual quality failure as its justification — not a general preference for quality.
