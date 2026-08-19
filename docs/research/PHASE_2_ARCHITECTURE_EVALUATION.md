# Phase 2 Architecture Evaluation — Full-Body Identity Replacement

**Status:** RESEARCH  
**Date:** 2026-08-19  
**Author:** Chameleon Engineering

---

## 1. Problem Statement

The system must replace a live performer's full-body appearance with a provided reference identity while:

- preserving the performer's motion (body, hands, face expression, head orientation)
- preserving the real background environment
- rendering the reference identity's appearance (face, hair, body, hands, clothing)

This is **not** a face swap problem. It is a **pose-conditioned full-body identity rendering problem** with real-time constraints.

---

## 2. Candidate Architecture Overview

Three fundamentally different approaches exist for this class of problem. Each is evaluated below on the same set of criteria derived from project invariants.

---

## 3. Candidate A — Parametric Avatar + Neural Appearance

### Description

A parametric body model (SMPL-X or equivalent) is fitted to the performer each frame. The reference identity's appearance is modeled as a learned neural texture or neural radiance field anchored to the parametric mesh. The posed mesh is then rendered differentiably.

### Component Stack

| Stage | Component | Notes |
|---|---|---|
| Body estimation | SMPLify-X, BEV, 4DHumans, or OSX | Fits SMPL-X parameters per frame |
| Hand estimation | MANO / SMPLify-X hand component | 15 DOF per hand |
| Face estimation | FLAME or SMPL-X face component | Coupled to body model |
| Appearance | Neural texture / NeRF on mesh | Requires reference fitting pass |
| Renderer | PyTorch3D / NVDiffRast | Differentiable rasterizer |
| Compositor | Alpha matte / segmentation | Blends render into real background |

### Strengths

- **Explicit motion control.** Body pose is a compact, interpretable parameter vector. Driving from performer to rendered avatar is mathematically clean.
- **Full body support.** SMPL-X natively includes face, body, and hands in a unified 10,475-vertex mesh.
- **No generator hallucination.** The avatar is driven geometrically — the generator cannot independently "guess" the pose.
- **Decomposed pipeline.** The offline identity fitting step is cleanly separable from real-time driving.

### Weaknesses

- **Appearance realism gap.** Neural texture models trained from a small reference set tend to produce blurry or over-smoothed skin and hair.
- **Hair is not modeled.** SMPL-X has no hair geometry. Hair requires separate strand-based rendering or a learned proxy.
- **Penetration artefacts.** Self-intersecting SMPL-X meshes produce visible geometry errors at elbows, hands, and armpits.
- **Fitting latency.** Per-frame SMPL-X optimization is not real-time without a learned regressor.

### Real-Time Feasibility

**Conditional.** With a learned single-pass regressor (e.g., 4DHumans) and a fixed neural texture renderer, ≤ 100 ms per frame is plausible on a mid-range GPU. Full photorealism is not currently achievable at real-time latency.

### License Assessment

| Component | License | Commercial Use |
|---|---|---|
| SMPL-X body model | Custom non-commercial | ❌ Requires separate commercial license |
| 4DHumans / HMR2.0 | CC BY-NC 4.0 | ❌ Non-commercial only |
| PyTorch3D | BSD | ✅ |
| NVDiffRast | Apache 2.0 | ✅ |

> [!CAUTION]
> SMPL-X itself requires a license from the MPI-IS SMPL-X download page. Commercial use requires explicit agreement. **4DHumans and HMR2.0 are CC BY-NC 4.0.** These license constraints must be verified before integrating into a production system.

---

## 4. Candidate B — Animatable Neural Human Representation

### Description

The reference identity is encoded directly as a 3D or quasi-3D neural representation (animatable NeRF, 3D Gaussian Splatting, or neural point cloud) during the offline identity preparation phase. At runtime, the performer's SMPL-X pose drives the deformation of the canonical neural representation.

### Representative Methods

| Method | Year | Representation | Rendering | Notes |
|---|---|---|---|---|
| Neural Body (NeuralBody) | 2021 | NeRF anchored to SMPL | Volume rendering | Identity-specific, requires ~300 video frames |
| HumanNeRF | 2022 | Deformable NeRF | Volume rendering | Requires per-person training (~1 hour) |
| Gaussian Avatar | 2024 | 3D Gaussian Splatting | Rasterization | Faster rendering, requires video |
| SplattingAvatar | 2024 | Gaussian Splatting + SMPL | Rasterization | 40–120 FPS on RTX GPU |
| HUGS | 2024 | Unified Gaussians + SMPL | Rasterization | Strong visual quality |

### Strengths

- **Highest potential identity fidelity.** Neural representations capture fine skin textures, hair, and clothing from the original video.
- **View-consistent novel pose rendering.** When properly trained, Gaussian avatars are view-consistent and handle occlusion.
- **Rasterization path is fast.** 3DGS rendering is real-time capable (40–120 FPS) once the avatar is fitted.

### Weaknesses

- **Per-identity offline training is slow.** Neural Body requires ~8 hours. Gaussian avatars take ~30–120 minutes.
- **Requires multi-view or dense video input.** A few photos are insufficient for most current methods.
- **Out-of-distribution poses fail.** Neural representations trained from limited views cannot generalize to poses unseen during fitting.
- **SMPL dependency inherited.** Most animatable Gaussians still require SMPL as a deformation skeleton.

### Real-Time Feasibility

**Conditional on offline training.** 3DGS-based avatars render at 40+ FPS on an RTX 3080 once trained. The bottleneck is per-identity fitting time, not inference time.

### License Assessment

| Component | License | Notes |
|---|---|---|
| HumanNeRF | MIT | ✅ |
| Gaussian Avatar | MIT | ✅ |
| HUGS | CC BY-NC 4.0 | ❌ Non-commercial |
| SplattingAvatar | MIT | ✅ |
| gaussian-splatting (Inria) | Research-only | ❌ Check terms carefully |

> [!WARNING]
> The original 3D Gaussian Splatting paper from Inria carries a research-only license. Downstream methods differ — check each repository individually.

---

## 5. Candidate C — Pose-Guided Generative Video Refinement

### Description

A pose-conditioned generative model (diffusion or flow-based) synthesizes the reference identity in the performer's pose directly from a pose signal and reference images. No explicit 3D avatar is constructed. The generator learns to map (pose, reference images) → synthesized frame.

### Representative Methods

| Method | Year | Conditioning | Backbone | Notes |
|---|---|---|---|---|
| ControlNet | 2023 | OpenPose skeleton | SD UNet | Face only; body skeleton |
| IP-Adapter | 2023 | CLIP/ViT reference image | SD UNet | Style/identity injection |
| AnimateAnyone | 2024 | DWPose + reference | AnimateDiff | Full-body video |
| Champ | 2024 | SMPL-X + rendering | AnimateDiff | Full-body with 3D guidance |
| MimicMotion | 2024 | DWPose confidence | SVD | Temporal coherence |
| Follow-Your-Emoji | 2024 | Facial landmarks | AnimateDiff | Face expression transfer |
| CogVideoX | 2024 | Prompt + reference | Video DiT | General video generation |

### Strengths

- **Best visual realism on in-distribution cases.** Generative refinement can produce photorealistic skin, hair, and clothing.
- **Handles hair naturally.** Unlike parametric models, the generator can synthesize complex hair from reference images.
- **Handles novel clothing.** The generator learns to reproduce the reference outfit from multi-view images.

### Weaknesses

- **Motion fidelity is architectural.** Without explicit 3D pose control, the generator may produce temporally incoherent or drifting results.
- **Temporal flickering.** Diffusion-based video models struggle with temporal consistency unless explicitly trained for it (SVD, AnimateDiff).
- **Cannot guarantee pose accuracy.** The generator may produce a "plausible" pose rather than the exact measured performer pose.
- **Latency.** A full denoising pass for a video clip (e.g., 25 steps) takes 5–30 seconds on a consumer GPU. Not real-time without significant optimization (distillation, few-step methods).
- **Identity leakage.** Cross-attention identity injection does not always isolate identity from pose — the generator can drift toward the training distribution.

### Real-Time Feasibility

**Not currently feasible for real-time.** Video diffusion models require multiple denoising steps. Recent consistency models and flow-based methods reduce this, but 30+ FPS on consumer hardware is not yet achievable without distillation/quantization engineering.

### License Assessment

| Component | License | Notes |
|---|---|---|
| AnimateAnyone | Restricted/custom | ❌ Code not fully released |
| Champ | Apache 2.0 | ✅ |
| MimicMotion | Apache 2.0 | ✅ |
| IP-Adapter | Apache 2.0 | ✅ |
| ControlNet | Apache 2.0 | ✅ |
| CogVideoX | Apache 2.0 | ✅ |

---

## 6. Comparative Evaluation Matrix

| Criterion | Candidate A (Parametric) | Candidate B (Neural 3D) | Candidate C (Generative) |
|---|---|---|---|
| **Full body support** | ✅ Explicit | ✅ Learned | ⚠️ Implicit |
| **Hand support** | ✅ MANO | ✅ Learned | ⚠️ Often absent |
| **Face support** | ✅ FLAME | ✅ Learned | ✅ Strong |
| **Hair support** | ❌ Not modeled | ✅ Learned | ✅ Strong |
| **Motion controllability** | ✅ Explicit parameters | ⚠️ Skeleton-driven | ⚠️ Pose-guided, not guaranteed |
| **Identity fidelity** | ⚠️ Neural texture quality | ✅ Highest | ✅ High on simple poses |
| **Real-time potential** | ✅ With regressor | ✅ With 3DGS | ❌ Not currently |
| **Per-identity offline cost** | Moderate (regressor fitting) | High (30–120 min training) | Low–moderate |
| **Input requirements** | 1–few photos | 300+ frames of video | 5–10 photos |
| **License compatibility** | ⚠️ SMPL-X license risk | ⚠️ Varies per method | ✅ Mostly permissive |
| **Temporal stability** | ✅ Parametric smoothing | ✅ Canonical space | ❌ Requires extra effort |
| **Background compositing** | ✅ Mask + render | ✅ Mask + render | ⚠️ Requires inpainting/mask |

---

## 7. Recommended Development Approach

> [!IMPORTANT]
> **No single candidate is the final answer.** The recommended approach is a **hybrid** that decouples motion from appearance:
>
> 1. **Motion Representation:** Use explicit tracked pose (MediaPipe + body pose estimation) as the authoritative motion signal. Never delegate motion control to a generator.
> 2. **Prototype with Candidate A** (parametric skeleton driving) to validate controllability before committing to a neural appearance model.
> 3. **Appearance refinement with Candidate C** only _after_ motion controllability is validated, using the parametric render as a conditioning signal.

### Recommended Phase Order

```
Phase 2.2 — Motion pipeline (MediaPipe face + body)
Phase 2.3 — Identity asset construction
Phase 2.4 — Parametric skeleton driving prototype (Candidate A motion control)
Phase 2.5 — Evaluate appearance failures → decide refinement path
Phase 2.6 — Generative refinement if appearance fails (Candidate C)
```

---

## 8. Open Questions / Stop Conditions

1. **SMPL-X license:** Must confirm whether SMPL-X or a fully permissive alternative (e.g., STAR, SMPL-A, or MediaPipe Pose alone) can replace SMPL-X in a commercial context.
2. **Full-body pose regressor:** 4DHumans is the strongest open regressor but CC BY-NC 4.0. Alternatives (ViTPose, RTMPose) must be evaluated for SMPL compatibility.
3. **Hand tracking:** MediaPipe Hands provides 21 3D landmarks per hand. Whether this is sufficient to drive MANO-level hand appearance must be tested empirically.
4. **Temporal stability of Gaussian avatars:** SplattingAvatar requires a stable per-frame SMPL estimate; temporal jitter in the pose estimate propagates into the rendered avatar.
