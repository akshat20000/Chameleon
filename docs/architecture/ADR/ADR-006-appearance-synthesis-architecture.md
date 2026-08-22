# ADR-006: Appearance Synthesis Architecture & Contracts

**Status:** APPROVED  
**Date:** 2026-08-21  
**Deciders:** Core Architecture Team  
**Replaces:** N/A (Establishes Phase 2.5 Architecture)

---

## 1. Context and Problem Statement

Phase 2.4D established a motion retargeting engine that transforms performer motion into `RetargetedActorState` (canonical 3D joint positions and $SO(3)$ anatomical orientations). Phase 2.3 established the offline `IdentityAsset` pipeline (512-D ArcFace embedding, selected reference views, quality metadata, segmentation masks, and manifest provenance).

Phase 2.5 addresses the synthesis of a visually identity-consistent character frame driven by `RetargetedActorState` and `IdentityAsset`. To avoid locking the project prematurely into a complex neural model (e.g. LivePortrait, ControlNet, IP-Adapter) before proving that conditioning contracts and metric evaluation harnesses are sound, Phase 2.5 is split into two distinct sub-phases:

- **Phase 2.5A:** Appearance Conditioning Infrastructure, Immutability Guarantees, Deterministic Baseline Synthesizer, and Metric Calibration Harness.
- **Phase 2.5B:** Candidate Synthesis Backend Evaluation & Benchmarking.

---

## 2. Decision Outcomes & Invariants

### 2.1 Immutable Kinematic Boundary Guarantee

> [!IMPORTANT]
> **Read-Only Kinematic Invariant:**
> `RetargetedActorState` is strictly **read-only** to all Phase 2.5 modules.
>
> Appearance synthesis MAY consume joint positions, joint orientations, actor proportions, and projected 2D keypoints.
> Appearance synthesis MUST NOT:
> 1. Modify joint positions or bone lengths.
> 2. Modify actor hierarchy or kinematic transforms.
> 3. Feed pose corrections back into the Phase 2.4D retargeting engine.

### 2.2 Conditioning & Temporal State Schema

```python
@dataclass
class AppearanceTemporalState:
    previous_frame_bgr: Optional[np.ndarray] = None
    previous_synthetic_result: Optional[Any] = None
    optical_flow: Optional[np.ndarray] = None
    frame_index: int = 0


@dataclass
class AppearanceConditioningState:
    identity_embedding: np.ndarray          # (512,) float32 ArcFace vector
    reference_views: List[Any]              # List[SegmentedReferenceView]
    pose_map_2d: np.ndarray                 # Rendered 2D skeletal line map (H, W, 3)
    keypoints_2d: np.ndarray                # (N, 2) float32 normalized keypoints
    joint_confidence: np.ndarray            # (N,) float32 joint confidence
    camera_parameters: Optional[Dict] = None
    region_guidance: Dict[str, np.ndarray] = field(default_factory=dict)
    frame_index: int = 0
    timestamp_s: float = 0.0
```

### 2.3 Synthesizer Interface & Diagnostic Result

```python
@dataclass
class SyntheticFrameResult:
    frame_bgr: np.ndarray                   # Synthesized output RGB/BGR frame
    synthetic_mask: Optional[np.ndarray]    # Subject mask uint8 (0/255)
    latency_ms: float
    metadata: Dict                          # Backend provenance & telemetry
    valid: bool = True
    warnings: List[str] = field(default_factory=list)


class BaseAppearanceSynthesizer(ABC):
    @abstractmethod
    def synthesize_frame(
        self,
        conditioning: AppearanceConditioningState,
        temporal_state: Optional[AppearanceTemporalState] = None,
        background_bgr: Optional[np.ndarray] = None,
    ) -> SyntheticFrameResult:
        pass
```

### 2.4 Baseline Articulated Renderer Transformation Policy

The deterministic `BaselineArticulatedSynthesizer` maps reference view patches to target keypoints:
- **Articulated Limbs:** Transformed using **2D Similarity Transformations** (translation, rotation, uniform scale) to prevent shear distortion.
- **Torso / Clothing:** Transformed using **2D Affine Transformations** to accommodate non-rigid clothing deformation.
- **Region Fallback:** Respects Phase 2.3 segmentation availability (`face`, `hair`, `clothing`, `body`). Missing optional masks produce diagnostic warnings in `SyntheticFrameResult.warnings` without crashing synthesis.

---

## 3. Evaluation Metric Protocols & Semantics

1. **Symmetric Body Scale Normalized Keypoint Error ($\text{NKE}_{\text{body}}$):**
   $$\text{NKE}_{\text{body}} = \frac{1}{K} \sum_{k=1}^{K} \frac{\|\mathbf{p}_k^{\text{gen}} - \mathbf{p}_k^{\text{target}}\|}{d_{\text{body}}}$$
   where $d_{\text{body}} = \frac{1}{2}\left(\|\mathbf{p}_{\text{left\_shoulder}} - \mathbf{p}_{\text{left\_hip}}\| + \|\mathbf{p}_{\text{right\_shoulder}} - \mathbf{p}_{\text{right\_hip}}\|\right)$.
   *Degenerate Safety:* If $d_{\text{body}} < 10^{-5}$, the metric reports `status = DEGENERATE_SCALE` rather than dividing by zero.

2. **Occlusion-Aware Valid Correspondence Perceptual Distance ($\text{WarpLPIPS}_{\text{valid}}$):**
   $$\text{WarpLPIPS}_{\text{valid}} = \text{LPIPS}(M_{\text{valid}} \odot I_t, M_{\text{valid}} \odot \text{Warp}(I_{t-1}, \mathbf{v}_{t-1 \to t}))$$
   where $M_{\text{valid}}$ excludes occlusions, disocclusions, and border boundary pixels.
   *Metric Availability Semantics:* If optical flow or PyTorch perceptual models are unavailable, the metric harness returns `available = False` without fabricating dummy scores.

3. **ArcFace Observational Calibration Protocol:**
   Evaluated on 112×112 aligned face chips for frames with $|\text{yaw}| \le 20^\circ$. ArcFace CosSim is reported as an observational/calibration metric during Phase 2.5A, not a hard pass/fail gate for the baseline.

---

## 4. Phase 2.5B Candidate Backend Evaluation Matrix

| Candidate Backend | Latency Target | Pose Controllability | Identity Fidelity | License Compliance | GPU Requirements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A: UV Mesh / Neural Texture Renderer** | Real-Time ($< 10\text{ ms}$) | High (direct skeletal mesh) | Moderate | 100% Permissive (MIT/Apache) | Low ($\le 2\text{ GB}$) |
| **Candidate B: LivePortrait / Latent Feature Warper** | Near Real-Time ($15\text{-}30\text{ ms}$) | High (keypoint-driven) | High | Check model weights license | Medium ($4\text{-}8\text{ GB}$) |
| **Candidate C: ControlNet + IP-Adapter Diffusion** | Offline ($> 500\text{ ms}$) | High (pose map conditioning) | Very High | Permissive weights available | High ($\ge 12\text{ GB}$) |
