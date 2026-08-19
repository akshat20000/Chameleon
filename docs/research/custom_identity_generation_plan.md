# Chameleon Phase 2.0 — Custom Identity-Preserving Generation Specification (First Principles)

**Date:** 2026-08-19  
**Status:** **RESEARCH & FIRST-PRINCIPLES ARCHITECTURAL PLAN**  
**Pre-Research Baseline Tag:** `pre-research-baseline`  
**Objective:** Design a custom, photorealistic, full-body identity-preserving generation and swap pipeline from first principles, overcoming the fundamental quality limitations of crop-based 2D UNet face swap models.

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 Why Previous Candidates (MobileFaceSwap, SimSwap, GhostV2) Failed
Our rigorous Phase 1.7 investigation demonstrated that crop-based 2D UNet face swap architectures suffer from three fundamental first-principles limitations:

1. **Information Bottleneck of 1D Identity Vectors:**  
   Compressing a human face into a single 512-dimensional vector ($\mathbf{z}_{\text{id}} \in \mathbb{R}^{512}$) discards high-frequency spatial geometry, fine skin textures, iris details, micro-expressions, and unique jawline contours. A 1D vector cannot reconstruct a 3D facial identity without severe feature blending.

2. **Rigid 256×256 Spatial Receptive Field Constraint:**  
   Models like AEI-Net and GhostV2 rely on a fixed $256 \times 256$ face crop. When an input face is slightly off-center, unaligned, or under extreme roll/pitch angles, facial landmarks land outside the UNet's expected spatial receptive fields, causing catastrophic warping, eye distortion, and border artifacts.

3. **Lack of Full-Body & Skin/Hair Boundary Awareness:**  
   Crop-based face swappers operate strictly within the face box, ignoring neck skin tone transitions, shoulder geometry, hair occlusion, and full-body lighting coherence.

### 1.2 The First-Principles Solution: Chameleon Phase 2.0
Instead of forcing a cropped 2D face swapper, Chameleon Phase 2.0 designs a **full-body, multi-scale identity-conditioned generative framework** powered by a Latent Diffusion / Flow-Matching backbone paired with dual identity encoders and explicit 3D pose/mesh guidance.

---

## 2. System Architecture Design

```
SOURCE IDENTITY IMAGE (S)                             TARGET POSE / SCENE IMAGE (T)
         │                                                           │
         ├──► [MobileFaceNet (512-D Semantic ID Vector)]             ├──► [MediaPipe 3D Mesh / DensePose Keypoints]
         │                                                           │
         └──► [DINOv2 / CLIP (Dense Spatial Feature Map)]            └──► [ControlNet / IP-Adapter Spatial Guidance]
                     │                                                           │
                     └───────────────────────────┬───────────────────────────────┘
                                                 │
                                                 ▼
                                  [Latent Diffusion / Flow Backbone]
                                  (Cross-Attention Identity Injection)
                                                 │
                                                 ▼
                                   SYNTHESIZED FULL-BODY IMAGE (Y)
                                  (Target Pose/Background + Source ID)
```

### 2.1 Core Architectural Components

1. **Dual Identity Encoder Stream ($\mathbf{E}_{\text{id}}$):**
   - **Global Semantic Vector ($\mathbf{z}_{\text{global}} \in \mathbb{R}^{512}$):** Extracted using MobileFaceNet / AdaFace for global identity matching and loss computation.
   - **Dense Spatial Feature Map ($\mathbf{F}_{\text{spatial}} \in \mathbb{R}^{H \times W \times C}$):** Extracted using DINOv2 / ViT patch tokens to capture high-frequency skin textures, mole positions, eye shapes, and fine facial structure.

2. **Pose & Geometry Control Stream ($\mathbf{E}_{\text{pose}}$):**
   - **MediaPipe 3D Face Mesh (478 landmarks):** Provides precise facial geometry, eye gaze, mouth openness, and head rotation.
   - **DensePose / Body Keypoints:** Controls full-body posture, neck alignment, shoulder position, and hand placement.

3. **Generative Backbone & Cross-Attention Conditioning:**
   - A Latent Diffusion / Flow Matching backbone (SDXL / Flux UNet-Transformer architecture).
   - **IP-Adapter Style Cross-Attention:** Injects $\mathbf{z}_{\text{global}}$ and $\mathbf{F}_{\text{spatial}}$ directly into cross-attention layers, decoupling identity representation from spatial background generation.

---

## 3. Training & Dataset Pipeline

### 3.1 Dataset Requirements
- **High-Resolution Multi-View Portraits (FFHQ, LAION-Face):** 100k+ high-quality images at $512 \times 512$ and $1024 \times 1024$.
- **Multi-Frame Video Keyframes (HD-TF, VoxCeleb2):** Pairs of frames of the same individual across different poses, lighting conditions, and expressions to supervise identity invariance under extreme motion.

### 3.2 Preprocessing & Augmentation Protocol
1. **Automated 3D Landmark & Mesh Extraction:** Run MediaPipe 3D Face Mesh + DensePose on all training pairs.
2. **Synthetic Identity-Pose Pair Creation:** For video keyframes $(I_A, I_B)$ of identity $i$:
   - Set Source $S = I_A$, Target Pose $T = \text{Pose}(I_B)$.
   - Ground truth output $Y = I_B$.

---

## 4. Loss Functions & Optimization Protocol

The overall loss function $\mathcal{L}_{\text{total}}$ combines identity preservation, pose alignment, perceptual quality, and generative realism:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{id}} \mathcal{L}_{\text{id}} + \lambda_{\text{pose}} \mathcal{L}_{\text{pose}} + \lambda_{\text{perceptual}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{diff}} \mathcal{L}_{\text{diff}}$$

1. **Identity Loss ($\mathcal{L}_{\text{id}}$):**
   $$\mathcal{L}_{\text{id}} = 1 - \cos\left(\text{MobileFaceNet}(Y), \text{MobileFaceNet}(S)\right)$$
   Enforces strict cosine similarity between the generated face $Y$ and source identity reference $S$.

2. **3D Pose & Expression Loss ($\mathcal{L}_{\text{pose}}$):**
   $$\mathcal{L}_{\text{pose}} = \frac{1}{N} \sum_{k=1}^{N} \left\| \text{Mesh}(Y)_k - \text{Mesh}(T)_k \right\|_2^2$$
   Measures Euclidean distance between 478 3D facial mesh points of generated output $Y$ and target pose $T$.

3. **Perceptual Reconstruction Loss ($\mathcal{L}_{\text{perceptual}}$):**
   $$\mathcal{L}_{\text{perceptual}} = \text{LPIPS}\left(Y \odot M_{\text{bg}}, T \odot M_{\text{bg}}\right)$$
   Preserves unswapped background, hair, clothing, and body features using masked LPIPS perceptual loss.

4. **Diffusion Noise Prediction Loss ($\mathcal{L}_{\text{diff}}$):**
   $$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, \mathbf{z}_{\text{id}}, \text{Pose}(T)) \right\|_2^2 \right]$$

---

## 5. Quantitative & Qualitative Evaluation Protocol

To pass the Phase 2.0 evaluation gate, any candidate pipeline must satisfy all five quantitative gates AND pass human visual quality review:

| Evaluation Metric | Target Gate Standard | Measurement Tool |
|---|---|---|
| **Identity Similarity ($\text{Sim} \rightarrow S$)** | **$\ge 0.8500$** | MobileFaceNet / AdaFace CosSim |
| **Pose Accuracy ($\Delta \text{Pitch, Yaw, Roll}$)** | **$\le 3.0^\circ$** | MediaPipe 3D Pose Estimator |
| **Expression MAE (52 Blendshapes)** | **$\le 0.0300$** | ARKit 52 Blendshape MAE |
| **Background Preservation (SSIM)** | **$\ge 0.9500$** | Masked SSIM vs Target $T$ |
| **Generative Realism (FID)** | **$\le 15.00$** | Fréchet Inception Distance |

### Human Visual Quality Gate (Pass / Fail Criteria)
- ❌ Zero facial warping or asymmetrical eye distortion.
- ❌ Zero ghosting artifacts at jawline/neck boundaries.
- ❌ Zero color/skin tone mismatch between face and neck.
- ✅ Photorealistic skin texture, lighting integration, and eye reflections.

---

## 6. Phased Execution Roadmap

- **Phase 2.0 (Current):** First-principles research specification & architecture design.
- **Phase 2.1:** Dataset mining & automated MediaPipe 3D Mesh / DensePose pipeline.
- **Phase 2.2:** Dual Identity Encoder ($\mathbf{z}_{\text{global}} + \mathbf{F}_{\text{spatial}}$) module construction & IP-Adapter cross-attention integration.
- **Phase 2.3:** Controlled training & quantitative/qualitative benchmarking against evaluation gates.
