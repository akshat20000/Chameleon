# ADR-007: Phase 2.5B Candidate Evaluation Strategy & License Audit Protocol

**Status:** APPROVED WITH REQUIRED LICENSING CORRECTIONS  
**Date:** 2026-08-21  
**Deciders:** Core Architecture Team  
**Replaces:** N/A (Establishes Phase 2.5B Candidate Evaluation & Primary Source License Audit Protocol)

---

## 1. Context & Problem Statement

Phase 2.5A successfully established the backend-neutral appearance conditioning schemas (`AppearanceConditioningState`), the read-only kinematic boundary (`RetargetedActorState`), deterministic region rendering (`BaselineArticulatedSynthesizer`), and metric safety protocols.

Phase 2.5B addresses the primary product hypothesis of Chameleon:
> **Can a legally compliant model take reference photos of Person A and motion from Person B, and produce a high-fidelity, identity-preserving animated video of Person A performing Person B's motion?**

Rather than prematurely attempting deep integration into the Chameleon runtime before proving that candidate neural synthesis backends satisfy the core product hypothesis, Phase 2.5B is structured as a **4-stage Feasibility Spike and Benchmarking Pipeline**:

1. **Phase 2.5B.0 — Primary Source Candidate Artifact & License Audit (Pre-Inference Gate)**
2. **Phase 2.5B.1 — Standalone Candidate Feasibility Spike (Isolated Evaluation)**
3. **Phase 2.5B.2 — Tiered Benchmark Selection (Tier 1 Face & Tier 2 Full-Body)**
4. **Phase 2.5B.3 — Chameleon Architecture Integration (Winner Only)**

---

## 2. Four-Stage Phase 2.5B Execution Strategy

```
Phase 2.5B.0: Primary Source Candidate Artifact & License Audit
  ├── Audit every code & weight dependency per candidate in phase_2_5b_candidates.yaml
  └── Require commercial_path_valid == true BEFORE production integration / backend selection
                          │
                          ▼
Phase 2.5B.1: Isolated Candidate Feasibility Spikes (experiments/phase_2_5b/)
  ├── Experiment 1: Candidate C (ControlNet + IP-Adapter) ──► Test Core Product Hypothesis
  ├── Experiment 2: Candidate B (LivePortrait Warper)     ──► Test Tier 1 Facial Isolation & Hybrid Need
  └── Experiment 3: Candidate A (UV / Mesh Renderer)     ──► Test Deterministic Control Lower Bound
                          │
                          ▼
Phase 2.5B.2: Tiered Benchmark Evaluation & Winner Selection
  ├── Tier 1 — Face/Head Feasibility (Pose, Expression, Identity, Temporal Face Stability)
  └── Tier 2 — Full-Body Feasibility (Torso, Arms, Legs, Walking Stride, Self-Occlusion)
                          │
                          ▼
Phase 2.5B.3: Chameleon Architecture Integration
  └── Winning Candidate ──► WinningAppearanceSynthesizer
```

---

## 2.1 Isolated Feasibility Sandbox Structure (`experiments/phase_2_5b/`)

To prevent premature coupling with the Chameleon runtime, Phase 2.5B.1 spikes are conducted in strictly isolated experiment directories:

```text
experiments/
└── phase_2_5b/
    ├── candidate_a/
    │   ├── run.py
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── inputs/
    │   └── outputs/
    │
    ├── candidate_b/
    │   ├── run.py
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── inputs/
    │   └── outputs/
    │
    └── candidate_c/
        ├── run.py
        ├── README.md
        ├── requirements.txt
        ├── inputs/
        └── outputs/
```

Outputs for each candidate are written to `test_data/outputs/phase_2_5b_feasibility/candidate_X/` along with a machine-readable `evidence.json` record:

```json
{
  "candidate_id": "candidate_c",
  "artifact_versions": {},
  "input_hashes": {},
  "output_hashes": {},
  "execution_status": "SUCCESS",
  "runtime_environment": {},
  "warnings": [],
  "license_status_at_execution": "RESEARCH_SPIKE_ONLY"
}
```

---

## 3. Candidate Backends & Primary Source Component License Audit Matrix

> [!IMPORTANT]
> **Primary Source Verification Rules:**
> 1. `verified: true` strictly signifies that the **exact artifact name, primary source URL, and governing license** have been independently verified from a primary source.
> 2. A repository's open-source code license (e.g. MIT or Apache-2.0) does **NOT** grant commercial rights to pre-trained model weights or third-party dependencies.
> 3. OpenRAIL licenses carry specific behavioral usage restrictions and must be recorded as their specific license class rather than generic permissive terms.

| Candidate Backend | Code License & Primary Source | Model Weights License & Source | Commercial License Notes | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Candidate A: UV Mesh / Neural Texture Renderer** | MIT / Apache 2.0 (Internal Chameleon) | MIT (`open_parametric_mesh` v1.0, commit `3a9f02e`) | Audited open MIT body mesh template (strictly avoiding non-commercial SMPL/SMPL-X) | **APPROVED_FOR_SPIKE** (`commercial_path_valid: false`) |
| **Candidate B: LivePortrait Latent Feature Warper** | MIT ([GitHub: KlingAIResearch/LivePortrait](https://github.com/KlingAIResearch/LivePortrait)) | Custom Research ([HuggingFace: KlingAI/LivePortrait](https://huggingface.co/KlingAI/LivePortrait)) | **WARNING:** `InsightFace buffalo_l` package (`2d106det.onnx`) is non-commercial research only. Must be replaced with specifically audited face detection/landmark artifacts whose exact code and model-weight licenses permit commercial use | **RESEARCH_ONLY** (`commercial_path_valid: false`) |
| **Candidate C: ControlNet + IP-Adapter Diffusion Stack** | Apache-2.0 ([GitHub: lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet), [GitHub: tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)) | OpenRAIL-M / Apache 2.0 (`ip-adapter-plus_sd15.safetensors` rev `9bf28b3`, sha256 `a1c250...`) | Exact artifact audit complete: ControlNet `734003d`, IP-Adapter Plus `9bf28b3`, SD1.5 base `6ce4eed`, CLIP encoder `b817502` | **APPROVED_FOR_SPIKE** (`commercial_path_valid: false`) |

---

## 4. Production Selection Invariant vs. Feasibility Spike Policy

To support research inspection without compromising commercial compliance, candidate artifacts are evaluated using three explicit status flags in [`docs/architecture/model_audit/phase_2_5b_candidates.yaml`](../model_audit/phase_2_5b_candidates.yaml):

- `download_allowed: true`: Permits downloading artifacts locally for license, hash, and metadata inspection.
- `execution_allowed: true`: Permits isolated standalone feasibility spike execution during Phase 2.5B.1.
- `commercial_path_valid: false`: Production selection flag.

> [!CAUTION]
> **Production Selection Invariant:**
> No artifact may enter the Chameleon production codebase path, be redistributed as part of the project, or be selected as an approved commercial backend until `commercial_path_valid: true`.

---

## 5. Tiered Benchmark Evaluation Protocol

Candidates are evaluated across two distinct benchmark tiers:

### Tier 1 — Face & Head Feasibility Benchmark
- **Identity Preservation:** ArcFace CosSim $\ge 0.80$ + Visual Human Review
- **Head Pose Transfer:** Yaw ($\pm 45^\circ$), Pitch ($\pm 20^\circ$)
- **Expression Transfer:** Mouth / eye motion adherence
- **Temporal Face Stability:** Low flicker across consecutive frames

### Tier 2 — Full-Body Feasibility Benchmark
- **Torso & Arm Articulation:** Raising, waving, and crossing arms
- **Leg Articulation & Walking:** Striding, stepping, body translation
- **Self-Occlusion & Disocclusion:** Arm crossing face or torso
- **Novel Body Pose Continuity:** Anatomical integrity under large pose deltas

> [!NOTE]
> A candidate that excels at Tier 1 (e.g. LivePortrait for facial warps) but fails Tier 2 full-body motion cannot be selected as a complete Phase 2.5B system winner without an accompanying body synthesis engine.

---

## 6. Evaluation Protocol & Decision Rules

```
                      Candidate Output
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
    AUTOMATED METRICS                  HUMAN VISUAL REVIEW
  ├── ArcFace CosSim (Identity)      ├── Identity Collapse Check
  ├── NKE_body (Pose Adherence)      ├── Artifact & Flicker Check
  └── WarpLPIPS (Temporal Stability) └── Natural Motion Quality Check
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                     CANDIDATE VERDICT
```

### Rejection Rules:
A candidate backend is **rejected** if:
1. License is non-commercial or incompatible with production (`commercial_path_valid == false`).
2. Identity visibly collapses during large head yaw/pitch angles.
3. Severe temporal flickering occurs across adjacent frames.
4. Large arm/body motion fails to follow driving skeleton.

### Selection Rule:
Among un-rejected candidates, select the backend demonstrating the **highest overall visual identity fidelity and motion adherence**. Performance optimization is applied during Phase 2.5B.3 / Phase 2.7.

---

## 7. Phase 2.5B.0 Exit Criteria & Outcome Classifications

Before moving to standalone inference spikes in Phase 2.5B.1, every candidate must be fully audited according to the 7 Phase 2.5B.0 exit criteria:

1. **Complete Runtime Dependency Graph:** Every runtime code, weight, base model, control model, encoder, and detection artifact is enumerated in [`phase_2_5b_candidates.yaml`](../model_audit/phase_2_5b_candidates.yaml).
2. **Primary Source Verification:** Every artifact has an exact primary source URL and exact commit hash / release version.
3. **Decoupled Code vs Weight Auditing:** Code repository licenses and model weight licenses are audited independently.
4. **Verified License:** License is independently verified from primary source or explicitly marked `UNKNOWN`.
5. **Separated Permissions:** Commercial use permissions (`commercial_use_allowed`) and redistribution rights (`redistribution_allowed`) are recorded separately.
6. **Exact tested revision/checkpoint recorded.**
7. **Single Outcome Assigned:** Candidate receives exactly one outcome classification:
   - **`APPROVED_FOR_SPIKE`**: Permissive or commercial-compatible path verified for isolated spike execution.
   - **`RESEARCH_ONLY`**: Non-commercial dependencies (e.g. CC BY-NC) present; spike permitted for research observation only.
   - **`REJECTED`**: License unacceptable or dependency graph unverifiable.
