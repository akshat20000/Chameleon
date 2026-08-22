# Candidate C Feasibility Spike: ControlNet + IP-Adapter Diffusion Stack

**Experiment ID:** Phase 2.5B.1.1 — Single-Frame Feasibility  
**Candidate Backend:** Candidate C (Pose-guided ControlNet + IP-Adapter Visual Identity Conditioning)  
**Primary Goal:** Evaluate single-frame identity reconstruction (Gate C1) and pose control (Gate C2/C3) before multi-frame video synthesis (Gate C4).

---

## 1. Single-Frame Feasibility Ladder & Review Criteria

Execution proceeds incrementally one gate at a time. Do **not** execute downstream gates or video generation if an earlier gate fails:

- **Gate C1 — Identity Reconstruction (Primary Focus):**
  - *Input:* Subject A reference (`inputs/identity/front.jpg`) + Subject A-like rest pose.
  - *Success Criteria:* Subject A visually recognizable + face geometry coherent + no catastrophic facial distortion.
- **Gate C2 — Basic Pose Control:**
  - *Input:* Subject A reference + novel OpenPose pose map (`inputs/driving/arm_raise.png`).
  - *Success Criteria:* Gate C1 pass + requested pose visibly followed + limbs anatomically coherent.
- **Gate C3 — Moderate Novel Pose:**
  - *Input:* Subject A reference + torso rotation (`inputs/driving/torso_turn.png`).
  - *Success Criteria:* Gate C2 pass + identity survives moderate novel pose + clothing identity preserved.
- **Gate C4 — Multi-Frame Sequence:**
  - *Condition:* Executed **only** if Gates C1 + C2 + C3 PASS.
  - *Success Criteria:* Low identity drift, minimal facial flicker, stable clothing and limbs.

---

## 2. Frozen Generation Config & Provenance Invariants

All feasibility experiments use frozen generation parameters recorded in `evidence.json`:

```json
{
  "generation_config": {
    "width": 512,
    "height": 512,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "controlnet_conditioning_scale": 1.0,
    "ip_adapter_scale": 0.7,
    "seed": 12345,
    "scheduler": "EulerDiscreteScheduler",
    "model_dtype": "float16"
  }
}
```

*Provenance Invariant:* Executed models must resolve local file paths and valid SHA-256 checksums at runtime. Loaded artifacts with missing SHA-256 trigger `execution_status = "INVALID_PROVENANCE"`.

---

## 3. Minimum Controlled Input Dataset

Place minimal benchmark assets in `experiments/phase_2_5b/candidate_c/inputs/`:

```text
candidate_c/
└── inputs/
    ├── identity/
    │   ├── front.jpg
    │   ├── left_45.jpg
    │   └── right_45.jpg
    │
    └── driving/
        ├── neutral.png
        ├── arm_raise.png
        ├── torso_turn.png
        └── short_motion.mp4
```

---

## 4. Execution Strategy

Execute Gate C1 feasibility test:
```bash
python experiments/phase_2_5b/candidate_c/run.py
```

Outputs will be saved to `test_data/outputs/phase_2_5b_feasibility/candidate_c/`:
- `gate_c1_identity.png`
- `evidence.json`
