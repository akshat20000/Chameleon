# Candidate B Feasibility Spike: LivePortrait Latent Feature Warper

**Experiment ID:** Phase 2.5B.1 / Experiment 2  
**Candidate Backend:** Candidate B (LivePortrait Latent Feature Warping Model)  
**Primary Goal:** Isolate Tier 1 facial/head pose and expression transfer to test facial identity preservation and evaluate whether a hybrid face/body architecture is required.

---

## 1. Input Requirements
Place benchmark input assets in `experiments/phase_2_5b/candidate_b/inputs/`:
- `subject_a_portrait.png` (Frontal portrait photo of Subject A)
- `subject_b_head_motion.mp4` (Head motion driving sequence)

---

## 2. Execution Strategy
Run standalone inference spike:
```bash
python experiments/phase_2_5b/candidate_b/run.py
```

Outputs will be saved to:
`test_data/outputs/phase_2_5b_feasibility/candidate_b/`
- `output_portrait_animation.mp4`
- `evidence.json`
