# Candidate A Feasibility Spike: UV Mesh / Neural Texture Renderer

**Experiment ID:** Phase 2.5B.1 / Experiment 3  
**Candidate Backend:** Candidate A (Rigid Mesh & Neural Texture Renderer)  
**Primary Goal:** Serve as a deterministic control experiment establishing the lower-bound baseline for pose controllability versus visual realism.

---

## 1. Input Requirements
Place benchmark input assets in `experiments/phase_2_5b/candidate_a/inputs/`:
- `subject_a_texture.png` (Neural texture map of Subject A)
- `motion_sequence.json` (Retargeted actor joint motion sequence)

---

## 2. Execution Strategy
Run standalone inference spike:
```bash
python experiments/phase_2_5b/candidate_a/run.py
```

Outputs will be saved to:
`test_data/outputs/phase_2_5b_feasibility/candidate_a/`
- `output_mesh_animation.mp4`
- `evidence.json`
