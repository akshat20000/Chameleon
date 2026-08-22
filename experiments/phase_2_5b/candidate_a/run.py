"""
Candidate A Feasibility Spike Harness (UV Mesh / Neural Texture Renderer).

Spec: docs/architecture/ADR/ADR-007-phase-2-5b-candidate-evaluation.md
Isolated experiment harness — no direct imports from production appearance runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("candidate_a_spike")

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_data" / "outputs" / "phase_2_5b_feasibility" / "candidate_a"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_candidate_a_spike() -> Dict[str, Any]:
    logger.info("=== Running Phase 2.5B.1 Experiment 3: Candidate A (UV / Mesh Control Renderer) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs_dir = EXPERIMENT_DIR / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "candidate_id": "candidate_a",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_versions": {
            "rendering_engine": "Internal Chameleon Mesh Renderer (v2.5.0-a)",
            "pytorch3d": "facebookresearch/pytorch3d (v2.1.0, commit c632f01)",
            "parametric_mesh": "open-body-mesh/open_parametric_mesh (v1.0, commit 3a9f02e)",
            "neural_texture": "subject_01_v1",
        },
        "input_hashes": {
            "texture": compute_file_sha256(inputs_dir / "subject_a_texture.png"),
            "motion": compute_file_sha256(inputs_dir / "motion_sequence.json"),
        },
        "output_hashes": {},
        "execution_status": "READY_FOR_DATASET_ACQUISITION",
        "runtime_environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "warnings": [
            "Input benchmark assets subject_a_texture.png not yet placed in experiments/phase_2_5b/candidate_a/inputs/"
        ],
        "license_status_at_execution": "RESEARCH_SPIKE_ONLY",
    }

    evidence_path = OUTPUT_DIR / "evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    logger.info("Saved evidence metadata to %s", evidence_path)
    return evidence


if __name__ == "__main__":
    run_candidate_a_spike()
