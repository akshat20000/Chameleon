"""
Candidate B Feasibility Spike Harness (LivePortrait Latent Feature Warper).

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
logger = logging.getLogger("candidate_b_spike")

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_data" / "outputs" / "phase_2_5b_feasibility" / "candidate_b"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_candidate_b_spike() -> Dict[str, Any]:
    logger.info("=== Running Phase 2.5B.1 Experiment 2: Candidate B (LivePortrait Warper) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs_dir = EXPERIMENT_DIR / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "candidate_id": "candidate_b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_versions": {
            "liveportrait_code": "KlingAIResearch/LivePortrait (commit e4a2b90)",
            "liveportrait_models": "KlingAI/LivePortrait (base_models/appearance_feature_extractor.safetensors, rev 8f192b1)",
            "insightface_detector": "deepinsight/insightface (buffalo_l/2d106det.onnx)",
            "arcface_embedding": "deepinsight/insightface (glintr100.onnx)",
        },
        "input_hashes": {
            "portrait": compute_file_sha256(inputs_dir / "subject_a_portrait.png"),
            "head_motion": compute_file_sha256(inputs_dir / "subject_b_head_motion.mp4"),
        },
        "output_hashes": {},
        "execution_status": "READY_FOR_DATASET_ACQUISITION",
        "runtime_environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "warnings": [
            "InsightFace dependency is non-commercial research only (CC BY-NC 4.0). Must be replaced before production integration."
        ],
        "license_status_at_execution": "RESEARCH_SPIKE_ONLY",
    }

    evidence_path = OUTPUT_DIR / "evidence.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    logger.info("Saved evidence metadata to %s", evidence_path)
    return evidence


if __name__ == "__main__":
    run_candidate_b_spike()
