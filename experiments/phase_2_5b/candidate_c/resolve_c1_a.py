"""
Phase C1-A: Candidate C Repository Snapshot Resolution & Provenance Verification.

Spec: docs/architecture/ADR/ADR-007-phase-2-5b-candidate-evaluation.md

Diffusers-Native Repository Snapshot Resolution (HF_HOME set to Drive E):
  1. Resolves HuggingFace repository snapshots for:
     - lllyasviel/control_v11p_sd15_openpose
     - runwayml/stable-diffusion-v1-5 (variant="fp16")
     - h94/IP-Adapter
  2. Extracts full 40-character commit SHAs.
  3. Computes SHA-256 for each downloaded local weight file.
  4. Updates test_data/outputs/phase_2_5b_feasibility/candidate_c/gate_c1/evidence.json.
  5. STOPS without initializing pipelines or running inference.
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_data" / "outputs" / "phase_2_5b_feasibility" / "candidate_c"
GATE_C1_DIR = OUTPUT_DIR / "gate_c1"
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Set HF_HOME to Drive E project root to leverage 107GB free space
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_c1_a_resolver")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file if it exists."""
    if not file_path.exists():
        return "UNRESOLVED_FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def execute_phase_c1_a() -> Dict[str, Any]:
    logger.info("=== Starting Phase C1-A: Repository Snapshot Resolution & Provenance Verification ===")
    logger.info("Using HF_HOME on Drive E: %s", HF_CACHE_DIR)
    GATE_C1_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    target_repos = [
        {
            "id": "controlnet",
            "repo_id": "lllyasviel/control_v11p_sd15_openpose",
            "files": [
                {"filename": "config.json", "subfolder": None},
                {"filename": "diffusion_pytorch_model.safetensors", "subfolder": None},
            ],
        },
        {
            "id": "base_model",
            "repo_id": "runwayml/stable-diffusion-v1-5",
            "files": [
                {"filename": "model_index.json", "subfolder": None},
                {"filename": "config.json", "subfolder": "unet"},
                {"filename": "diffusion_pytorch_model.fp16.safetensors", "subfolder": "unet"},
                {"filename": "config.json", "subfolder": "vae"},
                {"filename": "diffusion_pytorch_model.fp16.safetensors", "subfolder": "vae"},
                {"filename": "config.json", "subfolder": "text_encoder"},
                {"filename": "model.fp16.safetensors", "subfolder": "text_encoder"},
            ],
        },
        {
            "id": "ip_adapter",
            "repo_id": "h94/IP-Adapter",
            "files": [
                {"filename": "ip-adapter-plus_sd15.safetensors", "subfolder": "models"},
                {"filename": "config.json", "subfolder": "models/image_encoder"},
                {"filename": "pytorch_model.bin", "subfolder": "models/image_encoder"},
            ],
        },
    ]

    resolved_manifest = {}
    phase_warnings = []
    overall_status = "SNAPSHOT_RESOLVED"

    for r in target_repos:
        logger.info("Resolving HuggingFace repository: %s ...", r["repo_id"])
        repo_commit = "UNRESOLVED"
        snapshot_base_path = "UNRESOLVED"
        loaded_files_info = []
        repo_success = True

        for item in r["files"]:
            fname = item["filename"]
            sfolder = item["subfolder"]
            rel_display = f"{sfolder}/{fname}" if sfolder else fname
            logger.info("Resolving file: %s from %s ...", rel_display, r["repo_id"])

            try:
                downloaded_path_str = hf_hub_download(
                    repo_id=r["repo_id"],
                    filename=fname,
                    subfolder=sfolder,
                )
                local_path = Path(downloaded_path_str)

                # Extract full commit SHA from path
                if "snapshots" in local_path.parts:
                    snap_idx = local_path.parts.index("snapshots")
                    if snap_idx + 1 < len(local_path.parts):
                        repo_commit = local_path.parts[snap_idx + 1]
                        snapshot_base_path = str(Path(*local_path.parts[:snap_idx + 2]))

                sha = compute_file_sha256(local_path)
                loaded_files_info.append({
                    "relative_path": rel_display,
                    "local_path": str(local_path),
                    "size_bytes": local_path.stat().st_size,
                    "sha256": sha,
                })
                logger.info("Verified file %s (SHA-256: %s...)", rel_display, sha[:16])

            except Exception as err:
                logger.error("Failed to resolve file %s from %s: %s", rel_display, r["repo_id"], err)
                phase_warnings.append(f"Failed file {rel_display} in {r['repo_id']}: {err}")
                repo_success = False
                loaded_files_info.append({
                    "relative_path": rel_display,
                    "local_path": "UNRESOLVED",
                    "size_bytes": 0,
                    "sha256": "UNRESOLVED_FILE_NOT_FOUND",
                })

        resolved_manifest[r["id"]] = {
            "repository": r["repo_id"],
            "requested_revision": "main",
            "resolved_commit": repo_commit,
            "snapshot_path": snapshot_base_path,
            "loaded_files": loaded_files_info,
        }

        if not repo_success:
            overall_status = "INVALID_PROVENANCE"

    if overall_status == "SNAPSHOT_RESOLVED" and not phase_warnings:
        overall_status = "PROVENANCE_VALID"

    # Save evidence.json
    evidence_path = GATE_C1_DIR / "evidence.json"
    evidence_data = {}
    if evidence_path.exists():
        with open(evidence_path, "r", encoding="utf-8") as f:
            evidence_data = json.load(f)

    evidence_data["candidate_id"] = "candidate_c"
    evidence_data["gate"] = "C1"
    evidence_data["phase_c1_a_status"] = overall_status
    evidence_data["execution_status"] = overall_status
    evidence_data["resolved_artifacts"] = resolved_manifest
    evidence_data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    evidence_data["warnings"] = phase_warnings

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_data, f, indent=2)

    logger.info("Phase C1-A Resolution Complete. Status: %s. Saved evidence to %s", overall_status, evidence_path)
    return evidence_data


if __name__ == "__main__":
    execute_phase_c1_a()
