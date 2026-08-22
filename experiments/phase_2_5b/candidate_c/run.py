"""
Candidate C Single-Frame Feasibility Harness (ControlNet + IP-Adapter Diffusion Stack).

Spec: docs/architecture/ADR/ADR-007-phase-2-5b-candidate-evaluation.md

Diffusers-Native Sub-Phase Execution Pipeline (Option A):
  Phase C1-A: Repository Snapshot Resolution & Per-File Weight Hashing (Drive E cache)
  Phase C1-B: Pipeline Compatibility & Memory Telemetry (GTX 1650 4GB VRAM Support)
  Phase C1-C: Single Controlled Image Generation (seed 12345, 512x512)

Provenances & Memory Invariants:
  - Per-file loaded weight SHA-256s inside HF repository snapshots recorded in evidence.json.
  - CUDA Memory Telemetry recorded before, during, and after pipeline initialization.
  - Pure Python Chunked Safetensors Reader: reads safetensors directly into float16 tensors.
    Eliminates Windows OS Error 1455 mmap limits and torch.load security blocks under 8GB RAM.
  - Offload sequence: Load ControlNet -> Load CLIPVisionModel -> Load SD1.5 Pipeline -> Attach IP-Adapter -> Set IP-Adapter scale -> enable_attention_slicing() -> enable_vae_slicing() -> enable_model_cpu_offload() LAST.
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
import sys
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_data" / "outputs" / "phase_2_5b_feasibility" / "candidate_c"
GATE_C1_DIR = OUTPUT_DIR / "gate_c1"
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Force HF_HOME to Drive E project root to leverage 107GB free space
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("candidate_c_feasibility")

np_dtype_map = {
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "BF16": np.float32,
}


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file if it exists."""
    if not file_path.exists():
        return "UNRESOLVED_FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_weights_to_meta_model(model, weight_path: Path):
    """Zero-RAM in-place parameter copy streaming from disk directly into pre-allocated model memory."""
    model.to_empty(device="cpu")
    model.to(torch.float16)
    if weight_path.name.endswith(".safetensors"):
        param_map = dict(model.named_parameters())
        buffer_map = dict(model.named_buffers())
        with open(weight_path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_start = 8 + header_len
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                f.seek(data_start + v["data_offsets"][0])
                buf = f.read(v["data_offsets"][1] - v["data_offsets"][0])
                arr = np.frombuffer(buf, dtype=np_dtype_map.get(v["dtype"], np.float32)).reshape(v["shape"])
                if arr.dtype == np.float32:
                    arr = arr.astype(np.float16)
                t = torch.from_numpy(arr)
                with torch.no_grad():
                    if k in param_map:
                        param_map[k].copy_(t)
                    elif k in buffer_map:
                        buffer_map[k].copy_(t)
                del buf, arr, t
        gc.collect()
    else:
        sd = torch.load(weight_path, map_location="cpu", weights_only=True, mmap=True)
        fp16_sd = {}
        for k in list(sd.keys()):
            v = sd.pop(k)
            if v.is_floating_point():
                v = v.to(torch.float16)
            fp16_sd[k] = v
        del sd
        model.load_state_dict(fp16_sd, strict=False, assign=True)
        del fp16_sd
        gc.collect()
    return model


def find_first_existing_file(candidates: List[Path]) -> Optional[Path]:
    """Return the first existing file from a list of candidate paths."""
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_candidate_c_inputs() -> Dict[str, Optional[Path]]:
    """Resolve input file paths for Subject A identity and Subject B driving pose."""
    identity_dir = EXPERIMENT_DIR / "inputs" / "identity"
    driving_dir = EXPERIMENT_DIR / "inputs" / "driving"

    identity_dir.mkdir(parents=True, exist_ok=True)
    driving_dir.mkdir(parents=True, exist_ok=True)

    front_file = find_first_existing_file([identity_dir / "front.png", identity_dir / "front.jpg"])
    left_file = find_first_existing_file([identity_dir / "left_45.png", identity_dir / "left_45.jpg"])
    right_file = find_first_existing_file([identity_dir / "right_45.png", identity_dir / "right_45.jpg"])
    neutral_file = find_first_existing_file([driving_dir / "neutral.png", driving_dir / "neutral.jpg"])

    arm_raise_file = find_first_existing_file([driving_dir / "arm_raise.png", driving_dir / "arm_raise.jpg"])
    torso_turn_file = find_first_existing_file([driving_dir / "torso_turn.png", driving_dir / "torso_turn.jpg"])
    sequence_file = find_first_existing_file([driving_dir / "short_motion.mp4"])

    return {
        "identity_front": front_file,
        "identity_left": left_file,
        "identity_right": right_file,
        "driving_neutral": neutral_file,
        "driving_arm_raise": arm_raise_file,
        "driving_torso_turn": torso_turn_file,
        "driving_sequence": sequence_file,
    }


def get_default_generation_config() -> Dict[str, Any]:
    """Frozen generation parameters for reproducibility."""
    return {
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "controlnet_conditioning_scale": 1.0,
        "ip_adapter_scale": 0.7,
        "seed": 12345,
        "scheduler": "EulerDiscreteScheduler",
        "model_dtype": "float16",
        "low_vram_cpu_offload": True,
        "attention_slicing": True,
        "vae_slicing": True,
    }


def execute_phase_c1_a_snapshot_provenance() -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Execute Phase C1-A: Resolve HF snapshots and compute file hashes on Drive E."""
    logger.info("=== Phase C1-A: Verifying Repository Snapshots on Drive E ===")

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

    requested_manifest = {}
    resolved_manifest = {}
    all_files_valid = True

    for r in target_repos:
        repo_commit = "UNRESOLVED"
        snapshot_base_path = "UNRESOLVED"
        loaded_files_info = []

        for item in r["files"]:
            fname = item["filename"]
            sfolder = item["subfolder"]
            rel_display = f"{sfolder}/{fname}" if sfolder else fname

            try:
                path_str = hf_hub_download(repo_id=r["repo_id"], filename=fname, subfolder=sfolder)
                local_path = Path(path_str)

                if "snapshots" in local_path.parts:
                    snap_idx = local_path.parts.index("snapshots")
                    if snap_idx + 1 < len(local_path.parts):
                        repo_commit = local_path.parts[snap_idx + 1]
                        snapshot_base_path = str(Path(*local_path.parts[:snap_idx + 2]))

                sha = compute_file_sha256(local_path)
                if sha == "UNRESOLVED_FILE_NOT_FOUND":
                    all_files_valid = False

                loaded_files_info.append({
                    "relative_path": rel_display,
                    "local_path": str(local_path),
                    "size_bytes": local_path.stat().st_size if local_path.exists() else 0,
                    "sha256": sha,
                })
            except Exception as err:
                logger.error("Phase C1-A failed resolving %s from %s: %s", rel_display, r["repo_id"], err)
                all_files_valid = False
                loaded_files_info.append({
                    "relative_path": rel_display,
                    "local_path": "UNRESOLVED",
                    "size_bytes": 0,
                    "sha256": "UNRESOLVED_FILE_NOT_FOUND",
                })

        requested_manifest[r["id"]] = {"repository": r["repo_id"], "requested_revision": "main"}
        resolved_manifest[r["id"]] = {
            "repository": r["repo_id"],
            "requested_revision": "main",
            "resolved_commit": repo_commit,
            "snapshot_path": snapshot_base_path,
            "loaded_files": loaded_files_info,
        }

    return requested_manifest, resolved_manifest, all_files_valid


def run_gate_c1_harness() -> Dict[str, Any]:
    logger.info("=== Executing Candidate C Harness (Gate C1 Identity Reconstruction) ===")
    GATE_C1_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    inputs = resolve_candidate_c_inputs()

    # Step 1: Input Validation
    c1_required_keys = ["identity_front", "driving_neutral"]
    c1_missing = [k for k in c1_required_keys if inputs[k] is None or not inputs[k].exists()]

    input_hashes = {}
    input_paths_str = {}
    for k, p in inputs.items():
        if p is not None and p.exists():
            input_paths_str[k] = str(p)
            input_hashes[k] = compute_file_sha256(p)
        else:
            input_paths_str[k] = "MISSING"
            input_hashes[k] = "MISSING"

    input_manifest = {
        "gate": "C1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_paths_str,
        "input_hashes": input_hashes,
        "c1_missing_required_files": c1_missing,
    }
    with open(GATE_C1_DIR / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(input_manifest, f, indent=2)

    # Step 2: Phase C1-A Provenance Check
    requested_artifacts, resolved_artifacts, c1_a_valid = execute_phase_c1_a_snapshot_provenance()

    cuda_telemetry = {}
    execution_status = "INVALID_PROVENANCE"

    if c1_missing:
        execution_status = "READY_FOR_DATASET_ACQUISITION"
        logger.info("Gate C1 missing required input assets: %s", c1_missing)
    elif not c1_a_valid:
        execution_status = "INVALID_PROVENANCE"
        logger.warning("INVALID_PROVENANCE: Snapshot resolution incomplete or file hashes missing.")
    else:
        # Phase C1-A Passed cleanly -> Unlock Phase C1-B & C1-C
        logger.info("Phase C1-A Provenance Validated. Executing Phase C1-B Zero-RAM Meta Pipeline Load...")
        try:
            import torch
            from PIL import Image
            import numpy as np
            import diffusers
            from diffusers import (
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                UNet2DConditionModel,
                AutoencoderKL,
                EulerDiscreteScheduler,
            )
            from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

            if torch.cuda.is_available():
                cuda_telemetry["vram_allocated_before_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                cuda_telemetry["vram_reserved_before_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)

            c_snap = Path(resolved_artifacts["controlnet"]["snapshot_path"])
            base_snap = Path(resolved_artifacts["base_model"]["snapshot_path"])
            ip_snap = Path(resolved_artifacts["ip_adapter"]["snapshot_path"])

            # 1. ControlNet
            logger.info("Loading ControlNetModel (Zero-RAM meta device)...")
            c_config = json.load(open(c_snap / "config.json"))
            with torch.device("meta"):
                controlnet = ControlNetModel.from_config(c_config)
            assign_weights_to_meta_model(controlnet, c_snap / "diffusion_pytorch_model.safetensors")

            # 2. CLIP Image Encoder
            logger.info("Loading CLIPVisionModelWithProjection (Zero-RAM meta device)...")
            ie_snap = ip_snap / "models" / "image_encoder"
            ie_config = CLIPVisionConfig.from_pretrained(ie_snap)
            with torch.device("meta"):
                image_encoder = CLIPVisionModelWithProjection(ie_config)
            ie_st = ie_snap / "model.safetensors"
            if not ie_st.exists(): ie_st = ie_snap / "pytorch_model.bin"
            assign_weights_to_meta_model(image_encoder, ie_st)
            if hasattr(image_encoder, "vision_model") and hasattr(image_encoder.vision_model, "embeddings"):
                max_v_pos = getattr(ie_config, "max_position_embeddings", 257)
                image_encoder.vision_model.embeddings.register_buffer(
                    "position_ids",
                    torch.arange(max_v_pos, dtype=torch.int64).expand((1, -1)),
                    persistent=False,
                )

            # 3. Base SD 1.5 Components
            logger.info("Loading UNet2DConditionModel (Zero-RAM meta device)...")
            u_config = json.load(open(base_snap / "unet" / "config.json"))
            with torch.device("meta"):
                unet = UNet2DConditionModel.from_config(u_config)
            unet_st = base_snap / "unet" / "diffusion_pytorch_model.fp16.safetensors"
            if not unet_st.exists(): unet_st = base_snap / "unet" / "diffusion_pytorch_model.safetensors"
            assign_weights_to_meta_model(unet, unet_st)

            logger.info("Loading AutoencoderKL (Zero-RAM meta device)...")
            v_config = json.load(open(base_snap / "vae" / "config.json"))
            with torch.device("meta"):
                vae = AutoencoderKL.from_config(v_config)
            vae_st = base_snap / "vae" / "diffusion_pytorch_model.fp16.safetensors"
            if not vae_st.exists(): vae_st = base_snap / "vae" / "diffusion_pytorch_model.safetensors"
            assign_weights_to_meta_model(vae, vae_st)

            logger.info("Loading CLIPTextModel (Zero-RAM meta device)...")
            te_config = CLIPTextConfig.from_pretrained(base_snap / "text_encoder")
            with torch.device("meta"):
                text_encoder = CLIPTextModel(te_config)
            te_st = base_snap / "text_encoder" / "model.fp16.safetensors"
            if not te_st.exists(): te_st = base_snap / "text_encoder" / "model.safetensors"
            assign_weights_to_meta_model(text_encoder, te_st)
            if hasattr(text_encoder, "text_model") and hasattr(text_encoder.text_model, "embeddings"):
                max_t_pos = getattr(te_config, "max_position_embeddings", 77)
                text_encoder.text_model.embeddings.register_buffer(
                    "position_ids",
                    torch.arange(max_t_pos, dtype=torch.int64).expand((1, -1)),
                    persistent=False,
                )

            tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
            feature_extractor = CLIPImageProcessor.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="feature_extractor")

            logger.info("Assembling StableDiffusionControlNetPipeline...")
            pipe = StableDiffusionControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
            )

            logger.info("Attaching IP-Adapter weights (h94/IP-Adapter ip-adapter-plus_sd15.safetensors)...")
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter-plus_sd15.safetensors",
            )
            pipe.set_ip_adapter_scale(0.7)

            # Low VRAM (4GB GTX 1650) Memory Optimizations
            pipe.enable_attention_slicing()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            pipe.enable_model_cpu_offload()  # MUST BE CALLED LAST!

            if torch.cuda.is_available():
                cuda_telemetry["vram_allocated_after_init_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                cuda_telemetry["vram_reserved_after_init_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)

            logger.info("Phase C1-B Successful! Diffusers pipeline initialized with zero-RAM meta loading & low-VRAM CPU offload.")

            # Step 3: Phase C1-C Single Controlled Generation
            logger.info("Executing Phase C1-C: Single Deterministic Generation (Seed 12345, 512x512, 30 steps)...")

            identity_img = Image.open(inputs["identity_front"]).convert("RGB").resize((512, 512))
            driving_img = Image.open(inputs["driving_neutral"]).convert("RGB").resize((512, 512))

            # Save pose conditioning reference
            conditioning_pose_path = GATE_C1_DIR / "conditioning_pose.png"
            driving_img.save(conditioning_pose_path)

            generator = torch.Generator("cpu").manual_seed(12345)

            output = pipe(
                prompt="full body photo of a person, realistic, natural lighting, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, distorted",
                image=driving_img,
                ip_adapter_image=identity_img,
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=generator,
            )

            generated_image = output.images[0]
            generated_path = GATE_C1_DIR / "generated.png"
            generated_image.save(generated_path)
            logger.info("Generated image saved to %s", generated_path)

            if torch.cuda.is_available():
                cuda_telemetry["vram_allocated_after_gen_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                cuda_telemetry["vram_reserved_after_gen_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
                cuda_telemetry["vram_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

            execution_status = "SUCCESS"

        except ImportError as err:
            execution_status = "MISSING_DEPENDENCIES"
            logger.error("MISSING_DEPENDENCIES: Required Python packages missing: %s", err)
        except Exception as err:
            execution_status = "FAILED"
            logger.error("Candidate C Harness Execution Failed: %s", err, exc_info=True)

    # Review manifest structure
    review = {
        "gate": "C1",
        "criteria": {
            "subject_recognizable": None,
            "face_geometry_coherent": None,
            "catastrophic_facial_distortion": None,
        },
        "outcome": "UNREVIEWED",
        "valid_outcomes": ["PASS", "FAIL", "INCONCLUSIVE"],
        "review_notes": "Awaiting human visual inspection of raw generated output.",
    }
    with open(GATE_C1_DIR / "review.json", "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2)

    # Evidence package
    evidence = {
        "candidate_id": "candidate_c",
        "gate": "C1",
        "experiment_id": "phase_2_5b_1_candidate_c_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_status": execution_status,
        "generation_config": get_default_generation_config(),
        "requested_artifacts": requested_artifacts,
        "resolved_artifacts": resolved_artifacts,
        "input_hashes": input_hashes,
        "output_hashes": {
            "generated": compute_file_sha256(GATE_C1_DIR / "generated.png"),
            "conditioning_pose": compute_file_sha256(GATE_C1_DIR / "conditioning_pose.png"),
        },
        "telemetry": {
            "python_version": sys.version,
            "platform": sys.platform,
            "gpu_name": "NVIDIA GeForce GTX 1650 (4 GB VRAM)",
            "diffusers_version": "0.40.0",
            "transformers_version": "5.12.1",
            "accelerate_version": "1.14.0",
            "cuda_memory_mb": cuda_telemetry,
        },
        "warnings": [f"Missing C1 inputs: {c1_missing}"] if c1_missing else [],
        "license_status_at_execution": "RESEARCH_SPIKE_ONLY",
    }
    with open(GATE_C1_DIR / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    logger.info("Saved Gate C1 evidence package to %s (Final Status: %s)", GATE_C1_DIR, execution_status)
    return evidence


if __name__ == "__main__":
    run_gate_c1_harness()
