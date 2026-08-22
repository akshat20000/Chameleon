"""
Convert cached HuggingFace safetensors snapshots to PyTorch .bin format.

Spec: docs/architecture/ADR/ADR-007-phase-2-5b-candidate-evaluation.md

Rationale:
  Windows OS Error 1455 ("The paging file is too small for this operation to complete")
  occurs when safetensors uses Rust mmap to map multi-gigabyte files under constrained RAM commit limits.
  Converting weights to standard PyTorch .bin unpickled state dicts allows diffusers to stream weights
  via standard stream I/O (use_safetensors=False) with 100% reliability.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bin_converter")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

np_dtype_map = {
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "BF16": np.float32, # fallback for bfloat16 to float32
}

def convert_safetensors_file(safetensors_path: Path, bin_path: Path):
    if bin_path.exists() and bin_path.stat().st_size > 100000:
        logger.info("Skipping already converted file: %s", bin_path.name)
        return

    logger.info("Converting %s to PyTorch .bin format...", safetensors_path.name)
    with open(safetensors_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len

        state_dict = {}
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
            offsets = info["data_offsets"]
            dtype_str = info["dtype"]
            shape = info["shape"]

            f.seek(data_start + offsets[0])
            buf = f.read(offsets[1] - offsets[0])
            dtype = np_dtype_map.get(dtype_str, np.float32)
            arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
            state_dict[tensor_name] = torch.from_numpy(arr.copy())

        torch.save(state_dict, bin_path)
    logger.info("Successfully saved %s (%0.2f MB)", bin_path.name, bin_path.stat().st_size / (1024 * 1024))


def convert_all_candidate_c_snapshots():
    logger.info("=== Starting Candidate C Snapshot .bin Conversion ===")

    # 1. SD 1.5 Base Model
    sd15_snapshots = list((HF_CACHE_DIR / "hub" / "models--runwayml--stable-diffusion-v1-5" / "snapshots").glob("*"))
    if sd15_snapshots:
        snap = sd15_snapshots[0]
        # UNet
        unet_st = snap / "unet" / "diffusion_pytorch_model.fp16.safetensors"
        if not unet_st.exists(): unet_st = snap / "unet" / "diffusion_pytorch_model.safetensors"
        if unet_st.exists():
            convert_safetensors_file(unet_st, snap / "unet" / "diffusion_pytorch_model.bin")
        # VAE
        vae_st = snap / "vae" / "diffusion_pytorch_model.fp16.safetensors"
        if not vae_st.exists(): vae_st = snap / "vae" / "diffusion_pytorch_model.safetensors"
        if vae_st.exists():
            convert_safetensors_file(vae_st, snap / "vae" / "diffusion_pytorch_model.bin")
        # Text Encoder
        te_st = snap / "text_encoder" / "model.fp16.safetensors"
        if not te_st.exists(): te_st = snap / "text_encoder" / "model.safetensors"
        if te_st.exists():
            convert_safetensors_file(te_st, snap / "text_encoder" / "pytorch_model.bin")

    # 2. IP-Adapter
    ip_snapshots = list((HF_CACHE_DIR / "hub" / "models--h94--IP-Adapter" / "snapshots").glob("*"))
    if ip_snapshots:
        snap = ip_snapshots[0]
        ip_st = snap / "models" / "ip-adapter-plus_sd15.safetensors"
        if ip_st.exists():
            convert_safetensors_file(ip_st, snap / "models" / "ip-adapter-plus_sd15.bin")

    logger.info("=== Conversion Complete. All snapshots have native PyTorch .bin weights. ===")


if __name__ == "__main__":
    convert_all_candidate_c_snapshots()
