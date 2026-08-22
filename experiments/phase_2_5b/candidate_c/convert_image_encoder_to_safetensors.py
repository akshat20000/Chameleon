import json, numpy as np, torch
from pathlib import Path

bin_path = Path(r"E:\My_personal\Projects\ongoing\Chameleon\hf_cache\hub\models--h94--IP-Adapter\snapshots\018e402774aeeddd60609b4ecdb7e298259dc729\models\image_encoder\pytorch_model.bin")
st_path = bin_path.parent / "model.safetensors"

print(f"Reading {bin_path.name} state dict via mmap=True...", flush=True)
sd = torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)

header = {}
offset = 0

print("Calculating header offsets...", flush=True)
for k, v in sd.items():
    shape = list(v.shape)
    numel = v.numel()
    element_size = 2 if v.is_floating_point() else v.element_size()
    dtype_str = "F16" if v.is_floating_point() else ("I64" if v.dtype == torch.int64 else "I32")
    b_len = numel * element_size
    header[k] = {
        "dtype": dtype_str,
        "shape": shape,
        "data_offsets": [offset, offset + b_len]
    }
    offset += b_len

header_bytes = json.dumps(header).encode("utf-8")
header_len = len(header_bytes)

print(f"Streaming {st_path.name} to disk ({offset / (1024*1024):.2f} MB)...", flush=True)
with open(st_path, "wb") as f:
    f.write(header_len.to_bytes(8, "little"))
    f.write(header_bytes)
    for k in list(sd.keys()):
        v = sd.pop(k)
        if v.is_floating_point():
            arr = v.to(torch.float16).numpy()
        else:
            arr = v.numpy()
        f.write(arr.tobytes())
        del v, arr

print(f"SUCCESS! Created {st_path} ({st_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)
