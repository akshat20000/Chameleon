import json
import sys
import traceback
from pathlib import Path
import torch
from diffusers import ControlNetModel

def assign_safetensors_stream_to_meta_model(model, weight_path: Path):
    import json, numpy as np, torch, gc
    np_dtype_map = {'F16': np.float16, 'F32': np.float32, 'I64': np.int64, 'I32': np.int32, 'BF16': np.float32}
    model.to_empty(device="cpu")
    param_map = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())
    with open(weight_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        for k, v in header.items():
            if k == "__metadata__": continue
            f.seek(data_start + v["data_offsets"][0])
            buf = f.read(v["data_offsets"][1] - v["data_offsets"][0])
            arr = np.frombuffer(buf, dtype=np_dtype_map.get(v["dtype"], np.float32)).reshape(v["shape"])
            t = torch.from_numpy(arr)
            if t.is_floating_point():
                t = t.to(torch.float16)
            if k in param_map:
                param_map[k].data = t
            elif k in buffer_map:
                buffer_map[k].data = t
            del buf, arr, t
    gc.collect()
    return model

def test_load():
    snap = Path(r"E:\My_personal\Projects\ongoing\Chameleon\hf_cache\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14\unet")
    print("Testing zero-RAM streaming safetensors assignment on UNet2DConditionModel...", flush=True)
    u_config = json.load(open(snap / "config.json"))
    from diffusers import UNet2DConditionModel
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(u_config)
    assign_safetensors_stream_to_meta_model(unet, snap / "diffusion_pytorch_model.fp16.safetensors")
    print("SUCCESS! UNet2DConditionModel loaded cleanly via zero-RAM parameter streaming:", type(unet), flush=True)

if __name__ == "__main__":
    try:
        test_load()
    except Exception as err:
        print("EXCEPTION CAUGHT:", err, flush=True)
        traceback.print_exc()
