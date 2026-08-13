"""Checkpoint integrity and state dict validation for MobileFaceSwap."""
import hashlib
import os
import sys

files = [
    'services/inference/models/mobilefaceswap/MobileFaceSwap_224.pdparams',
    'services/inference/models/mobilefaceswap/arcface.pdparams',
]

print('=== CHECKPOINT FILE CHECKSUMS ===')
for fpath in files:
    sha256 = hashlib.sha256()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f'  {os.path.basename(fpath)}')
    print(f'    Size   : {size_mb:.2f} MB')
    print(f'    SHA256 : {sha256.hexdigest()}')

print()
print('=== PADDLE STATE DICT VERIFICATION ===')

try:
    import paddle
    paddle.set_device('cpu')

    state = paddle.load('services/inference/models/mobilefaceswap/arcface.pdparams')
    print('arcface.pdparams: VALID PaddlePaddle state dict')
    print('  Keys in state dict:', len(state))
    for k in list(state.keys())[:6]:
        v = state[k]
        shape = list(v.shape) if hasattr(v, 'shape') else 'N/A'
        print(f'    {k}: shape={shape}')

    state2 = paddle.load('services/inference/models/mobilefaceswap/MobileFaceSwap_224.pdparams')
    print('MobileFaceSwap_224.pdparams: VALID PaddlePaddle state dict')
    print('  Keys in state dict:', len(state2))
    for k in list(state2.keys())[:6]:
        v = state2[k]
        shape = list(v.shape) if hasattr(v, 'shape') else 'N/A'
        print(f'    {k}: shape={shape}')

except ImportError:
    print('PaddlePaddle not installed - skipping state dict verification.')
    print('(File checksums above are sufficient for archive integrity check.)')
except Exception as e:
    print(f'Load failed: {e}')
