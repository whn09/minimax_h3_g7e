"""Dequantize the NVFP4 checkpoint back to bf16 to separate weight error from activation error.

flashinfer's fp4 path is w4a4: ModelOptFp4LinearMethod.apply quantizes the activations too. Running
the SAME 4-bit weights with full-precision activations isolates which half of that costs the SSIM.
Dropping the *_scale keys is essential, otherwise the loader re-infers an NVFP4 config.
"""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/out/nvfp4_ref2va_fixed.safetensors"
DST = "/out/nvfp4_dequant_bf16.safetensors"
LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                   dtype=torch.float32)

out = {}
with safe_open(SRC, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    quant = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
    for i, k in enumerate(keys):
        mod, _, leaf = k.rpartition(".")
        if mod in quant and leaf in ("weight_scale", "weight_scale_2"):
            continue
        if mod in quant and leaf == "weight":
            q = f.get_tensor(k)
            s = f.get_tensor(mod + ".weight_scale").float()
            s2 = f.get_tensor(mod + ".weight_scale_2").float()
            v = torch.stack([LUT[(q & 0x0F).to(torch.long)], LUT[(q >> 4).to(torch.long)]], -1)
            v = v.reshape(q.shape[0], q.shape[1] * 2)
            out[k] = (v * (s * s2).repeat_interleave(v.shape[1] // s.shape[1], 1)).to(torch.bfloat16)
        else:
            out[k] = f.get_tensor(k)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)
print(f"writing {DST} ({len(out)} tensors)", flush=True)
save_file(out, DST)
print("done", flush=True)
