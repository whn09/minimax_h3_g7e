#!/usr/bin/env python3
"""Rewrite a ComfyUI-converter NVFP4 MiniMax-H3 checkpoint into the form sglang already expects.

    docker cp nvfp4_canonicalize.py h3:/out/ && docker exec h3 python3 /out/nvfp4_canonicalize.py

Three deviations, all measured against the official bf16 weights by nvfp4_diagnose.py (cos 0.9955 /
rel 0.094 with all three fixed; cos ~0.000 / rel ~1.41 with any one of them left wrong):

  1. `weight_scale` is stored SWIZZLED in 128x4 tiles; the loader infers `scale_layout=linear`.
  2. `weight` nibbles are HIGH-first (element 2i in the high nibble); the loader infers no swap.
  3. `qkv_proj` rows are blocked `[q_all; k_all; v_all]`; H3 wants per-head interleaved
     `[q_h, k_h, v_h]`. head_dim is 128 (21504 = 3 x 56 x 128).

Fixing this in the checkpoint rather than in the runtime, for two reasons. (1) There is no runtime
knob for the qkv row order at all -- `_build_nvfp4_config_from_safetensors_files`' `packed_qkv`
handling is FLUX-specific and its pattern does not match H3's module names -- so the other two knobs
alone cannot produce a correct model. (2) With the file canonical, the stock auto-inferred config is
already right, so nothing has to be passed on the command line and no `config.json` has to be
fabricated: point `--transformer-weights-path` at the rewritten file and leave `--quantization`
unset.

Peak host RAM is roughly the file size (24.4 GB) because save_file wants every tensor at once, so
stop the server first. All H3 scale shapes are already 128x4-tile aligned, hence the assert rather
than a padding path.
"""
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.environ.get("NVFP4_SRC", "/out/nvfp4_ref2va.safetensors")
DST = os.environ.get("NVFP4_DST", "/out/nvfp4_ref2va_fixed.safetensors")

#: (3, 56, 128) -> (56, 3, 128): blocked q/k/v -> per-head interleaved.
QKV_IDX = torch.arange(3 * 56 * 128).reshape(3, 56, 128).permute(1, 0, 2).reshape(-1)


def unswizzle(sc):
    """128x4-tile swizzled scale layout -> linear. Dtype-preserving (pure element permutation)."""
    m, k = sc.shape
    assert m % 128 == 0 and k % 4 == 0, f"unaligned scale shape {(m, k)}; needs a padding path"
    return (sc.reshape(m // 128, k // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
            .reshape(m, k).contiguous())


def main():
    out = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        quant_mods = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
        print(f"{len(keys)} tensors, {len(quant_mods)} quantized modules", flush=True)
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            mod, _, leaf = key.rpartition(".")
            if mod in quant_mods and leaf == "weight":
                assert tensor.dtype == torch.uint8, (key, tensor.dtype)
                tensor = ((tensor & 0x0F) << 4) | ((tensor >> 4) & 0x0F)
                if mod.endswith("qkv_proj"):
                    tensor = tensor[QKV_IDX]
            elif mod in quant_mods and leaf == "weight_scale":
                tensor = unswizzle(tensor)
                if mod.endswith("qkv_proj"):
                    tensor = tensor[QKV_IDX]
            out[key] = tensor.contiguous()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(keys)}", flush=True)

    print(f"writing {DST}", flush=True)
    save_file(out, DST, metadata=metadata or None)
    print("done", flush=True)


if __name__ == "__main__":
    main()
