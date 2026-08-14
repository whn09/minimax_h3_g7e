#!/usr/bin/env python3
"""Check that a canonicalized NVFP4 checkpoint decodes correctly under the LOADER'S OWN defaults.

    docker exec h3 python3 /out/nvfp4_verify.py

nvfp4_diagnose.py answers "how does this file deviate"; this answers "did the rewrite land", and it
deliberately hard-codes the interpretation the stock loader will use -- low nibble first, linear
scale layout, no row permutation -- so a pass here means the server needs no flags and no fabricated
config.json. Anything below cos 0.99 means do not bother starting the server: NVFP4 is either
~0.9955 (correct) or ~0.000 (one interpretation wrong), with nothing in between.
"""
import glob
import os

import torch
from safetensors import safe_open

NV = os.environ.get("NVFP4_FILE", "/out/nvfp4_ref2va_fixed.safetensors")
BASE = os.environ.get("H3_BF16_TRANSFORMER", "/models/MiniMax-H3/Ref2VA/transformer")
LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
SAMPLE_PREFIXES = ("blocks.0.", "blocks.25.", "blocks.49.", "token_refiner.blocks.0.")


def cos(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    index = {}
    for path in sorted(glob.glob(os.path.join(BASE, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                index[key] = path

    worst = 1.0
    with safe_open(NV, framework="pt", device="cpu") as f:
        mods = [k[: -len(".weight_scale")] for k in sorted(f.keys())
                if k.endswith(".weight_scale")]
        for mod in [m for m in mods if m.startswith(SAMPLE_PREFIXES)]:
            q = f.get_tensor(mod + ".weight")
            s = f.get_tensor(mod + ".weight_scale").float()
            s2 = f.get_tensor(mod + ".weight_scale_2").float()
            v = torch.stack([LUT[(q & 0x0F).to(torch.long)], LUT[(q >> 4).to(torch.long)]], dim=-1)
            v = v.reshape(q.shape[0], q.shape[1] * 2)
            d = v * (s * s2).repeat_interleave(v.shape[1] // s.shape[1], dim=1)
            with safe_open(index[mod + ".weight"], framework="pt", device="cpu") as bf:
                w = bf.get_tensor(mod + ".weight").float()
            c = cos(d, w)
            worst = min(worst, c)
            print(f"  {mod:44s} cos={c:+.5f} rel={float((d - w).norm() / w.norm()):.4f}"
                  f"{'' if c > 0.99 else '   <<< BAD'}")
    print(f"worst cos = {worst:+.5f} -> {'OK' if worst > 0.99 else 'FAILED'}")
    raise SystemExit(0 if worst > 0.99 else 1)


if __name__ == "__main__":
    main()
