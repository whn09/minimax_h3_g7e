#!/usr/bin/env python3
"""Identify how a third-party NVFP4 MiniMax-H3 checkpoint deviates from what sglang assumes.

Run inside the sglang container (needs the official bf16 weights to compare against):

    docker exec h3 python3 /out/nvfp4_diagnose.py

Why this exists
---------------
`lilcheaty/MiniMax-H3-NVFP4`'s `MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors` loads and runs, but the
video is pure mosaic (bitrate 20x, inter-frame motion energy 40x a healthy sample). The loader's
auto-inferred config logs

    group_size=16, 268 excluded modules, packed_qkv=False, comfy_quant=False,
    scale_layout=linear, swap_nibbles=False

and three of those guesses are wrong. Guessing knobs and re-running the 6-minute server start is a
bad loop; dequantizing one layer offline and correlating against the official bf16 tensor settles
each knob in seconds, and does it per-module so a partial mismatch is visible.

Method
------
Two independent probes, deliberately in this order:

1. **scale-only fingerprint** -- `weight_scale * weight_scale_2 * 6` should be the per-16-element
   block amax of the bf16 tensor. This touches no nibbles, so it isolates "is this even the same
   tensor / same base?" and "is the scale swizzled?" from the packing question. A `sorted_rowmax`
   cosine of ~1.0 with a matching maximum proves same tensor even when the aligned cosine is low,
   which is what says "permutation" rather than "different weights".
2. **full dequant** -- decode the nibbles and correlate. NVFP4 lands at cos ~0.9955 / rel ~0.094
   when every interpretation is right, and at cos ~0.000 / rel ~1.41 (orthogonal) when any single
   one is wrong. There is no ambiguous middle, so each knob is decidable on its own.

Measured verdict for that checkpoint (24/24 sampled modules, both stacks, all depths):
  * `weight_scale` is SWIZZLED (128x4 tiles), not linear.
  * nibbles are HIGH-first (element 2i in the high nibble), i.e. swap_weight_nibbles is needed.
  * `qkv_proj` rows are blocked `[q_all; k_all; v_all]`; H3 wants per-head interleaved
    `[q_h, k_h, v_h]` -- recovered by viewing the row index as (3, 56, 128) and permuting to
    (56, 3, 128). Note head_dim is 128 here (21504 = 3 x 56 x 128), not 96.

The first two are config knobs; the third has no knob at all, which is why the fix is to rewrite the
checkpoint (see nvfp4_canonicalize.py) instead of flipping runtime flags.
"""
import glob
import itertools
import os

import torch
from safetensors import safe_open

NV = os.environ.get("NVFP4_FILE", "/out/nvfp4_ref2va.safetensors")
BASE = os.environ.get("H3_BF16_TRANSFORMER", "/models/MiniMax-H3/Ref2VA/transformer")

#: e2m1, sign bit high. Index = the 4-bit code, value = what it decodes to.
LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])

#: (3, 56, 128) -> (56, 3, 128). `ckpt_rows[QKV_IDX]` is the official row order.
QKV_IDX = torch.arange(3 * 56 * 128).reshape(3, 56, 128).permute(1, 0, 2).reshape(-1)

SAMPLE_PREFIXES = ("blocks.0.", "blocks.13.", "blocks.25.", "blocks.49.",
                   "token_refiner.blocks.0.", "token_refiner.blocks.1.")


def build_base_index():
    index = {}
    for path in sorted(glob.glob(os.path.join(BASE, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                index[key] = path
    return index


def cos(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def unswizzle(sc):
    """128x4-tile swizzled scale layout -> linear, padding when a shape is not tile-aligned."""
    m, k = sc.shape
    mp = ((m + 127) // 128) * 128
    kp = ((k + 3) // 4) * 4
    padded = torch.zeros(mp, kp, dtype=sc.dtype)
    padded[:m, :k] = sc
    return (padded.reshape(mp // 128, kp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
            .reshape(mp, kp)[:m, :k])


def dequant(q, s, s2, low_nibble_first, swizzled, qkv_permute=False):
    lo = (q & 0x0F).to(torch.long)
    hi = (q >> 4).to(torch.long)
    first, second = (lo, hi) if low_nibble_first else (hi, lo)
    v = torch.stack([LUT[first], LUT[second]], dim=-1).reshape(q.shape[0], q.shape[1] * 2)
    sc = unswizzle(s) if swizzled else s
    if qkv_permute:
        v = v[QKV_IDX]
        sc = sc[QKV_IDX]
    return v * (sc * s2).repeat_interleave(v.shape[1] // sc.shape[1], dim=1)


def main():
    index = build_base_index()

    def base_get(key):
        with safe_open(index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key).float()

    with safe_open(NV, framework="pt", device="cpu") as f:
        mods = sorted(k[: -len(".weight_scale")] for k in f.keys() if k.endswith(".weight_scale"))
        print(f"{NV}: {len(list(f.keys()))} tensors, {len(mods)} quantized modules")

        def get(mod):
            return (f.get_tensor(mod + ".weight"),
                    f.get_tensor(mod + ".weight_scale").float(),
                    f.get_tensor(mod + ".weight_scale_2").float())

        # --- probe 1: scale-only fingerprint, no nibbles involved -------------------------------
        print("\n== probe 1: block-amax fingerprint (scale only) ==")
        for mod in [m for m in mods if m.startswith(SAMPLE_PREFIXES[:1])] + \
                   [m for m in mods if m.startswith("token_refiner.blocks.0.")]:
            w = base_get(mod + ".weight")
            q, s, s2 = get(mod)
            gs = w.shape[1] // s.shape[1]
            a_true = w.reshape(w.shape[0], -1, gs).abs().amax(-1)
            a_lin = s * float(s2) * 6.0
            a_swz = unswizzle(s) * float(s2) * 6.0
            row_t, row_c = a_true.amax(1), a_lin.amax(1)
            print(f"  {mod:44s} gs={gs} linear={cos(a_lin, a_true):+.4f} "
                  f"unswizzled={cos(a_swz, a_true):+.4f} "
                  f"sorted_rowmax={cos(row_c.sort().values, row_t.sort().values):+.4f} "
                  f"max {float(row_t.max()):.3f} vs {float(row_c.max()):.3f}")

        # --- probe 2: full dequant, one knob at a time -------------------------------------------
        print("\n== probe 2: full dequant, 2 nibble orders x 2 scale layouts ==")
        for mod in ("blocks.0.attn.out_proj", "blocks.0.mlp.fc1"):
            w = base_get(mod + ".weight")
            q, s, s2 = get(mod)
            for low_first in (True, False):
                for swz in (False, True):
                    d = dequant(q, s, s2, low_first, swz)
                    print(f"  {mod:26s} low_nibble_first={low_first!s:5s} swizzled={swz!s:5s} "
                          f"cos={cos(d, w):+.5f} rel={float((d - w).norm() / w.norm()):.4f}")

        # --- probe 3: the qkv row permutation ----------------------------------------------------
        print("\n== probe 3: qkv row layout (structured axis permutations) ==")
        mod = "blocks.0.attn.qkv_proj"
        w = base_get(mod + ".weight")
        q, s, s2 = get(mod)
        d = dequant(q, s, s2, low_nibble_first=False, swizzled=True)
        print(f"  {mod} identity cos={cos(d, w):+.5f}")
        n = w.shape[0]
        for dims in [(3, 56, 128), (56, 3, 128), (3, 128, 56), (4, 56, 96), (56, 4, 96)]:
            size = 1
            for dim in dims:
                size *= dim
            if size != n:
                continue
            grid = torch.arange(n).reshape(dims)
            for perm in itertools.permutations(range(len(dims))):
                if perm == tuple(range(len(dims))):
                    continue
                c = cos(d[grid.permute(perm).reshape(-1)], w)
                if abs(c) > 0.30:
                    print(f"    {dims} permute{perm} cos={c:+.5f}")

        # --- verdict: all three fixes together, across depth and both stacks --------------------
        print("\n== verdict: all three fixes applied ==")
        worst = 1.0
        for mod in [m for m in mods if m.startswith(SAMPLE_PREFIXES)]:
            w = base_get(mod + ".weight")
            q, s, s2 = get(mod)
            d = dequant(q, s, s2, low_nibble_first=False, swizzled=True,
                        qkv_permute=mod.endswith("qkv_proj"))
            c = cos(d, w)
            worst = min(worst, c)
            print(f"  {mod:44s} {tuple(q.shape)} cos={c:+.5f} "
                  f"rel={float((d - w).norm() / w.norm()):.4f}"
                  f"{'' if c > 0.99 else '   <<< BAD'}")
        print(f"worst cos = {worst:+.5f} -> {'OK' if worst > 0.99 else 'FAILED'}")


if __name__ == "__main__":
    main()
