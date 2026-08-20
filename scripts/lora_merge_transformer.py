#!/usr/bin/env python3
"""把一个 H3 Turbo LoRA **离线**合并进 bf16 transformer，输出还是一份普通 bf16 checkpoint。

    docker cp lora_merge_transformer.py h3n:/tmp/ && docker exec \
      -e SRC=/models/MiniMax-H3/FL2VA/transformer \
      -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
      -e DST=/out/turbo_v4_600_bf16 h3n python3 /tmp/lora_merge_transformer.py

为什么要离线合并而不用 sglang 的 `--lora-path`
-----------------------------------------------
运行时"合并式 LoRA"在**量化**权重上是坏的：`runtime/layers/lora/linear.py` 按 `[out, in]`
做 in-place add，而 fp8 把权重转置存、NVFP4 更是 `[N, K/2]` 的 e2m1 packed uint8 + 块标度 ——
形状对不上（fp8 上实测报 `21504 vs 5376`）。`--lora-merge-mode dynamic` 能跑但只快 3% 且画质
跌破重跑地板。所以想同时要 **NVFP4 + LoRA**，唯一干净的路是：先在 bf16 上合并，再拿合并后的
权重走同一套 NVFP4 量化配方（`nvfp4_quantize_transformer.py`）。量化器只看 SRC 目录，
所以输出目录长得和原 checkpoint 一模一样（同名 shard + index + config）就能直接接上。

键名对得上，不需要任何映射
--------------------------
LoRA（`larryvrh/MiniMax-H3-Turbo-Lora`，comfy 命名）与 diffusers 转出的 checkpoint 用的是同一套
名字：`blocks.N.attn.qkv_proj` / `mlp.fc{1,2}` / `adaln_proj.linear` / `token_refiner.blocks.N.*` /
`final_layer.adaln_proj.linear`。所以 `X.lora_A.weight` + `X.lora_B.weight` 直接落到 `X.weight`。
259 个模块（518 个张量），全部要命中；有一个没命中就是版本对不上，脚本会 exit 1 而不是静默少合。

`W_eff = W + strength * (B @ A)`，**没有 alpha/rank 缩放**（模型卡明说 alpha = rank），
strength 默认 1.0（模型卡：就是按 1.0 调的，别乱动）。delta 用 fp32 算完再 cast 回 bf16。
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src = os.environ["SRC"]
lora_path = os.environ["LORA"]
dst = os.environ["DST"]
strength = float(os.environ.get("STRENGTH", "1.0"))

os.makedirs(dst, exist_ok=True)
index_path = os.path.join(src, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))

# LoRA 整份进内存（bf16 ~780 MB），按目标权重名归组
lora = {}
with safe_open(lora_path, framework="pt") as f:
    for k in f.keys():
        if not (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")):
            print(f"UNEXPECTED lora key {k}", file=sys.stderr)
            sys.exit(1)
        base, side = k.rsplit(".lora_", 1)
        lora.setdefault(base + ".weight", {})[side[0]] = f.get_tensor(k)
missing_side = [k for k, v in lora.items() if set(v) != {"A", "B"}]
if missing_side:
    print(f"LORA 有单边的模块: {missing_side[:5]}", file=sys.stderr)
    sys.exit(1)
print(f"lora modules: {len(lora)}  strength={strength}")

applied, worst = set(), 0.0
for sh in shards:
    tensors = {}
    with safe_open(os.path.join(src, sh), framework="pt") as f:
        for k in f.keys():
            w = f.get_tensor(k)
            lw = lora.get(k)
            if lw is not None:
                a, b = lw["A"].float(), lw["B"].float()
                # A=[rank, in], B=[out, rank], W=[out, in]
                if a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != tuple(
                    w.shape
                ):
                    print(
                        f"SHAPE MISMATCH {k}: W{tuple(w.shape)} "
                        f"A{tuple(a.shape)} B{tuple(b.shape)}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                delta = (b @ a) * strength
                w32 = w.float()
                rel = (delta.norm() / w32.norm()).item()
                worst = max(worst, rel)
                w = (w32 + delta).to(w.dtype)
                applied.add(k)
                del delta, w32, a, b
            tensors[k] = w
    save_file(tensors, os.path.join(dst, sh), metadata={"format": "pt"})
    print(f"  {sh}: {len(tensors)} tensors, merged {len(applied)} so far", flush=True)
    del tensors

for extra in ("model.safetensors.index.json", "config.json"):
    p = os.path.join(src, extra)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(dst, extra))

unmatched = sorted(set(lora) - applied)
print(f"merged {len(applied)}/{len(lora)} modules, 最大 |delta|/|W| = {worst:.4f}")
if unmatched:
    print(f"NOT MATCHED ({len(unmatched)}): {unmatched[:8]}", file=sys.stderr)
    sys.exit(1)
print(f"wrote {dst}")
