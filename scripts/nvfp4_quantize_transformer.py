#!/usr/bin/env python3
"""把 H3 的 bf16 transformer 量化成 sglang 能直接加载的 NVFP4 checkpoint。

    docker cp nvfp4_quantize_transformer.py h3:/tmp/ && docker exec \
      -e SRC=/models/MiniMax-H3/FL2VA/transformer -e DST=/out/nvfp4_fl2va.safetensors \
      h3 python3 /tmp/nvfp4_quantize_transformer.py

为什么要自己写:Abiray 那个仓里 FL2VA 只有 `_pruned_nvfp4`(裁过的模型,和 stock 的
FL2VA/transformer 不是一回事),只有 Ref2VA 有 `_nvfp4_mixed`。想把 README 那张表的 fl2va 半边
补齐,就得从原始权重量化 —— 顺带这也是"从原始模型量化"那条路的证据。sglang 自带的
`tools/build_modelopt_nvfp4_transformer.py` 干不了这件事:它只是把 **已经** modelopt 导出好的
NVFP4 文件和 bf16 做混搭,不做量化。

配方 = 照抄 `/out/nvfp4_ref2va_fixed.safetensors` 的实测布局(那份是已验证能出正常片的)
--------------------------------------------------------------------------------------
208 个线性层(52 × qkv/out/fc1/fc2,52 = 50 blocks + 2 token_refiner.blocks):

    <name>.weight          U8       [N, K/2]    e2m1,**低 nibble 放偶数下标**
    <name>.weight_scale    F8_E4M3  [N, K/16]   每 16 元素一个块标度,**linear 布局不 swizzle**
    <name>.weight_scale_2  F32      []          per-tensor 全局标度 = amax/(448*6)

解量化是 **乘**:w ≈ e2m1 × weight_scale × weight_scale_2(modelopt 约定)。

50 个 `blocks.*.adaln_proj.linear.weight` 存成 **F8_E4M3 裸 cast**(无标度)。这不是我拍的,是
量过的:那份 ref2va 文件里的 adaln 对上原始 bf16 rel err 0.0269、amax 1.25 vs 1.3047,就是一次
不带标度的 e4m3 cast。adaln 占 DiT 权重 40% 但算力 0%,所以它是纯显存项 —— 留 bf16 会让文件从
23 GB 变 36 GB,而且和 ref2va 那半张表的配方就不一致了。

其余张量(norm、q_norm/k_norm、patch_proj、final_layer、rope.inv_freq、bias)原样 bf16 复制。

qkv 的行序:**不要动**
---------------------
原始 bf16 里 qkv 是每头 [q,k,v] 交错的,量化是逐行的,所以量化后行序还是 checkpoint 那套 ——
正好是 sglang loader 期望的输入。行序的置换由 loader + `patch_h3_qkv_scale_reorder.py` 负责
(权重和块标度都要换,少换一个就是一片灰糊)。这里多做一次 reorder 就是错。

`input_scale` 不写
------------------
ref2va 那份也没有。w4a4 的激活全局标度就留 sglang 的初始值;需要动态标度的话是
`patch_nvfp4_dynamic_act.py` + `H3_FP4_DYN_ACT=1` 那条路,和 checkpoint 无关。

自检
----
跑完对随机抽的层算 round-trip rel err,应该落在 **0.094±0.002**(group-16 round-to-nearest-e2m1
的地板)。明显更大 = 打包/标度算错了;明显更小 = 不可能,先怀疑参考张量取错了。
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# 只量化这 4 类线性层。`blocks.N.` 和 `token_refiner.blocks.N.` 都要,各 52 个。
QUANT_RE = re.compile(
    r"(?:^|\.)blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$"
)
FP8_RE = re.compile(r"(?:^|\.)blocks\.\d+\.adaln_proj\.linear\.weight$")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
#: 相邻幅值的中点 = round-to-nearest 的桶边界。bucketize 的 right=False 让恰好落在边界上的值
#: 归到上一档(与 fp8 cast 的 ties-to-even 不完全一致,但只影响 0 测度集合)。
E2M1_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
FP8_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16
ROWS = 4096          # 分行处理:fc1 整块 fp32 中间量 616 MB,没必要


def quantize_nvfp4(w):
    """bf16 [N, K] -> (packed U8 [N, K/2], scale F8_E4M3 [N, K/16], scale_2 F32 [])"""
    n, k = w.shape
    assert k % BLOCK == 0, (n, k)
    amax = w.abs().amax().float()
    w_s2 = (amax / (FP8_MAX * E2M1_MAX)).clamp(min=1e-30)
    packed = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // BLOCK), dtype=torch.float8_e4m3fn)
    for i in range(0, n, ROWS):
        j = min(i + ROWS, n)
        blk = w[i:j].float().reshape(j - i, k // BLOCK, BLOCK)
        # 块标度:让块内最大值正好落到 e2m1 的 6.0 上,再压回 fp8 能表示的范围
        sf = (blk.abs().amax(-1) / E2M1_MAX / w_s2).clamp(max=FP8_MAX)
        sf8 = sf.to(torch.float8_e4m3fn)
        scales[i:j] = sf8
        eff = (sf8.float() * w_s2).clamp(min=1e-30).unsqueeze(-1)
        q = (blk / eff).clamp(-E2M1_MAX, E2M1_MAX)
        idx = torch.bucketize(q.abs(), E2M1_EDGES).to(torch.uint8)
        code = (idx | (q < 0).to(torch.uint8) * 8).reshape(j - i, k)
        # 低 nibble = 偶数下标(与解量化端 `lo, hi = q & 0xF, q >> 4` 对齐)
        packed[i:j] = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, scales, w_s2.to(torch.float32).reshape(())


def dequant_nvfp4(q, sf, w_s2):
    lut = torch.tensor(E2M1, dtype=torch.float32)
    lo, hi = q & 0x0F, (q >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    vals = torch.where((nib & 0x08) > 0, -1.0, 1.0) * lut[(nib & 0x07).long()]
    s = sf.float() * w_s2.float()
    return (vals.reshape(vals.shape[0], sf.shape[-1], BLOCK) * s.unsqueeze(-1)).reshape(
        q.shape[0], -1
    )


def main():
    src = os.environ.get("SRC", "/models/MiniMax-H3/FL2VA/transformer")
    dst = os.environ.get("DST", "/out/nvfp4_fl2va.safetensors")
    check_every = int(os.environ.get("CHECK_EVERY", "40"))

    index = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))

    out = {}
    quant_meta = {}
    n_q = n_fp8 = n_copy = 0
    worst = (0.0, "")
    for shard in shards:
        with safe_open(os.path.join(src, shard), "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if QUANT_RE.search(name):
                    packed, scales, w_s2 = quantize_nvfp4(t)
                    out[name] = packed
                    out[name + "_scale"] = scales
                    out[name + "_scale_2"] = w_s2
                    quant_meta[name[: -len(".weight")]] = {"format": "nvfp4"}
                    n_q += 1
                    if n_q % check_every == 1:
                        ref = t.float()
                        rel = float(
                            (dequant_nvfp4(packed, scales, w_s2) - ref).norm()
                            / ref.norm()
                        )
                        print("  check %-46s rel=%.4f" % (name, rel), flush=True)
                        if rel > worst[0]:
                            worst = (rel, name)
                elif FP8_RE.search(name):
                    out[name] = t.to(torch.float8_e4m3fn)
                    n_fp8 += 1
                else:
                    out[name] = t
                    n_copy += 1
        print("shard %s done (q=%d fp8=%d copy=%d)" % (shard, n_q, n_fp8, n_copy),
              flush=True)

    if n_q != 208:
        print("!!! 量化了 %d 个线性层,预期 208 —— 层名规则和这个 checkpoint 不匹配" % n_q)
        return 1
    if worst[0] > 0.12:
        print("!!! round-trip rel err %.4f (%s) 超过 group-16 地板 0.094 太多" % worst)
        return 1

    save_file(
        out,
        dst,
        metadata={
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": quant_meta}
            ),
            "target_format": "NVFP4",
            "converted_by": "nvfp4_quantize_transformer.py",
        },
    )
    print("wrote %s: %d tensors (q=%d fp8=%d copy=%d) worst rel=%.4f (%s)"
          % (dst, len(out), n_q, n_fp8, n_copy, worst[0], worst[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
