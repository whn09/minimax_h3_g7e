#!/usr/bin/env python3
"""Sol-Attn（NVlabs/Sana @ sol-engine）在 g7e（sm_120）上的孤立微基准。

为什么要孤立测：sol_attn 是**动态阈值**稀疏（threshold = mean + tau*std，见
sol_attn/triton_ref/preprocess.py:_diag_threshold_kernel），tau 是 z-score 而不是 top-K 预算。
z-score 固定 ⇒ 保留**比例**大致与序列长度无关 ⇒ 保留块数 ∝ n ⇒ attention 仍是 O(n²)，
只是常数变小。这条断言必须在两个片长上同时量才算证实：如果 sol/dense 的加速比在
5 s 和 15 s 上基本相同，就说明它按比例砍、指数没动。

口径：H3 DiT 单卡形状 = heads 56 / head_dim 128 / bf16 / 一条 packed 序列（batch 1）。
序列长度用实测值：768p 5 s = 39760 token，15 s 按帧数比 362/124 放大。

    docker exec h3n python3 /tmp/sol_attn_micro.py            # 默认扫 tau
    SEQS=39760 TAUS=1.0 docker exec ...

输出每行：seq / 后端 / tau / ms / 相对 dense 的倍数 / 保留块比例（tau 档才有）。

**实测结论（g7e，56 头）：断言成立。** 保留块比例在两个片长上**一位不差地相同**
（tau 0.5/1.0/1.5/2.0 → 30.3% / 15.1% / 6.1% / 1.9%），对 sage 的加速比也一样
（tau=1.0：2.859× @5s vs 2.860× @15s）。隐含指数 sage n^1.98、sol tau=1.0 n^1.98 ——
它把 y 轴按比例压下来，没有动斜率。E2E 与画质在 README 的「稀疏 attention」一节。
注意：随机高斯 q/k 的尾比真实激活轻，所以这里的保留比例是乐观值（E2E 只拿到理论上限的 ~77%）。
"""

import os
import time

import torch

HEADS = int(os.environ.get("HEADS", 56))
DIM = 128
SEQS = [int(x) for x in os.environ.get("SEQS", "39760,116060").split(",")]
TAUS = [float(x) for x in os.environ.get("TAUS", "0.5,1.0,1.5,2.0,3.0").split(",")]
ITERS = int(os.environ.get("ITERS", 5))
dev = "cuda"


def bench(fn, iters=ITERS):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def retained_fraction(q, k, tau, scale):
    """复算 threshold 与 proxy score，量出"有多少 KV 块被判为要精确算"。

    与 kernel 同一套统计量：block 的 proxy score = q_centroid · kc_mean(block)。
    这不是 kernel 的内部计数（拿不到），而是同公式的 host 复算，用来看 tau 与稀疏度的映射。
    """
    B = 64
    T = q.shape[1]
    nb = (T + B - 1) // B
    pad = nb * B - T
    qq = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad)).float()
    kk = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)).float()
    qc = qq.view(1, nb, B, HEADS, DIM).mean(2)          # [1, nb, H, D] query 质心
    kc = kk.view(1, nb, B, HEADS, DIM).mean(2)          # [1, nb, H, D] KV 块均值
    mean = (qc * kc.mean(1, keepdim=True)).sum(-1) * scale
    var = (qc * qc * kc.var(1, unbiased=False, keepdim=True)).sum(-1) * scale * scale
    thr = mean + tau * var.clamp_min(0).add(1e-6).sqrt()          # [1, nb, H]
    score = torch.einsum("bqhd,bkhd->bqkh", qc, kc) * scale       # [1, nb, nb, H]
    return (score > thr.unsqueeze(2)).float().mean().item()


print(f"heads={HEADS} dim={DIM} dtype=bf16 dev={torch.cuda.get_device_name(0)} "
      f"cc={torch.cuda.get_device_capability()}")
from sol_attn import get_sol_attn_backend, sol_attn  # noqa: E402
print("sol_attn backend:", get_sol_attn_backend())

for T in SEQS:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 0.5
               for _ in range(3))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    scale = DIM ** -0.5
    mem = 3 * q.numel() * 2 / 2**30
    print(f"\n--- seq={T} ({mem:.2f} GiB for qkv)")

    # dense 参考：H3 实际走的是 flash_attn_varlen（sage 那条另算，见下面的 sage 分支）
    try:
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
        cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
        qt, kt, vt = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        def dense():
            o = flash_attn_varlen_func(qt, kt, vt, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                       max_seqlen_q=T, max_seqlen_k=T,
                                       softmax_scale=scale, causal=False)
            return o[0] if isinstance(o, tuple) else o
        ms_dense = bench(dense)
        print(f"  {'flash_varlen (dense)':28s}          {ms_dense:9.2f} ms  1.00x")
    except Exception as e:  # noqa: BLE001
        ms_dense = None
        print(f"  flash_varlen 不可用: {type(e).__name__}: {e}")

    try:
        from sageattention import sageattn
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))

        def sage():
            return sageattn(qs, ks, vs, tensor_layout="HND", is_causal=False)
        ms = bench(sage)
        rel = f"{ms_dense / ms:.2f}x" if ms_dense else "-"
        print(f"  {'sageattn (交付基线)':28s}          {ms:9.2f} ms  {rel}")
    except Exception as e:  # noqa: BLE001
        print(f"  sageattn 不可用: {type(e).__name__}: {e}")

    for tau in TAUS:
        try:
            def sol(tau=tau):
                return sol_attn(q, k, v, tau=tau, thresh_type="diag", kv_splits=1)
            ms = bench(sol)
            rel = f"{ms_dense / ms:.2f}x" if ms_dense else "-"
            frac = retained_fraction(q, k, tau, scale)
            print(f"  {'sol_attn diag':28s} tau={tau:<4.1f} {ms:9.2f} ms  {rel}"
                  f"   保留块 {frac * 100:5.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  sol_attn tau={tau}: {type(e).__name__}: {str(e)[:120]}")
    del q, k, v
    torch.cuda.empty_cache()
