#!/usr/bin/env python3
"""Bench SageAttention against the torch_sdpa path H3 runs, at H3's real DiT shapes, with fidelity.

    scp to the box, then: docker exec h3 python3 /out/attn_bench_sage.py

Why this and not attn_bench_fa4.py: FA4 is closed on this card both ways (BF16 dies inside the CuTe
DSL with `'NoneType' has no attribute '_trait'`; FP8 hard-asserts "only supported on SM100"). But
sglang ships SAGE_ATTN and SAGE_ATTN_3 attention backends, and its sage_attn.py already carries
H3-specific trailing-padding handling -- so this is a supported path, not a graft.

Stakes: attention is 59% of a 768p step and is the one part no quantization arm has touched. Our FP4
linears already run at 2.15x, so even *free* linears leave the step at 6.98 s -- above the 7.4 s/step
target for this resolution. Quantized attention is the missing lever.

Two things are measured for every candidate, and neither alone is sufficient:

  speed    median of ITERS, in-process only. This box has a 600 W wall that drifts 5.4% across
           restarts, so cross-process absolute numbers are not comparable.
  fidelity relative L2 error against an EXACT fp32 attention, at the production sequence length.
           Sage's INT8-QK / FP8-PV tricks are supposed to be near-lossless; "near" has to be a
           number before an E2E run is worth 4 minutes of GPU.

The fidelity reference is the subtle part. A full fp32 MATH reference is impossible at seq 41456
(the score matrix is 56 heads x 41456^2), so it is computed for a SLICE -- a few heads, a few
hundred query rows, all keys -- which is exact for those rows because attention does not mix rows.
Comparing bf16 SDPA against the same reference calibrates the scale: a candidate is only "lossy" if
it is materially worse than SDPA's own bf16 error, not if it is merely nonzero.

Layout: sglang's sdpa.py takes (b, s, h, d) and calls .transpose(1, 2), so SDPA sees a
NON-contiguous (b, h, s, d) and is timed that way. Sage's API takes (b, s, h, d) with tensor_layout
="NHD", which is what the backend passes.
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HEADS = 56
HEAD_DIM = 128
# 13700 is the 480p rung, 39760 is fl2va 768p, 41456 is ref2va 768p with a 1024 reference image --
# all three read off real traces, not estimated (see project_h3_dit_true_shapes_and_modulation).
SEQS = [int(s) for s in os.environ.get("SEQS", "13700,39760,41456").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
WARMUP = int(os.environ.get("WARMUP", "5"))
FID_HEADS = int(os.environ.get("FID_HEADS", "4"))
FID_ROWS = int(os.environ.get("FID_ROWS", "256"))


def _time(fn, *args, **kwargs):
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(ITERS):
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def sdpa_production(q_bshd, k_bshd, v_bshd):
    """Exactly what runtime/layers/attention/backends/sdpa.py does."""
    out = F.scaled_dot_product_attention(
        q_bshd.transpose(1, 2), k_bshd.transpose(1, 2), v_bshd.transpose(1, 2)
    )
    return out.transpose(1, 2)


def exact_slice(q, k, v, heads, rows):
    """Exact fp32 attention for `rows` query rows of `heads` heads -- rows do not mix, so this is
    the true answer for those rows, not an approximation of it."""
    qs = q[0, :rows, heads, :].float().transpose(0, 1)          # (h, rows, d)
    ks = k[0, :, heads, :].float().transpose(0, 1)              # (h, s, d)
    vs = v[0, :, heads, :].float().transpose(0, 1)
    scores = torch.matmul(qs, ks.transpose(1, 2)) / HEAD_DIM**0.5
    return torch.matmul(scores.softmax(-1), vs)                 # (h, rows, d)


def rel_err(got, ref, heads, rows):
    g = got[0, :rows, heads, :].float().transpose(0, 1)
    return (g - ref).norm().item() / ref.norm().item()


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")

    cands = []
    try:
        from sageattention import sageattn

        cands.append(("sageattn v1 (triton)", lambda q, k, v: sageattn(q, k, v, tensor_layout="NHD")))
        import sageattention as _sa

        print(f"sageattention at {_sa.__file__}")
        for fn_name in ("sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp16_cuda",
                        "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_sm90",
                        "sageattn_blackwell"):
            fn = getattr(_sa, fn_name, None)
            if fn is not None:
                cands.append((fn_name, lambda q, k, v, _f=fn: _f(q, k, v, tensor_layout="NHD")))
    except ImportError as exc:
        print(f"sageattention unavailable: {exc}")
    # sage 3 只在**这个 micro-bench 里**有意义：kernel 本身能编能跑，但 sglang 里
    # `--attention-backend sage_attn_3` 对 H3 是起不来的（H3 的 DiT 要 packed varlen，
    # 这个后端没实现 forward_varlen），所以这里量出来的数不能当成 E2E 可达。
    try:
        from sageattn3 import sageattn3_blackwell

        cands.append(("sageattn3 blackwell (fp4)",
                      lambda q, k, v: sageattn3_blackwell(q, k, v, tensor_layout="NHD")))
    except ImportError as exc:
        print(f"sageattn3 unavailable: {exc}")

    print("candidates: " + ", ".join(n for n, _ in cands))

    for seq in SEQS:
        q, k, v = (
            torch.randn(1, seq, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            for _ in range(3)
        )
        tflop = 4 * seq * seq * HEADS * HEAD_DIM / 1e12
        print(f"\n--- seq {seq}  ({tflop:.1f} TFLOP)")

        heads = torch.arange(0, HEADS, max(1, HEADS // FID_HEADS), device=dev)[:FID_HEADS]
        ref = exact_slice(q, k, v, heads, FID_ROWS)

        base = _time(sdpa_production, q, k, v)
        e = rel_err(sdpa_production(q, k, v), ref, heads, FID_ROWS)
        print(f"  {'torch_sdpa (production)':30s} {base:8.2f} ms  {tflop / (base / 1e3):7.1f} TFLOPS"
              f"  rel_l2 {e:.3e}")

        for name, fn in cands:
            try:
                t = _time(fn, q, k, v)
                e = rel_err(fn(q, k, v), ref, heads, FID_ROWS)
                print(f"  {name:30s} {t:8.2f} ms  {tflop / (t / 1e3):7.1f} TFLOPS"
                      f"  rel_l2 {e:.3e}  speedup {base / t:.3f}x")
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  {name:30s} FAILED: {type(exc).__name__}: {str(exc)[:150]}")

        del q, k, v, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
