#!/usr/bin/env python3
"""Time FA4's cute kernels against the torch_sdpa path H3 actually runs on, at H3's DiT shapes.

    scp to the box, then: docker exec h3 python3 /out/attn_bench_fa4.py

Why this exists, and why it is separate from attn_bench.py: attn_bench.py compared torch SDPA's own
three GPU backends (flash / mem-efficient / cuDNN) and found <=2.5% between them. It never tried
FlashAttention 4, because sglang cannot reach it -- platforms/cuda.py:357 resolves the FA backend to
TORCH_SDPA whenever platform.is_sm120(), unconditionally, so `--attention-backend fa4` is silently
downgraded. That gate's own comment says "not supported on SM12.x in this build", which is about the
sgl-kernel FA3 wheel; the separately installed flash-attn-4 wheel ships flash_fwd_sm120, i.e. a
dedicated sm_120 forward kernel. So the gate may be leaving a real kernel on the table.

Stakes: attention is 61.6% of a 768p step on this box (roofline notes), and every quantization arm
we have run -- fp8 w8a8, NVFP4 w4a4 -- leaves attention in BF16. That is exactly why fp8 only buys
1.075x at 768p. If attention itself can go faster, the 768p ceiling is 1/(0.616/2 + 0.384) = 1.44x.

Shapes: H3's DiT self-attention is packed and full -- no mask, no causal, heads=56, head_dim=128,
bf16, hidden 5376 with a 7168 attention inner dim (56*128). Sequence length is set by resolution and
frame count; 1344x768 / 124 frames lands near 41k with a reference image, ~34k without.

Layout matters and is easy to get wrong: sglang's sdpa.py receives (b, s, h, d) and calls
.transpose(1, 2), so SDPA sees a NON-contiguous (b, h, s, d). FA4's cute API takes (b, s, h, d)
directly. Allocating contiguous bhsd for SDPA would flatter it relative to production. So SDPA is
timed the way sdpa.py actually calls it.

Correctness is checked against SDPA's MATH backend at a small sequence only: MATH materializes the
h x s x s score matrix (15 GB at s=8192 x 56 heads), which OOMs next to anything else on the card.
A kernel that is fast because it computes something else is the failure mode that matters here.
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HEADS = 56
HEAD_DIM = 128
# 8192 is the correctness rung; 13700/34096 are the 480p/768p estimates; 41456 is 768p + a 1024
# reference image, which is the longest shape any of the eight benchmark cases produces.
SEQS = [int(s) for s in os.environ.get("SEQS", "8192,13700,34096,41456").split(",")]
ITERS = int(os.environ.get("ITERS", "20"))
WARMUP = int(os.environ.get("WARMUP", "5"))


def _time(fn, *args, **kwargs):
    """Median of ITERS, not mean: a single clock/power excursion should not set the number.

    This box has a 600 W power wall that drifts 5.4% across restarts, so absolute numbers are only
    comparable within one process -- which is why every candidate is timed in this one run.
    """
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
    """Exactly what sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py does."""
    out = F.scaled_dot_product_attention(
        q_bshd.transpose(1, 2), k_bshd.transpose(1, 2), v_bshd.transpose(1, 2)
    )
    return out.transpose(1, 2)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    try:
        import flash_attn.cute as fa4

        has_fa4 = True
        print(f"flash-attn-4 cute at {fa4.__file__}")
        print(f"  sm120 fwd kernel present: {'flash_fwd_sm120' in dir(fa4)}")
    except ImportError as exc:
        has_fa4 = False
        print(f"flash-attn-4 unavailable: {exc}")

    for seq in SEQS:
        q, k, v = (
            torch.randn(1, seq, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            for _ in range(3)
        )
        # 4 * s^2 * h * d MACs -> 2 flops each; this is the number the roofline notes use.
        tflop = 4 * seq * seq * HEADS * HEAD_DIM / 1e12
        print(f"\n--- seq {seq}  ({tflop:.1f} TFLOP)")

        base = _time(sdpa_production, q, k, v)
        print(f"  torch_sdpa (production path) {base:8.2f} ms  {tflop / (base / 1e3):7.1f} TFLOPS")

        if has_fa4:
            for name, kwargs in (("fa4 cute bf16", {}),):
                try:
                    t = _time(fa4.flash_attn_func, q, k, v, **kwargs)
                    print(
                        f"  {name:28s} {t:8.2f} ms  {tflop / (t / 1e3):7.1f} TFLOPS"
                        f"  speedup {base / t:.3f}x"
                    )
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    print(f"  {name:28s} FAILED: {type(exc).__name__}: {str(exc)[:160]}")

            # FP8: the wheel's own benchmark warns descale/max-offset scaling is implemented in the
            # SM100 kernel, so this is expected to fail on sm_120. Ask anyway -- a cheap question.
            try:
                qf, kf, vf = (t.to(torch.float8_e4m3fn) for t in (q, k, v))
                t = _time(fa4.flash_attn_func, qf, kf, vf)
                print(
                    f"  {'fa4 cute fp8_e4m3':28s} {t:8.2f} ms  {tflop / (t / 1e3):7.1f} TFLOPS"
                    f"  speedup {base / t:.3f}x"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {'fa4 cute fp8_e4m3':28s} FAILED: {type(exc).__name__}: {str(exc)[:160]}")

        if seq == SEQS[0] and has_fa4:
            # Correctness only at the smallest rung; MATH is O(h*s^2) memory.
            with sdpa_kernel(SDPBackend.MATH):
                ref = sdpa_production(q, k, v)
            try:
                got = fa4.flash_attn_func(q, k, v)
                err = (got.float() - ref.float()).abs().max().item()
                rel = err / ref.float().abs().max().item()
                print(f"  correctness vs MATH: max abs {err:.4g}  rel {rel:.4g}")
            except Exception as exc:  # noqa: BLE001
                print(f"  correctness vs MATH: skipped ({type(exc).__name__})")

        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
