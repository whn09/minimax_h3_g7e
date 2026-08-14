#!/usr/bin/env python3
"""Measure achieved bf16 GEMM throughput at H3's real DiT shapes, plus this card's practical peak.

    docker exec h3 python3 /out/gemm_bench.py

The shapes come from the profiler trace's `Input Dims` for aten::mm inside the denoise loop, not from
the config file: hidden 5376 but attention inner dim 7168 (56 heads x 128), so the qkv and o_proj
GEMMs are not square and cannot be inferred from hidden_size alone.

sglang's linear layer is plain F.linear (runtime/layers/linear.py:169), so these numbers are cuBLASLt
kernel selection on sm_120 -- there is no framework-level GEMM path to swap. The point of the
measurement is to decide whether the 36% of a denoise step that is matmul has headroom left, by
comparing against a big square GEMM (the practical peak) on the same card.
"""
import torch

# (M, K, N, label)
SHAPES = [
    (41472, 5376, 21504, "qkv        5376->3x7168"),
    (41472, 7168, 5376, "o_proj     7168->5376"),
    (41472, 5376, 28672, "ffn up/gate 5376->2x14336"),
    (41472, 14336, 5376, "ffn down   14336->5376"),
    (3, 2688, 96768, "adaln      M=3 (weight-bound)"),
    (16384, 16384, 16384, "square 16k (practical peak)"),
]
ITERS = 6


def timed(fn, warmup=3, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"{'shape':30s} {'ms':>9s} {'TFLOPS':>9s} {'GB/s(weights)':>14s}")
    for m, k, n, label in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        # F.linear takes (out, in), so the weight is (n, k); keep that exact layout.
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ms = timed(lambda: torch.nn.functional.linear(a, w))
        tflops = 2 * m * k * n / ms / 1e9
        gbs = w.numel() * 2 / (ms / 1e3) / 1e9
        print(f"{label:30s} {ms:9.3f} {tflops:9.1f} {gbs:14.0f}")
        del a, w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
