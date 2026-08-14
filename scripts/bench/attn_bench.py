#!/usr/bin/env python3
"""Micro-benchmark the attention kernels reachable on sm_120 at MiniMax-H3's real DiT shapes.

    docker exec h3 python3 /out/attn_bench.py

Why this exists: on this box sglang logs "FlashAttention is not supported on SM12.x in this build;
falling back to Torch SDPA" (platforms/cuda.py:357) and every H3 run so far has therefore used
torch_sdpa. That gate is a build-time guess about the sgl-kernel wheel, not a measurement, and
torch's own SDPA has three separate GPU kernels behind it (flash / mem-efficient / cuDNN) that the
dispatcher picks per shape. So: time each candidate directly at H3's shapes before paying 6 minutes
to restart a server for an E2E arm.

H3 DiT self-attention is packed and full (no mask, no causal), heads=56, head_dim=128, bf16 --
see the head-count/head_dim forensics in the NVFP4 notes. Sequence length is set by resolution and
frame count, hence the sweep; 1344x768 / 124 frames lands near the 32k row.

Correctness is checked against the MATH backend, not eyeballed: a kernel that is fast because it
computes something else is the failure mode that matters here. The check runs at a SMALL sequence
only -- MATH materializes the heads x seq x seq score matrix, which is 15 GB at seq 8192 x 56 heads
and OOMs next to a loaded server.
"""
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

#: "bhsd" allocates contiguous (b, h, s, d); "bshd" allocates (b, s, h, d) and transposes, which is
#: what sglang's sdpa.py actually hands to SDPA (`query.transpose(1, 2)` on a b,s,h,d tensor). The
#: two differ only in stride, and that is exactly where an isolated kernel win can evaporate.
LAYOUT = os.environ.get("ATTN_LAYOUT", "bhsd")
HEADS = int(os.environ.get("ATTN_HEADS", 56))
DIM = int(os.environ.get("ATTN_DIM", 128))
#: 41456 is H3's real packed sequence for ref2va 1344x768 / 124 frames, read off the profiler trace
#: (aten::scaled_dot_product_attention input dims [[1, 56, 41456, 128], ...]); each layer also does a
#: second, 16-token attention. Do not bench a round number and interpolate -- attention kernels are
#: tiled, so the tail block matters.
SEQS = [int(s) for s in os.environ.get("ATTN_SEQS", "41456").split(",")]
ITERS = int(os.environ.get("ATTN_ITERS", 8))


def timed(fn, warmup=3, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sdpa_runner(backend):
    def run(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v)
    return run


def flash_runner():
    from flash_attn import flash_attn_func

    def run(q, k, v):
        # flash_attn wants (b, s, h, d); our tensors are (b, h, s, d).
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    return run


def flashinfer_runner(backend):
    """flashinfer's single-request prefill, which is the shape H3 needs (batch 1, no mask).

    It is the one attention library actually installed here that ships its own sm_120 templates --
    ATen's flash kernel in the trace is `Flash_fwd_kernel_traits` with cutlass_80-class tiling, i.e.
    an SM80 kernel running on Blackwell. Its API takes (s, h, d) with no batch axis.
    """
    import flashinfer

    def run(q, k, v):
        o = flashinfer.single_prefill_with_kv_cache(
            q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1),
            causal=False, backend=backend)
        return o.transpose(0, 1).unsqueeze(0)
    return run


def make(seq, dev):
    """Return q, k, v shaped (b, h, s, d) in the requested memory layout."""
    if LAYOUT == "bshd":
        return tuple(torch.randn(1, seq, HEADS, DIM, device=dev, dtype=torch.bfloat16).transpose(1, 2)
                     for _ in range(3))
    return tuple(torch.randn(1, HEADS, seq, DIM, device=dev, dtype=torch.bfloat16) for _ in range(3))


def main():
    dev = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__} layout={LAYOUT}")
    cands = [
        ("sdpa_auto", lambda q, k, v: F.scaled_dot_product_attention(q, k, v)),
        ("sdpa_flash", sdpa_runner(SDPBackend.FLASH_ATTENTION)),
        ("sdpa_efficient", sdpa_runner(SDPBackend.EFFICIENT_ATTENTION)),
        ("sdpa_cudnn", sdpa_runner(SDPBackend.CUDNN_ATTENTION)),
    ]
    try:
        cands.append(("flash_attn_func", flash_runner()))
    except Exception as exc:  # noqa: BLE001 - reporting is the point
        print(f"flash_attn unavailable: {type(exc).__name__}: {exc}")
    for be in ("auto", "fa2", "cutlass"):
        try:
            cands.append((f"flashinfer_{be}", flashinfer_runner(be)))
        except Exception as exc:  # noqa: BLE001
            print(f"flashinfer[{be}] unavailable: {type(exc).__name__}: {exc}")

    # Correctness at a small sequence, where the MATH reference fits.
    q, k, v = make(1024, dev)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k, v).float()
    print("\ncorrectness @ seq=1024 vs MATH")
    for name, fn in cands:
        try:
            rel = float((fn(q, k, v).float() - ref).norm() / ref.norm())
            print(f"  {name:16s} rel={rel:.2e}{'' if rel < 5e-3 else '   <<< WRONG'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:16s} FAILED {type(exc).__name__}: {str(exc)[:110]}")
    del q, k, v, ref
    torch.cuda.empty_cache()

    for seq in SEQS:
        try:
            q, k, v = make(seq, dev)
        except torch.OutOfMemoryError:
            print(f"\nseq={seq}: OOM allocating inputs, skipped")
            continue
        flops = 4 * HEADS * seq * seq * DIM  # fwd only, non-causal
        print(f"\nseq={seq} heads={HEADS} dim={DIM}")
        for name, fn in cands:
            try:
                ms = timed(lambda: fn(q, k, v))
                print(f"  {name:16s} {ms:8.2f} ms  {flops / ms / 1e9:7.1f} TFLOPS")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:16s} FAILED {type(exc).__name__}: {str(exc)[:110]}")
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
