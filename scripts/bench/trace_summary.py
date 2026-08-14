#!/usr/bin/env python3
"""Split a torch-profiler trace of H3's denoise steps into GPU time per kernel class.

    trace_summary.py <trace.json[.gz]> [--top 25]

Reports GPU (device) time only, from the trace's kernel events -- not CPU operator time. On this
pipeline the two are not interchangeable: the denoise loop never synchronizes, so CPU-side op
durations measure launch cost and the CPU runs ~a full step ahead of the GPU.

Kernels are bucketed by name into gemm / attention / elementwise-and-norm / reduction / copy /
other, because the actionable question is "how much of the step is NOT matmul", i.e. how much could
in principle be fused away. Bucketing is by substring on the mangled kernel name; anything unmatched
lands in `other` and is printed in the top-N list so a mis-bucketed hot kernel is visible rather
than silently absorbed.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

BUCKETS = [
    # Attention MUST be tested before gemm: ATen's flash kernel is
    # `pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<...cutlass::...>>`, so a gemm rule
    # matching "cutlass" swallows the single largest kernel in the step and reports it as matmul.
    ("attention", re.compile(r"flash|fmha|attention|cudnn_generated|sdpa", re.I)),
    # cutlass/cublas GEMM kernels, incl. the sm120 bf16 tensor-core names
    ("gemm", re.compile(r"cutlass|gemm|s16816|sm\d+_xmma|nvjet|tensorop", re.I)),
    ("reduction", re.compile(r"reduce|isnan|argmax|_sum|norm_kernel", re.I)),
    ("copy", re.compile(r"memcpy|memset|copy_|cat_|transpose|permute", re.I)),
    ("elementwise/norm", re.compile(
        r"elementwise|vectorized|triton|scale_shift|gate|silu|rms|layer_norm|rope|mul|add|index",
        re.I)),
]


def bucket(name):
    for label, rx in BUCKETS:
        if rx.search(name):
            return label
    return "other"


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    ev = load(a.trace)["traceEvents"]
    per_kernel = defaultdict(lambda: [0.0, 0])
    for e in ev:
        # cat is "kernel" for device kernels; gpu_memcpy/gpu_memset are separate categories.
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
            rec = per_kernel[e["name"]]
            rec[0] += e["dur"] / 1000.0  # us -> ms
            rec[1] += 1

    total = sum(v[0] for v in per_kernel.values())
    per_bucket = defaultdict(lambda: [0.0, 0])
    for name, (ms, n) in per_kernel.items():
        rec = per_bucket[bucket(name)]
        rec[0] += ms
        rec[1] += n

    print(f"device kernel time {total:.1f} ms over {sum(v[1] for v in per_kernel.values())} launches"
          f"  ({len(per_kernel)} distinct kernels)\n")
    print(f"{'bucket':20s} {'ms':>10s} {'share':>7s} {'launches':>9s}")
    for label, (ms, n) in sorted(per_bucket.items(), key=lambda kv: -kv[1][0]):
        print(f"{label:20s} {ms:10.1f} {100 * ms / total:6.1f}% {n:9d}")

    print(f"\ntop {a.top} kernels")
    for name, (ms, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {ms:9.1f} ms {100 * ms / total:5.1f}% {n:6d}x  [{bucket(name):16s}] {name[:96]}")


if __name__ == "__main__":
    main()
