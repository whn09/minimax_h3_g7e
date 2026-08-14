#!/usr/bin/env python3
"""Make sglang's SAGE_ATTN backend pick its kernel by env var H3_SAGE_KERNEL.

    docker cp patch_sage_kernel_select.py h3:/tmp/ && docker exec h3 python3 /tmp/patch_sage_kernel_select.py
    docker exec h3 python3 /tmp/patch_sage_kernel_select.py --revert

Why: sage_attn.py calls the top-level `sageattn` dispatcher, which on sm_120 lands on the
INT8-QK / **FP8-PV** kernel. Measured at H3's real shapes (56 heads, d=128, seq 41456) by
attn_bench_sage.py:

    torch_sdpa                       364.2 TFLOPS   rel_l2 2.331e-03   (the bf16 baseline error)
    sageattn (auto -> fp8 PV)        646.7          rel_l2 3.941e-02   1.776x
    sageattn_qk_int8_pv_fp16_cuda    487.1          rel_l2 1.157e-02   1.337x   <- 3.4x cleaner
    sageattn_qk_int8_pv_fp16_triton  416.8          rel_l2 1.353e-02   1.144x

The FP8-PV path's 17x-worse-than-bf16 error is not academic: E2E on ref2va it gives SSIM 0.9501 /
0.9738 at 768p but only 0.8338 / 0.8697 at 480p, against a run-to-run floor of **1.000000** (single
GPU, fixed seed, bit-identical -- the 0.9444 floor in the notes is the 2-GPU Ulysses figure, where
the all-to-all reduction order varies). So every deviation here is real, and the fp16-PV kernel is
the arm worth measuring: it keeps most of the speedup at a third of the error.

The dispatcher is bypassed rather than reconfigured because `sageattn` has no kernel-choice
argument -- it picks by compute capability alone (sageattention/core.py).
"""
import argparse
import sys

TARGET = ("/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/layers/attention/"
          "backends/sage_attn.py")

ANCHOR = "from sageattention import sageattn\n"

NEW = '''import os as _os

import sageattention as _sa
from sageattention import sageattn as _sageattn_auto

# H3_SAGE_KERNEL names any sageattention entry point; unset keeps upstream's auto dispatch.
# See patch_sage_kernel_select.py for the accuracy/speed table that motivates the choice.
_H3_SAGE_KERNEL = _os.environ.get("H3_SAGE_KERNEL", "").strip()
if _H3_SAGE_KERNEL:
    sageattn = getattr(_sa, _H3_SAGE_KERNEL)
    _log_once = f"H3_SAGE_KERNEL={_H3_SAGE_KERNEL} (bypassing sageattn auto-dispatch)"
else:
    sageattn = _sageattn_auto
    _log_once = "sageattn auto-dispatch"
'''

LOG_ANCHOR = 'logger = init_logger(__name__)\n'
LOG_NEW = 'logger = init_logger(__name__)\nlogger.info(_log_once)\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    bak = a.file + ".h3bak"
    with open(a.file) as f:
        src = f.read()

    if a.revert:
        try:
            with open(bak) as f:
                orig = f.read()
        except FileNotFoundError:
            print("no backup; nothing to revert")
            return 1
        with open(a.file, "w") as f:
            f.write(orig)
        print(f"reverted {a.file}")
        return 0

    if "_H3_SAGE_KERNEL" in src:
        print("already patched")
        return 0
    for anchor in (ANCHOR, LOG_ANCHOR):
        if src.count(anchor) != 1:
            print(f"anchor {anchor!r} found {src.count(anchor)}x, expected 1")
            return 1

    with open(bak, "w") as f:
        f.write(src)
    src = src.replace(ANCHOR, NEW).replace(LOG_ANCHOR, LOG_NEW)
    with open(a.file, "w") as f:
        f.write(src)
    print(f"patched {a.file} (backup {bak}); select with H3_SAGE_KERNEL=<fn name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
