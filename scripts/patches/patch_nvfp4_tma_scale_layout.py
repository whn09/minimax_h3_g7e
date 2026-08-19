#!/usr/bin/env python3
"""sm_120 上 NVFP4 出噪声的真正原因：weight_scale 布局跟实际跑的 kernel 不匹配。env 开关 H3_FP4_TMA_SCALES=1。

    docker cp patch_nvfp4_tma_scale_layout.py h3:/tmp/ && \
      docker exec h3 python3 /tmp/patch_nvfp4_tma_scale_layout.py
    docker exec h3 python3 /tmp/patch_nvfp4_tma_scale_layout.py --revert

诊断
----
`ModelOptFp4LinearMethod.process_weights_after_loading` 按 backend 字符串决定 weight_scale 的布局：

    if flashinfer_backend == "trtllm":              -> shuffle_matrix_a / shuffle_matrix_sf_a，early return
    if flashinfer_backend is None or uses_flux1_scale_layout:
        # "CUTLASS and FLUX.1 CUDNN paths need the TMA scale layout"
        padded_scales = padded_scales.reshape(B, M/128, 4, 32, K/4, 4).permute(0,1,4,3,2,5)

trn... 不是，sm_120 (RTX PRO 6000 / g7e) 上 flashinfer 的 **trtllm fp4 GEMM 根本不存在**
（`mm_fp4 does not support backend 'trtllm' with capability 120`），所以必须传
`SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto`。而 "auto" 既不等于 "trtllm" 也不是 None
——**两个布局分支都不走**，scale 以最朴素的 padded 布局进 kernel，可 auto 在 sm_120 上落到的正是
cutlass kernel，它要的是 TMA 布局。

单层实测（`nvfp4_gemm_probe.py`，blocks.0.mlp.fc1，参考是同一批 4-bit 权重解回 bf16）：

    layout     backend    rel_err
    plain      auto        0.5513   <- sglang 现在的行为
    plain      cutlass     0.5513
    tma        auto        0.0951   <- 对（≈ 权重 fp4 量化误差 0.094，激活 4-bit 几乎不加噪）
    tma        cutlass     0.0951
    shuffled   auto        1.3689   <- trtllm 的布局，在这里是正交垃圾
    *          trtllm      不支持 capability 120

所以画面是纯噪声不是"4-bit 精度不够"，而是布局错。顺带推翻了之前"坏的是 4-bit 激活"的结论——
那条消融把整个文件反量化成 bf16，等于绕开了这个 GEMM，所以看不见这个 bug。
"""
import argparse
import re
import sys

TARGET = ("/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/layers/"
          "quantization/modelopt_quant.py")

ANCHOR = "        if flashinfer_backend is None or uses_flux1_scale_layout:\n"

NEW = '''        if (
            flashinfer_backend is None
            or uses_flux1_scale_layout
            # sm_120: "auto" 落到 cutlass kernel，它跟 backend=None 一样要 TMA 布局。
            # 见 patch_nvfp4_tma_scale_layout.py / nvfp4_gemm_probe.py。
            or (_H3_FP4_TMA_SCALES and flashinfer_backend in ("auto", "cutlass"))
        ):
'''

FLAG = '''
# Set by patch_nvfp4_tma_scale_layout.py; read once at import so the branch costs nothing per call.
_H3_FP4_TMA_SCALES = _os.environ.get("H3_FP4_TMA_SCALES", "0") == "1"
_h3_tma_layers = [0]
_h3_plain_layers = [0]
'''

# 记一笔谁走了哪个分支——否则"改了没生效"和"生效了但没用"看起来一模一样。
ANCHOR2 = """            padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)

        padded_scales = padded_scales.contiguous().cuda()
"""

NEW2 = """            padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)
            _h3_tma_layers[0] += 1
        else:
            _h3_plain_layers[0] += 1
        if (_h3_tma_layers[0] + _h3_plain_layers[0]) in (1, 200):
            logger.warning(
                "H3 nvfp4 scale layout: backend=%r flag=%s tma=%d plain=%d",
                flashinfer_backend,
                _H3_FP4_TMA_SCALES,
                _h3_tma_layers[0],
                _h3_plain_layers[0],
            )

        padded_scales = padded_scales.contiguous().cuda()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    with open(a.file) as f:
        src = f.read()
    bak = a.file + ".tmabak"

    if a.revert:
        try:
            with open(bak) as f:
                orig = f.read()
        except FileNotFoundError:
            print("no backup; nothing to revert")
            return 1
        with open(a.file, "w") as f:
            f.write(orig)
        print(f"reverted {a.file} from {bak}")
        return 0

    if "_H3_FP4_TMA_SCALES" in src:
        print("already patched")
        return 0
    for name, anc in (("ANCHOR", ANCHOR), ("ANCHOR2", ANCHOR2)):
        if src.count(anc) != 1:
            print(f"{name} found {src.count(anc)}x, expected 1 -- upstream changed, patch by hand")
            return 1

    with open(bak, "w") as f:
        f.write(src)

    m = re.search(r"^import torch$", src, re.M)
    if not m:
        print("no `import torch` line to anchor the flag on")
        return 1
    if "import os as _os" not in src:
        src = src[: m.end()] + "\nimport os as _os\n" + FLAG + src[m.end():]
    else:
        src = src[: m.end()] + FLAG + src[m.end():]

    src = src.replace(ANCHOR, NEW).replace(ANCHOR2, NEW2)
    with open(a.file, "w") as f:
        f.write(src)
    print(f"patched {a.file} (backup {bak}); enable with H3_FP4_TMA_SCALES=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
