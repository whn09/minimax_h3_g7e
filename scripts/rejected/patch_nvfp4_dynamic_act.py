#!/usr/bin/env python3
"""Give sglang's NVFP4 linear a DYNAMIC per-tensor activation scale, env-gated by H3_FP4_DYN_ACT=1.

    scp this to the box, then:
      docker cp patch_nvfp4_dynamic_act.py h3:/tmp/ && docker exec h3 python3 /tmp/patch_nvfp4_dynamic_act.py
      docker exec h3 python3 /tmp/patch_nvfp4_dynamic_act.py --revert

Why: the community NVFP4 H3 checkpoint (Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot) ships only
`weight` / `weight_scale` / `weight_scale_2` / `bias` -- there is NO `input_scale` tensor. The loader
declares that parameter with `missing_param_init: "ones"` (modelopt_quant.py:499), so the per-tensor
activation scale silently becomes 1.0 and `process_weights_after_loading` derives
`input_scale_inv = 1.0`, `alpha = 1.0 * weight_scale_2`.

That is not an algebra error -- x is quantized and dequantized with the same (unit) global scale --
it is a precision error, and a severe one. NVFP4 stores a per-16-element block scale in e4m3, and
the global scale exists to normalize those block scales into e4m3's range. The intended value is
`amax_x / (6 * 448)`, which maps the largest block scale onto e4m3's max of 448. With a global scale
of 1.0 the block scale is `block_amax / 6` instead, and DiT activations put that at roughly
0.002-2 -- straddling e4m3's smallest subnormal (2^-9 = 0.00195). Blocks with small amax therefore
round their scale to zero and the whole block of activations is annihilated.

This matches what was measured: from the same weights, w4a16 (activations untouched) scores
SSIM 0.9154, while w4a4 scores 0.585 with 5.2x the bitrate -- x264 spending bits on noise.

Dynamic rather than a calibrated static `input_scale`, for three reasons. (1) No calibration run and
no calibration set: a static amax is only valid for the geometry/prompt it was collected on, and H3's
two tasks do not even share a sequence length. (2) A per-call amax is tighter than any calibrated
upper bound. (3) The cost is one extra read of x: ~2.5 GB per layer at ~1.4 TB/s is ~1.8 ms/layer,
~90 ms/step against a ~9 s FP4 step, i.e. ~1%.

The `alpha` recovery is the one subtle part. `process_weights_after_loading` folded
`alpha = input_scale_2 * weight_scale_2` and `weight_scale_2` is not otherwise kept, but because this
checkpoint's `input_scale` is 1.0, `alpha` IS `weight_scale_2` -- so the dynamic alpha is
`input_scale_2_dyn * layer.alpha`. That identity only holds while input_scale is absent from the
checkpoint, so it is asserted at runtime rather than assumed; a checkpoint that ships real
`input_scale` values must not take this path.
"""
import argparse
import re
import sys

TARGET = ("/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/layers/"
          "quantization/modelopt_quant.py")

ANCHOR = "        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)\n"

# NVFP4: e2m1 max is 6, the e4m3 block-scale max is 448, so amax/(6*448) puts the largest block
# scale exactly at e4m3's top. `_h3_alpha` shadows layer.alpha only inside this call.
NEW = '''        if _H3_FP4_DYN_ACT:
            # See patch_nvfp4_dynamic_act.py. Valid only because this checkpoint ships no
            # `input_scale`, which makes layer.alpha == weight_scale_2.
            if not getattr(layer, "_h3_dyn_checked", False):
                assert float(layer.input_scale.max()) == 1.0, (
                    "H3_FP4_DYN_ACT assumes a checkpoint with no input_scale (so alpha == "
                    f"weight_scale_2), but input_scale is {float(layer.input_scale.max())}"
                )
                layer._h3_dyn_checked = True
            _h3_is2 = (x.abs().amax().to(torch.float32) / (6.0 * 448.0)).clamp_min(1e-8)
            x_fp4, x_scale_interleaved = fp4_quantize(x, 1.0 / _h3_is2)
            _h3_alpha = _h3_is2 * layer.alpha
        else:
            x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
            _h3_alpha = layer.alpha
'''

FLAG = '''
# Set by patch_nvfp4_dynamic_act.py; read once at import so the branch costs nothing per call.
_H3_FP4_DYN_ACT = _os.environ.get("H3_FP4_DYN_ACT", "0") == "1"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    with open(a.file) as f:
        src = f.read()
    bak = a.file + ".h3bak"

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

    if "_H3_FP4_DYN_ACT" in src:
        print("already patched")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"anchor found {src.count(ANCHOR)}x, expected 1 -- upstream changed, patch by hand")
        return 1

    with open(bak, "w") as f:
        f.write(src)

    # `os` may or may not already be imported under that name; import it privately to be sure.
    m = re.search(r"^import torch$", src, re.M)
    if not m:
        print("no `import torch` line to anchor the flag on")
        return 1
    src = src[: m.end()] + "\nimport os as _os\n" + FLAG + src[m.end():]

    # The GEMM call still says `layer.alpha`; point it at the possibly-dynamic one.
    src = src.replace(ANCHOR, NEW)
    n = src.count("            layer.alpha,\n")
    if n != 1:
        print(f"expected 1 `layer.alpha,` gemm arg, found {n}")
        return 1
    src = src.replace("            layer.alpha,\n", "            _h3_alpha,\n")

    with open(a.file, "w") as f:
        f.write(src)
    print(f"patched {a.file} (backup {bak}); enable with H3_FP4_DYN_ACT=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
