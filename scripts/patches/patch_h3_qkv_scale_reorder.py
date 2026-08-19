#!/usr/bin/env python3
"""修 sglang 的 H3 NVFP4:qkv 权重的行被 loader reorder 了,per-row 块标度没有。

    docker cp patch_h3_qkv_scale_reorder.py h3:/tmp/ && \
      docker exec h3 python3 /tmp/patch_h3_qkv_scale_reorder.py
    docker exec h3 python3 /tmp/patch_h3_qkv_scale_reorder.py --revert

**这是唯一必需的补丁**:不打它,NVFP4 出来的是一整片灰黑糊(w4a4 和 w4a16 都一样)。打上之后
w4a4 与 bf16 目视同构图、猫毛/金属件细节等同,768p/20 步 173.66 s vs bf16 219.71 s / fp8
204.58 s。它跟按层按步退回那套实验(`patch_nvfp4_mixed_precision.py`)无关,单独用就行 ——
**两个补丁不能同时开修复**,否则行序会被 reorder 两次(那边设 `H3_FP4_QKV_FIX=0`)。

缺陷本身
--------
`MiniMaxH3Attention._install_qkv_weight_loader` 只给 `qkv_proj.weight` 装了
`_reorder_grouped_qkv_to_qkv`(checkpoint 每头 [q,k,v] 交错,模块要 [q_all,k_all,v_all]),
NVFP4 的 `weight_scale` 走基础 loader、行序留在 checkpoint 那边 → 52 个 qkv 层每行都拿错块标度。
bf16 走的是 `_copy_grouped_qkv_tp_shard` 快路(它 `param.dtype != BF16` 就 return False),
做的是同一个置换,所以 bf16 没事、量化路径 100% 中招。

证据(见 nvfp4_qkv_row_order_probe.py):与原始 bf16 权重比,不修 rel 0.44~0.73,补上行序
rel 0.0943 = group-16 round-to-nearest-e2m1 的量化地板。

只对**行主序**的标度动手
------------------------
NVFP4 的块标度有两种存法:linear(每行一串,shape[0] == N)和 swizzled(128×4 分块打散,
shape[0] 不再是 N)。swizzled 的行搬不动,所以下面用 `shape[0] == N` 做闸门 —— 形状不对就
原样交给基础 loader,不猜。
"""
import argparse
import sys

TARGET = ("/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/"
          "minimax_h3.py")

ANCHOR = """        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = _weight_loader
        else:
            weight.weight_loader = _weight_loader
        # rank-local FSDP must reorder grouped QKV before selecting each shard
        weight.rank_local_weight_transform = _reorder_checkpoint_weight
"""

ADDITION = '''
        # NVFP4/INT4: the 4-bit weight rows are reordered above, so the per-row
        # block scales have to follow, or every row picks up another row's scale
        # (measured rel error 0.44-0.73 against the bf16 weights; the group-16
        # quantization floor is 0.094, and the model renders a flat grey blur).
        # Only row-major scale layouts can be permuted this way -- a swizzled
        # (128x4 blocked) scale tensor no longer has one row per output row, so
        # it is left to the base loader untouched.
        _qkv_rows = arch.num_attention_heads * 3 * arch.attention_head_dim
        for _scale_name in ("weight_scale", "weight_scale_inv"):
            _scale = getattr(self.qkv_proj, _scale_name, None)
            if _scale is None or _scale.dim() != 2 or _scale.shape[0] != _qkv_rows:
                continue
            _scale_base = getattr(_scale, "_weight_loader", None) or _scale.weight_loader

            def _scale_loader(param, loaded_weight, _base=_scale_base, _rows=_qkv_rows):
                if loaded_weight.dim() == 2 and loaded_weight.shape[0] == _rows:
                    loaded_weight = _reorder_checkpoint_weight(loaded_weight)
                _base(param, loaded_weight)

            if hasattr(_scale, "_weight_loader"):
                _scale._weight_loader = _scale_loader
            else:
                _scale.weight_loader = _scale_loader
            _scale.rank_local_weight_transform = _reorder_checkpoint_weight
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    with open(a.file) as f:
        src = f.read()
    bak = a.file + ".qkvscalebak"

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

    if "_qkv_rows" in src:
        print("already patched")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"anchor found {src.count(ANCHOR)}x, expected 1 -- upstream changed")
        return 1

    with open(bak, "w") as f:
        f.write(src)
    with open(a.file, "w") as f:
        f.write(src.replace(ANCHOR, ANCHOR + ADDITION))
    print(f"patched {a.file} (backup {bak})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
