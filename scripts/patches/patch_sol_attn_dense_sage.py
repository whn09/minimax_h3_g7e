#!/usr/bin/env python3
"""让 Sol-Attn 后端在这个镜像上能用：把它的 dense 回退路径从 flash-attn 换成 sage。

问题（c0b6474 + lmsysorg/sglang:dev 实测）：`backends/sol_attn.py` 顶部无条件
`from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func`，
而镜像里的 `flash_attn` 是个**空 namespace package**（装的是 flash-attn-4 4.0.0b19，
不导出这个符号）。导入本身是惰性的，所以 server 能起来、能过校验，直到第一次真的走
dense 路径才炸：

    Server warmup failed: cannot import name 'flash_attn_varlen_func' from 'flash_attn'

而 dense 路径是**默认就会走的**：`dense_steps` 默认 10（前 10 个去噪步全密集）、
`dense_layers` 默认 "0,1"。也就是说 sol_attn 后端在这个镜像上开箱即死，
除非把 dense_steps 设 0 且 dense_layers 设空 —— 但那等于放弃前几步的画质保护。

修法：用 sageattention 实现一个等价的 varlen 入口。sage 也是我们交付基线用的那个
（fp8 QK），所以"dense 步"用 sage、"稀疏步"用 sol，比较口径反而更干净：
两条路径的 dense 参考是同一个 kernel。

    docker exec h3n python3 /tmp/patch_sol_attn_dense_sage.py     # 幂等，打印 APPLIED / ALREADY
"""

import re
import sys

TARGET = (
    "/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/layers/"
    "attention/backends/sol_attn.py"
)
MARKER = "# PATCHED: dense fallback via sageattention"
OLD = (
    "from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func"
)
NEW = f'''{MARKER}
# 原来这里是 `from sglang.kernels.ops.attention.flash_attention import
# flash_attn_varlen_func`，在没有真 flash-attn 的镜像上会在第一个 dense 步炸。
def flash_attn_varlen_func(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """sage 版 varlen：逐段切开算再拼回去（THD 进 / THD 出）。

    H3 的 packed 序列通常只有一段，所以这个循环几乎总是只跑一次。
    cu_seqlens 在 device 上，.tolist() 会同步一次；dense 只用在头几步，可以忍。
    """
    from sageattention import sageattn

    bounds = cu_seqlens_q.tolist()
    outs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        qs, ks, vs = (
            x[s:e].transpose(0, 1).unsqueeze(0).contiguous() for x in (q, k, v)
        )  # [T,H,D] -> [1,H,T,D]
        o = sageattn(
            qs, ks, vs, tensor_layout="HND", is_causal=causal, sm_scale=softmax_scale
        )
        outs.append(o.squeeze(0).transpose(0, 1))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)
'''

src = open(TARGET).read()
if MARKER in src:
    print("ALREADY", TARGET)
    sys.exit(0)
if OLD not in src:
    print("MISSING anchor in", TARGET, file=sys.stderr)
    sys.exit(1)
open(TARGET, "w").write(src.replace(OLD, NEW, 1))
# 语法自检：这个文件是 server 启动时才导入的，语法错会变成很难读的 resolver ImportError
import ast  # noqa: E402

ast.parse(open(TARGET).read())
assert re.search(r"def flash_attn_varlen_func", open(TARGET).read())
print("APPLIED", TARGET)
