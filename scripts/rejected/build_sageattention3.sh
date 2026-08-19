#!/usr/bin/env bash
# **被否掉：SageAttention 3 对 H3 结构性不可用。** 留这个脚本是因为编译本身是成功的（证据在
# 下面），否掉的原因在 sglang/H3 那一侧，不在编译或这张卡上。别照着跑去指望性能。
#
# 实测（2026-08-19，g7e，容器 h3）：
#   编译 OK   sageattn3-1.0.0-cp312 / sm_120a / 约 2 分钟 / `from sageattn3 import
#             sageattn3_blackwell` 通过。python 3.12 也行（官方 README 写 >=3.13 是他们的测试
#             口径，扩展只要 torch>=2.8 + CUDA>=12.8）。
#   起服务失败 configs/pipeline_configs/minimax_h3.py:192 validate_server_args ->
#             get_attn_backend(..., AttentionRequirements(packed_varlen=True)) ->
#             ValueError: Attention backend 'sage_attn_3' does not implement packed varlen attention
#
# 原因：H3 的 DiT 走 **packed varlen**（视频+音频 token 拼成一条不定长序列），而
# `AttentionBackend.supports_packed_varlen()` 的判据是 impl 有没有覆写 `forward_varlen`。
# CUDA 上 20 个后端里只有 5 个覆写了：
#   sage_attn（= 交付用的 sage 2）、fa、torch_sdpa、sol_attn、subblock_sparse_attn
# `sage_attn_3` 不在其中，所以**报错而不是静默回落**——H3 的 pipeline config 在参数校验阶段就拒了。
# （`subblock_sparse_attn` 也用不上：它要 sm_100a，源码注释明写不在 10.3 / 12.x 上跑。）
#
# 想让 SA3 能用，唯一的路是给 `runtime/layers/attention/backends/sage_attn3.py` 实现
# `forward_varlen`（把 cu_seqlens 拆成逐段调用，或让 kernel 吃 varlen 布局）。那是改上游，
# 不是配置问题。
set -u
NAME=${NAME:-h3}
REF=${REF:-d1a57a5}
JOBS=${JOBS:-8}
REPO=${REPO:-https://github.com/thu-ml/SageAttention}

# 自检在源码目录之外跑（和 sage 2 同一个坑：在 sageattention3_blackwell/ 里 import 命中的是
# 没编好的源码包）。
docker exec -w /tmp "$NAME" bash -lc "
set -e
python3 -c 'import sageattn3' 2>/dev/null && { echo 'ALREADY_INSTALLED'; exit 0; }
cd /sgl-workspace
[ -d SageAttention ] || git clone --filter=blob:none $REPO SageAttention
cd SageAttention
git fetch --depth 50 origin || true
git checkout $REF
cd sageattention3_blackwell
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=$JOBS
export EXT_PARALLEL=$JOBS NVCC_APPEND_FLAGS='--threads 2'
pip install --no-build-isolation -v . 2>&1 | tail -30
python3 -c 'from sageattn3 import sageattn3_blackwell; print(\"sage3 ok\")'
"
