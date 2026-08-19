#!/usr/bin/env bash
# 在容器里从源码编 SageAttention（sm_120）。h3-validated 镜像里没有它，容器一重建就得再来一次。
#
#   ./build_sageattention.sh                    # 容器 h3，commit d1a57a5
#   NAME=h3 REF=main JOBS=16 ./build_sageattention.sh
#
# 为什么必须从源码：pip 上的 wheel 是 1.0.6 / 纯 Triton，在 sm_120 上只值 1.16×，源码这条是 1.776×
# （seq 41456 单点实测，rel_l2 3.9e-2）。d1a57a5 是 2026-08-13 验过、README 那张 FP8+sage 表的出处。
#
# JOBS 默认压到 8：这台是 48 vCPU，但**编译期间不要有正在计时的请求** —— nvcc 吃满核会把非 DiT
# 那几段推上去。最干净是趁 checkpoint 量化那种 CPU 窗口一起做。
set -u
NAME=${NAME:-h3}
REF=${REF:-d1a57a5}
JOBS=${JOBS:-8}
REPO=${REPO:-https://github.com/thu-ml/SageAttention}

# 自检和预检都必须在源码目录**之外**跑（末尾那个自检前有一句 `cd /tmp`，别删）。在
# /sgl-workspace/SageAttention 里 `import sageattention` 命中的是源码那个包(没有编好的 `_fused`
# 扩展),报的是 "cannot import name '_fused' ... partially initialized module" —— 看着像编译失败,
# 其实 pip 已经 Successfully installed 了。第一版少了这句 cd，h3n 容器那次就是这么"失败"的
# (log 里 Successfully installed sageattention-2.2.0 之后紧跟一个 ImportError)。
docker exec -w /tmp "$NAME" bash -lc "
set -e
python3 -c 'import sageattention' 2>/dev/null && { echo 'ALREADY_INSTALLED'; exit 0; }
cd /sgl-workspace
[ -d SageAttention ] || git clone --filter=blob:none $REPO SageAttention
cd SageAttention
git fetch --depth 50 origin || true
git checkout $REF
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=$JOBS
export EXT_PARALLEL=$JOBS NVCC_APPEND_FLAGS='--threads 2'
pip install --no-build-isolation -v . 2>&1 | tail -25
cd /tmp
python3 -c 'import sageattention, torch; print(\"sage ok\", sageattention.__file__, torch.__version__)'
"
