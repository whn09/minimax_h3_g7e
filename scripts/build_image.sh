#!/usr/bin/env bash
# 建 H3-on-g7e 的交付镜像（见 Dockerfile 头部：为什么烤镜像比运行时改容器好）。**在 g7e 机器上跑**。
#
#   ./build_image.sh                          # base 已在本地时约 2 分钟（实测 nvcc 编 SageAttention 117.6s）
#   TAG=h3-g7e:test ./build_image.sh
#   WITH_SOL=1 ./build_image.sh               # 顺带装 Sol-Attn（稀疏消融用）
#   BASE=lmsysorg/sglang:dev ./build_image.sh # 跟 sglang HEAD（**要先确认 PATCHES 还能打**）
#   JOBS=16 ./build_image.sh                  # 48 vCPU 上可以快一点；机器上有正在计时的请求时别这么干
#
# 建完自己做三件事：把真实 sglang commit 读回来打成第二个 tag（`h3-g7e:<sha7>`，不动的那个）、
# 打印镜像里到底烤了什么、以及提示 serve.sh 怎么用它。
#
# 不需要 GPU：SageAttention 靠 TORCH_CUDA_ARCH_LIST 交叉编译（setup.py 自己写了 "works without
# GPUs"），docker build 默认也拿不到 GPU。所以这个脚本在任何有 docker + 网络的盒子上都能跑，
# 只要 base 镜像拉得下来。
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

TAG=${TAG:-h3-g7e:local}
BASE=${BASE:-}
WITH_SOL=${WITH_SOL:-0}
JOBS=${JOBS:-8}
PATCHES=${PATCHES:-}
NOCACHE=${NOCACHE:-}

args=(--build-arg "WITH_SOL=$WITH_SOL" --build-arg "JOBS=$JOBS")
[ -n "$BASE" ]    && args+=(--build-arg "BASE=$BASE")
[ -n "$PATCHES" ] && args+=(--build-arg "PATCHES=$PATCHES")
[ -n "$NOCACHE" ] && args+=(--no-cache)

echo "== docker build -t $TAG （base=${BASE:-<Dockerfile 里钉的 digest>} with_sol=$WITH_SOL jobs=$JOBS）"
# context 是 scripts/，.dockerignore 把它收窄到只有 patches/ —— 不然 assets/ 里的 mp4/png 也会
# 被塞进 build context。
docker build -f Dockerfile -t "$TAG" "${args[@]}" .

# sglang 自己在镜像里留了 SGLANG_BUILD_COMMIT，拿它当第二个 tag，这样"这个镜像是哪版 sglang"
# 不依赖记性。用 --entrypoint bash 是因为 sglang 镜像的 entrypoint 不是 shell。
sha=$(docker run --rm --entrypoint bash "$TAG" -lc 'echo ${SGLANG_BUILD_COMMIT:0:7}')
if [ -n "$sha" ]; then
  docker tag "$TAG" "h3-g7e:$sha"
  echo "== 也打了 h3-g7e:$sha（sglang commit）"
fi

echo "== 镜像里烤了什么"
docker image inspect "$TAG" --format '{{range $k,$v := .Config.Labels}}{{$k}}={{$v}}
{{end}}' | grep '^h3\.'
docker run --rm --entrypoint bash "$TAG" -lc '
  echo "patches:  $(cat /sgl-workspace/.h3-image-patches)"
  echo "stamps:   $(ls /sgl-workspace/.h3-patches | tr "\n" " ")"
  cd /tmp
  d=$(python3 -c "import sageattention,os;print(os.path.dirname(sageattention.__file__))")
  echo "sage:     $(python3 -c "import sageattention;print(\"import ok\")") / cubin $(/usr/local/cuda/bin/cuobjdump --list-elf $d/_qattn_sm89*.so | sed -E "s/.*\.(sm_[0-9a-z]+)\.cubin/\1/" | sort -u | tr "\n" " ")"
  echo "sol:      $(python3 -c "from sol_attn import get_sol_attn_backend;print(get_sol_attn_backend())" 2>/dev/null || echo "未装")"
  echo "env:      RUNAI=$SGLANG_USE_RUNAI_MODEL_STREAMER TMA=$H3_FP4_TMA_SCALES QKV_FIX=$H3_FP4_QKV_FIX"
  echo "sglang:   $(git -C /sgl-workspace/sglang log --oneline -1)"'

cat <<EOF

== 用它起服务（serve.sh 会自己发现镜像里烤好的补丁清单，PATCHES 不用传）：
  IMAGE=$TAG NAME=h3 GPUS=1 ULYSSES=1 \\
    EXTRA="--layerwise-offload-components text_encoder \\
      --transformer-weights-path /out/nvfp4_fl2va.safetensors \\
      --attention-backend sage_attn \\
      --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa" \\
    WARMUP="864x480" ./serve.sh start
EOF
