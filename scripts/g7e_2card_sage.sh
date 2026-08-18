#!/usr/bin/env bash
# 2 卡 Ulysses=2 + FP8 权重 + SageAttention：在一台 g7e.12xlarge 上把加卡的收益量出来。
#
#   ./g7e_2card_sage.sh              # 1 卡控制组 + 2 卡四个场景（480/768 × 20/30 步）
#   CASES="768_30" ./g7e_2card_sage.sh
#   SKIP_G1=1 ./g7e_2card_sage.sh    # 跳过单卡控制组（只有在同一台机器上已经量过时才跳）
#
# 三件事按顺序做，顺序本身是有意义的：
#   1. 先在这台机器上量 1 卡，作为分母。跨机器直接拿别的机器的单卡数当分母是错的——`:dev` 是移动
#      标签，且同配置跨重启漂移实测 5.4%。加速比的分母必须和分子同机同镜像。
#   2. 再量 2 卡。serve.sh 在 GPUS=2 且不指定 TP 时把 ULYSSES 算成 2（切序列）。
#   3. 画质对比。2 卡的地板不是 SSIM 1.0：all-to-all 改归约序，实测地板在 0.944 量级。所以判据是
#      "在地板之上且码率不涨"，不是"接近 1.0"。
#
# 两个 g7e 必需项由 serve.sh 带上，别去掉：SGLANG_USE_RUNAI_MODEL_STREAMER=0（streamer 的匿名内存
# 打爆主机）、--layerwise-offload-components text_encoder。
#
# 实测（2026-08-18，东京 g7e.12xlarge spot，镜像 273d978b，sage d1a57a5，fl2va 5.175 s 成片）：
#   480p/20  39.612 -> 29.302  (1.352x)   480p/30  58.159 ->  43.069 (1.350x)
#   768p/20 144.392 -> 87.298  (1.654x)   768p/30 215.993 -> 130.732 (1.652x)
#   同机分母 219.762 -> 130.732 = 1.681x / 效率 84.0%；SSIM 0.971，码率 +0.3%（无损伤）
#   480p 只有 1.35x：那档每步通信 0.449 s 占 2 卡每步 1.377 s 的 33%。要加卡就在 768p 加。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:dev}
OUT=/opt/dlami/nvme/out
CASES=${CASES:-"480_20 480_30 768_20 768_30"}
PROMPT=${PROMPT:-"A white cat sitting on an open window ledge slowly turns its head toward the camera, blinks, and gently lifts one paw while a soft breeze moves the curtains. Natural afternoon light, subtle street ambience and soft paw sounds, realistic motion, static cinematic camera."}
SEED=${SEED:-6201}
mkdir -p "$OUT"

# 一个臂：起服务（FP8 + sage）-> 按 CASES 发请求 -> 停服务
run_arm() {  # run_arm <gpus> <tag_suffix> <cases...>
  local gpus=$1 suf=$2; shift 2
  echo "=== ARM gpus=$gpus ulysses=$gpus quant=fp8 attn=sage_attn $(date -u +%H:%M:%S)"
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  VARIANT=fl2va GPUS=$gpus IMAGE=$IMAGE \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0" \
    EXTRA="--layerwise-offload-components text_encoder --quantization fp8 --attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa" \
    LOG=/out/serve_2card_sage_g$gpus.log ./serve.sh start > "$OUT/start_2card_sage_g$gpus.log" 2>&1 \
    || { echo "SERVER_FAILED gpus=$gpus (看 $OUT/serve_2card_sage_g$gpus.log)"; return 1; }

  local c se st tag t0 t1 rc inf
  for c in "$@"; do
    se=${c%_*}; st=${c#*_}; tag="fl2va_${se}_${st}_fp8_sage_attn$suf"
    t0=$(date +%s)
    python3 h3gen.py --task fl2va --image assets/first.png --inline \
      --short-edge "$se" --aspect 16:9 --duration 5.0 --steps "$st" \
      --seed "$SEED" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$PROMPT" --out "$tag" > "${tag}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    # inference_time_s 是服务端自己报的纯推理时间（wall 还含请求往返和落盘）。它在 status.json 里
    # 的嵌套位置随服务端版本变过，所以按 key 递归找，不写死路径。
    inf=$(python3 - "$tag" <<'PY' 2>/dev/null
import json, sys
def dig(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "inference_time_s": return v
            r = dig(v)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = dig(v)
            if r is not None: return r
print(round(dig(json.load(open(sys.argv[1] + "_status.json"))), 3))
PY
)
    echo "REPRO $tag rc=$rc inference_time_s=${inf:-NA} wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
}

[ "${SKIP_G1:-0}" = 1 ] || run_arm 1 _g1 768_30      # 同机分母
run_arm 2 "" $CASES                                   # 2 卡 Ulysses=2

(unset VARIANT; ./serve.sh stop) >/dev/null 2>&1

# 画质：1 卡当参考，2 卡当候选。看 SSIM 是否在 0.944 的归约序地板之上、码率是否几乎不动。
# RUNDIR 必须传：quality_pair.sh 默认指向另一条产线的目录，不传就报 MISSING。
[ -f fl2va_768_30_fp8_sage_attn_g1.mp4 ] && [ -f fl2va_768_30_fp8_sage_attn.mp4 ] && \
  RUNDIR="$PWD" ./quality_pair.sh fl2va_768_30_fp8_sage_attn_g1 fl2va_768_30_fp8_sage_attn

echo "TWO_CARD_SAGE_DONE $(date -u +%FT%TZ)"
