#!/usr/bin/env bash
# Ref2VA 的 NVFP4 到底该用谁转的：**自量化** vs **第三方（lilcheaty + canonicalize）** vs **BF16 真值**。
#
#   ./g7e_ref2va_provenance.sh                        # 三条腿，480p/20 + 768p/20
#   ARMS="selfq bf16" CASES="768_20" ./g7e_ref2va_provenance.sh
#
# 为什么要这一跑：g7e 交付表的 ref2va 半边是在第三方文件
# （`lilcheaty/MiniMax-H3-NVFP4` 的 `minimax_h3_ref2va_nvfp4_mixed.safetensors` 过
# `nvfp4_canonicalize.py` -> `nvfp4_ref2va_fixed.safetensors`）上量的，而我们自己的
# `nvfp4_quantize_transformer.py` 与 partition 无关（B300 那半张 ref2va 表全是自量化的）。
# 判据不是"两个 NVFP4 文件互相像"，而是**谁离 BF16 更近**——所以 BF16 是参考腿，两个 NVFP4
# 各自对它做 SSIM。自检行两边都该是 951 张量 / worst rel 0.0951。
#
# 口径与交付表一致：ref 短边 1024、seed 8201、REF2VA_PROMPT、5.0 s、flow_shift 12/3、1 卡、sage。
# BF16 那条腿**不带** NVFP4 的三个 env 也不带 `--transformer-weights-path`（走 stock 分片），
# 但仍带 sage 和 text_encoder 豁免——否则比的就不止是 checkpoint 了。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:dev}
NAME=${NAME:-h3n}
OUT=/opt/dlami/nvme/out
COUT=/out
CASES=${CASES:-"480_20 768_20"}
ARMS=${ARMS:-"bf16 selfq third"}
# 本地素材目录里有 input_cat.jpg 就用它（和历史表同口径），没有就退到公开样例图。跑错输入是静默的，
# 所以 prompts.sh 会把实际用的图打成 PROMPT_PAIR 一行。
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
PORT=30030
SEED=${SEED:-8201}
RSE=${RSE:-1024}
# c0b6474 已经把 cpu-offload-inplace 收进上游了，serve.sh 的默认列表里还有它 → 打不上就 exit 1，
# 三条腿会各自"起服务失败"在 23 秒内全灭（serve 日志空文件，只有 start_prov_*.log 里那行
# DOES_NOT_APPLY 能看出来）。所以这里跟 g7e_dev_levers.sh 一样显式收窄补丁列表。
PATCHES=${PATCHES:-"minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch"}
. assets/prompts.sh   # prompt 按 $IMG 配对，别在这里写字面量
PROMPT=${PROMPT:-$REF2VA_PROMPT}

infer_s() { python3 - "$1" <<'PY' 2>/dev/null
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
}

echo "===== REF2VA_PROVENANCE arms='$ARMS' cases='$CASES' $(date -u +%FT%TZ)"
# 自量化文件：没有就现量化（纯 CPU，约 10 min，此时不能有计时请求）
if echo "$ARMS" | grep -qw selfq && [ ! -f "$OUT/nvfp4_ref2va.safetensors" ]; then
  (unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1
  NAME=$NAME PARTS=Ref2VA ./g7e_quant.sh || exit 1
fi
# 第三方文件：下载 + 规范化（规范化要一次拿住所有张量，峰值 host RAM ≈ 文件大小）
if echo "$ARMS" | grep -qw third && [ ! -f "$OUT/nvfp4_ref2va_fixed.safetensors" ]; then
  (unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1
  if [ ! -f "$OUT/nvfp4_ref2va_third.safetensors" ]; then
    echo "== download lilcheaty $(date -u +%H:%M:%S)"
    /opt/dlami/nvme/venv/bin/hf download lilcheaty/MiniMax-H3-NVFP4 \
      --include "minimax_h3_ref2va_nvfp4_mixed.safetensors" --local-dir "$OUT/nvfp4_dl" || exit 1
    ln -f "$OUT/nvfp4_dl/minimax_h3_ref2va_nvfp4_mixed.safetensors" "$OUT/nvfp4_ref2va_third.safetensors"
  fi
  echo "== canonicalize $(date -u +%H:%M:%S)"
  docker cp nvfp4_canonicalize.py "$NAME:/tmp/" >/dev/null
  docker exec -e NVFP4_SRC=$COUT/nvfp4_ref2va_third.safetensors \
              -e NVFP4_DST=$COUT/nvfp4_ref2va_fixed.safetensors \
    "$NAME" python3 /tmp/nvfp4_canonicalize.py || exit 1
fi
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  src=patches/$p; [ -f "$src" ] || src=$p
  docker cp "$src" "$NAME:/tmp/" >/dev/null && docker exec "$NAME" python3 "/tmp/$p" >/dev/null || exit 1
done

for arm in $ARMS; do
  NVENV="SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0"
  case $arm in
    bf16)  CK=""; ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0" ;;
    selfq) CK="--transformer-weights-path $COUT/nvfp4_ref2va.safetensors"
           ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 $NVENV" ;;
    third) CK="--transformer-weights-path $COUT/nvfp4_ref2va_fixed.safetensors"
           ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 $NVENV" ;;
    *) echo "SKIP unknown arm '$arm'"; continue ;;
  esac
  ENVX="$ENVX SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=$RSE"
  echo "=== ARM $arm $(date -u +%H:%M:%S)"
  (unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  VARIANT=ref2va GPUS=1 ULYSSES=1 IMAGE=$IMAGE NAME=$NAME ENVX="$ENVX" PATCHES="$PATCHES" \
    EXTRA="--layerwise-offload-components text_encoder $CK --attention-backend sage_attn \
           --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa" \
    LOG=$COUT/serve_prov_$arm.log ./serve.sh start > "$OUT/start_prov_$arm.log" 2>&1 \
    || { echo "ARM_FAILED $arm （看 $OUT/serve_prov_$arm.log）"; continue; }
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    name="prov_${arm}_${se}_${st}"
    t0=$(date +%s)
    python3 h3gen.py --task ref2va --image "$IMG" --inline --short-edge "$se" --aspect 16:9 \
      --duration 5.0 --steps "$st" --seed "$SEED" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$PROMPT" --port "$PORT" --out "$name" > "${name}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "PROV $arm ${se}_${st} rc=$rc inference_time_s=$(infer_s "$name") wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
done
(unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1

# 画质：BF16 当参考，两个 NVFP4 各自对它比。谁的 SSIM 高谁就该进交付。
for c in $CASES; do
  se=${c%_*}; st=${c#*_}
  for arm in selfq third; do
    [ -f "prov_bf16_${se}_${st}.mp4" ] && [ -f "prov_${arm}_${se}_${st}.mp4" ] && \
      RUNDIR="$PWD" NAME="$NAME" ./quality_pair.sh "prov_bf16_${se}_${st}" "prov_${arm}_${se}_${st}"
  done
done
echo "REF2VA_PROVENANCE_DONE $(date -u +%FT%TZ)"
