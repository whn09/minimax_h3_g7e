#!/usr/bin/env bash
# ref2va 的**参考图短边**扫描（1 卡，交付配置 NVFP4+sage）。这是 ref2va 上唯一一个"减少工作量"
# 而不是"算得更快"的旋钮，也是唯一还没扫到底的：上游默认 2048，我们已经钉到 1024（值 1.46×），
# 但 1024 相对 768p 输出画面仍然大 1.33×、相对 480p 大 2.13× —— 再往下走大概还有。
#
#   ./g7e_ref_edge_sweep.sh                       # 1024（分母）/ 768 / 512，两个几何
#   EDGES="768" CASES="768_20" ./g7e_ref_edge_sweep.sh
#
# 机制：`SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE` 决定参考图编码成多少 token，那些 token 全程参与
# packed full self-attention，所以它同时压 attention 的 seq 和每步的 KV 字节。**它不改输出画面几何**，
# 所以时间省下来是真省，不是偷分辨率。
#
# **这是有损的方向**：参考图给小了，身份/纹理保持会变差。判据不是 SSIM 接近 1.0（换 conditioning
# 一定会改轨迹），而是三条一起看 + 人眼看成片：
#   SSIM 对 1024 的片子、运动能量、码率（`quality_pair.sh`），再把 mp4 拉回本地逐帧看猫的花纹。
# 单卡同 seed 重跑是逐位相同的（SSIM 1.000000），所以这里 SSIM 的任何下降都是这个旋钮造成的。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:h3-validated}
NAME=${NAME:-h3}
OUT=/opt/dlami/nvme/out
COUT=/out
CKPT=${CKPT:-$COUT/nvfp4_ref2va_fixed.safetensors}
CASES=${CASES:-"768_20 480_20"}
EDGES=${EDGES:-"1024 768 512"}
# 本地素材目录里有 input_cat.jpg 就用它（和历史表同口径），没有就退到公开样例图。跑错输入是静默的，
# 所以 prompts.sh 会把实际用的图打成 PROMPT_PAIR 一行。
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
PORT=${PORT:-30030}
SEED=${SEED:-8201}
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

echo "===== REF_EDGE_SWEEP edges='$EDGES' cases='$CASES' $(date -u +%FT%TZ)"
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  src=patches/$p; [ -f "$src" ] || src=$p
  docker cp "$src" "$NAME:/tmp/" >/dev/null && docker exec "$NAME" python3 "/tmp/$p" || exit 1
done

for edge in $EDGES; do
  echo "=== EDGE $edge $(date -u +%H:%M:%S)"
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  VARIANT=ref2va GPUS=1 ULYSSES=1 IMAGE=$IMAGE \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=$edge \
          SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0" \
    EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path $CKPT \
           --attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa" \
    LOG=$COUT/serve_edge_$edge.log ./serve.sh start > "$OUT/start_edge_$edge.log" 2>&1 \
    || { echo "ARM_FAILED edge=$edge （看 $OUT/serve_edge_$edge.log）"; continue; }
  # 一次性开销按进程、按张量形状付；参考图短边也是形状的一部分，所以每个 edge 都要重新热
  for w in 480 768; do
    python3 h3gen.py --task ref2va --image "$IMG" --inline --short-edge "$w" --aspect 16:9 \
      --duration 5.0 --steps 4 --seed "$SEED" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$PROMPT" --port "$PORT" --out "warm_edge_${edge}_$w" >/dev/null 2>&1
  done
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    name="edge${edge}_${se}_${st}"
    t0=$(date +%s)
    python3 h3gen.py --task ref2va --image "$IMG" --inline --short-edge "$se" --aspect 16:9 \
      --duration 5.0 --steps "$st" --seed "$SEED" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$PROMPT" --port "$PORT" --out "$name" > "${name}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "EDGE $edge ${se}_${st} rc=$rc inference_time_s=$(infer_s "$name") wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
  # 回读服务端真的收到了这个短边（env 是静默的，写错了不会报错，只会白测一轮）
  docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_edge_$edge.log | grep -oiE 'ref.?image.?short.?edge[^,]*' | sort -u | head -3" 2>/dev/null
done
(unset VARIANT; ./serve.sh stop) >/dev/null 2>&1

# 画质：1024 当参考（现交付口径），其余当候选
for c in $CASES; do
  se=${c%_*}; st=${c#*_}
  for edge in $EDGES; do
    [ "$edge" = 1024 ] && continue
    [ -f "edge1024_${se}_${st}.mp4" ] && [ -f "edge${edge}_${se}_${st}.mp4" ] && \
      RUNDIR="$PWD" NAME="$NAME" ./quality_pair.sh "edge1024_${se}_${st}" "edge${edge}_${se}_${st}"
  done
done
echo "REF_EDGE_SWEEP_DONE $(date -u +%FT%TZ)"
