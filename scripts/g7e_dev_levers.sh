#!/usr/bin/env bash
# 在**最新 sglang** 上做的旋钮消融：`lmsysorg/sglang:dev` = c0b6474（2026-08-17），
# 对照交付镜像 `h3-validated` = 273d978be（2026-08-12）。容器另起一个（NAME=h3n），
# 交付容器 h3 原样保留，两个容器不能同时跑 server（96 GB 一张卡装不下两份）。
#
#   ./g7e_dev_levers.sh                            # 全部 arm
#   ARMS="cache_1" CASES="768_20" ./g7e_dev_levers.sh
#
# 一次性准备（本脚本会自己做，幂等）：
#   NAME=h3n IMAGE=lmsysorg/sglang:dev \
#     PATCHES="minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch" \
#     ./serve.sh prepare                           # 新镜像上 cpu-offload-inplace 已进上游、
#                                                  # target-width-height 要重新 diff
#   NAME=h3n ./build_sageattention.sh              # sage2 必须重编（镜像里没有）
#   两个 NVFP4 python 补丁在 c0b6474 上**仍然都要打、也都还能打**（实测 APPLIED）：
#   上游没修 sm_120 的标度布局分支（dev 里还是 `if flashinfer_backend is None or
#   uses_flux1_scale_layout:`），qkv 那个是模型侧缺陷。
#
# c0b6474 相对 273d978be 给 H3 新增的东西（源码 diff）：
#   * MiniMaxH3AdalnCache：按 fp32 timestep plan 预算 AdaLN，配 --minimax-h3-adaln-online /
#     --minimax-h3-adaln-cache-path，省掉常驻的 24.2 GiB adaln_proj。
#   * Cache-DiT 的 input preservation：H3 之前 residual 读出来是 0（静默），dev 才修对。
#   * 新 flag：--component-residency / --dit-layerwise-residency-policy /
#     --load-diffusion-decoder / 上面两个 adaln / --minimax-h3-adaln-plan-width。
#
# 各 arm 的先验：
#   cache    Cache-DiT 的**通用（env 驱动）**路径。注意：`--cache-dit-config` 是 diffusers 后端的，
#            H3 走 native pipeline，旋钮全在 env 上：SGLANG_CACHE_DIT_ENABLED/_FN/_BN/_WARMUP/
#            _RDT/_MC/_TAYLORSEER/_TS_ORDER（+ _SECONDARY_* 给音频那条 transformer）。
#            默认 Fn=1 Bn=0 W=4 RDT=0.24 MC=3。**有损**，所以每个 arm 都过 quality_pair.sh。
#            另一个坑：请求里只要带 `quality` 字段，H3 就把通用路径关掉
#            （`super()._cache_dit_requested() and "quality" not in explicit_fields`），
#            h3gen.py 默认不发 quality，别加 --quality。
#   cachehq  把 RDT/MC 收到上游为 H3 审过的那组（0.04 / 1，SSIM 0.931 / PSNR 28.16 dB），
#            但走通用路径 —— 直接用 quality="high" 会撞死在一个写死的部署门（4×H200 / 50 步 /
#            1344×768 / sm_9.0 才放行）。
#   adaln    --minimax-h3-adaln-online。**预期与 NVFP4 互斥**：minimax_h3.py:1468 是
#            `if (adaln_cache_path or adaln_weight_files) and quant_config is not None: raise`。
#            留这个 arm 只为把"预期报错"变成实测；它省的 24.2 GiB 是 BF16 口径，NVFP4 下
#            adaln_proj 也已经被量化，省的没那么多。
#
# 顺序跑，不并跑（两张卡各 600 W，并跑时功耗墙互相拖）。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:dev}
NAME=${NAME:-h3n}
PATCHES=${PATCHES:-"minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch"}
OUT=/opt/dlami/nvme/out
COUT=/out
VARIANT=${VARIANT:-fl2va}
CKPT=${CKPT:-$COUT/nvfp4_fl2va.safetensors}
CASES=${CASES:-"768_20 480_20"}
IMG=${IMG:-assets/first.png}
PORT=${PORT:-30010}
SEED=${SEED:-6201}
ARMS=${ARMS:-"base_1 cache_1 cachehq_1 adaln_1"}
PROMPT=${PROMPT:-"A white cat sitting on an open window ledge slowly turns its head toward the camera, blinks, and gently lifts one paw while a soft breeze moves the curtains. Natural afternoon light, subtle street ambience and soft paw sounds, realistic motion, static cinematic camera."}

req() {  # req <port> <short_edge> <steps> <out-tag>
  python3 h3gen.py --task "$VARIANT" --image "$IMG" --inline \
    --short-edge "$2" --aspect 16:9 --duration 5.0 --steps "$3" --seed "$SEED" \
    --flow-shift 12.0 --audio-flow-shift 3.0 --prompt "$PROMPT" --port "$1" --out "$4"; }

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

echo "===== G7E_DEV_LEVERS image=$IMAGE arms='$ARMS' cases='$CASES' $(date -u +%FT%TZ)"
# 交付容器必须先停：一张卡装不下两份权重。
(unset VARIANT; NAME=h3 ./serve.sh stop) >/dev/null 2>&1
NAME=$NAME IMAGE=$IMAGE PATCHES="$PATCHES" ./serve.sh prepare || exit 1
docker exec -w /tmp "$NAME" python3 -c 'import sageattention' 2>/dev/null \
  || { echo "sage 没装：先跑 NAME=$NAME ./build_sageattention.sh" >&2; exit 1; }
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  src=patches/$p; [ -f "$src" ] || src=$p
  [ -f "$src" ] || { echo "MISSING $p" >&2; exit 1; }
  docker cp "$src" "$NAME:/tmp/" >/dev/null && docker exec "$NAME" python3 "/tmp/$p" || exit 1
done
docker exec "$NAME" bash -lc 'python3 -c "import sglang;print(\"sglang\", sglang.__version__)"' 2>/dev/null | tail -1

for arm in $ARMS; do
  G=${arm##*_}; KNOB=${arm%_*}
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path $CKPT"
  ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto \
        H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0"
  case $KNOB in
    base) ;;
    cache)   ENVX="$ENVX SGLANG_CACHE_DIT_ENABLED=1" ;;
    cachehq) ENVX="$ENVX SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.04 \
                   SGLANG_CACHE_DIT_MC=1 SGLANG_CACHE_DIT_WARMUP=4 \
                   SGLANG_CACHE_DIT_SECONDARY_RDT=0.04 SGLANG_CACHE_DIT_SECONDARY_MC=1" ;;
    # cacheR<NN> = RDT 0.NN，MC 留默认 3。RDT 是"两步之间 residual 差多小就复用"的阈值，
    # 0.04 实测在 20 步上一次都没触发（和 base 逐格同分），0.24 触发到 1.92× —— 中间那段才是
    # 真正的性价比曲线，所以要扫。
    cacheR*) ENVX="$ENVX SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.${KNOB#cacheR} \
                   SGLANG_CACHE_DIT_SECONDARY_RDT=0.${KNOB#cacheR}" ;;
    adaln)   EXTRA="$EXTRA --minimax-h3-adaln-online --minimax-h3-adaln-plan-width 3" ;;
    *) echo "SKIP unknown knob '$KNOB'"; continue ;;
  esac
  # c0b6474 上要豁免的组件比 273d978be **多**：新镜像里 audio_vae 也去 selector 解 attention
  # 后端，而它只声明 ['fa','torch_sdpa']，于是全局 `--attention-backend sage_attn` 会让它
  # `Failed to load customized audio_vae`（`selector.py:300` ValueError → scheduler is dead）。
  # 三个非 DiT 组件一律 torch_sdpa（名字取自 model_index.json，**逗号分隔不是空格**）。
  EXTRA="$EXTRA --attention-backend sage_attn \
    --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa"

  echo "=== ARM $arm gpus=$G $(date -u +%H:%M:%S)"
  (unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  VARIANT=$VARIANT GPUS=$G ULYSSES=$G IMAGE=$IMAGE NAME=$NAME PATCHES="$PATCHES" \
    ENVX="$ENVX" EXTRA="$EXTRA" LOG=$COUT/serve_dev_$arm.log ./serve.sh start \
    > "$OUT/start_dev_$arm.log" 2>&1 \
    || { echo "ARM_FAILED $arm （看 $OUT/serve_dev_$arm.log）"
         docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_dev_$arm.log | grep -iE 'error|raise|Traceback' | tail -5" 2>/dev/null
         continue; }
  for w in 480 768; do req "$PORT" "$w" 4 "warm_dev_${arm}_$w" >/dev/null 2>&1; done
  echo "== $arm warm $(date -u +%H:%M:%S)"
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    name="dev_${arm}_${se}_${st}"
    t0=$(date +%s)
    req "$PORT" "$se" "$st" "$name" > "${name}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "DEVLEVER $arm ${se}_${st} rc=$rc inference_time_s=$(infer_s "$name") wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
  # 回读真正落地的东西：attention 后端（sm_120 上 fa 会被静默降级）+ Cache-DiT 到底挂上没有
  # （"Acceleration hooks is disabled for: BlockAdapter" = 没挂上，别把噪声当收益）。
  docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_dev_$arm.log | grep -oiE 'Enabling cache-dit[^\"]{0,120}|Acceleration hooks is disabled[^\"]{0,40}|SCM enabled[^\"]{0,80}|sage[a-z0-9_]*|\"attention_backend\": [^,]*' | sort -u | head -10" 2>/dev/null
done
(unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1

# 画质：新镜像的 base_1 当参考（同镜像同卡数，唯一变量是那个旋钮）。单卡同 seed 重跑逐位相同，
# 所以无损旋钮应当 SSIM 1.000000；cache_* 是有损的，这里量的就是损多少。
for c in $CASES; do
  se=${c%_*}; st=${c#*_}
  for arm in $ARMS; do
    [ "${arm##*_}" = 1 ] && [ "$arm" != base_1 ] || continue
    [ -f "dev_base_1_${se}_${st}.mp4" ] && [ -f "dev_${arm}_${se}_${st}.mp4" ] && \
      RUNDIR="$PWD" ./quality_pair.sh "dev_base_1_${se}_${st}" "dev_${arm}_${se}_${st}"
  done
done
echo "G7E_DEV_LEVERS_DONE $(date -u +%FT%TZ)"
