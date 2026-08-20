#!/usr/bin/env bash
# g7e 上还没动过的 sglang 性能旋钮消融（1 卡 / 2 卡，交付配置 NVFP4 之上再叠）。同一台机、同一
# session、同 prompt/seed/输入图，数字取 server 报的 inference_time_s。
#
#   ./g7e_levers.sh                                  # 全部 arm
#   ARMS="bcg_1" CASES="768_20" ./g7e_levers.sh
#
# 起点 = 交付配置 NVFP4+sage2（`runs/nvfp4_table/RESULTS.md`，fl2va）：
#   1 卡 480p/20 31.125 s、480p/30 45.145、768p/20 114.419、768p/30 170.278
#   2 卡 480p/20 22.375、768p/20  77.231
# 每个 arm 都重测 base，**不要跨 session 引用上面的数**：g7e 跨重启漂 5.4%（600 W 功耗墙），
# 同进程内才是 0.3%。所以 base_1/base_2 是这次的分母。
#
# 各旋钮先验：
#   bcg   --enable-breakable-cuda-graph：把 DiT forward 抓成 CUDA graph 段（在 attention 处断开），
#         省每 kernel 的 launch 开销，数值无损。B300 上实测白开（8 卡 768p 噪声内、480p 慢 3.8%）
#         且 1 卡直接崩（捕获成功，第一次真前向 illegal memory access），这里只是确认同样的结论。
#         显存上 1 卡塞得下：NVFP4+sage 峰值 52.0 GB + B300 实测 BCG 的 35 GB = 87 < 96。
#
# 已排除、不用测的：
#   sage_attn_3           **H3 结构性用不了，别加回来。** 编译是成功的（sageattn3-1.0.0/sm_120a），
#                         但起服务时 minimax_h3.py:192 validate_server_args 要求
#                         packed_varlen=True，`sage_attn_3` 没覆写 forward_varlen →
#                         ValueError，报错不是静默回落。CUDA 上覆写了 forward_varlen 的只有 5 个：
#                         sage_attn（=交付的 sage2）、fa、torch_sdpa、sol_attn、subblock_sparse_attn
#                         （最后这个要 sm_100a，g7e/B300 都不行）。证据见 rejected/build_sageattention3.sh。
#   --cfg-parallel-size   H3 的 deployment config 写死 supports_cfg_parallel=False（CFG 蒸馏）。
#   --batching-max-size   _can_dynamic_batch() 里 `image_path is not None -> return False`，
#                         fl2va/ref2va 永远进不了 batch；只有纯文本 t2va 能。
#   --enable-torch-compile 与 bcg 互斥，且 flag 自己的帮助文本写了 "will likely cause precision drifts"。
#
# 顺序跑，不并跑：两张卡各 600 W，并跑时功耗墙互相拖、时间数不可引用。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:h3-validated}
NAME=${NAME:-h3}
OUT=/opt/dlami/nvme/out
COUT=/out
VARIANT=${VARIANT:-fl2va}
CKPT=${CKPT:-$COUT/nvfp4_fl2va.safetensors}
CASES=${CASES:-"768_20 480_20"}
# 本地素材目录里有 input_cat.jpg 就用它（和历史表同口径），没有就退到公开样例图。跑错输入是静默的，
# 所以 prompts.sh 会把实际用的图打成 PROMPT_PAIR 一行。输入图只改内容不改形状，所以不影响时间，
# 但会影响画质比对（换图必须换 prompt，见 assets/prompts.sh）。
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
PORT=${PORT:-30010}
SEED=${SEED:-6201}
# arm 名 = <旋钮>_<卡数>，**下划线是必须的**（B300 那版写成 `sa38` + `${arm##*[a-z]}` 取卡数，
# glob 剥成 knob=sa/gpus=38，两个 sage arm 被静默 SKIP 掉了）。
ARMS=${ARMS:-"base_1 bcg_1 base_2"}
. assets/prompts.sh   # prompt 按 $IMG 配对，别在这里写字面量
PROMPT=${PROMPT:-$FL2VA_PROMPT}

req() {  # req <port> <short_edge> <steps> <out-tag>
  python3 h3gen.py --task "$VARIANT" --image "$IMG" --inline \
    --short-edge "$2" --aspect 16:9 --duration 5.0 --steps "$3" --seed "$SEED" \
    --flow-shift 12.0 --audio-flow-shift 3.0 --prompt "$PROMPT" --port "$1" --out "$4"
}

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

echo "===== G7E_LEVERS arms='$ARMS' cases='$CASES' $(date -u +%FT%TZ)"
# 两个 NVFP4 源码补丁（改容器里的源码，不走 serve.sh 的 .patch 流程），幂等。
# 两个位置都找：库里在 patches/，机器上历史遗留是平铺在 scripts/ 里的。
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  src=patches/$p; [ -f "$src" ] || src=$p
  [ -f "$src" ] || { echo "MISSING $p（patches/ 和当前目录都没有）" >&2; exit 1; }
  docker cp "$src" "$NAME:/tmp/" >/dev/null && docker exec "$NAME" python3 "/tmp/$p" || exit 1
done

for arm in $ARMS; do
  G=${arm##*_}; KNOB=${arm%_*}
  ATTN=sage_attn                     # 交付基线就带 sage2，所有 arm 的分母都含它
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path $CKPT"
  case $KNOB in
    base) ;;
    bcg)  EXTRA="$EXTRA --enable-breakable-cuda-graph" ;;
    *) echo "SKIP unknown knob '$KNOB'"; continue ;;
  esac
  # sage 只能全局设 + 必须把 text encoder 豁免：Qwen3VL 的 LocalAttention 只声明 {fa, torch_sdpa}，
  # 裸的全局 sage 会在构建 text encoder 时把服务打死。per-component 反过来设**什么都测不到**
  # （contextvar 只在组件 load 期间有效，H3 的 DiT 第一次 forward 才解析 backend，那时已经空了）。
  EXTRA="$EXTRA --attention-backend $ATTN --component-attention-backends text_encoder=torch_sdpa"

  echo "=== ARM $arm gpus=$G attn=$ATTN $(date -u +%H:%M:%S)"
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  VARIANT=$VARIANT GPUS=$G ULYSSES=$G IMAGE=$IMAGE \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto \
          H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0" \
    EXTRA="$EXTRA" LOG=$COUT/serve_lev_$arm.log ./serve.sh start \
    > "$OUT/start_lev_$arm.log" 2>&1 \
    || { echo "ARM_FAILED $arm （看 $OUT/serve_lev_$arm.log）"; continue; }
  # 一次性开销按 arm、按分辨率付（flashinfer fp4 GEMM autotune 缓存按张量形状、cudnn handle、
  # 首次分配）。不预热的话 480p/20 会比同 arm 的 480p/30 还慢。步数不影响形状，所以 4 步就够。
  for w in 480 768; do req "$PORT" "$w" 4 "warm_lev_${arm}_$w" >/dev/null 2>&1; done
  echo "== $arm warm $(date -u +%H:%M:%S)"
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    name="lev_${arm}_${se}_${st}"
    t0=$(date +%s)
    req "$PORT" "$se" "$st" "$name" > "${name}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "LEVER $arm ${se}_${st} rc=$rc inference_time_s=$(infer_s "$name") wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
  # **回读真正落地的后端**。`--attention-backend fa` 在 sm_120 是被静默降级的，只看时间会把
  # "后端没生效" 误读成 "这个后端更慢"。
  docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_lev_$arm.log | grep -oiE 'attention backend[^,\"]*|\"attention_backend\": [^,]*|sage[a-z0-9_]*' | sort -u | head -8" 2>/dev/null
done
(unset VARIANT; ./serve.sh stop) >/dev/null 2>&1

# 画质：base_1 当参考，其余 1 卡 arm 当候选（同卡数同几何，唯一变量是那个旋钮）。
# bcg 号称数值无损，所以这里期望 SSIM 1.000000（单卡同 seed 重跑是逐位相同的）；低于它就是有损。
for c in $CASES; do
  se=${c%_*}; st=${c#*_}
  for arm in $ARMS; do
    [ "${arm##*_}" = 1 ] && [ "$arm" != base_1 ] || continue
    [ -f "lev_base_1_${se}_${st}.mp4" ] && [ -f "lev_${arm}_${se}_${st}.mp4" ] && \
      RUNDIR="$PWD" NAME="$NAME" ./quality_pair.sh "lev_base_1_${se}_${st}" "lev_${arm}_${se}_${st}"
  done
done
echo "G7E_LEVERS_DONE $(date -u +%FT%TZ)"
