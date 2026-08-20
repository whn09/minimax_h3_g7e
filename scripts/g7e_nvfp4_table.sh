#!/usr/bin/env bash
# NVFP4 权重（可叠 SageAttention）在 g7e 上的 16 格表：{fl2va,ref2va} × {1,2 卡} × {480,768}p × {20,30} 步。
#
#   ./g7e_nvfp4_table.sh                       # 16 格，先 fl2va 后 ref2va，各自先 1 卡后 2 卡
#   ATTN=sage_attn ./g7e_nvfp4_table.sh        # 叠 SageAttention（**交付配置**，tag 带 _sage_attn）
#   ONLY=fl2va ./g7e_nvfp4_table.sh
#   GPUSET=2 ONLY=ref2va CASES="768_30" ./g7e_nvfp4_table.sh
#   FL2VA_CKPT=/out/x.safetensors REF2VA_CKPT=/out/y.safetensors ./g7e_nvfp4_table.sh
#
# 两个 variant 的 checkpoint 来路不同，两条路都得覆盖：
#   fl2va   `nvfp4_quantize_transformer.py` 从 stock 的 `FL2VA/transformer` bf16 现量化（~90 s）。
#           本脚本在 checkpoint 不存在时自动量化。
#   ref2va  lilcheaty/MiniMax-H3-NVFP4 的 `MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors`，
#           **必须先过 `nvfp4_canonicalize.py`**（那个文件的 scale 被 128×4 swizzle、nibble 高位在前、
#           qkv 行是 [q;k;v] 分块，三处都要掰回来）。第三方那边没有可用的 FL2VA，只有裁过的
#           `_pruned_nvfp4`，所以 fl2va 只能自己量化。
# 配方是同一套：208 个线性层 nvfp4 + 50 个 `adaln_proj.linear.weight` 裸 fp8 + 其余 bf16，
# 951 个张量，无 `input_scale`；round-trip 相对误差 0.094–0.0951 = group-16 e2m1 的量化地板。
#
# sm_120 上 NVFP4 需要的四件事，脚本都带上了，缺一个就是废片或起不来：
#   SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto  默认的 trtllm fp4 GEMM 在 capability 120 不存在
#   patches/patch_nvfp4_tma_scale_layout.py + H3_FP4_TMA_SCALES=1
#                                             auto 在 sm_120 落到 cutlass，它要 TMA 标度布局；
#                                             用朴素布局每层相对误差 0.55 而不是 0.095
#   patches/patch_h3_qkv_scale_reorder.py     qkv 的**行**被 DiT 自己的 loader 置换过，per-row 块标度
#                                             没跟着换 → 52 层全拿错标度 → 一整片灰糊
#   H3_FP4_QKV_FIX=0                          关掉 `patch_nvfp4_mixed_precision.py` 里的同一个修复
#                                             （如果那个补丁也在容器里）。两个一起开 = reorder 两次，
#                                             和一次都不做一样坏。
# **不要**传 `--quantization modelopt_fp4`：从 safetensors 自动推断出来的配置才是对的，显式给会建出
# 一个空的、不能用的 config。
#
# 实测（2026-08-19，东京 g7e.12xlarge spot，镜像 273d978b，sage d1a57a5，5.175 s 成片）：
#   NVFP4+sage 对 FP8+sage **32/32 格全赢**：1 卡 1.26–1.29×、2 卡 1.13–1.17×。
#   不叠 sage 时 480p 打平、768p 反而慢 15–19%——权重量化碰不到 attention，而 attention 是 768p
#   一步的 59%。所以 **NVFP4 必须配 sage**，单独上 NVFP4 在目标档位是退步。
#   2 卡的倍数系统性低于 1 卡（加卡收益 1.62× → 1.48×）：Ulysses 的 PCIe 通信是固定开销，DiT 算得
#   越快它占比越大。任何"让 DiT 更快"的优化都会同时稀释加卡收益。
#
# 顺序跑，不并跑：两张卡各 600 W，并跑时功耗墙会互相拖，时间数不可引用。
# checkpoint 量化是纯 CPU 的，脚本安排在没有计时请求的时候做——48 vCPU 被吃满会把非 DiT 那几段
# （VAE 出帧、封装）推上去。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
IMAGE=${IMAGE:-lmsysorg/sglang:h3-validated}   # 不用 :dev，那是会移动的 tag（g7e_bringup.sh 会建这个固定 tag）
NAME=${NAME:-h3}
OUT=/opt/dlami/nvme/out          # 主机视角
COUT=/out                        # 容器里的同一个目录
CASES=${CASES:-"480_20 480_30 768_20 768_30"}
# 本地素材目录里有 input_cat.jpg 就用它（和历史表同口径），没有就退到公开样例图。跑错输入是静默的，
# 所以 prompts.sh 会把实际用的图打成 PROMPT_PAIR 一行。
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
ATTN=${ATTN:-}
FL2VA_CKPT=${FL2VA_CKPT:-$COUT/nvfp4_fl2va.safetensors}
REF2VA_CKPT=${REF2VA_CKPT:-$COUT/nvfp4_ref2va_fixed.safetensors}
mkdir -p "$OUT"

. assets/prompts.sh   # prompt 按 $IMG 配对，别在这里写字面量

# 两个补丁是**改容器里的源码**，不是 git patch，所以不走 serve.sh 那套 .patch 流程。两个脚本自己幂等
# （已打过就打印 already patched），所以无条件跑。
echo "== applying the two NVFP4 patches (idempotent)"
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  [ -f "patches/$p" ] || { echo "MISSING patches/$p" >&2; exit 1; }
  docker cp "patches/$p" "$NAME:/tmp/" >/dev/null || exit 1
  docker exec "$NAME" python3 "/tmp/$p" || { echo "PATCH_FAILED $p" >&2; exit 1; }
done

# fl2va 的 checkpoint 没有就现量化。放在起服务之前，且此时没有计时请求在跑。
if [ "${ONLY:-both}" != ref2va ] && [ ! -f "${FL2VA_CKPT/#$COUT/$OUT}" ]; then
  echo "== quantizing FL2VA -> $FL2VA_CKPT $(date -u +%H:%M:%S)"
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  docker cp nvfp4_quantize_transformer.py "$NAME:/tmp/" >/dev/null
  docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer -e DST="$FL2VA_CKPT" \
    "$NAME" python3 /tmp/nvfp4_quantize_transformer.py || exit 1
fi

# 一个臂：起服务（NVFP4 权重 [+ sage]）-> 按 CASES 发请求 -> 下一个臂前由下一次 start 停掉
run_arm() {  # run_arm <variant> <gpus>
  local variant=$1 gpus=$2 port seed prompt tagmid refenv
  case $variant in
    fl2va)  port=30010; seed=${SEED:-6201}; prompt=$FL2VA_PROMPT; tagmid=""; refenv="" ;;
    ref2va) port=30030; seed=${SEED:-8201}; prompt=$REF2VA_PROMPT
            # 参考短边 1024 是 g7e 上 1.46× 的杠杆，也是这张表其它行的口径，所以钉住并写进 tag
            local rse=${REF_SHORT_EDGE:-1024}
            tagmid="_r${rse}"; refenv="SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=$rse" ;;
    *) echo "variant must be fl2va|ref2va" >&2; return 2 ;;
  esac
  local ck=$REF2VA_CKPT; [ "$variant" = fl2va ] && ck=$FL2VA_CKPT

  echo "=== ARM $variant gpus=$gpus ulysses=$gpus nvfp4=$ck attn=${ATTN:-torch_sdpa} $(date -u +%H:%M:%S)"
  # 停**所有**副本，不只是这个 variant：serve.sh 的 stop 只要环境里有 VARIANT 就会缩到一个副本，
  # 剩下的那个会一直占着显存，新副本就在加载中途死在 worker 管道的一个裸 EOFError 上。
  (unset VARIANT; ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  # sage 只能全局设，且必须把 text encoder 单独豁免：Qwen3VL 的 LocalAttention 只声明
  # {fa, torch_sdpa}，裸的全局 sage_attn 会在构建 text encoder 时把服务打死。反过来
  # `--attention-backend torch_sdpa --component-attention-backends transformer=sage_attn`
  # **什么都测不到**：per-component 是个 contextvar，只在组件 load 期间有效，而 H3 的 DiT 是
  # 第一次 forward 才惰性解析 backend，那时它已经空了，于是回落到全局值。
  VARIANT=$variant GPUS=$gpus ULYSSES=$gpus IMAGE=$IMAGE \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 $refenv \
          SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=${FP4BE:-auto} H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0" \
    EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path $ck \
           ${ATTN:+--attention-backend $ATTN --component-attention-backends text_encoder=torch_sdpa}" \
    LOG=$COUT/serve_nvfp4_${variant}_g$gpus.log ./serve.sh start \
    > "$OUT/start_nvfp4_${variant}_g$gpus.log" 2>&1 \
    || { echo "SERVER_FAILED $variant gpus=$gpus (看 $OUT/serve_nvfp4_${variant}_g$gpus.log)"; return 1; }

  local c se st tag t0 t1 rc inf
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    tag="${variant}_${se}_${st}${tagmid}_nvfp4${ATTN:+_$ATTN}"
    # 1 卡和 2 卡是这张表的两行，成片名不能撞
    [ "$gpus" != 2 ] && tag="${tag}_g$gpus"
    t0=$(date +%s)
    python3 h3gen.py --task "$variant" --image "$IMG" --inline \
      --short-edge "$se" --aspect 16:9 --duration 5.0 --steps "$st" \
      --seed "$seed" --flow-shift 12.0 --audio-flow-shift 3.0 \
      --prompt "$prompt" --port "$port" --out "$tag" > "${tag}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    # inference_time_s 是服务端报的纯推理时间（wall 还含请求往返和落盘）。它在 status.json 里的
    # 嵌套位置随服务端版本变过，所以按 key 递归找，不写死路径。
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

for variant in ${ONLY:-fl2va ref2va}; do
  for g in ${GPUSET:-1 2}; do
    run_arm "$variant" "$g"
  done
done
(unset VARIANT; ./serve.sh stop) >/dev/null 2>&1

# 画质：1 卡当参考、2 卡当候选。地板不是 SSIM 1.0——all-to-all 改归约序，实测地板 0.944 量级，
# 判据是"在地板之上且码率不涨"，不是"接近 1.0"。ref2va 的 SSIM 不能跨 prompt 当阈值。
for variant in ${ONLY:-fl2va ref2va}; do
  mid=""; [ "$variant" = ref2va ] && mid="_r${REF_SHORT_EDGE:-1024}"
  REF="${variant}_768_30${mid}_nvfp4${ATTN:+_$ATTN}_g1"; CAND="${variant}_768_30${mid}_nvfp4${ATTN:+_$ATTN}"
  [ -f "$REF.mp4" ] && [ -f "$CAND.mp4" ] && RUNDIR="$PWD" NAME="$NAME" ./quality_pair.sh "$REF" "$CAND"
done

echo "NVFP4_TABLE_DONE $(date -u +%FT%TZ)"
