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
# 本地素材目录里有 input_cat.jpg 就用它（和历史表同口径），没有就退到公开样例图。跑错输入是静默的，
# 所以 prompts.sh 会把实际用的图打成 PROMPT_PAIR 一行。
IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
# **端口必须跟着 VARIANT 走**：serve.sh 的 per-variant 默认是 fl2va→30010 / ref2va→30030
# （serve.sh:120）。写死 30010 去打 ref2va 的 server 会全程 Connection refused，而 serve.sh start
# 自己是 ready 的、脚本不报 ARM_FAILED —— 表现为整轮 rc=1、inference_time_s 空，白烧一轮 GPU。
PORT=${PORT:-$([ "$VARIANT" = ref2va ] && echo 30030 || echo 30010)}
SEED=${SEED:-6201}
ARMS=${ARMS:-"base_1 cache_1 cachehq_1 adaln_1"}
# 追加给每个 arm 的 env（不覆盖 arm 自己那组）。**ref2va 必须用它把参考短边钉到 1024**，
# 否则默认 2048 → 参考图编码像素是交付表口径的 4 倍，480p/20 会从 36 s 变成 65 s，
# 看着像"优化变慢了"其实是换了口径：
#   VARIANT=ref2va CKPT=/out/nvfp4_ref2va.safetensors \
#     ENVX_EXTRA="SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" ./g7e_dev_levers.sh
ENVX_EXTRA=${ENVX_EXTRA:-}
# FP4_UPSTREAM=1：容器里的 sglang 已经自带了两个 NVFP4 修复（把它们改成上游形态、无 env 开关
# 之后的样子）。此时既不能再打 patches/ 里那两个 python 补丁（会冲突），也不能设
# SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND / H3_FP4_* —— 新代码里默认值和 TMA 布局都是
# 无条件的，留着 env 等于绕过被测代码。g7e 上同配置逐格确定性，所以这一路的 md5 必须和
# 打补丁那一路完全相同。
FP4_UPSTREAM=${FP4_UPSTREAM:-0}
# 输出文件名前缀。**换 VARIANT / 换卡数时一定要换 TAG**，否则 dev_base_1_768_20.mp4 会被
# 另一个口径的同名文件覆盖，而且是静默覆盖（画质那一段照样能跑，只是比错了东西）。
TAG=${TAG:-dev}
. assets/prompts.sh   # prompt 按 $IMG 配对，别在这里写字面量
# prompt 也跟着 VARIANT 走：ref2va 的那条是「Use <Picture 1> as the visual subject…」，
# 拿 fl2va 的 prompt 去跑 ref2va 是能跑的，但等于没告诉模型那张图是主体。
PROMPT=${PROMPT:-$([ "$VARIANT" = ref2va ] && echo "$REF2VA_PROMPT" || echo "$FL2VA_PROMPT")}

# 片长（秒）。H3 按 17n+5 帧对齐：5.0 → 124 帧、10.0 → 243 帧。**换片长必须换 TAG**
# （文件名里不带片长，否则静默覆盖）。attention 是无 mask 的 packed full self-attention，
# 序列长度随片长线性涨、attention 项随平方涨，所以每步成本对帧数是超线性的 —— 这个旋钮就是
# 用来量那条曲线的。
DUR=${DUR:-5.0}
req() {  # req <port> <short_edge> <steps> <out-tag>
  python3 h3gen.py --task "$VARIANT" --image "$IMG" --inline \
    --short-edge "$2" --aspect 16:9 --duration "$DUR" --steps "$3" --seed "$SEED" \
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
if [ "$FP4_UPSTREAM" = 1 ]; then
  echo "FP4_UPSTREAM=1：跳过两个 NVFP4 python 补丁，也不设 FP4 env（用容器里现有的代码）"
else
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  src=patches/$p; [ -f "$src" ] || src=$p
  [ -f "$src" ] || { echo "MISSING $p" >&2; exit 1; }
  docker cp "$src" "$NAME:/tmp/" >/dev/null && docker exec "$NAME" python3 "/tmp/$p" || exit 1
done
fi
docker exec "$NAME" bash -lc 'python3 -c "import sglang;print(\"sglang\", sglang.__version__)"' 2>/dev/null | tail -1

for arm in $ARMS; do
  G=${arm##*_}; KNOB=${arm%_*}
  ATTN=sage_attn   # 每个 arm 重置（sol_* 会把它换掉）
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path $CKPT"
  ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0"
  [ "$FP4_UPSTREAM" = 1 ] || ENVX="$ENVX SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto \
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
    # cacheW<N>R<MM> = warmup N + RDT 0.MM。**步数少的时候必须调 warmup**：默认 W=4 意味着头 4 步
    # 一定全算，turbo 的 8 步就只剩 4 步可跳（MC=3 还限制连跳），默认档等于半残。
    cacheW*R*) w=${KNOB#cacheW}; w=${w%%R*}; r=${KNOB##*R}
             ENVX="$ENVX SGLANG_CACHE_DIT_ENABLED=1 \
                   SGLANG_CACHE_DIT_WARMUP=$w SGLANG_CACHE_DIT_SECONDARY_WARMUP=$w \
                   SGLANG_CACHE_DIT_RDT=0.$r SGLANG_CACHE_DIT_SECONDARY_RDT=0.$r" ;;
    adaln)   EXTRA="$EXTRA --minimax-h3-adaln-online --minimax-h3-adaln-plan-width 3" ;;
    # solT<tau×10>D<dense_steps> = Sol-Attn（NVlabs/Sana @ sol-engine）。**它是全局 attention
    # 后端，所以会把 sage 顶掉**（sage 是 fp8 QK 的，sol 是 bf16）—— 比较时必须记住这一条：
    # 对 sage 基线的净收益 = sol 的稀疏收益 − 丢掉 fp8 的损失。
    # 装：pip install --no-deps git+https://github.com/NVlabs/Sana.git@sol-engine#subdirectory=techniques/sparse_backends
    # （**必须 --no-deps**：它声明 torch>=2.10，不加会把容器里的 torch 换掉。cutlass-dsl 镜像里已有。）
    # **dense_steps 默认 10，turbo 只有 8 步 ⇒ 默认配置下 sol 一次都不触发**（和 Cache-DiT 的
    # warmup=4 一个坑）。tau 是 z-score（threshold = mean + tau*std），不是 top-K 预算。
    # 还要先打 patches/patch_sol_attn_dense_sage.py，否则第一个 dense 步就炸（镜像里 flash_attn
    # 是空 namespace package）。
    # 结论（768p 单卡 turbo 8 步，对同片长 sage base；全表见 README「稀疏 attention」一节）：
    #   收益随片长上升 —— tau=1.0 1.246×(5s)→1.377×(15s)、tau=1.5 1.343×→1.567×，
    #   指数从 n^1.576 按到 n^1.483 / n^1.432（加一张卡是 n^1.44，同量级）。
    #   **但单独用被 Cache-DiT 严格支配**：15 s 上 1.377× / SSIM 0.808 vs cache 1.340× / 0.893，
    #   而且 sol 的画质随片长恶化（0.876@5s → 0.808@15s），cache 不会。
    #   **两个能叠**：15 s 768p 148.561 s = 对 sage 1.720×（= 两者单独之积的 93%），$/成片秒 −22%。
    solT*D*)   t=${KNOB#solT}; t=${t%%D*}; d=${KNOB##*D}
             ATTN=sol_attn
             EXTRA="$EXTRA --attention-backend-config tau=$((t/10)).$((t%10)),dense_steps=$d,dense_layers=0-1" ;;
    *) echo "SKIP unknown knob '$KNOB'"; continue ;;
  esac
  [ -n "$ENVX_EXTRA" ] && ENVX="$ENVX $ENVX_EXTRA"
  # c0b6474 上要豁免的组件比 273d978be **多**：新镜像里 audio_vae 也去 selector 解 attention
  # 后端，而它只声明 ['fa','torch_sdpa']，于是全局 `--attention-backend sage_attn` 会让它
  # `Failed to load customized audio_vae`（`selector.py:300` ValueError → scheduler is dead）。
  # 三个非 DiT 组件一律 torch_sdpa（名字取自 model_index.json，**逗号分隔不是空格**）。
  EXTRA="$EXTRA --attention-backend $ATTN \
    --component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa"

  echo "=== ARM $arm gpus=$G dur=$DUR $(date -u +%H:%M:%S)"
  (unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1
  sleep 10
  VARIANT=$VARIANT GPUS=$G ULYSSES=$G IMAGE=$IMAGE NAME=$NAME PATCHES="$PATCHES" \
    ENVX="$ENVX" EXTRA="$EXTRA" LOG=$COUT/serve_${TAG}_$arm.log ./serve.sh start \
    > "$OUT/start_${TAG}_$arm.log" 2>&1 \
    || { echo "ARM_FAILED $arm （看 $OUT/serve_${TAG}_$arm.log）"
         docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_${TAG}_$arm.log | grep -iE 'error|raise|Traceback' | tail -5" 2>/dev/null
         continue; }
  for w in 480 768; do req "$PORT" "$w" 4 "warm_${TAG}_${arm}_$w" >/dev/null 2>&1; done
  echo "== $arm warm $(date -u +%H:%M:%S)"
  for c in $CASES; do
    se=${c%_*}; st=${c#*_}
    name="${TAG}_${arm}_${se}_${st}"
    t0=$(date +%s)
    req "$PORT" "$se" "$st" "$name" > "${name}_client.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "DEVLEVER $arm ${se}_${st} rc=$rc inference_time_s=$(infer_s "$name") wall_s=$((t1-t0)) mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/ -)"
  done
  # 回读真正落地的东西：attention 后端（sm_120 上 fa 会被静默降级）+ Cache-DiT 到底挂上没有
  # （"Acceleration hooks is disabled for: BlockAdapter" = 没挂上，别把噪声当收益）。
  docker exec "$NAME" bash -lc "tr '\r' '\n' < $COUT/serve_${TAG}_$arm.log | grep -oiE 'Enabling cache-dit[^\"]{0,120}|Acceleration hooks is disabled[^\"]{0,40}|SCM enabled[^\"]{0,80}|sage[a-z0-9_]*|\"attention_backend\": [^,]*' | sort -u | head -10" 2>/dev/null
done
(unset VARIANT; NAME=$NAME ./serve.sh stop) >/dev/null 2>&1

# 画质：**同卡数**的 base 当参考（同镜像同卡数，唯一变量是那个旋钮）。同 seed 同卡数重跑逐位相同，
# 所以无损旋钮应当 SSIM 1.000000；cache_* 是有损的，这里量的就是损多少。
# 跨卡数比没意义（Ulysses 换了 reduce 顺序，base_2 vs base_1 本身就不逐位相同）。
for c in $CASES; do
  se=${c%_*}; st=${c#*_}
  for arm in $ARMS; do
    G=${arm##*_}
    [ "$arm" != "base_$G" ] || continue
    [ -f "${TAG}_base_${G}_${se}_${st}.mp4" ] && [ -f "${TAG}_${arm}_${se}_${st}.mp4" ] && \
      RUNDIR="$PWD" NAME="$NAME" ./quality_pair.sh "${TAG}_base_${G}_${se}_${st}" "${TAG}_${arm}_${se}_${st}"
  done
done
echo "G7E_DEV_LEVERS_DONE $(date -u +%FT%TZ)"
