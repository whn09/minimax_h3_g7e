#!/usr/bin/env bash
# Turbo LoRA（`larryvrh/MiniMax-H3-Turbo-Lora`，v4_step600_ema）在 g7e 上能不能和 NVFP4 / sage /
# Cache-DiT 叠起来。
#
#   nohup ./g7e_turbo.sh > /opt/dlami/nvme/out/g7e_turbo.log 2>&1 &
#
# 前置（各做一次，都在 CPU 上，别和计时请求并跑）：
#   curl -sL -o /opt/dlami/nvme/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors
#   docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer \
#     -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     -e DST=/out/turbo_v4_600_bf16 h3n python3 /tmp/lora_merge_transformer.py   # 259/259 模块
#   docker exec -e SRC=/out/turbo_v4_600_bf16 -e DST=/out/nvfp4_fl2va_turbo.safetensors \
#     h3n python3 /tmp/nvfp4_quantize_transformer.py                            # worst rel 0.094x
#
# 为什么是"离线合并再量化"而不是 `--lora-path`：见 lora_merge_transformer.py 的注释
# （运行时合并式 LoRA 在量化权重上形状对不上，dynamic 模式画质跌破地板）。
# 对 serve 来说合并后的 NVFP4 文件就是一份普通 NVFP4 checkpoint，两个 python 补丁照旧要打，
# sage / Cache-DiT 也就自然能叠 —— 这条链上唯一的新东西是权重本身。
#
# 四个 phase（每个一个 TAG，别共用：同名 mp4 会被静默覆盖）。`PHASES="tbc2" ./g7e_turbo.sh`
# 可以只补一个：
#   tb0   stock NVFP4，20 步  —— 画质与倍数的共同参考（交付口径）
#   tbo   turbo NVFP4，4/6/8 步 —— 模型卡说 4-8 是有用区间、8 步最好，这里把曲线量出来
#   tbc   turbo NVFP4，8 步 + Cache-DiT 的**低 RDT 档** —— 结论：0.16 一次都不触发
#         （逐位相同、SSIM 1.0），warmup 收到 2 也不救。步数少 → 每步 residual 变化大 →
#         按 stock 20/30 步定的那两个推荐档（480p 0.20 / 768p 0.16）在 turbo 上整个偏低。
#   tbc2  接着往上扫 RDT，找 turbo 8 步自己的那个膝点 —— 结论：**膝点是 0.24**（正好是 sglang
#         默认值）。480p 11.882 s / 768p 36.400 s，对 stock 20 步 2.62× / 3.14×，而 SSIM
#         （参考 stock 20 步）比 turbo 自己的 8 步还高一丝（0.9034 / 0.8848 vs 0.9027 / 0.8839）：
#         cache 往前面的轨迹拉，把 turbo 冲过头的运动收回到 stock 附近。0.28 与 0.24 在 480p
#         输出逐位相同（平台效应）；0.32 再快 17% 但 768p SSIM 掉到 0.8606，不推荐。
#         全表在 h3_g7e_baseline/runs/turbo_cat/RESULTS_turbo.md。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
TURBO=${TURBO:-/out/nvfp4_fl2va_turbo.safetensors}
STOCK=${STOCK:-/out/nvfp4_fl2va.safetensors}
TURBO_REF=${TURBO_REF:-/out/nvfp4_ref2va_turbo.safetensors}
STOCK_REF=${STOCK_REF:-/out/nvfp4_ref2va.safetensors}
G=${G:-1}
PHASES=${PHASES:-"tb0 tbo tbc tbc2"}
# ref2va 必须钉参考短边，否则默认 2048 换了口径（见 g7e_dev_levers.sh 头部）
REFENV=${REFENV:-SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024}

has() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

echo "===== G7E_TURBO phases='$PHASES' $(date -u +%FT%TZ)"
has tb0 && { echo "##### tb0 (stock NVFP4, 20 步, 参考)"
  TAG=tb0 ARMS="base_$G" CASES="480_20 768_20" CKPT="$STOCK" ./g7e_dev_levers.sh; }

has tbo && { echo "##### tbo (turbo NVFP4, 步数曲线)"
  TAG=tbo ARMS="base_$G" CASES="480_4 480_6 480_8 768_4 768_6 768_8" CKPT="$TURBO" \
    ./g7e_dev_levers.sh; }

has tbc && { echo "##### tbc (turbo NVFP4 + Cache-DiT 低 RDT, 8 步)"
  TAG=tbc ARMS="base_$G cacheW2R16_$G cacheW2R20_$G cacheR16_$G" CASES="480_8 768_8" \
    CKPT="$TURBO" ./g7e_dev_levers.sh; }

has tbc2 && { echo "##### tbc2 (turbo NVFP4 + Cache-DiT 高 RDT, 8 步)"
  TAG=tbc2 ARMS="base_$G cacheW2R24_$G cacheW2R28_$G cacheW2R32_$G" CASES="480_8 768_8" \
    CKPT="$TURBO" ./g7e_dev_levers.sh; }

# ---- 2 卡（fl2va，Ulysses=2）。turbo 把 DiT 压得更薄，加卡收益会不会被稀释，只能量。
# 结论：**没被稀释、还微升**。加卡倍数沿 stock→turbo→turbo+cache 单调上升
# （480p 1.24→1.30→1.32、768p 1.48→1.50→1.52），因为 cache 跳整块时把那一块的 all-to-all
# 一起跳了，每步算力:通信比不变。对比 sage：它只压算力，768p 加卡从 1.62× 掉到 1.50×。
has tb20 && { echo "##### tb20 (stock NVFP4, 2 卡, 20 步, 参考)"
  TAG=tb20 ARMS="base_2" CASES="480_20 768_20" CKPT="$STOCK" ./g7e_dev_levers.sh; }
has tb2 && { echo "##### tb2 (turbo NVFP4, 2 卡, 8 步 ± cache)"
  TAG=tb2 ARMS="base_2 cacheW2R24_2" CASES="480_8 768_8" CKPT="$TURBO" ./g7e_dev_levers.sh; }

# ---- ref2va。**这个 LoRA 库里没有 ref2va 专用权重**（整库就一份 t2v 命名的 LoRA），
# 但同一份 LoRA 合进 Ref2VA 分区是可用的（g7e 实测 merged 259/259、量化 worst rel 0.0950，
# 8 步+R24 对同卡数 stock 20 步 2.70×（480p 1 卡）/ 3.15×（768p 1 卡）/ 3.22×（768p 2 卡），
# 倍数比 fl2va 还高一点，因为参考图编码那份固定开销不随步数缩）。
# **ref2va 的 SSIM 不能当画质判据**（只给主体图不给首帧，构图本身允许不同，motion 0.78–1.36
# 是 fl2va 的 2–3 倍），只能目视；实测三行同构图同光照、猫的花纹与毛发一致、无 artifact。
# 前置（各一次，CPU）：
#   docker exec -e SRC=/models/MiniMax-H3/Ref2VA/transformer \
#     -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
#     -e DST=/out/turbo_ref2va_bf16 h3n python3 /tmp/lora_merge_transformer.py
#   docker exec -e SRC=/out/turbo_ref2va_bf16 -e DST=/out/nvfp4_ref2va_turbo.safetensors \
#     h3n python3 /tmp/nvfp4_quantize_transformer.py
has tbr0 && { echo "##### tbr0 (stock NVFP4 ref2va, 1/2 卡, 20 步, 参考)"
  TAG=tbr0 ARMS="base_1 base_2" CASES="480_20 768_20" CKPT="$STOCK_REF" \
    VARIANT=ref2va ENVX_EXTRA="$REFENV" ./g7e_dev_levers.sh; }
has tbr && { echo "##### tbr (turbo NVFP4 ref2va, 1/2 卡, 8 步 ± cache)"
  TAG=tbr ARMS="base_1 cacheW2R24_1 base_2 cacheW2R24_2" CASES="480_8 768_8" CKPT="$TURBO_REF" \
    VARIANT=ref2va ENVX_EXTRA="$REFENV" ./g7e_dev_levers.sh; }

# ---- 片长曲线（"10 秒为什么不是 5 秒的 2 倍"）。同一轮里量 5 s 和 10 s，1 卡和 2 卡，
# 用 turbo 8 步当载体（最便宜、且步数不是变量）。attention 是无 mask 的 packed full
# self-attention：序列随片长线性涨、attention 项随平方涨，其余（QKV/O 投影 + SwiGLU FFN）是线性的
# ⇒ 每步成本介于 2× 和 4× 之间。加卡切的正是序列轴，所以长片上加卡效率应当更高 —— 一起量。
# 结论：**超线性但远不到平方**。相对 5 s 按帧数比（1 : 1.960 : 2.919）取指数：480p 单卡
# n^1.27 / n^1.34，768p 单卡 n^1.53 / n^1.57（纯 O(n²) 时 15 s 该是 5 s 的 8.52×，实测 4.20× / 5.40×）。
# **加卡越长越划算**：480p 1.30→1.45→1.54、768p 1.50→1.68→1.73（效率 87%），2 卡把 768p 指数
# 从 n^1.57 按到 n^1.44。Cache-DiT 收益也随片长上升（768p 1.30×→1.33×→1.34×）。
# 15 s/768p 单卡不 OOM（52.6 GB base / 55.5 GB +cache），2 卡峰值 64.3–64.7 GB。
has tbd && { echo "##### tbd (片长 10 s / 15 s × 1/2 卡 × base/cache, turbo 8 步)"
  # 5 s 那一列不重跑：1 卡在 tbo/tbc2、2 卡在 tb2，同一台机同一 checkpoint，逐位可复现。
  # 每档都跑 base 和 cache R24 两条 —— 交付主力就是 LoRA+NVFP4+sage+Cache-DiT，
  # 曲线必须按交付配置量，不能只量 base。
  for d in 10.0 15.0; do
    TAG=tbd${d%.*} DUR=$d ARMS="base_1 cacheW2R24_1 base_2 cacheW2R24_2" CASES="480_8 768_8" \
      CKPT="$TURBO" ./g7e_dev_levers.sh
  done; }
echo "===== G7E_TURBO_DONE $(date -u +%FT%TZ)"
