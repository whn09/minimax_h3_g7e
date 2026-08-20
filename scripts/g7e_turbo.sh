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
G=${G:-1}
PHASES=${PHASES:-"tb0 tbo tbc tbc2"}

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
echo "===== G7E_TURBO_DONE $(date -u +%FT%TZ)"
