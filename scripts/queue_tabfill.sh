#!/usr/bin/env bash
# 把交付表里 Cache-DiT 那一档的空格补齐。fl2va 单卡已经量过（20/30 步全有），缺的是
# **ref2va 单卡 / fl2va 2卡 / ref2va 2卡**。
#
#   nohup ./queue_tabfill.sh > /opt/dlami/nvme/out/queue_tabfill.log 2>&1 &
#
# 三条规矩：
#  1. 每个配置的 base 和 cache 在**同一次调用里**跑完（同一进程、同一 session），倍数只在自己那组
#     内部算。跨重启漂 5.4%，跨 session 拿 base 当分母会把 2.2× 算成 2.1× 或 2.3×。
#  2. 每个配置一个 TAG，否则 dev_base_1_768_20.mp4 这种名字会被另一个口径静默覆盖。
#  3. ref2va 必须钉 `SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024` + seed 8201 + REF2VA_PROMPT，
#     跟交付表同口径；默认 2048 会把 480p/20 从 36 s 顶到 65 s。
#
# 每个配置 3 个 arm（base / RDT 0.16 / RDT 0.20）× 4 个几何。768p 推荐 0.16、480p 推荐 0.20，
# 但两个 RDT 都在两个分辨率上跑，顺手复核 30 步那轮发现的**平台效应**（768p 0.16≡0.20 逐格相同）。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
CASES4=${CASES4:-"480_20 480_30 768_20 768_30"}

fl2va_arms() {  # fl2va_arms <TAG> <GPUS>
  TAG=$1 ARMS="base_$2 cacheR16_$2 cacheR20_$2" CASES="$CASES4" ./g7e_dev_levers.sh
}
ref2va_arms() {  # ref2va_arms <TAG> <GPUS>
  IMG=${IMG:-$([ -f assets/input_cat.jpg ] && echo assets/input_cat.jpg || echo assets/first.png)}
  . assets/prompts.sh
  TAG=$1 ARMS="base_$2 cacheR16_$2 cacheR20_$2" CASES="$CASES4" \
    VARIANT=ref2va CKPT=/out/nvfp4_ref2va.safetensors PORT=30030 SEED=8201 \
    PROMPT="$REF2VA_PROMPT" ENVX_EXTRA=SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024 \
    ./g7e_dev_levers.sh
}

echo "===== QUEUE_TABFILL $(date -u +%FT%TZ)"
echo "##### r2c1 (ref2va 1卡)";  ref2va_arms r2c1 1
echo "##### f2c2 (fl2va 2卡)";   fl2va_arms  f2c2 2
echo "##### r2c2 (ref2va 2卡)";  ref2va_arms r2c2 2
echo "===== QUEUE_TABFILL_DONE $(date -u +%FT%TZ)"
