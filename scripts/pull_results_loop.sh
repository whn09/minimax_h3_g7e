#!/usr/bin/env bash
# 在**笔记本**上跑：每 60 秒把机器上的成片和日志拉回本地，直到 spot 被回收或手动停。
#
#   ./pull_results_loop.sh <本地目录> [ssh别名] [间隔秒]
#   ./pull_results_loop.sh ~/…/h3_g7e_baseline/runs/cache_dit_cat G7E 60
#   停：kill $(cat <本地目录>/.pull.pid)
#
# 为什么需要它：g7e 的盘是 **instance store**，spot 一回收全没。已经栽两次了 ——
#   * 2026-08-19 17:01Z：dev 消融的成片没下，全丢。
#   * 2026-08-19 23:40Z：input_cat 那轮 RDT 跑完 1 分钟后被回收，只剩日志里的时间数，画质列全丢。
# 两次都是"等整队列跑完再下"。跑完再下就是赌，所以改成边跑边拉。
#
# 只拉小文件（mp4/json/log），不碰权重和 NEFF。没写完的 mp4 下一轮 size/mtime 变了会重传。
#
# **别加 `--append-verify`**：现在的 macOS 装的是 openrsync（`rsync --version` 报
# "protocol version 29 / rsync version 2.6.9 compatible"），不认这个 flag，直接吐 usage 就退出。
# 因为下面两条 rsync 都 `2>/dev/null`（远端被回收时不想刷屏），这个错是**完全静默**的：
# 循环照样每 60 秒打印 mp4=0，本地一个文件都没落。2026-08-20 就这么白等了 20 分钟。
# 改动这两行时先 `rsync -az --dry-run ... ` 手工验一次。
set -u
DEST=${1:?用法: pull_results_loop.sh <本地目录> [ssh别名] [间隔秒]}
HOST=${2:-G7E}
EVERY=${3:-60}
REMOTE_RUN=${REMOTE_RUN:-/opt/dlami/nvme/h3run/scripts}
REMOTE_OUT=${REMOTE_OUT:-/opt/dlami/nvme/out}
mkdir -p "$DEST"
echo $$ > "$DEST/.pull.pid"
echo "== pull loop: $HOST -> $DEST 每 ${EVERY}s（pid $$，停：kill \$(cat $DEST/.pull.pid)）"
while :; do
  rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" \
    --include='*/' --include='*.mp4' --include='*.json' --include='*.log' --exclude='*' \
    "$HOST:$REMOTE_RUN/" "$DEST/" 2>/dev/null
  rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -o BatchMode=yes" \
    --include='*.log' --exclude='*' "$HOST:$REMOTE_OUT/" "$DEST/logs/" 2>/dev/null
  n=$(ls "$DEST"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
  printf '\r[%s] mp4=%s  ' "$(date -u +%H:%M:%S)" "$n"
  sleep "$EVERY"
done
