#!/usr/bin/env bash
# Bring a fresh g7e DLAMI box to the point where scripts/serve.sh can start a server.
# Run it detached (`setsid nohup ./g7e_bringup.sh > ~/bringup.log 2>&1 < /dev/null &`): the weights
# are 269 GB and the image is another 48 GB, so it runs well past any interactive SSH timeout.
#
# BOTH partitions are downloaded by default, because the deliverable covers all three tasks and the
# task -> partition map is a hard gate: Ref2VA/ serves ref2va, FL2VA/ serves t2va AND fl2va, and one
# server process can only load one of them. Ref2VA/ goes first so a ref2va server can be started
# while FL2VA/ is still coming down. PARTS="Ref2VA" trims it to one partition (135 GB instead of
# 269 GB) when only that half is needed.
#
# The trailing `--include "*.json"` pass is NOT optional: it fetches the root model_index.json,
# without which the pipeline class cannot be resolved.
#
# python3.12-venv MUST be installed before `python3 -m venv`. Without it venv creation fails on
# ensurepip but still leaves the directory behind, so pip/hf are simply missing and the download
# "succeeds" instantly with an empty target dir.
set -euo pipefail

VENV=/opt/dlami/nvme/venv
DEST=/opt/dlami/nvme/h3
IMAGE=${IMAGE:-lmsysorg/sglang:dev}
PARTS=${PARTS:-"Ref2VA FL2VA"}

sudo mkdir -p /opt/dlami/nvme/out
sudo chown -R "$(id -u):$(id -g)" /opt/dlami/nvme
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12-venv ffmpeg

# **必须关掉 unattended-upgrades**，否则 DLAMI 会在开机 20 分钟左右自己升包 + 重启，把整队列吃掉。
# 2026-08-20 00:03Z 实测：升级 docker 时 docker.sock 权限翻掉 → 正在跑的 arm 报
# "permission denied while trying to connect to the docker API"（服务日志是空的，只有 start_*.log
# 里有那一行），紧接着 systemd-logind 广播 "The system will reboot at ..."，1 分钟后真重启。
# spot 请求还是 fulfilled、实例还是 running，**看起来像被回收但不是**：instance store 全都还在，
# 只是容器 Exited(137) 且里面装的 sageattention 随容器没了（docker rm -f h3n 后重建 + 重编）。
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
sudo systemctl mask unattended-upgrades 2>/dev/null || true
printf 'APT::Periodic::Unattended-Upgrade "0";\nAPT::Periodic::Update-Package-Lists "0";\n' \
  | sudo tee /etc/apt/apt.conf.d/99-no-auto-upgrade >/dev/null
sudo rm -f /var/run/reboot-required /var/run/reboot-required.pkgs

[ -x $VENV/bin/hf ] || { rm -rf $VENV; python3 -m venv $VENV; $VENV/bin/pip -q install -U pip "huggingface_hub[hf_transfer,cli]"; }

# Pull the image in parallel with the download -- they contend for nothing but the NIC.
#
# `:dev` IS A MOVING TAG. Re-running this script on a box that already works will silently fetch a
# newer sglang and re-point the tag, leaving the validated image dangling (one `docker image prune`
# from gone). Measured on 2026-08-14: sglang 273d978bed -> c4271c3fe1. Running containers are
# unaffected (they keep their own filesystem), but a NEW container would come up on the new commit,
# where the four git patches are unverified and sageattention would have to be rebuilt. Keep a
# durable tag on the image you validated. This script creates that tag for you (below, after the
# pull) and every driver script defaults to it, so a later re-run of THIS script can pull a newer
# `:dev` without moving `:h3-validated` -- the tag is only created when it does not already exist.
docker pull "$IMAGE" > /tmp/pull.log 2>&1 &
PULL=$!

# hf_transfer is gone; huggingface_hub now warns that HF_HUB_ENABLE_HF_TRANSFER does nothing and
# points at the Xet path instead. Both are exported so this works on older hub versions too.
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
for part in $PARTS; do
  $VENV/bin/hf download MiniMaxAI/MiniMax-H3 --include "$part/*" --local-dir $DEST
  echo "${part}_DONE $(date -u +%FT%TZ) size=$(du -sh $DEST | cut -f1)"
done
$VENV/bin/hf download MiniMaxAI/MiniMax-H3 --include "*.json" --local-dir $DEST
echo "JSON_DONE $(date -u +%FT%TZ) size=$(du -sh $DEST | cut -f1)"

wait $PULL && echo "IMAGE_DONE $(docker images --format '{{.ID}}' "$IMAGE")"
# Pin what we just pulled under a tag that never moves. `|| true` on the inspect: only tag when the
# durable name is absent, so re-running bringup on a working box cannot clobber a validated image.
if ! docker image inspect lmsysorg/sglang:h3-validated >/dev/null 2>&1; then
  docker tag "$IMAGE" lmsysorg/sglang:h3-validated
  echo "TAGGED lmsysorg/sglang:h3-validated -> $(docker images --format '{{.ID}}' "$IMAGE")"
else
  echo "KEPT existing lmsysorg/sglang:h3-validated ($(docker images --format '{{.ID}}' lmsysorg/sglang:h3-validated))"
fi
echo "BRINGUP_DONE $(date -u +%FT%TZ)"
