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

[ -x $VENV/bin/hf ] || { rm -rf $VENV; python3 -m venv $VENV; $VENV/bin/pip -q install -U pip "huggingface_hub[hf_transfer,cli]"; }

# Pull the image in parallel with the download -- they contend for nothing but the NIC.
#
# `:dev` IS A MOVING TAG. Re-running this script on a box that already works will silently fetch a
# newer sglang and re-point the tag, leaving the validated image dangling (one `docker image prune`
# from gone). Measured on 2026-08-14: sglang 273d978bed -> c4271c3fe1. Running containers are
# unaffected (they keep their own filesystem), but a NEW container would come up on the new commit,
# where the four git patches are unverified and sageattention would have to be rebuilt. Keep a
# durable tag on the image you validated -- `docker tag <id> lmsysorg/sglang:h3-validated` -- and
# pass IMAGE=lmsysorg/sglang:h3-validated to serve.sh.
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
echo "BRINGUP_DONE $(date -u +%FT%TZ)"
