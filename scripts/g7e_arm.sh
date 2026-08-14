#!/bin/bash
# One optimization arm on the g7e box (RTX PRO 6000, sm_120), paired against the BF16
# baseline: ref2va, image reference at short edge 1024, 1344x768, 5.175 s, steps from $STEPS.
#
#   g7e_arm.sh <tag> [extra sglang flags...]           e.g.  g7e_arm.sh cudnn --attention-backend torch_cudnn_sdpa
#   STEPS="20 30" g7e_arm.sh <tag> ...                 sweep (default "20 30")
#   ENVX="A=1 B=2" g7e_arm.sh <tag> ...                extra env for the server process
#   GPUS=2 g7e_arm.sh ulysses2 ...                     Ulysses degree follows GPUS unless ULYSSES is set
#   WORKDIR=~/h3run/scripts g7e_arm.sh ...             where serve.sh / h3gen.py / assets live
#
# GPUS/ULYSSES/WORKDIR are overridable so the 1-, 2- and 4-GPU arms can be measured on ONE box
# under ONE image. Cross-box pairing is worthless here: the `:dev` tag is mutable (it moved
# c7c03ec53b -> 273d978bed mid-project) and cross-restart drift alone is 5.4%.
#
# Pairing baselines on this box (bf16 / torch_sdpa / ref short edge 1024):
#   20 steps 224.05 s, 30 steps 337.61 s, peak 88108 MiB
#
# Two g7e-specific requirements are baked in and must not be dropped:
#   SGLANG_USE_RUNAI_MODEL_STREAMER=0        the streamer's anonymous memory blows up the host
#   --layerwise-offload-components text_encoder   the only offload shape that survives ref2va
# The reference short edge is pinned to 1024 (1.95x/step cheaper than the released 2048 default,
# quality validated) so every arm is compared against the recommended serving configuration.
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
TAG=$1; shift
EXTRA_IN="$*"
OUT=/opt/dlami/nvme/out   # host view
COUT=/out                 # the SAME directory as seen inside the container

echo "=== ARM $TAG   EXTRA='$EXTRA_IN'   ENVX='${ENVX:-}'   $(date -u +%H:%M:%S)"
VARIANT=ref2va ./serve.sh stop >/dev/null 2>&1
sleep 5
docker exec h3 pkill -f "sglang serve" 2>/dev/null
sleep 8

NG=${GPUS:-1}
if ! VARIANT=ref2va GPUS=$NG ULYSSES=${ULYSSES:-$NG} \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024 ${ENVX:-}" \
    EXTRA="--layerwise-offload-components text_encoder $EXTRA_IN" \
    LOG=$COUT/serve_${TAG}.log ./serve.sh start > $OUT/arm_${TAG}_start.log 2>&1; then
  # ^ LOG is redirected INSIDE the container; a host path like /opt/dlami/nvme/out/... does not
  # exist there, the redirect fails, and the server dies before it prints anything.
  echo "ARM $TAG START_FAILED"; tail -30 $OUT/arm_${TAG}_start.log; exit 1
fi

for st in ${STEPS:-20 30}; do
  out=${TAG}_768p_${st}st
  python3 h3gen.py --task ref2va --image assets/first.png --inline \
      --width 1344 --height 768 --steps "$st" --duration 5.175 --port 30030 \
      --out "$out" > ${out}_client.log 2>&1
  rc=$?
  t=$(python3 - <<PY
import json
def dig(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "inference_time_s":
                return v
            r = dig(v)
            if r is not None:
                return r
    return None
try:
    print(dig(json.load(open("${out}_status.json"))))
except Exception:
    print("NA")
PY
)
  echo "ARM $TAG steps=$st rc=$rc inference_time_s=$t peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done
grep -m3 -i "attention backend\|Attention backends for" $OUT/serve_${TAG}.log || true
