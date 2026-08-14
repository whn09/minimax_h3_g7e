#!/usr/bin/env bash
# Capture a torch-profiler trace of TWO real denoise steps on g7e, so the per-step cost can be split
# by CUDA kernel instead of guessed from FLOP arithmetic.
#
#   prof_step.sh [tag] [extra sglang server flags...]
#
# Why the server has to be restarted for this: the trace directory comes from
# SGLANG_DIFFUSION_TORCH_PROFILER_DIR, read at profiler construction from the SERVER's environment.
# The trigger itself is request-scoped (`profile` / `num_profiled_timesteps` are declared fields of
# the video request), so one server can then be profiled per request.
#
# num_profiled_timesteps=2 with the profiler's built-in warmup=1 means 3 steps are recorded; keep it
# small, a 50-layer step with record_shapes+with_stack is ~100 MB of trace per step.
#
# py-spy is the wrong tool for this question and was already tried: the denoise loop contains no
# synchronize/.item()/.cpu() at all, so the CPU runs a full step ahead of the GPU and its stack shows
# which launch is *blocked*, i.e. it is weighted by kernel COUNT, not kernel TIME.
#
# TASK is a knob because the two tasks do NOT share a sequence length, and the whole point of the
# trace is to read the real one instead of estimating it. ref2va appends the reference image as extra
# tokens (that is what REF_SHORT_EDGE scales), fl2va conditions through the latent and adds none. So
# a per-step cost model calibrated on a ref2va trace cannot be applied to an fl2va step time --
# doing exactly that is what produced a bogus "33% of the step is not matmul".
#   TASK=fl2va prof_step.sh fl2va768
set -u
cd "$(dirname "$0")/.."
TAG=${1:-prof}; shift || true
TASK=${TASK:-ref2va}
# fl2va's image is a keyframe, ref2va's is a reference; the ports differ because serve.sh runs one
# variant per port (30010 fl2va / 30030 ref2va).
case $TASK in
  fl2va) PORT=${PORT:-30010}; ROLE_ARGS="--task fl2va";;
  *)     PORT=${PORT:-30030}; ROLE_ARGS="--task ref2va";;
esac
OUT=/opt/dlami/nvme/out
COUT=/out            # same directory inside the container; LOG/profiler paths must be this form

sudo mkdir -p $OUT/prof && sudo chmod 777 $OUT/prof

VARIANT=$TASK ./serve.sh stop >/dev/null 2>&1
sleep 5
docker exec h3 pkill -f "sglang serve" 2>/dev/null
sleep 8

if ! VARIANT=$TASK GPUS=1 ULYSSES=1 \
    ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024 SGLANG_DIFFUSION_TORCH_PROFILER_DIR=$COUT/prof" \
    EXTRA="--layerwise-offload-components text_encoder $*" \
    LOG=$COUT/serve_${TAG}.log ./serve.sh start > $OUT/${TAG}_start.log 2>&1; then
  echo "PROF $TAG START_FAILED"; tail -30 $OUT/${TAG}_start.log; exit 1
fi

# 6 steps is enough: the profiler only records the first 1+2 of them, and the rest just finish the
# request. Same geometry as every other g7e arm so the numbers are comparable.
python3 h3gen.py $ROLE_ARGS --image assets/first.png --inline \
  --width ${W:-1344} --height ${H:-768} --steps 6 --duration 5.175 --port $PORT \
  --extra profile=true --extra num_profiled_timesteps=2 \
  --out ${TAG}_trace > $OUT/${TAG}_client.log 2>&1
echo "rc=$? traces:"; ls -la $OUT/prof/
