# Start the ref2va server on the canonicalized NVFP4 checkpoint (see nvfp4_canonicalize.py).
# --quantization is deliberately NOT passed: the auto-inferred config from the safetensors is the
# correct one, while an explicit --quantization modelopt_fp4 builds an empty, unusable config.
# FP4BE=auto is required on sm_120 -- the default trtllm fp4 GEMM does not exist there.
cd "$(dirname "$0")/.."
./serve.sh stop >/dev/null 2>&1
sleep 5
docker exec h3 pkill -f "sglang serve" 2>/dev/null
sleep 8
VARIANT=ref2va GPUS=1 ULYSSES=1 \
  ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024 SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=${FP4BE:-auto}" \
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path ${NVFP4:-/out/nvfp4_ref2va_fixed.safetensors}" \
  LOG=${LOG:-/out/serve_nvfp4_fixed.log} ./serve.sh start
