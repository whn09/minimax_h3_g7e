#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1
/opt/dlami/nvme/venv/bin/hf download Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot \
  --include "MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors" --include "MiniMax_H3_FL2VA_pruned_nvfp4.safetensors" \
  --local-dir /opt/dlami/nvme/nvfp4
echo "NVFP4_DONE $(date -u +%FT%TZ)"
