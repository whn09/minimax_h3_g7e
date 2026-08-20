#!/usr/bin/env bash
# 把 H3 写死的参考图短边常量（2048）改成读 env 的形式。**在容器里跑**（serve.sh 和 Dockerfile
# 都调它，所以这段逻辑只有一份）：
#
#   bash /patches/inplace_ref_short_edge.sh
#
# 为什么是 in-place sed 而不是 .patch：它只改一行常量，而 context patch 为这一行付的代价很高 ——
# `minimax-h3-ref-image-short-edge-env.patch` 里那个 `import os` hunk 在文件自己长出
# `import os` 的那一版（c7c03ec53b → 273d978bed）就失效了，而它关心的那行根本没动过。
#
# 打完之后默认行为不变（仍是发布的 2048），只有设了 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE
# 才生效。这是 g7e 上 ref2va 的 1.46× 杠杆。
set -eu
SGL=${SGL:-/sgl-workspace/sglang}
f=$SGL/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py

if grep -q "^MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048$" "$f"; then
  grep -q "^import os$" "$f" || sed -i "0,/^import math$/s//import math\nimport os/" "$f"
  sed -i "s|^MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048$|MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = int(os.environ.get(\"SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE\", 2048))|" "$f"
  echo "APPLIED       ref-image-short-edge env override (in-place)"
elif grep -q "SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE" "$f"; then
  echo "ALREADY       ref-image-short-edge env override"
else
  echo "DOES_NOT_APPLY ref-image-short-edge: constant not found in $f" >&2
  exit 1
fi
