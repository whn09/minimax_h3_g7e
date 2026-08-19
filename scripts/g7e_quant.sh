#!/usr/bin/env bash
# 从 stock BF16 权重把 **两个** partition 都量化成 sglang 能直接加载的 NVFP4：
#
#   ./g7e_quant.sh                 # FL2VA + Ref2VA -> /out/nvfp4_{fl2va,ref2va}.safetensors
#   PARTS=Ref2VA ./g7e_quant.sh    # 只补一个
#
# 为什么有这个脚本：g7e 上 Ref2VA 长期用的是第三方文件（lilcheaty 的
# `MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors` + `nvfp4_canonicalize.py` 掰布局 ->
# `nvfp4_ref2va_fixed.safetensors`）。那只是当时最快的路：FL2VA 在 Abiray/lilcheaty 那边只有
# `_pruned_nvfp4`（裁过的模型），我们才写了 `nvfp4_quantize_transformer.py`。而那个量化器与
# partition 无关（只看 SRC 目录），B300 上 `runquant.sh` 就是 `for v in FL2VA Ref2VA` 两个都自己
# 量的，那半张 ref2va NVFP4 表（16 格，2.28–2.33× vs BF16）全部出自自量化文件。
# 所以 g7e 这边没有理由继续依赖第三方文件：同一套配方、少一道 canonicalize、无第三方来源问题。
#
# 纯 CPU，一个 partition 约 10 分钟，峰值 host RAM ≈ 单文件大小。**别和计时请求并跑**
# （吃满核会把非 DiT 那几段推上去），趁编 sage 那种窗口一起做。
#
# 期望自检行（两个 partition 都是这个量级）：
#   wrote /out/nvfp4_<v>.safetensors: 951 tensors (q=208 fp8=50 copy=277) worst rel=0.09xx
# worst rel 落在 0.094±0.002 = group-16 round-to-nearest-e2m1 的地板；明显更大 = 打包/标度算错。
#
# 换到自量化文件后 serve 的 `--transformer-weights-path` 从
# `/out/nvfp4_ref2va_fixed.safetensors` 改成 `/out/nvfp4_ref2va.safetensors`，两个 python 补丁
# （TMA 标度布局 + qkv 行序）照旧都要打 —— 它们修的是 sglang 的加载/GEMM 路径，与文件来源无关。
set -u
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
NAME=${NAME:-h3}
PARTS=${PARTS:-"FL2VA Ref2VA"}
LOGDIR=${LOGDIR:-/opt/dlami/nvme/out}

docker cp nvfp4_quantize_transformer.py "$NAME:/tmp/" >/dev/null || exit 1
for v in $PARTS; do
  lv=$(echo "$v" | tr 'A-Z' 'a-z')
  echo "== quantize $v $(date -u +%FT%TZ)"
  docker exec -e SRC=/models/MiniMax-H3/$v/transformer -e DST=/out/nvfp4_${lv}.safetensors \
    "$NAME" python3 /tmp/nvfp4_quantize_transformer.py > "$LOGDIR/quant_$lv.log" 2>&1
  rc=$?
  echo "QUANT_$v rc=$rc $(tail -1 "$LOGDIR/quant_$lv.log")"
  [ $rc = 0 ] || { echo "看 $LOGDIR/quant_$lv.log" >&2; exit 1; }
done
docker exec "$NAME" bash -lc 'ls -l /out/nvfp4_*.safetensors'
