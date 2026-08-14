#!/usr/bin/env bash
# make_patch.sh   -- regenerate minimax-h3-target-width-height.patch inside the h3 container.
#
# The width/height patch has to be diffed on top of minimax-h3-short-edge.patch, because both edit
# `_validate_target` in request_validation.py. Diffing straight against the image's HEAD would fold
# the short-edge change into this patch, and applying both in sequence then conflicts.
#
# So: reverse the width/height edit, make a temporary commit that captures the short-edge (and
# cpu-offload) state as the baseline, re-apply forward, diff against it, then drop the temp commit
# with `reset --soft` so the working tree and HEAD both look exactly as they did before.
set -euo pipefail
C=${CONTAINER:-h3}
REPO=/sgl-workspace/sglang
OUT=${OUT:-/out/minimax-h3-target-width-height.patch}
MG=python/sglang/multimodal_gen
# Keep this on one line: it is interpolated into the docker exec command string below, so a newline
# here would end the `git diff` line and run the second path as its own command.
FILES="$MG/configs/sample/minimax_h3.py $MG/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py"

docker exec "$C" bash -lc "
set -euo pipefail
cd $REPO
# Tolerate a leftover temp commit from an interrupted run: the real baseline is the image commit,
# which is the first ancestor whose subject is not ours.
while [ \"\$(git log -1 --format=%s)\" = 'temp: short-edge baseline' ]; do git reset -q --mixed HEAD~1; done
BASE=\$(git rev-parse HEAD)
python3 /patches/make_width_height_patch.py --reverse
git -c user.email=x@x -c user.name=x commit -q -am 'temp: short-edge baseline'
python3 /patches/make_width_height_patch.py
git diff -- $FILES > $OUT
# --mixed, not --soft: it must also reset the index back to the image commit, otherwise the repo is
# left with the short-edge state staged and a later plain \`git diff\` reports the wrong baseline.
git reset -q --mixed \$BASE
echo '== HEAD back to' \$(git rev-parse --short HEAD)
grep -c '^+' $OUT | sed 's/^/added lines: /'
"
echo "== the regenerated patch applies on top of the short-edge patch, not instead of it"
