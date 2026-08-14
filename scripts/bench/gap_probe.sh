#!/usr/bin/env bash
# Attribute the SILENT ~12 s gap between "[MiniMaxH3DenoisingStage] finished" and
# "[MiniMaxH3DecodingStage] started...". That gap is ~6% of a 20-step g7e request, nothing logs
# inside it, and it is the largest lossless item left after denoise itself.
#
#   gap_probe.sh [steps]        default 20; fires one request and samples the worker's stack
#
# Method: fire the request, then blind-sample py-spy on a fixed schedule and correlate the
# timestamps with the server log afterwards. Do NOT try to trigger off a log tail -- tqdm writes the
# denoise progress bar with carriage returns and no newline, so the "finished" text does not reach a
# line-oriented reader until the NEXT newline, which is the "DecodingStage started" line, i.e. after
# the gap is already over.
#
# Run py-spy from the HOST, not inside the container: the container has no CAP_SYS_PTRACE, so
# `docker exec h3 py-spy dump` fails with "Failed to copy Py_Version symbol: Permission denied", and
# caps cannot be added to an already-running container. Copy the binary out once
# (docker cp h3:/usr/local/bin/py-spy /usr/local/bin/) and sudo it against the host-namespace pid
# from nvidia-smi -- that pid is the `sgl_diffusion::scheduler` process, which is where the pipeline
# stages actually run.
set -u
STEPS=${1:-20}
OUT=/opt/dlami/nvme/out/gap_probe.txt
cd "$(dirname "$0")/.."
: > "$OUT"

pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1 | tr -d ' ')
echo "sampling host pid $pid, steps=$STEPS" >> "$OUT"

python3 h3gen.py --task ref2va --image assets/first.png --inline \
  --width 1344 --height 768 --steps "$STEPS" --duration 5.175 --port 30030 \
  --out gapprobe_${STEPS}st > /opt/dlami/nvme/out/gap_probe_client.log 2>&1 &

# Denoise is ~10.2 s/step here; start sampling a little before it can possibly end.
sleep $((STEPS * 10 - 25))
for i in $(seq 1 24); do
  echo "--- sample $i $(date -u +%H:%M:%S)" >> "$OUT"
  sudo py-spy dump --pid "$pid" >> "$OUT" 2>&1
  sleep 2
done
echo "=== done sampling $(date -u +%H:%M:%S)" >> "$OUT"
wait
