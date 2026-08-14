#!/usr/bin/env bash
# Paired quality check for one optimization arm: SSIM against the BF16 reference clip PLUS the
# inter-frame motion energy of both.
#
#   quality_pair.sh <ref-name> <cand-name> [more cand-names...]     (names without .mp4)
#
# Both numbers are required, and this is not belt-and-braces. A distillation or step-skipping arm
# fails by collapsing motion, which makes frames more similar to each other and can push SSIM UP
# while the video is visibly worse; a numerics arm (quantization, kernel swap) fails by adding
# noise, which shows up in SSIM but can leave motion energy untouched. The 4-bit-activation arm in
# this project scored SSIM 0.585 with motion 1.764 (4x the reference) -- one metric alone would have
# been ambiguous.
#
# The run-to-run floor depends on the REPLICA SHAPE, not on the clip, and the two cases are far
# apart -- so check which one an arm was measured under before calling it lossless:
#   1 GPU, fixed seed    SSIM 1.000000, bit-identical output, motion and bitrate equal to the digit
#                        (measured on 480_20 / 480_30 / 768_20 ref2va r1024)
#   2 GPUs (Ulysses=2)   SSIM 0.9444 on the 768p clip -- the all-to-all reduction order varies
# Healthy motion energy for ref2va 1344x768 / 5.175 s: 0.40-0.50 (480p runs hotter, 1.17-1.18).
#
# Same two gotchas as nvfp4_probe_quality.sh: ffmpeg/ffprobe live in the CONTAINER, and the run
# directory is not mounted, so files are hardlinked into /opt/dlami/nvme/out (= /out) first.
set -u
RUNDIR=${RUNDIR:-/opt/dlami/nvme/minimax_h3_h200}

link() {
  local f="$RUNDIR/$1.mp4"
  [ -f "$f" ] || { echo "MISSING $f" >&2; return 1; }
  # A hardlink only works when RUNDIR is on the same filesystem as /opt/dlami/nvme; RUNDIR=$HOME
  # (where the sweep scripts write) is not, and `ln` fails with EXDEV. Copy in that case.
  sudo ln -f "$f" /opt/dlami/nvme/out/ 2>/dev/null || sudo cp -f "$f" /opt/dlami/nvme/out/
}

# Bitrate is the cheapest quantization-damage detector and the one that caught the broken w4a4 arm:
# x264 at fixed quality spends bits on noise, so a numerics regression inflates the file even when
# the eye is unsure. BF16 ref2va 768p is ~466 kbps; the broken arm was 2421 kbps (5.2x).
bitrate() {
  docker exec h3 ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate \
    -of csv=p=0 "/out/$1.mp4" 2>/dev/null | awk '{printf "%dk", $1/1000}'
}

motion() {
  docker exec h3 ffprobe -f lavfi \
    "movie=/out/$1.mp4,tblend=all_mode=difference,signalstats" \
    -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0 2>/dev/null \
    | awk -F, 'NF&&$1!=""{s+=$1;n++} END{if(n)printf "%.4f",s/n; else print "n/a"}'
}

REF=$1; shift
link "$REF" || exit 1
echo "ref  $REF   motion=$(motion "$REF")   bitrate=$(bitrate "$REF")"

for CAND in "$@"; do
  link "$CAND" || continue
  # -v error keeps ffmpeg's own progress noise out but still prints the SSIM summary line.
  # Two traps in one line: the ssim filter prints its summary at ffmpeg's INFO level, so `-v error`
  # silently swallows it and the check reports nothing; and the line reads
  # "SSIM Y:0.99 (18.3) U:... All:0.99 (19.3)", so a contiguous "Y:.. U:.. V:.. All:.." pattern
  # never matches either.
  s=$(docker exec h3 ffmpeg -hide_banner -i "/out/$CAND.mp4" -i "/out/$REF.mp4" \
        -lavfi "[0:v][1:v]ssim" -f null - 2>&1 | grep -o "SSIM .*")
  echo "cand $CAND   motion=$(motion "$CAND")   bitrate=$(bitrate "$CAND")   $s"
done
