#!/usr/bin/env bash
# quality_pair.sh, but run on the laptop against already-downloaded mp4s. Same two metrics and the
# same reasoning (see quality_pair.sh header): SSIM against a reference clip PLUS inter-frame motion
# energy of every clip, because a motion-collapse failure can push SSIM UP.
#
#   quality_pair_local.sh <ref.mp4> <cand.mp4> [more...]
#
# Exists because a g7e spot box can vanish between generating a clip and checking it -- which
# happened twice here. Pull the mp4 the moment it lands and grade it locally; homebrew ffmpeg has
# both filters, so no box is needed.
#
# Reference points for ref2va 1344x768 / 5.175 s:
#   1-GPU run-to-run floor            SSIM 1.0 (bit-identical)
#   multi-GPU reduction-order floor   SSIM ~0.944 on Trn2/H200; measure it, do not assume it
#   healthy motion energy             0.40-0.50
set -u

motion() {
  ffprobe -f lavfi \
    "movie=$1,tblend=all_mode=difference,signalstats" \
    -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0 2>/dev/null \
    | awk -F, 'NF&&$1!=""{s+=$1;n++} END{if(n)printf "%.4f",s/n; else print "n/a"}'
}

REF=$1; shift
printf 'ref  %-28s motion=%s  %s\n' "$(basename "$REF")" "$(motion "$REF")" \
  "$(ffprobe -v error -select_streams v -show_entries stream=width,height,nb_frames -of csv=p=0 "$REF")"

for CAND in "$@"; do
  s=$(ffmpeg -hide_banner -i "$CAND" -i "$REF" -lavfi "[0:v][1:v]ssim" -f null - 2>&1 | grep -o "SSIM .*")
  printf 'cand %-28s motion=%s  %s\n' "$(basename "$CAND")" "$(motion "$CAND")" "$s"
done
