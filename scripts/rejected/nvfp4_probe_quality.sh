#!/usr/bin/env bash
# Probe an mp4's validity without downloading it: bitrate + inter-frame motion energy.
#
# Two gotchas this encodes. (1) ffprobe lives in the CONTAINER, not on the g7e host. (2) the run
# directory /opt/dlami/nvme/minimax_h3_h200 is NOT mounted into the container, so hardlink the file
# into /opt/dlami/nvme/out (which is mounted as /out) first.
#
# Healthy ref2va 768p sample: bitrate ~470-620 kbps, motion ~0.40-0.50.
# Broken (mosaic) sample:     bitrate ~12-14 Mbps, motion ~14-16.
set -u
RUNDIR=${RUNDIR:-/opt/dlami/nvme/minimax_h3_h200}
for name in "$@"; do
  f="$RUNDIR/$name.mp4"
  [ -f "$f" ] || { echo "$name  MISSING"; continue; }
  sudo ln -f "$f" /opt/dlami/nvme/out/ 2>/dev/null
  br=$(docker exec h3 ffprobe -v quiet -show_entries format=bit_rate -of csv=p=0 "/out/$name.mp4")
  st=$(docker exec h3 ffprobe -v quiet -select_streams v:0 \
        -show_entries stream=width,height,nb_frames -of csv=p=0 "/out/$name.mp4")
  mo=$(docker exec h3 ffprobe -f lavfi \
        "movie=/out/$name.mp4,tblend=all_mode=difference,signalstats" \
        -show_entries frame_tags=lavfi.signalstats.YAVG -of csv=p=0 2>/dev/null \
       | awk -F, 'NF&&$1!=""{s+=$1;n++} END{if(n)printf "%.4f (%d frames)",s/n,n; else print "n/a"}')
  echo "$name  bitrate=$br  stream=$st  motion=$mo"
done
