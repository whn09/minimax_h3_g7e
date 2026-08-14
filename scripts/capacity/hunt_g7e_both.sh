#!/usr/bin/env bash
# Hunt g7e.24xlarge (4 GPU) and g7e.48xlarge (8 GPU) at the same time and keep only the first one that
# lands. Runs on the Jump box, where ~/launch_g7e.sh and the per-region AMI/subnet/SG/key table live.
#
# !!! DO NOT RUN THIS UNTIL THE QUOTA IS RAISED (measured 2026-08-14). The account's "All G and VT Spot
# Instance Requests" quota (L-3819A6DF, counted in vCPU) is 64 in ap-northeast-1 and 48-95 in us-east-2,
# while g7e.24xlarge needs 96 and g7e.48xlarge needs 192. No amount of retrying can win: the wall is the
# quota, not capacity. Measured with ./probe_g_quota.sh (titrates the shared G/VT quota with cheap g6
# spot) and ./probe_g7e_capacity.sh (attempts the real type in all 20 offering AZs). Capacity for the
# 8-GPU box was present in 6 AZs and for the 4-GPU box in 3 AZs at probe time -- all of them returned
# MaxSpotInstanceCountExceeded, i.e. they got past the capacity check and died on quota.
# Re-run both probes after any quota increase; resume hunting only in AZs that showed capacity.
#
#   ./hunt_g7e_both.sh              # both hunts, 90 s between rounds, spot only
#   SLEEP=300 ./hunt_g7e_both.sh    # gentler polling
#
# Why both: measured spot placement scores (2026-08-14, 10 AZs across 6 regions) are
#   g7e.24xlarge  1/10 everywhere            <- the cheap 4-GPU shape, but effectively unobtainable
#   g7e.48xlarge  3/10 at apne1-az1, euc1-az2
#   g7e.12xlarge  3/10 at apne1-az1/az4, euc1-az2   <- the score class we actually landed one at
# So the 8-GPU box is EASIER to get than the 4-GPU box, and 48xlarge also gets us Ulysses=8 in the same
# session. 24xlarge stays in the hunt because it is 3.2x cheaper per hour and 55% of 4xlarge per GPU.
#
# In THIS account apne1-az1 = ap-northeast-1c (az-id != az-name; the mapping is per account, so resolve
# it with `describe-availability-zones --query 'AvailabilityZones[].[ZoneName,ZoneId]'` before trusting
# any hard-coded AZ name). ap-northeast-1c is already in launch_g7e.sh's TARGETS.
#
# --spot-only on BOTH is deliberate: launch_g7e.sh defaults to MODE=both, which would silently fall
# back to on-demand -- an on-demand 48xlarge is several times the $15.8346/h spot price and would be a
# very expensive surprise for a capacity hunt that is meant to run unattended.
#
# us-east-1 is left in the 24xlarge target list but it cannot win there: the account's G-family spot
# quota in us-east-1 is below 96 vCPU (`MaxSpotInstanceCountExceeded`), and this role cannot even read
# the quota (`servicequotas:GetServiceQuota` denied), let alone raise it. eu-central-1 (euc1-az2, the
# other SPS-3 AZ) is NOT covered: it has no AMI/subnet/SG/keypair row yet. Add one if Tokyo stays dry.
set -uo pipefail
cd ~
SLEEP=${SLEEP:-90}

INSTANCE_TYPE=g7e.24xlarge NAME_TAG=G7E-h3-24 \
  setsid ./launch_g7e.sh --loop "$SLEEP" --spot-only            > ~/hunt24.log 2>&1 &
P24=$!
# The 48xlarge hunt is NOT restricted to ap-northeast-1 even though that is the only SPS-3 AZ we have a
# target row for: RunInstances evaluates capacity BEFORE quota, so "no capacity" vs "QUOTA exceeded"
# tells us which wall we are on per region -- and 192 vCPU is big enough that some regions will hit the
# quota wall with nothing running at all. Letting it sweep every region turns the hunt into a free
# quota probe. (Observed 2026-08-14: ap-northeast-1c returned QUOTA, i.e. capacity WAS there.)
INSTANCE_TYPE=g7e.48xlarge NAME_TAG=G7E-h3-48 \
  setsid ./launch_g7e.sh --loop "$SLEEP" --spot-only > ~/hunt48.log 2>&1 &
P48=$!
echo "hunting: 24xl pid=$P24 (~/hunt24.log)   48xl pid=$P48 (~/hunt48.log)   round=${SLEEP}s"

# launch_g7e.sh --loop only returns when it has an instance, so "exited" == "landed" (or died).
while true; do
  for pair in "$P24:$P48:24xl:48xl" "$P48:$P24:48xl:24xl"; do
    IFS=: read -r me other mine theirs <<<"$pair"
    if ! kill -0 "$me" 2>/dev/null; then
      echo "$mine hunt exited -- killing the $theirs hunt so we do not pay for two boxes"
      pkill -P "$other" 2>/dev/null; kill "$other" 2>/dev/null
      tail -25 ~/hunt${mine%xl}.log
      exit 0
    fi
  done
  sleep 20
done
