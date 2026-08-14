#!/usr/bin/env bash
# Sweep every AZ that offers a g7e shape and classify WHICH wall we hit there, using the real instance
# type rather than a proxy. RunInstances validates params -> capacity -> quota, so the error code splits
# the two walls cleanly:
#
#   InsufficientInstanceCapacity   no machines in that AZ right now (says nothing about quota)
#   MaxSpotInstanceCountExceeded   capacity WAS there; only the G/VT spot vCPU quota blocks us  <-- gold
#   i-...                          it actually launched (terminated immediately; relaunch properly)
#
# Uses the default VPC and an SSM-resolved AL2023 AMI so it works in regions where we have no
# AMI/subnet/SG/keypair row -- the box is never meant to boot into anything, it only has to be accepted.
#
#   ./probe_g7e_capacity.sh                      # g7e.24xlarge across all offering AZs
#   TYPE=g7e.48xlarge ./probe_g7e_capacity.sh
#   MARKET=ondemand ./probe_g7e_capacity.sh      # on-demand pool + on-demand quota (both separate!)
#
# MARKET matters twice over: the on-demand capacity pool is a different pool from spot, AND on-demand
# counts against a different quota (L-DB2E81BA "Running On-Demand G and VT instances") than spot
# (L-3819A6DF). On-demand quota exhaustion shows up as VcpuLimitExceeded, not MaxSpotInstanceCountExceeded.
set -uo pipefail
TYPE=${TYPE:-g7e.24xlarge}
MARKET=${MARKET:-spot}
KEEP=${KEEP:-0}     # KEEP=1 leaves a successful launch running instead of terminating it
MKT_ARGS=( --instance-market-options 'MarketType=spot' )
[ "$MARKET" = "ondemand" ] && MKT_ARGS=()

for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
  azs=$(aws ec2 describe-instance-type-offerings --region "$r" --location-type availability-zone \
          --filters Name=instance-type,Values="$TYPE" \
          --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null)
  [ -z "$azs" ] && continue
  ami=$(aws ssm get-parameter --region "$r" \
          --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
          --query 'Parameter.Value' --output text 2>/dev/null)
  if [[ "$ami" != ami-* ]]; then
    ami=$(aws ec2 describe-images --region "$r" --owners amazon \
            --filters 'Name=name,Values=al2023-ami-2023*-x86_64' Name=state,Values=available \
            --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text 2>/dev/null)
  fi
  [[ "$ami" != ami-* ]] && { echo "$r  SKIP (no AMI resolvable)"; continue; }

  for az in $azs; do
    sn=$(aws ec2 describe-subnets --region "$r" \
           --filters Name=availability-zone,Values="$az" Name=default-for-az,Values=true \
           --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
    [[ "$sn" != subnet-* ]] && { printf '%-16s %-16s %s\n' "$r" "$az" "SKIP (no default subnet)"; continue; }
    out=$(aws ec2 run-instances --region "$r" --image-id "$ami" --instance-type "$TYPE" \
            --subnet-id "$sn" --count 1 "${MKT_ARGS[@]}" \
            --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=G7E-capacity-probe}]' \
            --query 'Instances[0].InstanceId' --output text 2>&1)
    if [[ "$out" == i-* ]]; then
      printf '%-16s %-16s %s\n' "$r" "$az" "LAUNCHED $out  <<< CAPACITY AND QUOTA BOTH OK"
      [ "$KEEP" = "1" ] || aws ec2 terminate-instances --region "$r" --instance-ids "$out" >/dev/null
    elif grep -qE 'MaxSpotInstanceCountExceeded|VcpuLimitExceeded|InstanceLimitExceeded' <<<"$out"; then
      printf '%-16s %-16s %s\n' "$r" "$az" "QUOTA  <<< capacity exists here"
    elif grep -q InsufficientInstanceCapacity <<<"$out"; then
      printf '%-16s %-16s %s\n' "$r" "$az" "no capacity"
    else
      printf '%-16s %-16s %s\n' "$r" "$az" "$(tr '\n' ' ' <<<"$out" | cut -c1-150)"
    fi
  done
done
