#!/usr/bin/env bash
# Titrate the account's "All G and VT Spot Instance Requests" vCPU quota in one region, to answer:
# is the quota big enough for a g7e.48xlarge (192 vCPU) at all?
#
# Why this is needed: the quota cannot be read with the Jump role (servicequotas:GetServiceQuota is
# AccessDenied), and RunInstances error text never states the limit. But the quota is shared across the
# whole G/VT family, so cheap g6 instances are a legal probe for what a g7e.48xlarge would need.
#
# Method: RunInstances validates in the order params -> capacity -> quota, so a launch that returns
# MaxSpotInstanceCountExceeded proves capacity existed and the quota is the wall. Add G-family vCPUs in
# 48-vCPU chunks until either TARGET_VCPU is reached or a launch is quota-rejected; then refine the
# ceiling downward with 16/8/4-vCPU shapes. Everything launched here is terminated on exit (trap), and
# only instance-ids this script created are ever touched.
#
#   ./probe_g_quota.sh                       # Tokyo, probe up to 192 total vCPU
#   TARGET_VCPU=240 REGION=us-west-2 ./probe_g_quota.sh
#
# Run it with the capacity hunt STOPPED: probe instances consume the same quota, so an in-flight
# g7e.48xlarge attempt would get a spurious QUOTA answer while the probe holds vCPUs.
set -uo pipefail

REGION=${REGION:-ap-northeast-1}
AZ=${AZ:-ap-northeast-1c}
SUBNET=${SUBNET:-subnet-32e7bf69}
SG=${SG:-sg-0db09eea89497051b}
AMI=${AMI:-ami-04d7d24927f2a914c}
TARGET_VCPU=${TARGET_VCPU:-192}
# vCPUs already held by instances we must not disturb (the running g7e.12xlarge).
BASE_VCPU=${BASE_VCPU:-0}

LAUNCHED=()
cleanup() {
  if [ ${#LAUNCHED[@]} -gt 0 ]; then
    echo "--- terminating probe instances: ${LAUNCHED[*]}"
    aws ec2 terminate-instances --region "$REGION" --instance-ids "${LAUNCHED[@]}" \
      --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
  fi
}
trap cleanup EXIT INT TERM

# Pre-existing G/VT spot vCPUs, so the probe reports absolute quota usage rather than its own delta.
declare -A VCPU=( [g6.xlarge]=4 [g6.2xlarge]=8 [g6.4xlarge]=16 [g6.12xlarge]=48 [g6.24xlarge]=96 )
if [ "$BASE_VCPU" = "0" ]; then
  while read -r t _; do
    [ -z "$t" ] && continue
    n=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$t" \
          --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text 2>/dev/null)
    [[ "$n" =~ ^[0-9]+$ ]] && BASE_VCPU=$((BASE_VCPU + n)) && echo "pre-existing: $t = $n vCPU"
  done < <(aws ec2 describe-instances --region "$REGION" \
             --filters Name=instance-state-name,Values=running,pending \
                       Name=instance-lifecycle,Values=spot \
             --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)].[InstanceType]' \
             --output text)
fi
echo "=== region=$REGION az=$AZ  baseline G-spot usage=${BASE_VCPU} vCPU  target=${TARGET_VCPU} vCPU"

used=$BASE_VCPU
try_shape() {   # $1 = instance type -> 0 launched, 1 quota wall, 2 no capacity / other
  local t=$1 n=${VCPU[$1]}
  echo "--- attempt $t (+$n -> $((used + n)) vCPU)"
  local out
  out=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$t" \
          --subnet-id "$SUBNET" --security-group-ids "$SG" --count 1 \
          --instance-market-options 'MarketType=spot' \
          --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=G-quota-probe}]' \
          --query 'Instances[0].InstanceId' --output text 2>&1)
  if [[ "$out" == i-* ]]; then
    LAUNCHED+=("$out"); used=$((used + n))
    echo "    LAUNCHED $out   usage now ${used} vCPU  -> quota >= ${used}"
    return 0
  fi
  echo "    $out" | tail -2
  if grep -qi 'MaxSpotInstanceCountExceeded\|VcpuLimitExceeded\|InstanceLimitExceeded' <<<"$out"; then
    echo "    QUOTA WALL at $((used + n)) vCPU  -> quota < $((used + n))"
    return 1
  fi
  echo "    inconclusive (capacity/other), not a quota signal"
  return 2
}

wall=""
while [ $((used + 48)) -le "$TARGET_VCPU" ]; do
  try_shape g6.12xlarge; rc=$?
  [ $rc -eq 1 ] && { wall=$((used + 48)); break; }
  [ $rc -eq 2 ] && { echo "!!! g6.12xlarge unavailable, cannot titrate further"; break; }
done

# Refine: if the 48-chunk hit the wall, walk up in smaller shapes to bracket the exact ceiling.
if [ -n "$wall" ]; then
  for t in g6.4xlarge g6.2xlarge g6.xlarge; do
    while [ $((used + VCPU[$t])) -lt "$wall" ]; do
      try_shape "$t"; rc=$?
      [ $rc -eq 1 ] && { wall=$((used + VCPU[$t])); break; }
      [ $rc -ne 0 ] && break
    done
  done
fi

echo "=== VERDICT"
if [ -n "$wall" ]; then
  echo "    G/VT spot quota in $REGION is in [${used}, $((wall - 1))] vCPU"
  [ "$wall" -le 192 ] && echo "    -> 192 vCPU (g7e.48xlarge) does NOT fit, even with everything else stopped"
else
  echo "    reached ${used} vCPU with no quota rejection -> quota >= ${used} vCPU"
  [ "$used" -ge 192 ] && echo "    -> 192 vCPU (g7e.48xlarge) DOES fit once other G instances are stopped"
fi
