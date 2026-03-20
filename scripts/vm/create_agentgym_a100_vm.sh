#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-odion-voice-agent1}"
NAME="${NAME:-}"
NAME_PREFIX="${NAME_PREFIX:-odion-agentgym-rl}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-2000}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-8g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-tesla-a100}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-8}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
IMAGE_NAME="${IMAGE_NAME:-ubuntu-accelerator-2204-amd64-with-nvidia-580-v20260226}"
TAGS="${TAGS:-agentgym-rl,visual-inspector}"
LABELS="${LABELS:-}"
STARTUP_SCRIPT="${STARTUP_SCRIPT:-$(cd "$(dirname "$0")" && pwd)/agentgym_a100_startup.sh}"
RESULT_FILE="${RESULT_FILE:-}"
DESCRIBE_FORMAT="${DESCRIBE_FORMAT:-yaml(name,status,zone,machineType,guestAccelerators,networkInterfaces[0].accessConfigs[0].natIP,disks,labels)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${CANDIDATE_ZONES:-}" ]]; then
  read -r -a CANDIDATE_ZONE_ARRAY <<<"${CANDIDATE_ZONES}"
else
  CANDIDATE_ZONE_ARRAY=(
    us-central1-a
    us-central1-c
    us-central1-b
    us-central1-f
    us-east5-b
    us-east5-a
    us-east4-c
    us-east1-b
    us-west1-b
  )
fi

sanitize_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-'
}

zone_basename() {
  printf '%s' "$1" | awk -F/ '{print $NF}'
}

instance_name_for_zone() {
  local zone="$1"
  if [[ -n "$NAME" ]]; then
    printf '%s\n' "$NAME"
    return
  fi
  local clean_zone base
  clean_zone=$(sanitize_name "$zone")
  base="${NAME_PREFIX}-${clean_zone}"
  base="${base:0:63}"
  printf '%s\n' "${base%-}"
}

write_result() {
  local name="$1"
  local zone="$2"
  [[ -n "$RESULT_FILE" ]] || return 0
  cat > "$RESULT_FILE" <<RESULT
NAME=${name}
ZONE=${zone}
PROJECT=${PROJECT}
MACHINE_TYPE=${MACHINE_TYPE}
ACCELERATOR_TYPE=${ACCELERATOR_TYPE}
ACCELERATOR_COUNT=${ACCELERATOR_COUNT}
RESULT
}

for zone in "${CANDIDATE_ZONE_ARRAY[@]}"; do
  instance_name=$(instance_name_for_zone "$zone")
  echo "Trying ${zone} with ${MACHINE_TYPE} (${ACCELERATOR_TYPE} x${ACCELERATOR_COUNT}) as ${instance_name}..."

  cmd=(
    gcloud compute instances create "$instance_name"
    --project="$PROJECT"
    --zone="$zone"
    --machine-type="$MACHINE_TYPE"
    --maintenance-policy=TERMINATE
    --restart-on-failure
    --provisioning-model=STANDARD
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}"
    --create-disk="boot=yes,auto-delete=yes,device-name=${instance_name},image=projects/${IMAGE_PROJECT}/global/images/${IMAGE_NAME},mode=rw,size=${BOOT_DISK_SIZE_GB},type=projects/${PROJECT}/zones/${zone}/diskTypes/${BOOT_DISK_TYPE}"
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default
    --metadata=enable-osconfig=TRUE,enable-oslogin=FALSE,serial-port-enable=true
    --metadata-from-file="startup-script=${STARTUP_SCRIPT}"
    --tags="$TAGS"
    --quiet
  )

  if [[ -n "$LABELS" ]]; then
    cmd+=(--labels="$LABELS")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN zone=%s name=%s labels=%s machine=%s accelerator=%s count=%s\n' \
      "$zone" "$instance_name" "$LABELS" "$MACHINE_TYPE" "$ACCELERATOR_TYPE" "$ACCELERATOR_COUNT"
    write_result "$instance_name" "$zone"
    continue
  fi

  set +e
  CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" "${cmd[@]}"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "Created ${instance_name} in ${zone}"
    write_result "$instance_name" "$zone"
    CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
      gcloud compute instances describe "$instance_name" \
      --project="$PROJECT" \
      --zone="$zone" \
      --format="$DESCRIBE_FORMAT"
    exit 0
  fi

  echo "Zone ${zone} failed, trying next zone..."
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

echo "No candidate zone succeeded." >&2
exit 1
