#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CREATE_SCRIPT="$SCRIPT_DIR/create_agentgym_a100_vm.sh"
REMOTE_PREP_SCRIPT="$SCRIPT_DIR/remote_prepare_agentgym_training.sh"
REMOTE_LAUNCH_SCRIPT="$SCRIPT_DIR/remote_launch_agentgym_training.sh"

PROJECT="${PROJECT:-odion-voice-agent1}"
SSH_USER="${SSH_USER:-mavinomichael}"
IMAGE_NAME="${IMAGE_NAME:-ubuntu-accelerator-2204-amd64-with-nvidia-580-v20260313}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-2000}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
INSTANCE_PREFIX="${INSTANCE_PREFIX:-odion-agentgym}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d%H%M%S)-$RANDOM}"
WORKDIR="${WORKDIR:-/tmp/agentgym-acquire-${BATCH_ID}}"
WINNER_TIMEOUT_SECONDS="${WINNER_TIMEOUT_SECONDS:-7200}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-900}"
VERIFY_INTERVAL_SECONDS="${VERIFY_INTERVAL_SECONDS:-20}"
TRAIN_START_TIMEOUT_SECONDS="${TRAIN_START_TIMEOUT_SECONDS:-7200}"
TRAIN_EXP_NAME="${TRAIN_EXP_NAME:-babyai_2agent_scaling_200_8gpu}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/${SSH_USER}/AgentGym-RL}"
DELETE_OLD_VM_ON_SUCCESS="${DELETE_OLD_VM_ON_SUCCESS:-1}"
OLD_VM_NAME="${OLD_VM_NAME:-odion-agentgym-rl}"
OLD_VM_ZONE="${OLD_VM_ZONE:-us-east5-a}"
BOOTSTRAP_TRAINING="${BOOTSTRAP_TRAINING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SIMULATED_WINNER_GROUP="${SIMULATED_WINNER_GROUP:-}"
SIMULATED_WINNER_ZONE="${SIMULATED_WINNER_ZONE:-}"

mkdir -p "$WORKDIR/logs" "$WORKDIR/results" "$WORKDIR/state"

WORKER_GROUPS=(h200-us h200-intl b200-us b200-intl)
WORKER_MACHINE_h200_us="a3-ultragpu-8g"
WORKER_ACCEL_h200_us="nvidia-h200-141gb"
WORKER_SHAPE_h200_us="h200"
WORKER_ZONES_h200_us="us-east5-a us-east4-b us-central1-b us-west1-c"

WORKER_MACHINE_h200_intl="a3-ultragpu-8g"
WORKER_ACCEL_h200_intl="nvidia-h200-141gb"
WORKER_SHAPE_h200_intl="h200"
WORKER_ZONES_h200_intl="europe-west4-a europe-west1-b asia-south1-b"

WORKER_MACHINE_b200_us="a4-highgpu-8g"
WORKER_ACCEL_b200_us="nvidia-b200"
WORKER_SHAPE_b200_us="b200"
WORKER_ZONES_b200_us="us-east4-b us-east1-b us-central1-b us-west3-b us-west3-c us-south1-b"

WORKER_MACHINE_b200_intl="a4-highgpu-8g"
WORKER_ACCEL_b200_intl="nvidia-b200"
WORKER_SHAPE_b200_intl="b200"
WORKER_ZONES_b200_intl="europe-west4-b europe-north1-b asia-southeast1-b asia-northeast1-b"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

worker_var_name() {
  printf '%s' "$1" | tr '-' '_'
}

worker_field() {
  local group_key
  group_key=$(worker_var_name "$1")
  local field="$2"
  eval "printf '%s' \"\${WORKER_${field}_${group_key}}\""
}

candidate_name_for() {
  local group="$1"
  local zone="$2"
  local shape clean_zone name_prefix base
  shape=$(worker_field "$group" SHAPE)
  clean_zone=$(printf '%s' "$zone" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-')
  name_prefix="${INSTANCE_PREFIX}-${shape}-${group}-${BATCH_ID}"
  base="${name_prefix}-${clean_zone}"
  base="${base:0:63}"
  printf '%s\n' "${base%-}"
}

batch_filter() {
  printf 'labels.agentgym-batch=%s' "$BATCH_ID"
}

print_dry_run() {
  log "Dry run for batch ${BATCH_ID}"
  for group in "${WORKER_GROUPS[@]}"; do
    local shape machine accel zones labels
    shape=$(worker_field "$group" SHAPE)
    machine=$(worker_field "$group" MACHINE)
    accel=$(worker_field "$group" ACCEL)
    zones=$(worker_field "$group" ZONES)
    labels="agentgym-batch=${BATCH_ID},agentgym-shape=${shape},agentgym-group=${group}"
    printf 'GROUP=%s SHAPE=%s MACHINE=%s ACCEL=%s LABELS=%s\n' "$group" "$shape" "$machine" "$accel" "$labels"
    for zone in $zones; do
      printf '  zone=%s name=%s\n' "$zone" "$(candidate_name_for "$group" "$zone")"
    done
  done
  if [[ -n "$SIMULATED_WINNER_GROUP" && -n "$SIMULATED_WINNER_ZONE" ]]; then
    printf 'SIMULATED_WINNER=%s %s\n' "$SIMULATED_WINNER_GROUP" "$SIMULATED_WINNER_ZONE"
    printf 'SIMULATED_WINNER_NAME=%s\n' "$(candidate_name_for "$SIMULATED_WINNER_GROUP" "$SIMULATED_WINNER_ZONE")"
    printf 'CLEANUP_FILTER=%s\n' "$(batch_filter)"
  fi
}

launch_worker() {
  local group="$1"
  local group_key shape machine accel zones labels name_prefix result_file log_file pid_file
  group_key=$(worker_var_name "$group")
  shape=$(worker_field "$group" SHAPE)
  machine=$(worker_field "$group" MACHINE)
  accel=$(worker_field "$group" ACCEL)
  zones=$(worker_field "$group" ZONES)
  labels="agentgym-batch=${BATCH_ID},agentgym-shape=${shape},agentgym-group=${group}"
  name_prefix="${INSTANCE_PREFIX}-${shape}-${group}-${BATCH_ID}"
  result_file="$WORKDIR/results/${group}.env"
  log_file="$WORKDIR/logs/${group}.log"
  pid_file="$WORKDIR/state/${group}.pid"

  (
    export PROJECT NAME_PREFIX="$name_prefix" MACHINE_TYPE="$machine" ACCELERATOR_TYPE="$accel" ACCELERATOR_COUNT=8
    export IMAGE_NAME BOOT_DISK_SIZE_GB BOOT_DISK_TYPE LABELS="$labels" CANDIDATE_ZONES="$zones"
    export RESULT_FILE="$result_file" STARTUP_SCRIPT="$SCRIPT_DIR/agentgym_a100_startup.sh"
    export CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}"
    "$CREATE_SCRIPT"
  ) >"$log_file" 2>&1 &
  echo "$!" > "$pid_file"
}

stop_workers() {
  for group in "${WORKER_GROUPS[@]}"; do
    local pid_file="$WORKDIR/state/${group}.pid"
    [[ -f "$pid_file" ]] || continue
    pid=$(cat "$pid_file")
    kill "$pid" >/dev/null 2>&1 || true
    pkill -TERM -P "$pid" >/dev/null 2>&1 || true
  done
}

list_batch_instances() {
  CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
    gcloud compute instances list \
    --project="$PROJECT" \
    --filter="$(batch_filter)" \
    --format='value(name,zone.basename(),status)'
}

verify_instance() {
  local name="$1"
  local zone="$2"
  local started now cmd output gpu_count
  started=$(date +%s)
  while true; do
    now=$(date +%s)
    if (( now - started >= VERIFY_TIMEOUT_SECONDS )); then
      return 1
    fi
    output=$(CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
      gcloud compute ssh "${SSH_USER}@${name}" \
      --project="$PROJECT" \
      --zone="$zone" \
      --ssh-flag='-o StrictHostKeyChecking=no' \
      --quiet \
      --command "bash -lc 'test -f ~/.agentgym_startup_done && nvidia-smi --query-gpu=name --format=csv,noheader'" 2>/dev/null || true)
    gpu_count=$(printf '%s\n' "$output" | sed '/^$/d' | wc -l | tr -d ' ')
    if [[ "$gpu_count" == "8" ]]; then
      return 0
    fi
    sleep "$VERIFY_INTERVAL_SECONDS"
  done
}

cleanup_losers() {
  local winner_name="$1"
  local winner_zone="$2"
  while read -r name zone status; do
    [[ -n "$name" ]] || continue
    if [[ "$name" == "$winner_name" && "$zone" == "$winner_zone" ]]; then
      continue
    fi
    log "Deleting losing instance ${name} in ${zone}"
    CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
      gcloud compute instances delete "$name" \
      --project="$PROJECT" \
      --zone="$zone" \
      --quiet || true
  done < <(list_batch_instances)
}

sync_repo_to_winner() {
  local name="$1"
  local zone="$2"
  tar \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='output' \
    --exclude='reports' \
    --exclude='papers' \
    --exclude='AgentGym/env-visualization/node_modules' \
    --exclude='AgentGym/env-visualization/dist' \
    --exclude='AgentGym/env-visualization/.env.local' \
    -czf - -C "$REPO_ROOT" . | \
    CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
    gcloud compute ssh "${SSH_USER}@${name}" \
      --project="$PROJECT" \
      --zone="$zone" \
      --ssh-flag='-o StrictHostKeyChecking=no' \
      --quiet \
      --command "bash -lc 'rm -rf \"$REMOTE_REPO_ROOT\" && mkdir -p \"$REMOTE_REPO_ROOT\" && tar -xzf - -C \"$REMOTE_REPO_ROOT\"'"
}

prepare_training() {
  local name="$1"
  local zone="$2"
  sync_repo_to_winner "$name" "$zone"
  CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
    gcloud compute ssh "${SSH_USER}@${name}" \
    --project="$PROJECT" \
    --zone="$zone" \
    --ssh-flag='-o StrictHostKeyChecking=no' \
    --quiet \
    --command "bash -lc '$REMOTE_REPO_ROOT/scripts/vm/remote_prepare_agentgym_training.sh'"
  CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
    gcloud compute ssh "${SSH_USER}@${name}" \
    --project="$PROJECT" \
    --zone="$zone" \
    --ssh-flag='-o StrictHostKeyChecking=no' \
    --quiet \
    --command "bash -lc '$REMOTE_REPO_ROOT/scripts/vm/remote_launch_agentgym_training.sh'"
}

wait_for_first_step() {
  local name="$1"
  local zone="$2"
  local started now output
  started=$(date +%s)
  while true; do
    now=$(date +%s)
    if (( now - started >= TRAIN_START_TIMEOUT_SECONDS )); then
      return 1
    fi
    output=$(CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
      gcloud compute ssh "${SSH_USER}@${name}" \
      --project="$PROJECT" \
      --zone="$zone" \
      --ssh-flag='-o StrictHostKeyChecking=no' \
      --quiet \
      --command "bash -lc 'grep -m1 \"step:\" /dev/shm/agentgym_logs/${TRAIN_EXP_NAME}/train_step200.log 2>/dev/null || true'" 2>/dev/null || true)
    if [[ -n "$output" ]]; then
      printf '%s\n' "$output"
      return 0
    fi
    sleep 30
  done
}

delete_old_vm_if_needed() {
  if [[ "$DELETE_OLD_VM_ON_SUCCESS" != "1" ]]; then
    return 0
  fi
  log "Deleting old VM instance ${OLD_VM_NAME} in ${OLD_VM_ZONE}"
  CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/opt/anaconda3/bin/python3}" \
    gcloud compute instances delete "$OLD_VM_NAME" \
    --project="$PROJECT" \
    --zone="$OLD_VM_ZONE" \
    --quiet
}

all_workers_done() {
  for group in "${WORKER_GROUPS[@]}"; do
    local pid_file="$WORKDIR/state/${group}.pid"
    [[ -f "$pid_file" ]] || continue
    pid=$(cat "$pid_file")
    if kill -0 "$pid" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

acquire_winner() {
  local started now name zone status line
  started=$(date +%s)
  while true; do
    now=$(date +%s)
    if (( now - started >= WINNER_TIMEOUT_SECONDS )); then
      return 1
    fi
    while read -r name zone status; do
      [[ "$status" == "RUNNING" ]] || continue
      [[ -n "$name" ]] || continue
      log "Verifying candidate ${name} in ${zone}"
      if verify_instance "$name" "$zone"; then
        printf '%s %s\n' "$name" "$zone"
        return 0
      fi
    done < <(list_batch_instances)

    if all_workers_done; then
      return 1
    fi
    sleep 20
  done
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_dry_run
  exit 0
fi

log "Starting acquisition batch ${BATCH_ID}"
for group in "${WORKER_GROUPS[@]}"; do
  launch_worker "$group"
  log "Launched worker ${group}"
done

if ! winner=$(acquire_winner); then
  stop_workers
  log "No VM acquired for batch ${BATCH_ID}"
  exit 1
fi

winner_name=$(printf '%s' "$winner" | awk '{print $1}')
winner_zone=$(printf '%s' "$winner" | awk '{print $2}')
log "Winner: ${winner_name} in ${winner_zone}"

stop_workers
cleanup_losers "$winner_name" "$winner_zone"
sleep 10
cleanup_losers "$winner_name" "$winner_zone"

cat > "$WORKDIR/winner.env" <<EOF
WINNER_NAME=$winner_name
WINNER_ZONE=$winner_zone
BATCH_ID=$BATCH_ID
WORKDIR=$WORKDIR
EOF

if [[ "$BOOTSTRAP_TRAINING" == "1" ]]; then
  log "Preparing repo and training environment on ${winner_name}"
  prepare_training "$winner_name" "$winner_zone"
  log "Waiting for first training step"
  first_step=$(wait_for_first_step "$winner_name" "$winner_zone") || {
    log "Training did not reach first step in time"
    exit 1
  }
  log "Training started: ${first_step}"
  delete_old_vm_if_needed
fi

log "Acquisition complete. Winner=${winner_name} Zone=${winner_zone} Batch=${BATCH_ID}"
printf 'WINNER_NAME=%s\nWINNER_ZONE=%s\nBATCH_ID=%s\nWORKDIR=%s\n' "$winner_name" "$winner_zone" "$BATCH_ID" "$WORKDIR"
