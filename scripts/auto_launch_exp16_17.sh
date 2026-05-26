#!/bin/bash
# Polls nodeset3 GPU state and auto-launches exp16 (DBPO) + exp17 (scalar
# self-sim GRPO) once at least 4 GPUs drop below 50% util with >=50 GB free.
#
# Intended to be run on nodeset3 inside a detached tmux session:
#   ssh nodeset3
#   tmux new-session -d -s auto-launch-exp16-17 -c /home/ubuntu/rl/verifiers \
#     'bash scripts/auto_launch_exp16_17.sh 2>&1 | tee /tmp/auto_launch_exp16_17.log'
#
# Once both launches fire the poller exits.
set -euo pipefail

WORK_DIR="/home/ubuntu/rl/verifiers"
POLL_INTERVAL_S=300
FREE_MEM_MIB_MIN=50000     # >=50 GB free per candidate GPU
UTIL_MAX_PCT=50            # <50% util per candidate GPU
MIN_FREE_GPUS=4            # need 4 free GPUs (2 per exp)
LOG_FILE="/tmp/auto_launch_exp16_17.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_free_gpus() {
  nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v fmin="$FREE_MEM_MIB_MIN" -v umax="$UTIL_MAX_PCT" '
        { gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($2+0 >= fmin && $3+0 < umax) print $1 }' \
    | wc -l
}

log "Auto-launch poller starting (interval=${POLL_INTERVAL_S}s, need ${MIN_FREE_GPUS}+ GPUs w/ <${UTIL_MAX_PCT}% util and >=${FREE_MEM_MIB_MIN} MiB free)."
log "Working dir: $WORK_DIR"

cd "$WORK_DIR"

while true; do
  free_count=$(count_free_gpus || echo 0)
  snapshot=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '|')
  log "free_gpus=${free_count} snapshot=${snapshot}"

  if [ "$free_count" -ge "$MIN_FREE_GPUS" ]; then
    log "Threshold met — pulling latest and launching exp16 and exp17."
    git pull --rebase origin main 2>&1 | tee -a "$LOG_FILE" || true

    log "Launching exp16 (DBPO)…"
    bash scripts/launch_exp16.sh 2>&1 | tee -a "$LOG_FILE" || log "exp16 launch script returned non-zero"
    sleep 15
    log "Launching exp17 (scalar self-sim GRPO baseline)…"
    bash scripts/launch_exp17.sh 2>&1 | tee -a "$LOG_FILE" || log "exp17 launch script returned non-zero"

    log "Both launches dispatched. Tmux sessions:"
    tmux ls 2>&1 | grep -E "exp16|exp17" | tee -a "$LOG_FILE" || true
    log "Poller exiting."
    exit 0
  fi

  sleep "$POLL_INTERVAL_S"
done
