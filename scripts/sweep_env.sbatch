#!/bin/bash
#SBATCH --job-name=tcr_sweepenv
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Generic 4-wide sweep over (config x seed) for any verifiers env (gsm8k/math/...)
# via scripts/train_generic.py. Node 7 only. Submit with --export=ALL,HOME=/home/ubuntu.
# Env: CONFIGS="path1 path2 ..."  SEEDS="1 2 3"  SMOKE=1.
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SMOKE="${SMOKE:-0}"; SEEDS="${SEEDS:-1 2 3}"; CONFIGS="${CONFIGS:?set CONFIGS}"; JID="$SLURM_JOB_ID"

launch() {  # config seed vllm_gpu trainer_gpu port
  local CONFIG="$1" SEED="$2" VG="$3" TG="$4" PORT="$5"
  local NCCL=$(( PORT + 1000 ))
  local CN; CN="$(basename "$CONFIG" .toml)"
  local RUN="${CN}-s${SEED}"
  local TMP="/tmp/${CN}_s${SEED}_${JID}.toml"
  sed -e "s|^run_name = .*|run_name = \"$RUN\"|" \
      -e "s/^vllm_server_port = .*/vllm_server_port = $PORT/" \
      -e "s/^nccl_group_port = .*/nccl_group_port = $NCCL/" \
      -e "/^\[trainer\.args\]/a seed = $SEED" "$CONFIG" > "$TMP"
  grep -q "reward_vector_dump_path" "$TMP" && sed -i "s|^reward_vector_dump_path = .*|reward_vector_dump_path = \"/tmp/${RUN}_rvecs.jsonl\"|" "$TMP"
  [ "$SMOKE" = "1" ] && sed -i "s/^max_steps = .*/max_steps = 6/" "$TMP"
  (
    CUDA_VISIBLE_DEVICES=$VG vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
      > "logs/vllm_${RUN}_${JID}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$RUN] vLLM died"; exit 1; }
    done
    CUDA_VISIBLE_DEVICES=$TG PYTORCH_ALLOC_CONF=expandable_segments:True \
      python scripts/train_generic.py --config "$TMP" \
      > "logs/trainer_${RUN}_${JID}.log" 2>&1
    echo "[$RUN] trainer done rc=$?"; kill $V 2>/dev/null
  ) &
}

ITEMS=(); for c in $CONFIGS; do for s in $SEEDS; do ITEMS+=("$c|$s"); done; done
PV=(0 2 4 6); PT=(1 3 5 7); N=${#ITEMS[@]}; i=0
echo "[sweep_env] $N runs (4/wave): $CONFIGS x [$SEEDS]"
while [ $i -lt $N ]; do
  for p in 0 1 2 3; do
    idx=$(( i + p )); [ $idx -ge $N ] && break
    launch "${ITEMS[$idx]%|*}" "${ITEMS[$idx]#*|}" "${PV[$p]}" "${PT[$p]}" "$(( 8001 + p ))"
  done
  wait; echo "[sweep_env] wave through $((i+4)) done ($(date +%H:%M:%S))"; i=$(( i + 4 ))
done
echo "[sweep_env] ALL DONE: $N runs"
