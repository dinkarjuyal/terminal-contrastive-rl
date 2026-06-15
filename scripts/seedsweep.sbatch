#!/bin/bash
#SBATCH --job-name=tcr_seed
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Multi-seed robustness: for each seed, run all 4 configs in parallel (4 GPU pairs),
# wait, then next seed. Per-seed run_name -> separate checkpoint dirs.
# Env: SEEDS (default "1 2"), SMOKE=1 -> 6 steps + only seed in SEEDS[0].
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="Qwen/Qwen2.5-1.5B-Instruct"
SMOKE="${SMOKE:-0}"; SEEDS="${SEEDS:-1 2}"; JID="$SLURM_JOB_ID"
[ "$SMOKE" = "1" ] && SEEDS="$(echo $SEEDS | awk '{print $1}')"

declare -A CFG=(
  [mgda]=configs/rl/bash_agent_tc_exp19a_mgda.toml
  [ctrl]=configs/rl/bash_agent_tc_exp19_ctrl_v2k3.toml
  [k5]=configs/rl/bash_agent_tc_exp19b.toml
  [corr]=configs/rl/bash_agent_tc_exp19d_corrected.toml
)

launch() {  # <abbrev> <seed> <vllm_gpu> <trainer_gpu> <port>
  local AB="$1" SEED="$2" VG="$3" TG="$4" PORT="$5"
  local NCCL=$(( PORT + 1000 ))
  local RUN="bash-agent-tc-${AB}-s${SEED}"
  local TMP="/tmp/${AB}_s${SEED}_${JID}.toml"
  sed -e "s|^run_name = .*|run_name = \"$RUN\"|" \
      -e "s/^vllm_server_port = .*/vllm_server_port = $PORT/" \
      -e "s/^nccl_group_port = .*/nccl_group_port = $NCCL/" \
      -e "/^\[trainer\.args\]/a seed = $SEED" \
      -e "s|^reward_vector_dump_path = .*|reward_vector_dump_path = \"/tmp/${AB}_s${SEED}_rvecs.jsonl\"|" \
      "${CFG[$AB]}" > "$TMP"
  [ "$SMOKE" = "1" ] && sed -i -e "s/^max_steps = .*/max_steps = 6/" -e "s/^save_steps = .*/save_steps = 6/" "$TMP"
  (
    CUDA_VISIBLE_DEVICES=$VG vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
      > "logs/vllm_${AB}_s${SEED}_${JID}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$AB s$SEED] vLLM died"; exit 1; }
    done
    CUDA_VISIBLE_DEVICES=$TG PYTORCH_ALLOC_CONF=expandable_segments:True \
      python environments/bash_agent/bash_agent.py --config "$TMP" \
      > "logs/trainer_${AB}_s${SEED}_${JID}.log" 2>&1
    echo "[$AB s$SEED] trainer done rc=$?"; kill $V 2>/dev/null
  ) &
}

for SEED in $SEEDS; do
  echo "=== seed $SEED wave ($(date +%H:%M:%S)) ==="
  launch mgda "$SEED" 0 1 8001
  launch ctrl "$SEED" 2 3 8002
  launch k5   "$SEED" 4 5 8003
  launch corr "$SEED" 6 7 8004
  wait
  echo "=== seed $SEED wave done ==="
done
echo "[seedsweep] all seeds done: $SEEDS"
