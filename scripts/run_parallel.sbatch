#!/bin/bash
#SBATCH --job-name=tcr_par
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Partition a3mega is OverSubscribe=EXCLUSIVE (whole node per job), so to use all
# 8 GPUs of node 7 in parallel we run ONE job that launches 4 experiments, each on
# its own GPU pair (vLLM + trainer). SMOKE=1 -> 6 steps each.
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"
mkdir -p logs
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
SMOKE="${SMOKE:-0}"
JID="$SLURM_JOB_ID"

launch() {  # <config> <suffix> <vllm_gpu> <trainer_gpu> <port>
  local CONFIG="$1" SUF="$2" VG="$3" TG="$4" PORT="$5"
  local NCCL=$(( PORT + 1000 ))
  local TMPCFG="/tmp/${SUF}_${JID}.toml"
  sed -e "s/^vllm_server_port = .*/vllm_server_port = $PORT/" \
      -e "s/^nccl_group_port = .*/nccl_group_port = $NCCL/" "$CONFIG" > "$TMPCFG"
  [ "$SMOKE" = "1" ] && sed -i -e "s/^max_steps = .*/max_steps = 6/" -e "s/^save_steps = .*/save_steps = 6/" "$TMPCFG"
  (
    echo "[$SUF] vLLM=GPU$VG:$PORT trainer=GPU$TG nccl=$NCCL"
    CUDA_VISIBLE_DEVICES=$VG vf-vllm --model Qwen/Qwen2.5-1.5B-Instruct --enforce-eager \
      --port "$PORT" --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
      > "logs/vllm_${SUF}_${JID}.log" 2>&1 &
    local VPID=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $VPID 2>/dev/null || { echo "[$SUF] vLLM died"; exit 1; }
    done
    echo "[$SUF] vLLM ready"
    CUDA_VISIBLE_DEVICES=$TG PYTORCH_ALLOC_CONF=expandable_segments:True \
      python environments/bash_agent/bash_agent.py --config "$TMPCFG" \
      > "logs/trainer_${SUF}_${JID}.log" 2>&1
    echo "[$SUF] trainer done rc=$?"
    kill $VPID 2>/dev/null
  ) &
}

launch configs/rl/bash_agent_tc_exp19a_mgda.toml      exp19a 0 1 8001
launch configs/rl/bash_agent_tc_exp19_ctrl_v2k3.toml  ctrl   2 3 8002
launch configs/rl/bash_agent_tc_exp19b.toml           k5     4 5 8003
launch configs/rl/bash_agent_tc_exp19d_corrected.toml corr   6 7 8004
wait
echo "[parallel] all 4 experiments finished"
