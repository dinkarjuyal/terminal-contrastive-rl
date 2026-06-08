#!/bin/bash
#SBATCH --job-name=tcrl
#SBATCH --partition=a3mega
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Usage: sbatch --job-name=<suf> -w <node> run.sbatch <config.toml> <suffix>
# Terminal-contrastive owns nodes 4-7; pass -w nucla3m-a3meganodeset-{4,5,6,7}.
# Multiple 2-GPU jobs co-reside on one node (8 GPUs -> 4 jobs/node); ports are
# auto-derived from SLURM_JOB_ID so they don't collide.
# Runs vLLM (1 GPU) + GRPO trainer (1 GPU) on the 2 GPUs Slurm allocates.
# Jobs run under slurmd, so they survive SSH disconnects (unlike nohup/tmux here).
set -uo pipefail

CONFIG="${1:?usage: sbatch run.sbatch <config> <suffix>}"
SUF="${2:?usage: sbatch run.sbatch <config> <suffix>}"
WORK="$HOME/terminal-contrastive-rl"
cd "$WORK"
mkdir -p logs
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$WORK:$WORK/environments"
source .venv/bin/activate

# Slurm puts the 2 allocated GPU ids in CUDA_VISIBLE_DEVICES (e.g. "2,3").
# Split: first -> vLLM, second -> trainer. Derive unique ports from the job id
# so multiple jobs can co-reside on node 7.
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
VG="${GPUS[0]}"; TG="${GPUS[1]:-${GPUS[0]}}"
unset CUDA_VISIBLE_DEVICES
PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
NCCL_PORT=$(( 51000 + (SLURM_JOB_ID % 900) ))

echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) vLLM=GPU$VG:$PORT trainer=GPU$TG nccl=$NCCL_PORT config=$CONFIG"

# Override the config's ports at launch via env the trainer/vllm read? The config
# hardcodes vllm_server_port; we instead pass the port through a temp config copy.
TMPCFG="/tmp/${SUF}_${SLURM_JOB_ID}.toml"
sed -e "s/^vllm_server_port = .*/vllm_server_port = $PORT/" \
    -e "s/^nccl_group_port = .*/nccl_group_port = $NCCL_PORT/" "$CONFIG" > "$TMPCFG"
# Smoke mode (rule: verify small before scaling): SMOKE=1 -> 6 steps only.
if [ "${SMOKE:-0}" = "1" ]; then
  sed -i -e "s/^max_steps = .*/max_steps = 6/" -e "s/^save_steps = .*/save_steps = 6/" "$TMPCFG"
  echo "[sbatch] SMOKE mode: max_steps=6"
fi

CUDA_VISIBLE_DEVICES=$VG vf-vllm --model Qwen/Qwen2.5-1.5B-Instruct --enforce-eager \
  --port "$PORT" --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
  > "logs/vllm_${SUF}_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  sleep 5
  kill -0 $VLLM_PID 2>/dev/null || { echo "[sbatch] vLLM died"; exit 1; }
done
echo "[sbatch] vLLM ready on :$PORT"

CUDA_VISIBLE_DEVICES=$TG PYTORCH_ALLOC_CONF=expandable_segments:True \
  python environments/bash_agent/bash_agent.py --config "$TMPCFG"
echo "[sbatch] trainer finished rc=$?"
