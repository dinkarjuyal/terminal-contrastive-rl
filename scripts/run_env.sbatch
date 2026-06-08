#!/bin/bash
#SBATCH --job-name=tcr_env
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Generic single-run launcher (any verifiers env via scripts/train_generic.py).
# vLLM on one GPU + trainer on the other of the 2 Slurm-allocated GPUs. Node 7 only.
#   sbatch --job-name=<suf> run_env.sbatch <config.toml> <suffix>
# SMOKE=1 -> max_steps=6.
set -uo pipefail
CONFIG="${1:?config}"; SUF="${2:?suffix}"
WORK="$HOME/terminal-contrastive-rl"; cd "$WORK"; mkdir -p logs
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$WORK:$WORK/environments"
source .venv/bin/activate
BASE="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
IFS=',' read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1}"; VG="${G[0]}"; TG="${G[1]:-${G[0]}}"; unset CUDA_VISIBLE_DEVICES
PORT=$(( 8000 + SLURM_JOB_ID % 1000 )); NCCL=$(( 51000 + SLURM_JOB_ID % 900 ))
TMP="/tmp/${SUF}_${SLURM_JOB_ID}.toml"
sed -e "s/^vllm_server_port = .*/vllm_server_port = $PORT/" \
    -e "s/^nccl_group_port = .*/nccl_group_port = $NCCL/" "$CONFIG" > "$TMP"
[ "${SMOKE:-0}" = "1" ] && sed -i "s/^max_steps = .*/max_steps = 6/" "$TMP"
echo "[run_env] job=$SLURM_JOB_ID vLLM=GPU$VG:$PORT trainer=GPU$TG config=$CONFIG"

CUDA_VISIBLE_DEVICES=$VG vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
  --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
  > "logs/vllm_${SUF}_${SLURM_JOB_ID}.log" 2>&1 &
VPID=$!; trap "kill $VPID 2>/dev/null" EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  sleep 5; kill -0 $VPID 2>/dev/null || { echo "[run_env] vLLM died"; exit 1; }
done
echo "[run_env] vLLM ready"
CUDA_VISIBLE_DEVICES=$TG PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/train_generic.py --config "$TMP"
echo "[run_env] trainer rc=$?"
