#!/bin/bash
# Run a config on a nodeset node: vLLM (one GPU) + GRPO trainer (another GPU),
# detached via nohup so it survives SSH disconnect. Sets CUDA_HOME (nvcc present).
#   bash scripts/run_nodeset.sh <config.toml> [suffix]
#   env overrides: VLLM_GPU (default 0) TRAIN_GPU (default 1) PORT (default 8000)
set -uo pipefail

CONFIG="${1:?usage: run_nodeset.sh <config> [suffix]}"
SUF="${2:-$(basename "$CONFIG" .toml)}"
WORK="$HOME/terminal-contrastive-rl"
VLLM_GPU="${VLLM_GPU:-0}"; TRAIN_GPU="${TRAIN_GPU:-1}"; PORT="${PORT:-8000}"

cd "$WORK"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$WORK:$WORK/environments"
source .venv/bin/activate

echo "[run] $SUF : vLLM=GPU$VLLM_GPU:$PORT trainer=GPU$TRAIN_GPU config=$CONFIG"
CUDA_VISIBLE_DEVICES=$VLLM_GPU nohup .venv/bin/vf-vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct --enforce-eager --port "$PORT" \
  --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
  > "/tmp/vllm_${SUF}.log" 2>&1 &
echo "[run] vllm pid $!"

until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  sleep 5
  if ! pgrep -f "vf-vllm.*port $PORT" >/dev/null; then echo "[run] vllm died; see /tmp/vllm_${SUF}.log"; exit 1; fi
done
echo "[run] vllm ready on :$PORT"

CUDA_VISIBLE_DEVICES=$TRAIN_GPU PYTORCH_ALLOC_CONF=expandable_segments:True nohup \
  python environments/bash_agent/bash_agent.py --config "$CONFIG" \
  > "/tmp/trainer_${SUF}.log" 2>&1 &
echo "[run] trainer pid $!  (log: /tmp/trainer_${SUF}.log)"
