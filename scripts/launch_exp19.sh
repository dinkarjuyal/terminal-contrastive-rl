#!/bin/bash
# Generic Exp19 launcher for a 2-GPU Prime pod (uv .venv layout).
# vf-vllm on GPU0:8000, trainer on GPU1. Pass the config as $1.
#   bash scripts/launch_exp19.sh configs/rl/bash_agent_tc_exp19b.toml [session_suffix]
set -euo pipefail

CONFIG="${1:-configs/rl/bash_agent_tc_exp19b.toml}"
SUFFIX="${2:-$(basename "$CONFIG" .toml)}"
SESSION="exp19-${SUFFIX}"
WORK_DIR="$HOME/terminal-contrastive-rl"
PY="$WORK_DIR/.venv/bin/python"
VF_VLLM="$WORK_DIR/.venv/bin/vf-vllm"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2
tmux new-session -d -s "$SESSION" -c "$WORK_DIR" bash

VLLM_CMD="CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $VF_VLLM --model $MODEL --enforce-eager --port 8000 \
  --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee /tmp/vllm_${SUFFIX}.log"
tmux send-keys -t "$SESSION:0.0" "$VLLM_CMD" C-m

tmux split-window -v -t "$SESSION:0" -c "$WORK_DIR"
TRAIN_CMD=": > /tmp/trainer_${SUFFIX}.log
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do echo 'waiting for vllm...'; sleep 5; done
echo 'vf-vllm ready on :8000'
CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $PY environments/bash_agent/bash_agent.py --config $CONFIG \
  2>&1 | tee /tmp/trainer_${SUFFIX}.log"
tmux send-keys -t "$SESSION:0.1" "$TRAIN_CMD" C-m

echo "Session '$SESSION' launched. vf-vllm=GPU0:8000, trainer=GPU1, config=$CONFIG"
