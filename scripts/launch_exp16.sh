#!/bin/bash
# Experiment 16: DBPO (Density-Bootstrap Policy Optimization, §4c).
# KDE log-density reward replaces V1's mean-sim. GPU 3 = vf-vllm, GPU 4 = trainer.
set -euo pipefail

SESSION="bash-agent-tc-exp16"
WORK_DIR="/home/ubuntu/rl/verifiers"
PYTHON="/home/ubuntu/miniconda3/envs/vllm/bin/python"
VF_VLLM="/home/ubuntu/miniconda3/envs/vllm/bin/vf-vllm"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONFIG="configs/rl/bash_agent_tc_exp16.toml"

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

tmux new-session -d -s "$SESSION" -c "$WORK_DIR" bash

VLLM_CMD="CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $VF_VLLM --model $MODEL --enforce-eager --port 8002 \
  --gpu-memory-utilization 0.4 --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee /tmp/vllm_exp16.log"
tmux send-keys -t "$SESSION:0.0" "$VLLM_CMD" C-m

tmux split-window -v -t "$SESSION:0" -c "$WORK_DIR"
TRAIN_CMD="until curl -sf http://localhost:8002/health > /dev/null 2>&1; do echo 'waiting for vllm...'; sleep 5; done
echo 'vf-vllm ready on port 8002'
CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $PYTHON environments/bash_agent/bash_agent.py --config $CONFIG \
  2>&1 | tee /tmp/trainer_exp16.log"
tmux send-keys -t "$SESSION:0.1" "$TRAIN_CMD" C-m

echo "Session '$SESSION' launched. vf-vllm=GPU3:8002, trainer=GPU4."
