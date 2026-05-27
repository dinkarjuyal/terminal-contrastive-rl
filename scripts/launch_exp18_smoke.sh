#!/bin/bash
# Smoke launcher for exp18 (max_steps=2) -- validates speed optimizations.
# Diff vs launch_exp16.sh: drops --enforce-eager, bumps gpu-mem-util 0.4 -> 0.7.
# Uses different port (8004) and NCCL group (51219) so it can run alongside
# nothing else; do NOT run concurrently with exp16/exp17 on the same pod.
set -euo pipefail

SESSION="bash-agent-tc-exp18-smoke"
WORK_DIR="/home/ubuntu/rl/verifiers"
PYTHON="/home/ubuntu/miniconda3/envs/vllm/bin/python"
VF_VLLM="/home/ubuntu/miniconda3/envs/vllm/bin/vf-vllm"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
CONFIG="configs/rl/bash_agent_tc_exp18_smoke.toml"

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

tmux new-session -d -s "$SESSION" -c "$WORK_DIR" bash

VLLM_CMD="CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $VF_VLLM --model $MODEL --port 8004 \
  --gpu-memory-utilization 0.7 --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee /tmp/vllm_exp18_smoke.log"
tmux send-keys -t "$SESSION:0.0" "$VLLM_CMD" C-m

tmux split-window -v -t "$SESSION:0" -c "$WORK_DIR"
TRAIN_CMD="until curl -sf http://localhost:8004/health > /dev/null 2>&1; do echo 'waiting for vllm...'; sleep 5; done
echo 'vf-vllm ready on port 8004'
CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$WORK_DIR:$WORK_DIR/environments \
  $PYTHON environments/bash_agent/bash_agent.py --config $CONFIG \
  2>&1 | tee /tmp/trainer_exp18_smoke.log"
tmux send-keys -t "$SESSION:0.1" "$TRAIN_CMD" C-m

echo "Session '$SESSION' launched. vf-vllm=GPU3:8004, trainer=GPU4."
echo "Expected runtime: ~3-4 min for max_steps=2."
echo "Compare wall_clock/generate_s vs exp16 step-1 (138s) to estimate speedup."
