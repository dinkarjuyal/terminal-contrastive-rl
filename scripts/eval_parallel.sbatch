#!/bin/bash
#SBATCH --job-name=tcr_eval
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# best@k eval of base + each trained LoRA policy, parallel across node 7's GPUs.
# Each policy: one vLLM (base + its LoRA adapter) on its own GPU, then eval_best_at_k.
# SMOKE=1 -> n=2 and only base+mgda (validate the LoRA-serving eval path first).
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="Qwen/Qwen2.5-1.5B-Instruct"
SMOKE="${SMOKE:-0}"; N=16; [ "$SMOKE" = "1" ] && N=2

# name  gpu  port  adapter_path("-" = base model, no LoRA)
SPECS=(
  "base   0 8001 -"
  "mgda   1 8002 outputs/bash-agent-tc-exp19a-mgda/checkpoint-200"
  "ctrl   2 8003 outputs/bash-agent-tc-exp19-ctrl-v2k3/checkpoint-200"
  "k5     3 8004 outputs/bash-agent-tc-exp19b/checkpoint-200"
  "corr   4 8005 outputs/bash-agent-tc-exp19d-corrected/checkpoint-200"
)
[ "$SMOKE" = "1" ] && SPECS=("base 0 8001 -" "mgda 1 8002 outputs/bash-agent-tc-exp19a-mgda/checkpoint-200")

serve_and_eval() {
  local NAME="$1" GPU="$2" PORT="$3" ADAPTER="$4"
  local MODELARG REQNAME EXTRA=""
  if [ "$ADAPTER" = "-" ]; then
    REQNAME="$BASE"
  else
    EXTRA="--enable-lora --max-lora-rank 16 --lora-modules ${NAME}=${ADAPTER}"
    REQNAME="$NAME"
  fi
  (
    CUDA_VISIBLE_DEVICES=$GPU vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes $EXTRA \
      > "logs/evalvllm_${NAME}.log" 2>&1 &
    local VPID=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $VPID 2>/dev/null || { echo "[$NAME] vLLM died"; exit 1; }
    done
    echo "[$NAME] serving on :$PORT (model=$REQNAME); evaluating best@$N"
    python scripts/eval_best_at_k.py --base-url "http://localhost:$PORT/v1" \
      --model "$REQNAME" --n "$N" --out "results/bestk_${NAME}_nodeset.json" \
      > "logs/eval_${NAME}.log" 2>&1
    echo "[$NAME] eval done rc=$?"
    kill $VPID 2>/dev/null
  ) &
}

for spec in "${SPECS[@]}"; do
  serve_and_eval $spec
done
wait
echo "[eval] all done"
for spec in "${SPECS[@]}"; do
  set -- $spec
  echo "=== $1 ==="; grep -E "pass@|slope" "logs/eval_$1.log" 2>/dev/null | tail -5
done
