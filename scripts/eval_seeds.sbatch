#!/bin/bash
#SBATCH --job-name=tcr_evalseed
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# best@k eval of the per-seed checkpoints (seeds 1,2 x 4 configs = 8), one GPU each,
# all parallel. Outputs results/bestk_<cfg>_s<seed>_nodeset.json for aggregate_seeds.py.
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="Qwen/Qwen2.5-1.5B-Instruct"
SMOKE="${SMOKE:-0}"; N=16; [ "$SMOKE" = "1" ] && N=2

# cfg seed gpu port
SPECS=(
  "mgda 1 0 8001" "ctrl 1 1 8002" "k5 1 2 8003" "corr 1 3 8004"
  "mgda 2 4 8005" "ctrl 2 5 8006" "k5 2 6 8007" "corr 2 7 8008"
)

run_one() {  # cfg seed gpu port
  local CFG="$1" SEED="$2" GPU="$3" PORT="$4"
  local CKPT="outputs/bash-agent-tc-${CFG}-s${SEED}/checkpoint-200"
  local NAME="${CFG}s${SEED}"
  if [ ! -d "$CKPT" ]; then echo "[$NAME] MISSING $CKPT — skip"; return; fi
  (
    CUDA_VISIBLE_DEVICES=$GPU vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
      --enable-lora --max-lora-rank 16 --lora-modules ${NAME}=${CKPT} \
      > "logs/evalvllm_${NAME}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$NAME] vLLM died"; exit 1; }
    done
    python scripts/eval_best_at_k.py --base-url "http://localhost:$PORT/v1" \
      --model "$NAME" --n "$N" --out "results/bestk_${CFG}_s${SEED}_nodeset.json" \
      > "logs/eval_${NAME}.log" 2>&1
    echo "[$NAME] eval done rc=$?"; kill $V 2>/dev/null
  ) &
}

for spec in "${SPECS[@]}"; do run_one $spec; done
wait
echo "[eval_seeds] all done"
