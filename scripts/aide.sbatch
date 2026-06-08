#!/bin/bash
#SBATCH --job-name=tcr_aide
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Idea: AIDE-style feedback-guided search (paper sec:remaining). Does sequential
# propose->execute->read-error->revise search recover performance that i.i.d.
# best@k says is lost? Run on base, the format-collapsed K5, and a healthy policy.
# Reports sequential solve@B vs i.i.d. pass@B. SMOKE=1 -> budget 2, base only.
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="Qwen/Qwen2.5-1.5B-Instruct"
SMOKE="${SMOKE:-0}"; B="${B:-8}"; [ "$SMOKE" = "1" ] && B=2

# name gpu port adapter(-=base)
SPECS=(
  "base    0 8001 -"
  "k5      1 8002 outputs/bash-agent-tc-exp19b/checkpoint-200"
  "healthy 2 8003 outputs/bash-agent-tc-corrmgda-s1/checkpoint-200"
)
[ "$SMOKE" = "1" ] && SPECS=("base 0 8001 -")

run_one() {  # name gpu port adapter
  local NAME="$1" GPU="$2" PORT="$3" ADP="$4" EXTRA="" REQ
  if [ "$ADP" = "-" ]; then REQ="$BASE"; else EXTRA="--enable-lora --max-lora-rank 16 --lora-modules ${NAME}=${ADP}"; REQ="$NAME"; fi
  (
    CUDA_VISIBLE_DEVICES=$GPU vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes $EXTRA \
      > "logs/aidevllm_${NAME}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$NAME] vLLM died"; exit 1; }
    done
    python scripts/aide_search.py --base-url "http://localhost:$PORT/v1" \
      --model "$REQ" --budget "$B" --out "results/aide_${NAME}.json" \
      > "logs/aide_${NAME}.log" 2>&1
    echo "[$NAME] aide rc=$?"; kill $V 2>/dev/null
  ) &
}
for spec in "${SPECS[@]}"; do run_one $spec; done
wait
echo "[aide] done"
for spec in "${SPECS[@]}"; do set -- $spec; echo "=== $1 ==="; grep -E "solve@B|pass@B|lift" "logs/aide_$1.log" 2>/dev/null | tail -4; done
