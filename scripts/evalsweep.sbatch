#!/bin/bash
#SBATCH --job-name=tcr_evalsweep
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Generalized best@k eval sweep. MANIFEST file: lines "<name> <ckpt_or_-> ".
# ckpt "-" = base model (no LoRA). 8 evals parallel per wave; outputs
# results/bestk_<name>_nodeset.json (with per_task_correct for #4 ensembling).
# Env: MANIFEST=path  N=16  SMOKE=1(->N=2)
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="Qwen/Qwen2.5-1.5B-Instruct"
MANIFEST="${MANIFEST:?set MANIFEST}"; N="${N:-16}"; [ "${SMOKE:-0}" = "1" ] && N=2

mapfile -t LINES < <(grep -vE '^\s*$|^#' "$MANIFEST")

run_one() {  # name ckpt gpu port
  local NAME="$1" CKPT="$2" GPU="$3" PORT="$4" EXTRA="" REQ
  if [ "$CKPT" = "-" ]; then REQ="$BASE"; else EXTRA="--enable-lora --max-lora-rank 16 --lora-modules ${NAME}=${CKPT}"; REQ="$NAME"; fi
  if [ "$CKPT" != "-" ] && [ ! -d "$CKPT" ]; then echo "[$NAME] MISSING $CKPT"; return; fi
  (
    CUDA_VISIBLE_DEVICES=$GPU vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes $EXTRA \
      > "logs/evalvllm_${NAME}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$NAME] vLLM died"; exit 1; }
    done
    python scripts/eval_best_at_k.py --base-url "http://localhost:$PORT/v1" \
      --model "$REQ" --n "$N" --out "results/bestk_${NAME}_nodeset.json" \
      > "logs/eval_${NAME}.log" 2>&1
    echo "[$NAME] eval rc=$?"; kill $V 2>/dev/null
  ) &
}

GPUS=(0 1 2 3 4 5 6 7); M=${#LINES[@]}; i=0
echo "[evalsweep] $M evals, 8 per wave"
while [ $i -lt $M ]; do
  for p in $(seq 0 7); do
    idx=$(( i + p )); [ $idx -ge $M ] && break
    set -- ${LINES[$idx]}
    run_one "$1" "$2" "${GPUS[$p]}" "$(( 8001 + p ))"
  done
  wait
  echo "[evalsweep] wave through $((i+8)) done"
  i=$(( i + 8 ))
done
echo "[evalsweep] ALL DONE"
