#!/bin/bash
#SBATCH --job-name=tcr_evalsweepenv
#SBATCH --partition=a3mega
#SBATCH --nodelist=nucla3m-a3meganodeset-7
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/terminal-contrastive-rl/logs/slurm-%x-%j.out
# Env-generic best@k eval sweep. MANIFEST lines: "<name> <env_id> <ckpt_or_->".
# 8 evals/wave; each serves base(+LoRA) on one GPU and runs eval_best_at_k_env.
# Output results/bestk_<name>_nodeset.json. Submit with --export=ALL,HOME=/home/ubuntu.
set -uo pipefail
cd "$HOME/terminal-contrastive-rl"; mkdir -p logs results
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD:$PWD/environments"
source .venv/bin/activate
BASE="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MANIFEST="${MANIFEST:?set MANIFEST}"; N="${N:-16}"; NUMEX="${NUMEX:-100}"; [ "${SMOKE:-0}" = "1" ] && { N=2; NUMEX=10; }
mapfile -t LINES < <(grep -vE '^\s*$|^#' "$MANIFEST")

run_one() {  # name env_id ckpt gpu port
  local NAME="$1" ENVID="$2" CKPT="$3" GPU="$4" PORT="$5" EXTRA="" REQ
  if [ "$CKPT" = "-" ]; then REQ="$BASE"; else
    [ -d "$CKPT" ] || { echo "[$NAME] MISSING $CKPT"; return; }
    EXTRA="--enable-lora --max-lora-rank 16 --lora-modules ${NAME}=${CKPT}"; REQ="$NAME"
  fi
  (
    CUDA_VISIBLE_DEVICES=$GPU vf-vllm --model "$BASE" --enforce-eager --port "$PORT" \
      --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes $EXTRA \
      > "logs/evalvllm_${NAME}.log" 2>&1 &
    local V=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
      sleep 5; kill -0 $V 2>/dev/null || { echo "[$NAME] vLLM died"; exit 1; }
    done
    python scripts/eval_best_at_k_env.py --env-id "$ENVID" --base-url "http://localhost:$PORT/v1" \
      --model "$REQ" --n "$N" --num-examples "$NUMEX" --out "results/bestk_${NAME}_nodeset.json" \
      > "logs/eval_${NAME}.log" 2>&1
    echo "[$NAME] eval rc=$?"; kill $V 2>/dev/null
  ) &
}

GPUS=(0 1 2 3 4 5 6 7); M=${#LINES[@]}; i=0
echo "[evalsweep_env] $M evals, 8/wave"
while [ $i -lt $M ]; do
  for p in $(seq 0 7); do
    idx=$(( i + p )); [ $idx -ge $M ] && break
    set -- ${LINES[$idx]}; run_one "$1" "$2" "$3" "${GPUS[$p]}" "$(( 8001 + p ))"
  done
  wait; echo "[evalsweep_env] wave through $((i+8)) done"; i=$(( i + 8 ))
done
echo "[evalsweep_env] ALL DONE"
