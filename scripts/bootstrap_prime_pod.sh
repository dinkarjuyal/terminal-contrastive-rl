#!/bin/bash
# One-shot bootstrap: pick cheapest available H100/A100, create a Prime Intellect
# pod, wait for SSH, clone the repo, install dependencies, and launch exp16 + exp17.
#
# Prereqs (locally):
#   - prime CLI authenticated:    prime login
#   - SSH key configured:         ~/.prime/config.json -> ssh_key_path
#   - github SSH access to:       github.com/dinkarjuyal/terminal-contrastive-rl
#
# Usage:
#   bash scripts/bootstrap_prime_pod.sh            # default: 1xH100_80GB, $1.99/hr cap
#   GPU=A100_80GB MAX_HRLY=1.50 bash scripts/bootstrap_prime_pod.sh
set -euo pipefail

GPU=${GPU:-RTX4090}          # Qwen2.5-1.5B + LoRA fits comfortably in 24 GB
GPU_COUNT=${GPU_COUNT:-2}    # 2 GPUs because weight sync needs separate physical GPUs
DISK=${DISK:-80}
POD_NAME=${POD_NAME:-tc-rl-exp16-17}
REPO_URL=${REPO_URL:-git@github.com:dinkarjuyal/terminal-contrastive-rl.git}

echo "[bootstrap] GPU=$GPU x$GPU_COUNT  pod_name=$POD_NAME"

# 1. Verify auth.
prime whoami >/dev/null 2>&1 || { echo "[bootstrap] prime CLI not authenticated. Run 'prime login' first."; exit 1; }

# 2. List availability + grab cheapest single-GPU offering.
echo "[bootstrap] Querying availability for $GPU..."
prime availability list --gpu-type "$GPU" --gpu-count "$GPU_COUNT" --output json > /tmp/prime_avail.json
SHORT_ID=$(python3 -c "
import json, sys
data = json.load(open('/tmp/prime_avail.json'))
# data shape varies; try common keys
rows = data if isinstance(data, list) else data.get('results', data.get('availability', []))
if not rows:
    print('NO_AVAILABILITY', file=sys.stderr); sys.exit(1)
# pick cheapest price
def price(r):
    return r.get('price_per_hour') or r.get('hourly_price') or r.get('price') or float('inf')
rows.sort(key=price)
print(rows[0].get('short_id') or rows[0].get('id') or rows[0].get('cloudId'))
")
echo "[bootstrap] Cheapest offering: short_id=$SHORT_ID"

# 3. Create the pod (non-interactive).
echo "[bootstrap] Creating pod..."
prime pods create --id "$SHORT_ID" --name "$POD_NAME" --disk-size "$DISK" --yes

# 4. Poll until SSH is available.
echo "[bootstrap] Waiting for pod to come online..."
for i in $(seq 1 60); do
    sleep 15
    POD_INFO=$(prime pods list --output json 2>/dev/null || echo "[]")
    SSH_HOST=$(python3 -c "
import json
rows = json.loads('''$POD_INFO''') if '''$POD_INFO''' else []
for p in (rows if isinstance(rows, list) else rows.get('pods', [])):
    if p.get('name') == '$POD_NAME' and p.get('status') in ('ACTIVE','RUNNING','running','active'):
        print(p.get('ssh_connection') or p.get('ssh') or '')
        break
" 2>/dev/null || echo "")
    if [ -n "$SSH_HOST" ]; then
        echo "[bootstrap] Pod online: $SSH_HOST"
        break
    fi
    echo "[bootstrap]   (poll $i/60) not ready yet"
done

if [ -z "${SSH_HOST:-}" ]; then
    echo "[bootstrap] Pod failed to come online in 15 min. Inspect: prime pods list"
    exit 2
fi

# 5. Remote bootstrap: clone, install, launch.
REMOTE_SCRIPT=$(cat <<'REMOTE'
set -euo pipefail
cd ~
if [ ! -d terminal-contrastive-rl ]; then
    git clone git@github.com:dinkarjuyal/terminal-contrastive-rl.git || \
        git clone https://github.com/dinkarjuyal/terminal-contrastive-rl.git
fi
cd terminal-contrastive-rl
git pull --rebase origin main
# Install uv if missing
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.12 2>/dev/null || true
source .venv/bin/activate
uv pip install -e ".[all]" 2>&1 | tail -5
uv pip install flash-attn --no-build-isolation 2>&1 | tail -3 || echo "flash-attn install skipped"
# Rewrite launch scripts to use GPUs 0 (vf-vllm) and 1 (trainer), since pod has 2 GPUs.
sed -i 's/CUDA_VISIBLE_DEVICES=3 /CUDA_VISIBLE_DEVICES=0 /' scripts/launch_exp16.sh
sed -i 's/CUDA_VISIBLE_DEVICES=4 /CUDA_VISIBLE_DEVICES=1 /' scripts/launch_exp16.sh
sed -i 's/CUDA_VISIBLE_DEVICES=5 /CUDA_VISIBLE_DEVICES=0 /' scripts/launch_exp17.sh
sed -i 's/CUDA_VISIBLE_DEVICES=6 /CUDA_VISIBLE_DEVICES=1 /' scripts/launch_exp17.sh
# Update paths from nodeset3 layout to pod layout: WORK_DIR + PYTHON/VF_VLLM live in venv.
sed -i 's|/home/ubuntu/rl/verifiers|'"$HOME/terminal-contrastive-rl"'|g' scripts/launch_exp16.sh scripts/launch_exp17.sh
sed -i 's|/home/ubuntu/miniconda3/envs/vllm/bin/python|'"$HOME/terminal-contrastive-rl/.venv/bin/python"'|g' scripts/launch_exp16.sh scripts/launch_exp17.sh
sed -i 's|/home/ubuntu/miniconda3/envs/vllm/bin/vf-vllm|'"$HOME/terminal-contrastive-rl/.venv/bin/vf-vllm"'|g' scripts/launch_exp16.sh scripts/launch_exp17.sh
# Launch exp16 first (exp17 will be triggered after exp16 finishes via the orchestrator below).
tmux new-session -d -s exp16-launcher "bash scripts/launch_exp16.sh 2>&1 | tee /tmp/exp16_bootstrap.log"
# Sequential orchestrator: waits for exp16 trainer tmux session to exit, then launches exp17.
tmux new-session -d -s seq-orchestrator "while tmux has-session -t bash-agent-tc-exp16 2>/dev/null; do sleep 60; done; echo 'exp16 done, launching exp17'; bash $HOME/terminal-contrastive-rl/scripts/launch_exp17.sh 2>&1 | tee /tmp/exp17_bootstrap.log"
tmux ls
REMOTE
)

echo "[bootstrap] Remote bootstrap on pod..."
ssh -o StrictHostKeyChecking=no $SSH_HOST "$REMOTE_SCRIPT"

echo "[bootstrap] Done. To attach:"
echo "  prime pods ssh $POD_NAME"
echo "  tmux attach -t bash-agent-tc-exp16"
echo "  tmux attach -t bash-agent-tc-exp17"
