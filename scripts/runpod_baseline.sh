#!/usr/bin/env bash
# Sync code and start/restart baseline training on RunPod (on-demand by default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=runpod_common.sh
source "${SCRIPT_DIR}/runpod_common.sh"

POD_ID="${RUNPOD_POD_ID:-so7xcl5men3ywg}"
BID="${RUNPOD_BID_PER_GPU:-5.0}"
TMUX_SESSION="${RUNPOD_TMUX_SESSION:-baseline}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"

RUNPOD_KEY="$(runpod_api_key)"
WANDB_KEY="$(wandb_api_key)"

if [[ -z "$RUNPOD_KEY" ]]; then
  echo "Missing RunPod API key (workspace .env or keychain: runpod-api-key)" >&2
  exit 1
fi

RESP=$(runpod_ensure_running "$RUNPOD_KEY" "$POD_ID" "$BID")

for _ in $(seq 1 30); do
  if eval "$(echo "$RESP" | runpod_ssh_from_response 2>/dev/null)"; then
    break
  fi
  sleep 10
  RESP=$(runpod_graphql "$RUNPOD_KEY" "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort } } } }\"}")
done

if [[ -z "${IP:-}" || -z "${PORT:-}" ]]; then
  echo "SSH not ready; pod may still be provisioning. Retry later." >&2
  exit 1
fi

echo "SSH: ssh -i ${SSH_KEY} -p ${PORT} root@${IP}"

RUNPOD_POD_ID="$POD_ID" "${SCRIPT_DIR}/sync_to_runpod.sh" || true

REMOTE_START=$(cat <<'EOS'
set -euo pipefail
if [[ -f /workspace/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/.env
  set +a
fi
cd /workspace/modded-nanogpt
export PATH="/usr/local/bin:$PATH"
PY=/usr/local/bin/python
if ! $PY -c 'import wandb' 2>/dev/null; then
  $PY -m pip install --break-system-packages -q -r requirements.txt
fi
if [[ ! -f data/fineweb10B/fineweb_train_*.bin ]]; then
  echo "Downloading fineweb10B shard (27 = ~2.7B tokens)..."
  $PY data/cached_fineweb10B.py 27
fi
mkdir -p logs
if tmux has-session -t baseline 2>/dev/null; then
  tmux kill-session -t baseline
fi
tmux new-session -d -s baseline "export WANDB_PROJECT=modded-nanogpt WANDB_RUN_NAME=baseline-h100-1gpu; \
  ./run_1gpu_baseline.sh 2>&1 | tee logs/baseline-h100-1gpu.log"
echo "tmux session baseline started"
EOS
)

ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$PORT" "root@${IP}" \
  "export WANDB_API_KEY='${WANDB_KEY}'; ${REMOTE_START}"

echo "Done. Attach: ssh -i ${SSH_KEY} -p ${PORT} root@${IP} -t tmux attach -t ${TMUX_SESSION}"
