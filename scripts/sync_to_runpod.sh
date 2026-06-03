#!/usr/bin/env bash
# Sync code to RunPod /workspace/modded-nanogpt (not training data; download on-pod).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=runpod_common.sh
source "${SCRIPT_DIR}/runpod_common.sh"

# On-demand default; RUNPOD_SPOT=1 uses interruptible spot + podBidResume
POD_ID="${RUNPOD_POD_ID:-so7xcl5men3ywg}"
SRC="${RUNPOD_REPO_ROOT}"
RUNPOD_KEY="$(runpod_api_key)"
BID="${RUNPOD_BID_PER_GPU:-5.0}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ -z "$RUNPOD_KEY" ]]; then
  echo "Set RUNPOD_API_KEY, workspace .env, or keychain runpod-api-key" >&2
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
  echo "SSH not ready yet; retry in a minute." >&2
  exit 1
fi

sync_runpod_env_file "$IP" "$PORT" "$SSH_KEY" || true

echo "Syncing $SRC -> root@$IP:$PORT:/workspace/modded-nanogpt/"
rsync -az --no-owner --no-group --omit-dir-times \
  -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY} -p $PORT" \
  --exclude '.git' \
  --exclude 'records' \
  --exclude 'logs' \
  --exclude 'wandb' \
  --exclude 'data/fineweb10B/*.bin' \
  --exclude 'data/fineweb10B/.cache' \
  "$SRC/" "root@${IP}:/workspace/modded-nanogpt/"
echo "rsync ok"
