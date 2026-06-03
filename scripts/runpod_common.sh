#!/usr/bin/env bash
# Shared RunPod helpers (source from other scripts; do not execute directly).
set -euo pipefail

# Repo root: modded-nanogpt/
RUNPOD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Parent workspace (nanochat-stewy/) when present
RUNPOD_WORKSPACE_ROOT="$(cd "${RUNPOD_REPO_ROOT}/.." && pwd)"

# Default on-demand; set RUNPOD_SPOT=1 to use interruptible spot + podBidResume
RUNPOD_SPOT="${RUNPOD_SPOT:-0}"

load_runpod_env() {
  local env_file="${RUNPOD_ENV_FILE:-${RUNPOD_WORKSPACE_ROOT}/.env}"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

runpod_api_key() {
  load_runpod_env
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    printf '%s' "$RUNPOD_API_KEY"
    return 0
  fi
  security find-generic-password -a "$USER" -s runpod-api-key -w 2>/dev/null || true
}

wandb_api_key() {
  load_runpod_env
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    printf '%s' "$WANDB_API_KEY"
    return 0
  fi
  security find-generic-password -a "$USER" -s wandb-api-key -w 2>/dev/null || true
}

runpod_graphql() {
  local key="$1"
  shift
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${key}" \
    -H "Content-Type: application/json" \
    -d "$1"
}

runpod_ssh_from_response() {
  python3 -c "
import json,sys
d=json.load(sys.stdin)['data']['pod']
for p in (d.get('runtime') or {}).get('ports') or []:
    if p.get('privatePort')==22:
        print(f'export IP={p[\"ip\"]} PORT={p[\"publicPort\"]}')
        sys.exit(0)
sys.exit(1)
"
}

runpod_ensure_running() {
  local key="$1" pod_id="$2" bid="${3:-5.0}"
  local resp status
  resp=$(runpod_graphql "$key" "{\"query\":\"query { pod(input: {podId: \\\"${pod_id}\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort } } } }\"}")
  status=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['pod']['desiredStatus'])")

  if [[ "$status" == "EXITED" ]]; then
    if [[ "$RUNPOD_SPOT" == "1" ]]; then
      echo "Resuming spot pod ${pod_id} (bid \$${bid}/GPU)..."
      runpod_graphql "$key" "{\"query\":\"mutation { podBidResume(input: { podId: \\\"${pod_id}\\\", bidPerGpu: ${bid}, gpuCount: 1 }) { id desiredStatus } }\"}" >/dev/null
    else
      echo "On-demand pod ${pod_id} is EXITED; starting with podResume..."
      runpod_graphql "$key" "{\"query\":\"mutation { podResume(input: { podId: \\\"${pod_id}\\\" }) { id desiredStatus } }\"}" >/dev/null
    fi
  fi
  printf '%s' "$resp"
}

sync_runpod_env_file() {
  local ip="$1" port="$2" ssh_key="$3"
  local env_file="${RUNPOD_ENV_FILE:-${RUNPOD_WORKSPACE_ROOT}/.env}"
  if [[ ! -f "$env_file" ]]; then
    return 0
  fi
  scp -q -o StrictHostKeyChecking=no -i "$ssh_key" -P "$port" \
    "$env_file" "root@${ip}:/workspace/.env"
  ssh -o StrictHostKeyChecking=no -i "$ssh_key" -p "$port" "root@${ip}" \
    'chmod 600 /workspace/.env'
}
