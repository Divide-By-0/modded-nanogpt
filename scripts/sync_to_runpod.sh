#!/usr/bin/env bash
# Sync code to RunPod /workspace/modded-nanogpt (not training data; download on-pod).
set -euo pipefail

POD_ID="${RUNPOD_POD_ID:-37xe2dpyf7o933}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
RUNPOD_KEY="${RUNPOD_API_KEY:-$(security find-generic-password -a "$USER" -s runpod-api-key -w 2>/dev/null)}"
if [[ -z "$RUNPOD_KEY" ]]; then
  echo "Set RUNPOD_API_KEY or add keychain service runpod-api-key" >&2
  exit 1
fi

RESP=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort } } } }\"}")

STATUS=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['pod']['desiredStatus'])")
if [[ "$STATUS" == "EXITED" ]]; then
  echo "Pod exited; resuming spot instance..."
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podBidResume(input: { podId: \\\"${POD_ID}\\\", bidPerGpu: 4.5, gpuCount: 1 }) { id desiredStatus } }\"}" >/dev/null
  sleep 15
  RESP=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort } } } }\"}")
fi

eval "$(echo "$RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data']['pod']
for p in (d.get('runtime') or {}).get('ports') or []:
    if p.get('privatePort')==22:
        print(f'IP={p[\"ip\"]} PORT={p[\"publicPort\"]}')
        break
else:
    raise SystemExit('no SSH port yet')
")"

echo "Syncing $SRC -> root@$IP:$PORT:/workspace/modded-nanogpt/"
rsync -az --no-owner --no-group --omit-dir-times \
  -e "ssh -o StrictHostKeyChecking=no -i ${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519} -p $PORT" \
  --exclude '.git' \
  --exclude 'records' \
  --exclude 'logs' \
  --exclude 'data/fineweb10B/*.bin' \
  --exclude 'data/fineweb10B/.cache' \
  "$SRC/" "root@${IP}:/workspace/modded-nanogpt/"
echo "rsync ok"
