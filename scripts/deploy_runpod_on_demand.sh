#!/usr/bin/env bash
# Create on-demand H100 pod (podFindAndDeployOnDemand). Terminates same-named pods first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=runpod_common.sh
source "${SCRIPT_DIR}/runpod_common.sh"

POD_NAME="${RUNPOD_POD_NAME:-modded-nanogpt-baseline}"
RUNPOD_KEY="$(runpod_api_key)"
if [[ -z "$RUNPOD_KEY" ]]; then
  echo "Set RUNPOD_API_KEY, add workspace .env, or keychain runpod-api-key" >&2
  exit 1
fi

export RUNPOD_POD_NAME="$POD_NAME"

echo "Terminating existing pods named ${POD_NAME}..."
runpod_graphql "$RUNPOD_KEY" '{"query":"query { myself { pods { id name podType } } }"}' | python3 -c "
import json, os, sys
name = os.environ['RUNPOD_POD_NAME']
for p in json.load(sys.stdin)['data']['myself']['pods']:
    if p.get('name') == name:
        print(p['id'])
" | while read -r old_id; do
  [[ -z "$old_id" ]] && continue
  echo "  terminate ${old_id}"
  runpod_graphql "$RUNPOD_KEY" "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${old_id}\\\"}) }\"}" >/dev/null
done

PUBKEY=$(cat "${RUNPOD_SSH_PUB:-$HOME/.ssh/id_ed25519.pub}")
BODY=$(python3 -c "
import json, os, sys
pub = sys.argv[1]
variables = {
    'input': {
        'cloudType': 'SECURE',
        'gpuCount': 1,
        'gpuTypeId': os.environ.get('RUNPOD_GPU_TYPE', 'NVIDIA H100 80GB HBM3'),
        'name': os.environ.get('RUNPOD_POD_NAME', 'modded-nanogpt-baseline'),
        'imageName': os.environ.get('RUNPOD_IMAGE', 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'),
        'volumeInGb': int(os.environ.get('RUNPOD_VOLUME_GB', '100')),
        'containerDiskInGb': int(os.environ.get('RUNPOD_CONTAINER_DISK_GB', '100')),
        'volumeMountPath': '/workspace',
        'minVcpuCount': 8,
        'minMemoryInGb': 80,
        'ports': '22/tcp',
        'startSsh': True,
        'supportPublicIp': True,
        'env': [{'key': 'SSH_PUBLIC_KEY', 'value': pub}],
    }
}
query = 'mutation(\$input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: \$input) { id desiredStatus podType costPerHr } }'
print(json.dumps({'query': query, 'variables': variables}))
" "$PUBKEY")

RESP=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_KEY}" \
  -H "Content-Type: application/json" \
  -d "$BODY")

NEW_ID=$(echo "$RESP" | python3 -c "
import json,sys
out=json.load(sys.stdin)
if out.get('errors'):
    print(json.dumps(out, indent=2), file=sys.stderr)
    sys.exit(1)
pod=out['data']['podFindAndDeployOnDemand']
print(pod['id'])
print(f\"Created on-demand pod {pod['id']} type={pod.get('podType')} status={pod.get('desiredStatus')} ~\${pod.get('costPerHr')}/hr\", file=sys.stderr)
")

echo "RUNPOD_POD_ID=${NEW_ID}"
echo "Export: export RUNPOD_POD_ID=${NEW_ID}"
