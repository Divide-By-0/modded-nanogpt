#!/usr/bin/env bash
# Pull run RESULTS (logs + rebuilt A/B dashboard) DOWN from the pod to this Mac.
#
# Direction matters: sync_to_runpod.sh PUSHES source code mac->pod and deliberately
# EXCLUDES logs/, records/, wandb/ -- those are generated ON the pod, so pushing the
# Mac's (empty) copies would clobber the pod's real outputs. To view results we do the
# opposite: pull pod->mac here, then open logs/ab_dashboard.html locally.
#
# Uses the `runpod-nano` ssh alias (see ~/.ssh/config) so there are no option strings to
# mis-escape and all transfers reuse one multiplexed connection.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${RUNPOD_SSH_ALIAS:-runpod-nano}"
REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/workspace/modded-nanogpt}"

# Rebuild the dashboard on the pod from the latest logs before pulling it.
ssh "$HOST" "cd '$REMOTE_DIR' && python3 scripts/ab_dashboard.py || true"

mkdir -p logs
echo "Pulling logs/ from $HOST:$REMOTE_DIR ..."
# Exclude *.pt: model+optimizer checkpoints are ~1 GB each and not needed for the dashboard
# (which only parses the *.txt loss logs). Pulling them is slow and can fill the local disk.
rsync -az --no-owner --no-group --omit-dir-times \
  -e "ssh" --exclude '*.pt' \
  "$HOST:$REMOTE_DIR/logs/" ./logs/
echo "Pulled. Dashboard: $(pwd)/logs/ab_dashboard.html"

# Print the sweep summary tables if present.
for f in logs/sweep_s50_summary.tsv logs/sweep_5m_summary.tsv; do
  if [[ -f "$f" ]]; then
    echo ""; echo "=== $f ==="
    { head -1 "$f"; tail -n +2 "$f" | sort -t$'\t' -k2,2g; } | column -t -s$'\t' 2>/dev/null || cat "$f"
  fi
done
