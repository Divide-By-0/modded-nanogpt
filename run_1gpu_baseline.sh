#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export WANDB_PROJECT="${WANDB_PROJECT:-modded-nanogpt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline-h100-1gpu}"
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  if command -v uuidgen >/dev/null 2>&1; then
    WANDB_RUN_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
  else
    WANDB_RUN_ID="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
  fi
  export WANDB_RUN_ID
fi

if [[ -z "${WANDB_API_KEY:-}" ]] && security find-generic-password -a "$USER" -s wandb-api-key -w &>/dev/null; then
  export WANDB_API_KEY="$(security find-generic-password -a "$USER" -s wandb-api-key -w)"
fi

# 1x H100: keep global batch 512 (8*64) via gradient accumulation.
torchrun --standalone --nproc_per_node=1 train_gpt2.py
