#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export WANDB_PROJECT="${WANDB_PROJECT:-modded-nanogpt}"
# REASON: wandb writes its run-data dir to $WANDB_DIR/wandb (default: CWD -> ./wandb).
# A ./wandb dir in the repo root is dangerous here: if wandb is ever NOT pip-installed,
# `import wandb` silently resolves to that empty dir as a PEP-420 namespace package
# (wandb.__file__ is None, no .login) instead of erroring -- which both crashes runs at
# wandb.login() and tricks `if ! python -c 'import wandb'` install-guards into skipping
# the install. Park run-data outside the repo so the repo dir never shadows the package.
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-modded}"
mkdir -p "$WANDB_DIR"
export AB_TAG="${AB_TAG:-baseline}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-120}"
export TRAIN_SEED="${TRAIN_SEED:-1337}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${AB_TAG}-h100-1gpu}"
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
# More frequent val than train_gpt2.py default (125); use VAL_LOSS_EVERY=125 for leaderboard-comparable logging.
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-25}"

torchrun --standalone --nproc_per_node=1 train_gpt2.py
