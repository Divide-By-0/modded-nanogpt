#!/usr/bin/env bash
# Run a compact 1xH100 A/B matrix. Override MAX_TRAIN_SECONDS to fit the budget.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

# The Python-side limit exits cleanly; the shell timeout is the hard total cap.
COMMON_MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-120}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-145}"
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-10}"
COMMON_TRAIN_SEED="${TRAIN_SEED:-1337}"
COMMON_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-25}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_KERNEL_AUTOTUNE="${RUN_KERNEL_AUTOTUNE:-0}"

run_variant() {
  local tag="$1"
  local desc="$2"
  shift 2
  echo "=== ${tag}: ${desc} ==="
  set +e
  timeout --kill-after="${RUN_KILL_AFTER_SECONDS}s" "${RUN_TIMEOUT_SECONDS}s" \
    env \
    MAX_TRAIN_SECONDS="${COMMON_MAX_TRAIN_SECONDS}" \
    TRAIN_SEED="${COMMON_TRAIN_SEED}" \
    VAL_LOSS_EVERY="${COMMON_VAL_LOSS_EVERY}" \
    AB_TAG="${tag}" \
    EXPERIMENT_DESC="${desc}" \
    WANDB_RUN_NAME="${tag}-h100-1gpu" \
    "$@" \
    ./run_1gpu_baseline.sh 2>&1 | tee "logs/${tag}-h100-1gpu.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
    echo "=== ${tag}: hard timeout after ${RUN_TIMEOUT_SECONDS}s (+${RUN_KILL_AFTER_SECONDS}s kill window) ==="
  elif [[ "$status" -ne 0 ]]; then
    echo "=== ${tag}: exited with status ${status} ==="
  fi
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  run_variant baseline "Current default: Muon blocks + AdamW tied embedding/head; QK norm before RoPE"
fi
run_variant muon-steps3 "Muon Newton-Schulz backend steps 5 -> 3" MUON_BACKEND_STEPS=3
run_variant muon-mom98 "Muon momentum 0.95 -> 0.98" MUON_MOMENTUM=0.98
run_variant adamw-all "Replace Muon blocks with AdamW everywhere; LR 0.0036 -> 0.0018" OPTIMIZER_MODE=adamw_all LEARNING_RATE=0.0018
run_variant qk-after-rope "Move QK RMSNorm from before RoPE to after RoPE" QK_NORM_MODE=after_rope
run_variant embed-rmsnorm "Add RMSNorm immediately after token embedding lookup" EMBED_RMSNORM=1
if [[ "$RUN_KERNEL_AUTOTUNE" == "1" ]]; then
  run_variant triton-gemm-autotune "Try Inductor max-autotune with Triton GEMM kernels" \
    TORCH_COMPILE_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON
fi

python3 scripts/ab_dashboard.py || true
