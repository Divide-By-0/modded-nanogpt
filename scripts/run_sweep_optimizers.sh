#!/usr/bin/env bash
# Modern-optimizer 50-step sweep: vanilla Muon vs Muon2 (adaptive 2nd-moment
# preconditioning before orthogonalization) vs Aurora (leverage-aware orthogonalization).
#
# Aurora's published LR (~0.0375-0.05) is far above this repo's Muon LR convention
# (muon_lr_multiplier * learning_rate = 0.1 * 0.0036), and the post-NS scaling differs,
# so we DON'T assume one LR -- we sweep MUON_LR_MULTIPLIER for Aurora. muon2 reuses the
# tuned Muon LR (its preconditioner is roughly scale-preserving).
#
# Same harness as run_sweep_50step.sh (50 steps, seed 1337, labeled AB_TAG + W&B run).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
TRAIN_SEED="${TRAIN_SEED:-1337}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-420}"  # aurora compiles two polar loops -> a bit slower
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-15}"
SWEEP_TAG_SUFFIX="${SWEEP_TAG_SUFFIX:-opt}"

SUMMARY="logs/sweep_${SWEEP_TAG_SUFFIX}_summary.tsv"
printf 'tag\tfinal_val_loss\tstep_avg_ms\tstatus\tdesc\n' > "$SUMMARY"

run_variant() {
  local tag="$1"; local desc="$2"; shift 2
  local full_tag="${tag}-${SWEEP_TAG_SUFFIX}"
  local log="logs/${full_tag}-h100-1gpu.log"
  echo "=== ${full_tag}: ${desc} ==="
  set +e
  timeout --kill-after="${RUN_KILL_AFTER_SECONDS}s" "${RUN_TIMEOUT_SECONDS}s" \
    env \
      MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" MAX_TRAIN_SECONDS=0 \
      TRAIN_SEED="${TRAIN_SEED}" VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      AB_TAG="${full_tag}" EXPERIMENT_DESC="${desc}" \
      WANDB_RUN_NAME="${full_tag}-h100-1gpu" \
      "$@" \
      ./run_1gpu_baseline.sh 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  local val ms statestr
  val=$(grep -oE 'val_loss:[0-9.]+' "$log" | tail -1 | cut -d: -f2 || true)
  ms=$(grep -oE 'step_avg:[0-9.]+ms' "$log" | tail -1 | grep -oE '[0-9.]+' || true)
  if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then statestr="TIMEOUT"
  elif [[ "$status" -ne 0 ]]; then statestr="exit${status}"
  else statestr="ok"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$full_tag" "${val:-NA}" "${ms:-NA}" "$statestr" "$desc" >> "$SUMMARY"
}

# Reference: vanilla Muon (same as baseline) so the optimizer comparison is self-contained.
run_variant muon "Vanilla Muon (reference)" MUON_VARIANT=muon

# Muon2: adaptive second-moment preconditioning before the polar step.
run_variant muon2      "Muon2: Adam-style 2nd-moment preconditioning before orthogonalization" MUON_VARIANT=muon2
run_variant muon2b299  "Muon2 with beta2 0.95 -> 0.99 (slower 2nd-moment EMA)" MUON_VARIANT=muon2 MUON_BETA2=0.99

# Aurora: leverage-aware orthogonalization. Sweep the Muon LR multiplier because Aurora's
# effective step size / LR convention differs from vanilla Muon.
run_variant auroram01 "Aurora, muon_lr_mult 0.10 (same as Muon)"  MUON_VARIANT=aurora MUON_LR_MULTIPLIER=0.10
run_variant auroram03 "Aurora, muon_lr_mult 0.30"                 MUON_VARIANT=aurora MUON_LR_MULTIPLIER=0.30
run_variant auroram10 "Aurora, muon_lr_mult 1.00 (~paper LR)"     MUON_VARIANT=aurora MUON_LR_MULTIPLIER=1.00

echo ""
echo "=== optimizer sweep summary (${SUMMARY}) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2,2g; } | column -t -s$'\t' 2>/dev/null || cat "$SUMMARY"
python3 scripts/ab_dashboard.py || true
