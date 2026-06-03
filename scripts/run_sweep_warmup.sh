#!/usr/bin/env bash
# Warmup length mini-sweep at 50 steps: 0 (baseline) vs 32 vs 64 linear-warmup steps.
# Interpretation note: warmup64 is still mid-ramp at step 50 (LR only 50/64 of full), so its
# 50-step loss is sandbagged and not comparable. warmup32 FINISHES warming up by step 32 and
# then trains 18 steps at full LR, so its step-50 number is a fair reading. baseline (0) is the
# same-seed reference so the comparison is noise-controlled within one session.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
TRAIN_SEED="${TRAIN_SEED:-1337}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-360}"
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-15}"
SWEEP_TAG_SUFFIX="${SWEEP_TAG_SUFFIX:-warm}"

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

run_variant warmup0  "No warmup (baseline reference)"              WARMUP_ITERS=0
run_variant warmup32 "32-step linear warmup (done by step 32)"     WARMUP_ITERS=32
run_variant warmup64 "64-step linear warmup (still ramping at 50)" WARMUP_ITERS=64

echo ""
echo "=== warmup sweep summary (${SUMMARY}) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2,2g; } | column -t -s$'\t' 2>/dev/null || cat "$SUMMARY"
python3 scripts/ab_dashboard.py || true
