#!/usr/bin/env bash
# Phase 2: full 5-minute (wall-clock) runs for the finalists that won the 50-step
# sweep (scripts/run_sweep_50step.sh), always alongside a fresh baseline for a clean
# same-seed comparison. Reuses MAX_TRAIN_SECONDS (not MAX_TRAIN_STEPS) so each variant
# trains as far as the GPU allows in the budget; we compare final val_loss + step_avg_ms.
#
# Finalists are passed as positional args of the form:  TAG|DESC|ENV1=v ENV2=v
# e.g.  ./scripts/run_sweep_5min.sh "lr0048|LR 0.0048|LEARNING_RATE=0.0048" \
#                                   "warmup64|64-step warmup|WARMUP_ITERS=64"
# Baseline is always run first unless RUN_BASELINE=0.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-300}"
TRAIN_SEED="${TRAIN_SEED:-1337}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-25}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-420}"   # 300s train + compile + eval headroom
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-20}"
SWEEP_TAG_SUFFIX="${SWEEP_TAG_SUFFIX:-5m}"
RUN_BASELINE="${RUN_BASELINE:-1}"

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
      MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS}" \
      MAX_TRAIN_STEPS=0 \
      TRAIN_SEED="${TRAIN_SEED}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      AB_TAG="${full_tag}" \
      EXPERIMENT_DESC="${desc}" \
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

if [[ "$RUN_BASELINE" == "1" ]]; then
  run_variant baseline "Default: Muon + AdamW head, LR 0.0036, no warmup, QK norm before RoPE"
fi

# Parse finalist specs: TAG|DESC|ENV...
for spec in "$@"; do
  IFS='|' read -r f_tag f_desc f_env <<< "$spec"
  # shellcheck disable=SC2086
  run_variant "$f_tag" "$f_desc" $f_env
done

echo ""
echo "=== 5-min sweep summary (${SUMMARY}) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2,2g; } | column -t -s$'\t'
python3 scripts/ab_dashboard.py || true
