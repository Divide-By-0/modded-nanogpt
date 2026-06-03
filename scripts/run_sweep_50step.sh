#!/usr/bin/env bash
# Phase 1: cheap 50-step A/B sweep on 1xH100 to surface early-curve winners before
# spending wall-clock on full 5-minute runs. Each variant runs MAX_TRAIN_STEPS steps
# on a fixed seed; we capture final val_loss and per-step latency (step_avg_ms) into a
# summary table. Promote the best 2-3 (plus baseline) to scripts/run_sweep_5min.sh.
#
# Every variant gets a distinct AB_TAG + EXPERIMENT_DESC + WANDB_RUN_NAME so it shows up
# labeled-by-change in W&B and in logs/ab_dashboard.html (matches the codex labeling setup).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
TRAIN_SEED="${TRAIN_SEED:-1337}"
# Dense val so a 50-step curve still has several points; 0 wall cap (step cap governs).
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10}"
# Hard safety timeout per run (compile ~60-120s + 50 fast steps). 0 disables.
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-360}"
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-15}"
SWEEP_TAG_SUFFIX="${SWEEP_TAG_SUFFIX:-s50}"

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
      MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
      MAX_TRAIN_SECONDS=0 \
      TRAIN_SEED="${TRAIN_SEED}" \
      VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
      AB_TAG="${full_tag}" \
      EXPERIMENT_DESC="${desc}" \
      WANDB_RUN_NAME="${full_tag}-h100-1gpu" \
      "$@" \
      ./run_1gpu_baseline.sh 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e

  # Pull the last val_loss and step_avg line out of the run log for the summary table.
  local val ms statestr
  val=$(grep -oE 'val_loss:[0-9.]+' "$log" | tail -1 | cut -d: -f2 || true)
  ms=$(grep -oE 'step_avg:[0-9.]+ms' "$log" | tail -1 | grep -oE '[0-9.]+' || true)
  if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
    statestr="TIMEOUT"
  elif [[ "$status" -ne 0 ]]; then
    statestr="exit${status}"
  else
    statestr="ok"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$full_tag" "${val:-NA}" "${ms:-NA}" "$statestr" "$desc" >> "$SUMMARY"
}

# ---- baseline ----
run_variant baseline "Default: Muon blocks + AdamW head, LR 0.0036, no warmup, QK norm before RoPE"

# ---- learning-rate sweep (Muon+AdamW head LR) ----
run_variant lr0024 "LR 0.0036 -> 0.0024 (lower base LR)" LEARNING_RATE=0.0024
run_variant lr0048 "LR 0.0036 -> 0.0048 (higher base LR)" LEARNING_RATE=0.0048
run_variant lr0060 "LR 0.0036 -> 0.0060 (aggressive base LR)" LEARNING_RATE=0.0060

# ---- warmup (baseline uses 0 warmup; recent recipes add a short warmup for stability) ----
run_variant warmup64 "Add 64-step linear warmup (warmup_iters 0 -> 64)" WARMUP_ITERS=64

# ---- weight decay (baseline 0; AdamW-style decoupled WD on the head) ----
run_variant wd010 "Weight decay 0 -> 0.1 on AdamW head" WEIGHT_DECAY=0.1

# ---- Muon optimizer tweaks ----
run_variant mom098 "Muon momentum 0.95 -> 0.98" MUON_MOMENTUM=0.98
run_variant muonlr015 "Muon LR multiplier 0.10 -> 0.15" MUON_LR_MULTIPLIER=0.15
run_variant muonlr008 "Muon LR multiplier 0.10 -> 0.08" MUON_LR_MULTIPLIER=0.08

# ---- architecture tweaks (recent-paper inspired) ----
run_variant qkafterrope "QK RMSNorm after RoPE instead of before" QK_NORM_MODE=after_rope
run_variant embedrmsnorm "RMSNorm on token embeddings (nGPT-like conditioning)" EMBED_RMSNORM=1

echo ""
echo "=== sweep summary (${SUMMARY}) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2,2g; } | column -t -s$'\t'
python3 scripts/ab_dashboard.py || true
