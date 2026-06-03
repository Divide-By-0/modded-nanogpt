#!/usr/bin/env bash
# Practical adaptive-Muon sweep (round 2). The paper-faithful Muon2 (2nd moment BEFORE
# orthogonalization) lost in round 1 -- expected, since that pre-scaling largely cancels
# under the polar normalization. This round tests what the community/GitHub repos actually
# run and report as best empirically:
#   - AdaMuon  (https://github.com/Chongjie-Si/AdaMuon): element-wise 2nd moment AFTER
#              orthogonalization, BIAS-CORRECTED, then RMS-realigned to Adam's update size.
#   - NorMuon  (https://github.com/zichongli5/NorMuon): same but per output-neuron (row).
# Both keep update RMS==1 so muon_lr_multiplier transfers; we still bracket the LR
# (0.05 / 0.10 / 0.15) per your request to test "its own lower learning rate".
#
# NOTE: a separate "muon2 + bias correction" run is intentionally omitted -- bias correction
# is a global scalar that cancels under the pre-orthogonalization polar norm, so it would be
# numerically identical to plain muon2. The meaningful bias-corrected recipe IS adamuon/normuon.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
TRAIN_SEED="${TRAIN_SEED:-1337}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-360}"
RUN_KILL_AFTER_SECONDS="${RUN_KILL_AFTER_SECONDS:-15}"
SWEEP_TAG_SUFFIX="${SWEEP_TAG_SUFFIX:-opt2}"

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

run_variant muonref   "Vanilla Muon (reference, round 2)" MUON_VARIANT=muon

# AdaMuon: element-wise post-orthogonalization 2nd moment, bias-corrected, RMS-realigned.
run_variant adamuon   "AdaMuon: bias-corrected element-wise 2nd moment after NS, lr_mult 0.10" MUON_VARIANT=adamuon MUON_LR_MULTIPLIER=0.10
run_variant adamuonlo "AdaMuon, lower lr_mult 0.05"  MUON_VARIANT=adamuon MUON_LR_MULTIPLIER=0.05
run_variant adamuonhi "AdaMuon, higher lr_mult 0.15" MUON_VARIANT=adamuon MUON_LR_MULTIPLIER=0.15

# NorMuon: per output-neuron (row) post-orthogonalization 2nd moment, bias-corrected.
run_variant normuon   "NorMuon: bias-corrected per-neuron 2nd moment after NS, lr_mult 0.10" MUON_VARIANT=normuon MUON_LR_MULTIPLIER=0.10
run_variant normuonlo "NorMuon, lower lr_mult 0.05"  MUON_VARIANT=normuon MUON_LR_MULTIPLIER=0.05

# Aurora refinement around round-1's interior optimum (best was lr_mult 0.30; 0.10 worse, 1.00 worse).
run_variant auroram02 "Aurora, lr_mult 0.20" MUON_VARIANT=aurora MUON_LR_MULTIPLIER=0.20
run_variant auroram05 "Aurora, lr_mult 0.50" MUON_VARIANT=aurora MUON_LR_MULTIPLIER=0.50

echo ""
echo "=== practical optimizer sweep summary (${SUMMARY}) ==="
{ head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -t$'\t' -k2,2g; } | column -t -s$'\t' 2>/dev/null || cat "$SUMMARY"
python3 scripts/ab_dashboard.py || true
