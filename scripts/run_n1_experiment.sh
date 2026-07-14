#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for the N1 radiologist-only LR / auxiliary-loss sweep.
#
# Default sequence:
#   1) low_lr
#   2) outside_gland
#   3) patient_risk
#   4) outside_gland_patient_risk
#   5) low_lr_outside_gland
#   6) low_lr_patient_risk
#   7) low_lr_outside_gland_patient_risk
#
# The completed default run (LR=1e-4, no auxiliary loss) is intentionally
# omitted. Every variant also checks EXP_DIR and skips itself when a matching
# run already has last_checkpoint.pth plus internal/external rows in test_log.csv.
#
# Each Python invocation calls train.main(), and train.main() runs final test
# after training. The N1 wrapper uses the shared evaluation protocol:
#   - common_internal_test.csv
#   - N4_mixed_PROMIS_external_val.csv (canonical PROMIS cohort)
#
# Examples:
#   bash scripts/run_n1_experiment.sh
#   EPOCHS=5 bash scripts/run_n1_experiment.sh
#   bash scripts/run_n1_experiment.sh low_lr outside_gland
#   FORCE=1 bash scripts/run_n1_experiment.sh patient_risk
#   RUN_MODE=slurm bash scripts/run_n1_experiment.sh
#
# Optional environment variables:
#   PYTHON_BIN      Python executable, default: python
#   RP_BASE_DIR     Project base dir on training machine
#   RP_DATASET_ROOT Dataset root
#   RP_EXP_DIR      Output experiment dir
#   EPOCHS          Optional NUM_EPOCHS override
#   LR              Optional LR override for every selected variant
#   LOW_LR_VALUE    LR for low_lr variants, default: 5e-5
#   OUTSIDE_GLAND_WEIGHT Outside-gland loss weight, default: 0.05
#   PATIENT_RISK_WEIGHT  Patient-risk loss weight, default: 0.05
#   DROPOUT_RATE    Optional DROPOUT_RATE override
#   FORCE           Set to 1 to rerun completed matching experiments
#   RUN_MODE        local or slurm, default: local
#   SLURM_PARTITION SLURM partition, optional
#   SLURM_GPUS      GPUs per job, default: 1
#   SLURM_CPUS      CPUs per job, default: 8
#   SLURM_MEM       Memory per job, default: 48G
#   SLURM_TIME      Wall time, default: 2-00:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"
LOW_LR_VALUE="${LOW_LR_VALUE:-5e-5}"
OUTSIDE_GLAND_WEIGHT="${OUTSIDE_GLAND_WEIGHT:-0.05}"
PATIENT_RISK_WEIGHT="${PATIENT_RISK_WEIGHT:-0.05}"

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(
    low_lr
    outside_gland
    patient_risk
    outside_gland_patient_risk
    low_lr_outside_gland
    low_lr_patient_risk
    low_lr_outside_gland_patient_risk
  )
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=(
    "${SCRIPT_DIR}/run_n1_experiment.py"
    --experiment "$exp"
    --low-lr-value "$LOW_LR_VALUE"
    --outside-gland-weight "$OUTSIDE_GLAND_WEIGHT"
    --patient-risk-weight "$PATIENT_RISK_WEIGHT"
  )
  if [[ -n "${RP_BASE_DIR:-}" ]]; then
    RUN_ARGS+=(--base-dir "$RP_BASE_DIR")
  fi
  if [[ -n "${RP_DATASET_ROOT:-}" ]]; then
    RUN_ARGS+=(--dataset-root "$RP_DATASET_ROOT")
  fi
  if [[ -n "${RP_EXP_DIR:-}" ]]; then
    RUN_ARGS+=(--exp-dir "$RP_EXP_DIR")
  fi
  if [[ -n "${EPOCHS:-}" ]]; then
    RUN_ARGS+=(--epochs "$EPOCHS")
  fi
  if [[ -n "${LR:-}" ]]; then
    RUN_ARGS+=(--lr "$LR")
  fi
  if [[ -n "${DROPOUT_RATE:-}" ]]; then
    RUN_ARGS+=(--dropout-rate "$DROPOUT_RATE")
  fi
  if [[ "${FORCE:-0}" == "1" ]]; then
    RUN_ARGS+=(--force)
  fi
}

quote_args() {
  printf '%q ' "$@"
}

cd "$PROJECT_DIR"

for exp in "${EXPERIMENTS[@]}"; do
  echo "=== Launching N1 experiment: ${exp} ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    job_name="RP_N1_${exp}"
    cmd="cd $(printf '%q' "$PROJECT_DIR") && $(printf '%q' "$PYTHON_BIN") $(quote_args "${RUN_ARGS[@]}")"
    sbatch_args=(--job-name "$job_name")
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
      sbatch_args+=(--partition "$SLURM_PARTITION")
    fi
    sbatch_args+=(--gres "gpu:${SLURM_GPUS:-1}")
    sbatch_args+=(--cpus-per-task "${SLURM_CPUS:-8}")
    sbatch_args+=(--mem "${SLURM_MEM:-48G}")
    sbatch_args+=(--time "${SLURM_TIME:-2-00:00:00}")
    sbatch_args+=(--output "${PROJECT_DIR}/slurm-%x-%j.out")
    sbatch "${sbatch_args[@]}" --wrap "$cmd"
  else
    "$PYTHON_BIN" "${RUN_ARGS[@]}"
  fi
done
