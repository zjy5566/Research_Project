#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for the B1 LR / positive-weight sweep.
#
# Default sequence:
#   1) default
#   2) low_lr
#   3) low_pos_weight
#   4) low_lr_low_pos_weight
#
# Each Python invocation calls train.main(), and train.main() runs final test
# after training. Because local mode is sequential, every experiment completes
# train -> test before the next experiment starts.
#
# Examples:
#   bash scripts/run_b1_lr_posweight_sweep.sh
#   EPOCHS=5 bash scripts/run_b1_lr_posweight_sweep.sh
#   bash scripts/run_b1_lr_posweight_sweep.sh default low_lr
#   RUN_MODE=slurm bash scripts/run_b1_lr_posweight_sweep.sh
#
# Optional environment variables:
#   PYTHON_BIN            Python executable, default: python
#   RP_BASE_DIR           Project base dir on training machine
#   RP_DATASET_ROOT       Dataset root
#   RP_EXP_DIR            Output experiment dir
#   EPOCHS                Optional NUM_EPOCHS override
#   LOW_LR_VALUE          Low LR value, default: 5e-5
#   LOW_POS_WEIGHT_VALUE  Low POS_WEIGHT_VAL, default: 1.0
#   RUN_MODE              local or slurm, default: local
#   SLURM_PARTITION       SLURM partition, optional
#   SLURM_GPUS            GPUs per job, default: 1
#   SLURM_CPUS            CPUs per job, default: 8
#   SLURM_MEM             Memory per job, default: 48G
#   SLURM_TIME            Wall time, default: 2-00:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"
LOW_LR_VALUE="${LOW_LR_VALUE:-5e-5}"
LOW_POS_WEIGHT_VALUE="${LOW_POS_WEIGHT_VALUE:-1.0}"

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(default low_lr low_pos_weight low_lr_low_pos_weight)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=(
    "${SCRIPT_DIR}/run_b1_lr_posweight_sweep.py"
    --experiment "$exp"
    --low-lr-value "$LOW_LR_VALUE"
    --low-pos-weight-value "$LOW_POS_WEIGHT_VALUE"
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
}

quote_args() {
  printf '%q ' "$@"
}

cd "$PROJECT_DIR"

for exp in "${EXPERIMENTS[@]}"; do
  echo "=== Launching B1 sweep: ${exp} ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    job_name="RP_B1_${exp}"
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
