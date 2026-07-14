#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for B2/B3 TCIA biopsy-supervision experiments.
#
# Default sequence:
#   1) b2: TCIA SBx region supervision only
#   2) b3: TCIA TBx ROI + TCIA SBx region supervision
#
# Each Python invocation calls train.main(), and train.main() runs final test
# after training. In local mode, every experiment completes train -> internal
# test -> external test before the next experiment starts.
#
# Examples:
#   bash scripts/run_b2_b3_experiments.sh
#   bash scripts/run_b2_b3_experiments.sh b2
#   EPOCHS=5 bash scripts/run_b2_b3_experiments.sh
#   LR=5e-5 SYS_POS_WEIGHT=0.5 bash scripts/run_b2_b3_experiments.sh b3
#   RUN_MODE=slurm bash scripts/run_b2_b3_experiments.sh
#
# Optional environment variables:
#   PYTHON_BIN      Python executable, default: python
#   RP_BASE_DIR     Project base dir on training machine
#   RP_DATASET_ROOT Dataset root
#   RP_EXP_DIR      Output experiment dir
#   EPOCHS          Optional NUM_EPOCHS override
#   LR              Optional LR override
#   POS_WEIGHT      Optional POS_WEIGHT_VAL override for TBx ROI BCE
#   SYS_POS_WEIGHT  Optional SYS_POS_WEIGHT_VAL override for SBx region BCE
#   DROPOUT_RATE    Optional DROPOUT_RATE override
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

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(b2 b3)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=("${SCRIPT_DIR}/run_b2_b3_experiments.py" --experiment "$exp")
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
  if [[ -n "${POS_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--pos-weight "$POS_WEIGHT")
  fi
  if [[ -n "${SYS_POS_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--sys-pos-weight "$SYS_POS_WEIGHT")
  fi
  if [[ -n "${DROPOUT_RATE:-}" ]]; then
    RUN_ARGS+=(--dropout-rate "$DROPOUT_RATE")
  fi
}

quote_args() {
  printf '%q ' "$@"
}

cd "$PROJECT_DIR"

for exp in "${EXPERIMENTS[@]}"; do
  exp_upper="$(printf '%s' "$exp" | tr '[:lower:]' '[:upper:]')"
  echo "=== Launching ${exp_upper} experiment ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    job_name="RP_${exp_upper}"
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
