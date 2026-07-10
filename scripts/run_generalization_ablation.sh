#!/usr/bin/env bash
set -euo pipefail

# Remote runner for the B1 generalization ablations.
#
# Examples:
#   bash scripts/run_generalization_ablation.sh
#   bash scripts/run_generalization_ablation.sh G1 G3
#   RUN_MODE=slurm bash scripts/run_generalization_ablation.sh G1 G2 G3 G4
#
# Optional environment variables:
#   PYTHON_BIN      Python executable on the training machine, default: python
#   RP_BASE_DIR     Project base dir, default: config.py BASE_DIR
#   RP_DATASET_ROOT Dataset root, default: config.py DATASET_ROOT
#   RP_EXP_DIR      Output experiment dir, default: config.py EXP_DIR
#   EPOCHS          Optional NUM_EPOCHS override
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
  EXPERIMENTS=(G1 G2 G3 G4)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=(run_generalization_ablation.py --experiment "$exp")
  # Forward only explicitly provided environment overrides; otherwise the
  # Python wrapper keeps the checked-in Config defaults for the training host.
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
  echo "=== Launching ${exp} ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    job_name="RP_${exp}"
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
