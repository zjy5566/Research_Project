#!/usr/bin/env bash
set -euo pipefail

# Sequential RA+TBx (N2) and RA+SBx (N3) runner. Each Python invocation runs
# training followed by multi-checkpoint internal and external final tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(n2 n3)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=("${SCRIPT_DIR}/run_n2_n3_experiments.py" --experiment "$exp")
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
  if [[ -n "${DENSE_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--dense-weight "$DENSE_WEIGHT")
  fi
  if [[ -n "${BIOPSY_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--biopsy-weight "$BIOPSY_WEIGHT")
  fi
  if [[ -n "${OUTSIDE_GLAND_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--outside-gland-weight "$OUTSIDE_GLAND_WEIGHT")
  fi
  if [[ -n "${PATIENT_RISK_WEIGHT:-}" ]]; then
    RUN_ARGS+=(--patient-risk-weight "$PATIENT_RISK_WEIGHT")
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
  if [[ "${DISABLE_OUTSIDE_GLAND:-0}" == "1" ]]; then
    RUN_ARGS+=(--disable-outside-gland)
  fi
  if [[ "${DISABLE_PATIENT_RISK:-0}" == "1" ]]; then
    RUN_ARGS+=(--disable-patient-risk)
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
  exp_upper="$(printf '%s' "$exp" | tr '[:lower:]' '[:upper:]')"
  echo "=== Launching ${exp_upper} mixed-supervision experiment ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    job_name="RP_${exp_upper}_MIXED"
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
