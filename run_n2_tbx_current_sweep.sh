#!/usr/bin/env bash
set -euo pipefail

# Sequential N2 TBx sweep for the current soft-label sparse loss.
# Run on the server from this repository:
#   bash run_n2_tbx_current_sweep.sh
#
# Training outputs are created by train.py under:
#   /raid/candi/jiayi/RP/Experiments
# This script also keeps per-run tee logs under:
#   /raid/candi/jiayi/RP/Experiments/sweep_logs/N2_tbx_current_method

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SERVER_BASE_DIR="${SERVER_BASE_DIR:-/raid/candi/jiayi/RP}"
SWEEP_LOG_DIR="${SERVER_BASE_DIR}/Experiments/sweep_logs/N2_tbx_current_method"

mkdir -p "$SWEEP_LOG_DIR"

BACKUP_CONFIG="$(mktemp config.py.bak_n2_tbx_sweep.XXXXXX)"
cp config.py "$BACKUP_CONFIG"

restore_config() {
  cp "$BACKUP_CONFIG" config.py
}
trap restore_config EXIT

set_n2_config() {
  local tag="$1"
  local use_em="$2"
  local use_curriculum="$3"
  local sparse_start="$4"
  local sparse_weight="$5"
  local pos_soft="$6"
  local neg_soft="$7"
  local em_lr="$8"
  local use_clamp="$9"
  local logvar_min="${10}"
  local logvar_max="${11}"

  "$PYTHON_BIN" - "$tag" "$use_em" "$use_curriculum" "$sparse_start" \
    "$sparse_weight" "$pos_soft" "$neg_soft" "$em_lr" "$use_clamp" \
    "$logvar_min" "$logvar_max" "$SERVER_BASE_DIR" <<'PY'
import re
import sys
from pathlib import Path

(
    tag,
    use_em,
    use_curriculum,
    sparse_start,
    sparse_weight,
    pos_soft,
    neg_soft,
    em_lr,
    use_clamp,
    logvar_min,
    logvar_max,
    server_base_dir,
) = sys.argv[1:]

path = Path("config.py")
text = path.read_text()


def replace_global(src: str, key: str, value: str) -> str:
    pattern = rf"(^\s*{key}\s*=\s*).*$"
    new_src, count = re.subn(
        pattern,
        lambda match: match.group(1) + value,
        src,
        flags=re.M,
    )
    if count == 0:
        raise RuntimeError(f"Global config key not found: {key}")
    return new_src


def replace_in_n2(src: str, key: str, value: str) -> str:
    pattern = (
        r'(elif EXPERIMENT_MODE == "N2_PUB_TCIA_TBX_ONLY":.*?)'
        r'(?=\n    elif EXPERIMENT_MODE == "N3_PUB_TCIA_SBX_ONLY":)'
    )
    match = re.search(pattern, src, flags=re.S)
    if not match:
        raise RuntimeError("N2 config block not found in config.py")

    block, count = re.subn(
        rf"(^\s*{key}\s*=\s*).*$",
        lambda item: item.group(1) + value,
        match.group(1),
        flags=re.M,
    )
    if count == 0:
        raise RuntimeError(f"N2 config key not found: {key}")
    return src[: match.start(1)] + block + src[match.end(1) :]


def replace_sparse_weight(src: str, value: str) -> str:
    new_src, count = re.subn(
        r'("lesion_sparse"\s*:\s*)[0-9.]+',
        lambda match: match.group(1) + value,
        src,
        count=1,
    )
    if count == 0:
        raise RuntimeError('FIXED_LOSS_WEIGHTS["lesion_sparse"] not found')
    return new_src


text = replace_global(text, "BASE_DIR", f'r"{server_base_dir}"')
text = replace_global(text, "EXPERIMENT_MODE", '"N2_PUB_TCIA_TBX_ONLY"')

text = replace_in_n2(text, "EXPERIMENT_TAG", f'"{tag}"')
text = replace_in_n2(text, "USE_EM_WEIGHTING", use_em)
text = replace_in_n2(text, "USE_CURRICULUM", use_curriculum)
text = replace_in_n2(text, "LESION_SPARSE_START_EPOCH", sparse_start)

text = replace_global(text, "EM_LR_MULTIPLIER", em_lr)
text = replace_global(text, "USE_LOGVAR_CLAMP", use_clamp)
text = replace_global(text, "LOGVAR_MIN", logvar_min)
text = replace_global(text, "LOGVAR_MAX", logvar_max)
text = replace_global(text, "TBX_POSITIVE_SOFT_LABEL", pos_soft)
text = replace_global(text, "TBX_NEGATIVE_SOFT_LABEL", neg_soft)
text = replace_sparse_weight(text, sparse_weight)

path.write_text(text)
PY
}

run_exp() {
  local tag="$1"
  echo "===== Starting ${tag} at $(date) ====="
  "$PYTHON_BIN" -u train.py 2>&1 | tee "${SWEEP_LOG_DIR}/${tag}.log"
  echo "===== Finished ${tag} at $(date) ====="
}

# Current-method sweep.
# Args:
#   tag use_em use_curriculum sparse_start sparse_weight pos_soft neg_soft
#   em_lr use_clamp logvar_min logvar_max
#
# Note: in the current Loss_function.py, FIXED_LOSS_WEIGHTS are ignored when
# USE_EM_WEIGHTING=True, so sparse_weight only affects the FixedW experiments.

set_n2_config "N2_CM01_FixedW025_NoCurr_P09N005" "False" "False" "1" "0.25" "0.9" "0.05" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM01_FixedW025_NoCurr_P09N005"

set_n2_config "N2_CM02_FixedW025_Curr10_P09N005" "False" "True" "10" "0.25" "0.9" "0.05" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM02_FixedW025_Curr10_P09N005"

set_n2_config "N2_CM03_FixedW010_Curr10_P09N005" "False" "True" "10" "0.10" "0.9" "0.05" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM03_FixedW010_Curr10_P09N005"

set_n2_config "N2_CM04_FixedW010_Curr15_P09N005" "False" "True" "15" "0.10" "0.9" "0.05" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM04_FixedW010_Curr15_P09N005"

set_n2_config "N2_CM05_FixedW005_Curr15_P09N005" "False" "True" "15" "0.05" "0.9" "0.05" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM05_FixedW005_Curr15_P09N005"

set_n2_config "N2_CM06_FixedW010_Curr10_P085N010" "False" "True" "10" "0.10" "0.85" "0.10" "1.0" "False" "-2.0" "2.0"
run_exp "N2_CM06_FixedW010_Curr10_P085N010"

set_n2_config "N2_CM07_EM1_Curr15_ClampN05P1_P09N005" "True" "True" "15" "0.25" "0.9" "0.05" "1.0" "True" "-0.5" "1.0"
run_exp "N2_CM07_EM1_Curr15_ClampN05P1_P09N005"

echo "All N2 current-method experiments finished at $(date)."
