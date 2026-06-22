#!/usr/bin/env bash
set -euo pipefail

# Sequential N4 mixed-supervision sweep.
# N4 uses PUB dense lesion masks + TCIA TBx sparse tracks + TCIA SBx MIL labels.
#
# Run on the server from this repository:
#   bash run_n4_mixed_sweep.sh
#
# Training outputs are created by train.py under:
#   /raid/candi/jiayi/RP/Experiments
# This script also keeps per-run tee logs under:
#   /raid/candi/jiayi/RP/Experiments/sweep_logs/N4_mixed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SERVER_BASE_DIR="${SERVER_BASE_DIR:-/raid/candi/jiayi/RP}"
SWEEP_LOG_DIR="${SERVER_BASE_DIR}/Experiments/sweep_logs/N4_mixed"

mkdir -p "$SWEEP_LOG_DIR"

BACKUP_CONFIG="$(mktemp config.py.bak_n4_mixed_sweep.XXXXXX)"
cp config.py "$BACKUP_CONFIG"

restore_config() {
  cp "$BACKUP_CONFIG" config.py
}
trap restore_config EXIT

set_n4_config() {
  local tag="$1"
  local use_em="$2"
  local use_curriculum="$3"
  local sparse_start="$4"
  local sys_start="$5"
  local sparse_weight="$6"
  local sys_weight="$7"
  local pos_soft="$8"
  local neg_soft="$9"
  local em_lr="${10}"
  local use_clamp="${11}"
  local logvar_min="${12}"
  local logvar_max="${13}"
  local sys_pos_weight="${14}"
  local sys_focal_alpha="${15}"
  local sys_focal_gamma="${16}"
  local mask_target_in_sys="${17}"

  "$PYTHON_BIN" - "$tag" "$use_em" "$use_curriculum" "$sparse_start" \
    "$sys_start" "$sparse_weight" "$sys_weight" "$pos_soft" "$neg_soft" \
    "$em_lr" "$use_clamp" "$logvar_min" "$logvar_max" "$sys_pos_weight" \
    "$sys_focal_alpha" "$sys_focal_gamma" "$mask_target_in_sys" \
    "$SERVER_BASE_DIR" <<'PY'
import re
import sys
from pathlib import Path

(
    tag,
    use_em,
    use_curriculum,
    sparse_start,
    sys_start,
    sparse_weight,
    sys_weight,
    pos_soft,
    neg_soft,
    em_lr,
    use_clamp,
    logvar_min,
    logvar_max,
    sys_pos_weight,
    sys_focal_alpha,
    sys_focal_gamma,
    mask_target_in_sys,
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


def replace_in_n4(src: str, key: str, value: str) -> str:
    pattern = (
        r'(elif EXPERIMENT_MODE == "N4_MIXED":.*?)'
        r'(?=\n    else:)'
    )
    match = re.search(pattern, src, flags=re.S)
    if not match:
        raise RuntimeError("N4 config block not found in config.py")

    block, count = re.subn(
        rf"(^\s*{key}\s*=\s*).*$",
        lambda item: item.group(1) + value,
        match.group(1),
        flags=re.M,
    )
    if count == 0:
        raise RuntimeError(f"N4 config key not found: {key}")
    return src[: match.start(1)] + block + src[match.end(1) :]


def replace_loss_weight(src: str, key: str, value: str) -> str:
    new_src, count = re.subn(
        rf'("{re.escape(key)}"\s*:\s*)[0-9.]+',
        lambda match: match.group(1) + value,
        src,
        count=1,
    )
    if count == 0:
        raise RuntimeError(f'FIXED_LOSS_WEIGHTS["{key}"] not found')
    return new_src


text = replace_global(text, "BASE_DIR", f'r"{server_base_dir}"')
text = replace_global(text, "EXPERIMENT_MODE", '"N4_MIXED"')

text = replace_in_n4(text, "EXPERIMENT_TAG", f'"{tag}"')
text = replace_in_n4(text, "TASK", '"mixed"')
text = replace_in_n4(text, "DATASET_TASK", '"mixed"')
text = replace_in_n4(text, "USE_LESION_DENSE_TASK", "True")
text = replace_in_n4(text, "USE_LESION_SPARSE_TASK", "True")
text = replace_in_n4(text, "USE_LESION_SYS_TASK", "True")
text = replace_in_n4(text, "USE_EM_WEIGHTING", use_em)
text = replace_in_n4(text, "USE_CURRICULUM", use_curriculum)
text = replace_in_n4(text, "LESION_DENSE_START_EPOCH", "1")
text = replace_in_n4(text, "LESION_SPARSE_START_EPOCH", sparse_start)
text = replace_in_n4(text, "LESION_SYS_START_EPOCH", sys_start)

text = replace_global(text, "EM_LR_MULTIPLIER", em_lr)
text = replace_global(text, "USE_LOGVAR_CLAMP", use_clamp)
text = replace_global(text, "LOGVAR_MIN", logvar_min)
text = replace_global(text, "LOGVAR_MAX", logvar_max)
text = replace_global(text, "TBX_POSITIVE_SOFT_LABEL", pos_soft)
text = replace_global(text, "TBX_NEGATIVE_SOFT_LABEL", neg_soft)
text = replace_global(text, "SYS_POS_WEIGHT_VAL", sys_pos_weight)
text = replace_global(text, "SYS_FOCAL_ALPHA", sys_focal_alpha)
text = replace_global(text, "SYS_FOCAL_GAMMA", sys_focal_gamma)
text = replace_global(text, "USE_SYS_CLASS_BALANCED_BCE", "True")
text = replace_global(text, "MASK_TARGET_IN_SYS", mask_target_in_sys)

text = replace_loss_weight(text, "lesion_dense", "1.0")
text = replace_loss_weight(text, "lesion_sparse", sparse_weight)
text = replace_loss_weight(text, "lesion_sys", sys_weight)

path.write_text(text)
PY
}

run_exp() {
  local tag="$1"
  echo "===== Starting ${tag} at $(date) ====="
  "$PYTHON_BIN" -u train.py 2>&1 | tee "${SWEEP_LOG_DIR}/${tag}.log"
  echo "===== Finished ${tag} at $(date) ====="
}

# Args:
#   tag use_em use_curriculum sparse_start sys_start sparse_weight sys_weight
#   pos_soft neg_soft em_lr use_clamp logvar_min logvar_max
#   sys_pos_weight sys_focal_alpha sys_focal_gamma mask_target_in_sys
#
# Notes:
#   - TBx is lesion_sparse; SBx is lesion_sys.
#   - FIXED_LOSS_WEIGHTS only affect runs with USE_EM_WEIGHTING=False.
#   - mask_target_in_sys=True removes TBx voxels from SBx zone masks when both
#     labels are present, avoiding duplicate supervision on the same voxels.

set_n4_config "N4_M01_FixedW025_050_Curr10_30_P09N005" "False" "True" "10" "30" "0.25" "0.50" "0.9" "0.05" "1.0" "False" "-2.0" "2.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M01_FixedW025_050_Curr10_30_P09N005"

set_n4_config "N4_M02_FixedW010_025_Curr10_20_P09N005" "False" "True" "10" "20" "0.10" "0.25" "0.9" "0.05" "1.0" "False" "-2.0" "2.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M02_FixedW010_025_Curr10_20_P09N005"

set_n4_config "N4_M03_EM1_Curr10_30_ClampN05P1_P09N005" "True" "True" "10" "30" "0.25" "0.50" "0.9" "0.05" "1.0" "True" "-0.5" "1.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M03_EM1_Curr10_30_ClampN05P1_P09N005"

set_n4_config "N4_M04_EM5_Curr10_20_ClampN1P2_P09N005" "True" "True" "10" "20" "0.25" "0.50" "0.9" "0.05" "5.0" "True" "-1.0" "2.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M04_EM5_Curr10_20_ClampN1P2_P09N005"

set_n4_config "N4_M05_EM10_Curr10_30_ClampN2P2_P085N010" "True" "True" "10" "30" "0.25" "0.50" "0.85" "0.10" "10.0" "True" "-2.0" "2.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M05_EM10_Curr10_30_ClampN2P2_P085N010"

set_n4_config "N4_M06_EM1_NoCurr_ClampN05P1_P09N005" "True" "False" "1" "1" "0.25" "0.50" "0.9" "0.05" "1.0" "True" "-0.5" "1.0" "1.0" "0.75" "2.0" "True"
run_exp "N4_M06_EM1_NoCurr_ClampN05P1_P09N005"

echo "All N4 mixed experiments finished at $(date)."
