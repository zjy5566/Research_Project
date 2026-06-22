#!/usr/bin/env bash
set -euo pipefail

# Run several N3 experiments sequentially. Each experiment edits config.py,
# runs train.py to completion, then starts the next configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p exp/N3/server_runs

BACKUP_CONFIG="config.py.bak_n3_sweep"
cp config.py "$BACKUP_CONFIG"

restore_config() {
  cp "$BACKUP_CONFIG" config.py
}
trap restore_config EXIT

set_n3_config() {
  local tag="$1"
  local use_em="$2"
  local em_lr="$3"
  local sys_start="$4"
  local logvar_min="$5"
  local sys_weight="$6"

  python - "$tag" "$use_em" "$em_lr" "$sys_start" "$logvar_min" "$sys_weight" <<'PY'
import re
import sys
from pathlib import Path

tag, use_em, em_lr, sys_start, logvar_min, sys_weight = sys.argv[1:]
path = Path("config.py")
text = path.read_text()


def replace_global(src: str, key: str, value: str) -> str:
    return re.sub(
        rf"(^\s*{key}\s*=\s*).*$",
        lambda match: match.group(1) + value,
        src,
        flags=re.M,
    )


def replace_in_n3(src: str, key: str, value: str) -> str:
    pattern = (
        r'(elif EXPERIMENT_MODE == "N3_PUB_TCIA_SBX_ONLY":.*?)'
        r'(?=\n    elif EXPERIMENT_MODE == "N4_MIXED":)'
    )
    match = re.search(pattern, src, flags=re.S)
    if not match:
        raise RuntimeError("N3 config block not found in config.py")

    block = re.sub(
        rf"(^\s*{key}\s*=\s*).*$",
        lambda item: item.group(1) + value,
        match.group(1),
        flags=re.M,
    )
    return src[: match.start(1)] + block + src[match.end(1) :]


text = replace_global(text, "EXPERIMENT_MODE", '"N3_PUB_TCIA_SBX_ONLY"')

text = replace_in_n3(text, "EXPERIMENT_TAG", f'"{tag}"')
text = replace_in_n3(text, "USE_EM_WEIGHTING", use_em)
text = replace_in_n3(text, "USE_CURRICULUM", "True")
text = replace_in_n3(text, "LESION_SYS_START_EPOCH", sys_start)

text = replace_global(text, "EM_LR_MULTIPLIER", em_lr)
text = replace_global(text, "USE_LOGVAR_CLAMP", "True")
text = replace_global(text, "LOGVAR_MIN", logvar_min)
text = replace_global(text, "LOGVAR_MAX", "2.0")

text = replace_global(text, "SYS_POS_WEIGHT_VAL", "1.0")
text = replace_global(text, "SYS_FOCAL_ALPHA", "0.75")
text = replace_global(text, "SYS_FOCAL_GAMMA", "2.0")
text = replace_global(text, "USE_SYS_CLASS_BALANCED_BCE", "True")

text = re.sub(
    r'("lesion_sys"\s*:\s*)[0-9.]+',
    lambda match: match.group(1) + sys_weight,
    text,
)

path.write_text(text)
PY
}

run_exp() {
  local tag="$1"
  echo "===== Starting ${tag} at $(date) ====="
  python -u train.py 2>&1 | tee "exp/N3/server_runs/${tag}.log"
  echo "===== Finished ${tag} at $(date) ====="
}

set_n3_config "N3_Exp3_EM10_Start10" "True" "10.0" "10" "-2.0" "1.0"
run_exp "N3_Exp3_EM10_Start10"

set_n3_config "N3_Exp4_EM1_Start15" "True" "1.0" "15" "-0.5" "1.0"
run_exp "N3_Exp4_EM1_Start15"

set_n3_config "N3_Exp5_FixedSys025_Start15" "False" "1.0" "15" "-0.5" "0.25"
run_exp "N3_Exp5_FixedSys025_Start15"

echo "All experiments finished at $(date)."
