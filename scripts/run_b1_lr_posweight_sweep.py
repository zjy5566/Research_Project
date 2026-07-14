#!/usr/bin/env python3
"""Run B1 LR / positive-weight sweep without editing config.py.

Each invocation runs exactly one B1 configuration. train.main() already runs
the final test after training, so the companion shell script can call this file
sequentially to guarantee train -> test before the next experiment starts.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "default": {
        "suffix": "B1_Default",
        "description": "B1 default parameters",
        "overrides": {},
    },
    "low_lr": {
        "suffix": "B1_LR5e-5",
        "description": "B1 with LR 1e-4 -> 5e-5",
        "overrides": {"LR": 5e-5},
    },
    "low_pos_weight": {
        "suffix": "B1_PosW1",
        "description": "B1 with POS_WEIGHT_VAL 2 -> 1",
        "overrides": {"POS_WEIGHT_VAL": 1.0},
    },
    "low_lr_low_pos_weight": {
        "suffix": "B1_LR5e-5_PosW1",
        "description": "B1 with LR 1e-4 -> 5e-5 and POS_WEIGHT_VAL 2 -> 1",
        "overrides": {"LR": 5e-5, "POS_WEIGHT_VAL": 1.0},
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one B1 LR / positive-weight experiment."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        required=True,
        help="One experiment from the B1 sweep.",
    )
    parser.add_argument(
        "--low-lr-value",
        type=float,
        default=5e-5,
        help="LR used by low_lr variants. Default: 5e-5.",
    )
    parser.add_argument(
        "--low-pos-weight-value",
        type=float,
        default=1.0,
        help="POS_WEIGHT_VAL used by low_pos_weight variants. Default: 1.0.",
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("RP_BASE_DIR"),
        help="Optional project base dir on the training machine.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("RP_DATASET_ROOT"),
        help="Optional dataset root. Also forwarded through RP_DATASET_ROOT.",
    )
    parser.add_argument(
        "--exp-dir",
        default=os.environ.get("RP_EXP_DIR"),
        help="Optional output experiment directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional NUM_EPOCHS override for quick checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit before training.",
    )
    return parser.parse_args()


def _refresh_b1_paths(
    Config: Any,
    *,
    base_dir: Optional[str],
    dataset_root: Optional[str],
    exp_dir: Optional[str],
) -> None:
    if base_dir:
        Config.BASE_DIR = base_dir
    if dataset_root:
        Config.DATASET_ROOT = dataset_root
    elif base_dir:
        Config.DATASET_ROOT = os.path.join(Config.BASE_DIR, "data")

    if base_dir or dataset_root:
        Config.UNIFIED_DATA_DIR = os.path.join(Config.DATASET_ROOT, "Unified_Dataset")
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        Config.COMMON_INTERNAL_VAL_CSV = os.path.join(
            Config.SPLIT_DIR, "common_internal_evaluation.csv"
        )
        Config.COMMON_INTERNAL_TEST_CSV = os.path.join(
            Config.SPLIT_DIR, "common_internal_test.csv"
        )
        Config.COMMON_EXTERNAL_TEST_CSV = os.path.join(
            Config.SPLIT_DIR, "N4_mixed_PROMIS_external_val.csv"
        )

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, "B1_TCIA_TBx_baseline_train.csv")
    Config.VAL_CSV = Config.COMMON_INTERNAL_VAL_CSV
    Config.INTERNAL_TEST_CSV = Config.COMMON_INTERNAL_TEST_CSV
    Config.TEST_CSV = Config.COMMON_EXTERNAL_TEST_CSV
    Config.COMMON_FINAL_TEST_DATASETS = (
        ("internal", Config.INTERNAL_TEST_CSV),
        ("external", Config.TEST_CSV),
    )
    Config.FINAL_TEST_DATASETS = Config.COMMON_FINAL_TEST_DATASETS

    if exp_dir:
        Config.EXP_DIR = exp_dir
    elif base_dir:
        Config.EXP_DIR = os.path.join(Config.BASE_DIR, "Experiments")


def _refresh_fixed_loss_weights(Config: Any) -> None:
    weights = dict(getattr(Config, "FIXED_LOSS_WEIGHTS", {}) or {})
    weights.update(
        {
            "lesion_dense": float(getattr(Config, "LESION_DENSE_LOSS_WEIGHT", 0.0)),
            "lesion_sparse": float(getattr(Config, "LESION_SPARSE_LOSS_WEIGHT", 0.0)),
            "lesion_sys": float(getattr(Config, "LESION_SYS_LOSS_WEIGHT", 0.0)),
            "lesion_outside_gland": float(getattr(Config, "OUTSIDE_GLAND_LOSS_WEIGHT", 0.0)),
            "lesion_patient": float(getattr(Config, "PATIENT_RISK_LOSS_WEIGHT", 0.0)),
        }
    )
    Config.FIXED_LOSS_WEIGHTS = weights


def _resolved_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = dict(EXPERIMENTS[args.experiment]["overrides"])
    if "LR" in overrides:
        overrides["LR"] = float(args.low_lr_value)
    if "POS_WEIGHT_VAL" in overrides:
        overrides["POS_WEIGHT_VAL"] = float(args.low_pos_weight_value)
    return overrides


def _apply_experiment(Config: Any, args: argparse.Namespace) -> None:
    if getattr(Config, "EXPERIMENT_MODE", "") != "B1_TCIA_TBX_BASELINE":
        raise RuntimeError(
            "This sweep expects Config.EXPERIMENT_MODE='B1_TCIA_TBX_BASELINE'. "
            f"Got {getattr(Config, 'EXPERIMENT_MODE', None)!r}."
        )

    for key, value in _resolved_overrides(args).items():
        setattr(Config, key, value)
    if args.epochs is not None:
        Config.NUM_EPOCHS = int(args.epochs)

    base_tag = str(getattr(Config, "EXPERIMENT_TAG", "B1_TCIA_TBxROI"))
    suffix = EXPERIMENTS[args.experiment]["suffix"]
    if suffix not in base_tag:
        Config.EXPERIMENT_TAG = f"{base_tag}_{suffix}"

    _refresh_fixed_loss_weights(Config)


def _print_resolved_config(Config: Any, experiment: str) -> None:
    keys: Iterable[str] = [
        "EXPERIMENT_MODE",
        "EXPERIMENT_TAG",
        "TRAIN_CSV",
        "VAL_CSV",
        "INTERNAL_TEST_CSV",
        "TEST_CSV",
        "FINAL_TEST_DATASETS",
        "EXP_DIR",
        "NUM_EPOCHS",
        "LR",
        "POS_WEIGHT_VAL",
        "SYS_POS_WEIGHT_VAL",
        "BEST_MODEL_METRIC",
        "VALIDATION_COMPUTE_OPERATING_METRICS",
        "VALIDATION_COMPUTE_FROC_METRICS",
        "FINAL_TEST_COMPUTE_OPERATING_METRICS",
        "FINAL_TEST_COMPUTE_FROC_METRICS",
        "FINAL_TEST_CHECKPOINT_EPOCHS",
        "FINAL_TEST_INCLUDE_BEST",
        "FINAL_TEST_INCLUDE_LAST",
        "USE_OUTSIDE_GLAND_PENALTY",
        "USE_PATIENT_RISK_LOSS",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected B1 experiment: {experiment} - {EXPERIMENTS[experiment]['description']}")
    print("Final test: enabled by train.main() after every training run")
    for key in keys:
        print(f"{key:<32}: {getattr(Config, key, None)}")


def main() -> None:
    args = _parse_args()
    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = args.dataset_root

    from config import Config

    _refresh_b1_paths(
        Config,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    _apply_experiment(Config, args)
    _print_resolved_config(Config, args.experiment)

    if args.dry_run:
        return

    import train

    train.main()


if __name__ == "__main__":
    main()
