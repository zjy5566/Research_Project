#!/usr/bin/env python3
"""Run N1 radiologist-only experiments without editing config.py.

N1 trains on PUB dense radiologist lesion masks only, then final-tests every
selected checkpoint on both internal and external cohorts.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": "N1 PUB dense-only default parameters",
        "low_lr": False,
        "outside_gland": False,
        "patient_risk": False,
    },
    "low_lr": {
        "description": "Lower LR; auxiliary losses disabled",
        "low_lr": True,
        "outside_gland": False,
        "patient_risk": False,
    },
    "outside_gland": {
        "description": "Default LR with outside-gland penalty",
        "low_lr": False,
        "outside_gland": True,
        "patient_risk": False,
    },
    "patient_risk": {
        "description": "Default LR with patient-risk auxiliary loss",
        "low_lr": False,
        "outside_gland": False,
        "patient_risk": True,
    },
    "outside_gland_patient_risk": {
        "description": "Default LR with both auxiliary losses",
        "low_lr": False,
        "outside_gland": True,
        "patient_risk": True,
    },
    "low_lr_outside_gland": {
        "description": "Lower LR with outside-gland penalty",
        "low_lr": True,
        "outside_gland": True,
        "patient_risk": False,
    },
    "low_lr_patient_risk": {
        "description": "Lower LR with patient-risk auxiliary loss",
        "low_lr": True,
        "outside_gland": False,
        "patient_risk": True,
    },
    "low_lr_outside_gland_patient_risk": {
        "description": "Lower LR with both auxiliary losses",
        "low_lr": True,
        "outside_gland": True,
        "patient_risk": True,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N1 experiment.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="default",
        help="N1 experiment variant. Default: default.",
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
        "--lr",
        type=float,
        default=None,
        help="Optional LR override for the selected variant.",
    )
    parser.add_argument(
        "--low-lr-value",
        type=float,
        default=5e-5,
        help="LR used by low_lr variants. Default: 5e-5.",
    )
    parser.add_argument(
        "--outside-gland-weight",
        type=float,
        default=0.05,
        help="Outside-gland penalty weight when enabled. Default: 0.05.",
    )
    parser.add_argument(
        "--patient-risk-weight",
        type=float,
        default=0.05,
        help="Patient-risk auxiliary loss weight when enabled. Default: 0.05.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=None,
        help="Optional DROPOUT_RATE override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit before training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a completed matching experiment already exists.",
    )
    return parser.parse_args()


def _format_value(value: float) -> str:
    return f"{value:g}"


def _resolved_variant(args: argparse.Namespace) -> Dict[str, Any]:
    spec = EXPERIMENTS[args.experiment]
    lr = args.lr
    if lr is None:
        lr = args.low_lr_value if spec["low_lr"] else 1e-4

    outside_weight = args.outside_gland_weight if spec["outside_gland"] else 0.0
    patient_weight = args.patient_risk_weight if spec["patient_risk"] else 0.0
    is_historical_default = args.experiment == "default" and lr == 1e-4
    if is_historical_default:
        suffix_parts = ["N1_Default"]
    else:
        suffix_parts = [f"N1Sweep_{args.experiment}", f"LR{_format_value(lr)}"]
    if spec["outside_gland"]:
        suffix_parts.append(f"OutGlandW{_format_value(outside_weight)}")
    if spec["patient_risk"]:
        suffix_parts.append(f"PatientRiskW{_format_value(patient_weight)}")
    if not is_historical_default and not spec["outside_gland"] and not spec["patient_risk"]:
        suffix_parts.append("NoAux")

    return {
        **spec,
        "lr": float(lr),
        "outside_gland_weight": float(outside_weight),
        "patient_risk_weight": float(patient_weight),
        "suffix": "_".join(suffix_parts),
    }


def _resolved_exp_dir(args: argparse.Namespace) -> str:
    if args.exp_dir:
        return os.path.abspath(args.exp_dir)
    if args.base_dir:
        return os.path.join(os.path.abspath(args.base_dir), "Experiments")
    return os.path.join("/raid/candi/jiayi/RP", "Experiments")


def _completed_run(exp_dir: str, suffix: str) -> Optional[str]:
    """Return a matching run with completed internal and external final tests."""
    root = Path(exp_dir)
    if not root.is_dir():
        return None

    for run_dir in sorted(root.glob(f"*{suffix}*"), reverse=True):
        test_log = run_dir / "test_log.csv"
        last_checkpoint = run_dir / "last_checkpoint.pth"
        if not test_log.is_file() or not last_checkpoint.is_file():
            continue
        try:
            with test_log.open(newline="") as handle:
                labels = {
                    str(row.get("test_dataset_label", "")).strip().lower()
                    for row in csv.DictReader(handle)
                }
        except (OSError, csv.Error):
            continue
        if {"internal", "external"}.issubset(labels):
            return str(run_dir)
    return None


def _refresh_paths(
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

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, "N1_radiologist_only_train.csv")
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
            "grade_tbx": 0.0,
            "grade_sbx": 0.0,
            "lesion_dense": float(getattr(Config, "LESION_DENSE_LOSS_WEIGHT", 0.0)),
            "lesion_sparse": float(getattr(Config, "LESION_SPARSE_LOSS_WEIGHT", 0.0)),
            "lesion_sys": float(getattr(Config, "LESION_SYS_LOSS_WEIGHT", 0.0)),
            "lesion_outside_gland": float(getattr(Config, "OUTSIDE_GLAND_LOSS_WEIGHT", 0.0)),
            "lesion_patient": float(getattr(Config, "PATIENT_RISK_LOSS_WEIGHT", 0.0)),
            "gland": 0.0,
        }
    )
    Config.FIXED_LOSS_WEIGHTS = weights


def _apply_n1_config(
    Config: Any,
    args: argparse.Namespace,
    variant: Dict[str, Any],
) -> None:
    Config.EXPERIMENT_MODE = "N1_RADIOLOGIST_ONLY"

    Config.TASK = "mixed"
    Config.DATASET_TASK = "mixed"
    Config.TRAIN_DATASET_TASK = "radiologist_only"
    Config.VAL_DATASET_TASK = "mixed"
    Config.TEST_DATASET_TASK = "mixed"

    Config.USE_GRADE_TBX_TASK = False
    Config.USE_GRADE_SBX_TASK = False
    Config.USE_LESION_DENSE_TASK = True
    Config.USE_LESION_SPARSE_TASK = False
    Config.USE_LESION_SYS_TASK = False
    Config.USE_GLAND_TASK = False

    Config.LESION_DENSE_LOSS_WEIGHT = 1.0
    Config.LESION_SPARSE_LOSS_WEIGHT = 0.0
    Config.LESION_SYS_LOSS_WEIGHT = 0.0
    Config.OUTSIDE_GLAND_LOSS_WEIGHT = 0.0
    Config.PATIENT_RISK_LOSS_WEIGHT = 0.0

    Config.USE_EM_WEIGHTING = False
    Config.USE_CURRICULUM = False
    Config.USE_OUTSIDE_GLAND_PENALTY = False
    Config.USE_PATIENT_RISK_LOSS = False

    Config.LESION_DENSE_START_EPOCH = 1
    Config.LESION_SPARSE_START_EPOCH = 1
    Config.LESION_SYS_START_EPOCH = 1
    Config.OUTSIDE_GLAND_START_EPOCH = 1
    Config.PATIENT_RISK_START_EPOCH = 1

    Config.BEST_MODEL_METRIC = "lesion_dice"

    Config.LR = variant["lr"]
    Config.USE_OUTSIDE_GLAND_PENALTY = bool(variant["outside_gland"])
    Config.OUTSIDE_GLAND_LOSS_WEIGHT = variant["outside_gland_weight"]
    Config.USE_PATIENT_RISK_LOSS = bool(variant["patient_risk"])
    Config.PATIENT_RISK_LOSS_WEIGHT = variant["patient_risk_weight"]
    if args.epochs is not None:
        Config.NUM_EPOCHS = int(args.epochs)
    if args.dropout_rate is not None:
        Config.DROPOUT_RATE = float(args.dropout_rate)

    base_tag = "N1_PUBDenseOnly_CommonEval"
    suffix = variant["suffix"]
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
        "DROPOUT_RATE",
        "BEST_MODEL_METRIC",
        "TRAIN_DATASET_TASK",
        "VAL_DATASET_TASK",
        "TEST_DATASET_TASK",
        "USE_LESION_DENSE_TASK",
        "USE_LESION_SPARSE_TASK",
        "USE_LESION_SYS_TASK",
        "USE_OUTSIDE_GLAND_PENALTY",
        "OUTSIDE_GLAND_LOSS_WEIGHT",
        "USE_PATIENT_RISK_LOSS",
        "PATIENT_RISK_LOSS_WEIGHT",
        "PATIENT_RISK_POOLING",
        "PATIENT_RISK_USE_GLAND_MASK",
        "VALIDATION_COMPUTE_OPERATING_METRICS",
        "VALIDATION_COMPUTE_FROC_METRICS",
        "FINAL_TEST_COMPUTE_OPERATING_METRICS",
        "FINAL_TEST_COMPUTE_FROC_METRICS",
        "FINAL_TEST_CHECKPOINT_EPOCHS",
        "FINAL_TEST_INCLUDE_BEST",
        "FINAL_TEST_INCLUDE_LAST",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected N1 experiment: {experiment} - {EXPERIMENTS[experiment]['description']}")
    print("Final test: internal + external datasets after every training run")
    for key in keys:
        print(f"{key:<32}: {getattr(Config, key, None)}")


def main() -> None:
    args = _parse_args()
    variant = _resolved_variant(args)
    if not args.force:
        completed = _completed_run(_resolved_exp_dir(args), variant["suffix"])
        if completed:
            print(f"Skipping completed N1 experiment '{args.experiment}': {completed}")
            return

    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = args.dataset_root

    from config import Config

    _refresh_paths(
        Config,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    _apply_n1_config(Config, args, variant)
    _print_resolved_config(Config, args.experiment)

    if args.dry_run:
        return

    import train

    train.main()


if __name__ == "__main__":
    main()
