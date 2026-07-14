#!/usr/bin/env python3
"""Run B2/B3 TCIA biopsy-supervision experiments without editing config.py.

Each invocation runs exactly one experiment. train.main() performs training and
then final-tests the selected checkpoints, so the companion shell script can run
B2 and B3 sequentially while preserving train -> test ordering.
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
    "b2": {
        "mode": "B2_TCIA_SBX_ONLY",
        "suffix": "B2_Default",
        "description": "TCIA systematic-biopsy region supervision only",
        "train_csv": "B2_TCIA_SBx_only_train.csv",
        "use_sparse": False,
        "use_sys": True,
        "sparse_weight": 0.0,
        "sys_weight": 1.0,
        "use_curriculum": False,
        "sparse_start": 1,
        "sys_start": 1,
        "best_metric": "region_auprc",
        "base_tag": "B2_TCIA_SBxOnly",
    },
    "b3": {
        "mode": "B3_TCIA_TBX_SBX",
        "suffix": "B3_Default",
        "description": "TCIA target-biopsy ROI + systematic-biopsy region supervision",
        "train_csv": "B3_TCIA_TBx_SBx_train.csv",
        "use_sparse": True,
        "use_sys": True,
        "sparse_weight": 1.0,
        "sys_weight": 1.0,
        "use_curriculum": True,
        "sparse_start": 1,
        "sys_start": 10,
        "best_metric": "biopsy_auprc_composite",
        "base_tag": "B3_TCIA_TBxROI_SBx_PosNeg",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one B2 or B3 experiment.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        required=True,
        help="Experiment to run: b2 or b3.",
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
        help="Optional LR override.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional POS_WEIGHT_VAL override for TBx ROI BCE.",
    )
    parser.add_argument(
        "--sys-pos-weight",
        type=float,
        default=None,
        help="Optional SYS_POS_WEIGHT_VAL override for SBx region BCE.",
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
    return parser.parse_args()


def _refresh_paths(
    Config: Any,
    experiment_cfg: Dict[str, Any],
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

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, experiment_cfg["train_csv"])
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


def _apply_experiment(Config: Any, args: argparse.Namespace) -> None:
    experiment_cfg = EXPERIMENTS[args.experiment]

    Config.EXPERIMENT_MODE = experiment_cfg["mode"]

    Config.TASK = "mixed"
    Config.DATASET_TASK = "mixed"
    Config.TRAIN_DATASET_TASK = "mixed"
    Config.VAL_DATASET_TASK = "mixed"
    Config.TEST_DATASET_TASK = "mixed"

    Config.USE_GRADE_TBX_TASK = False
    Config.USE_GRADE_SBX_TASK = False
    Config.USE_LESION_DENSE_TASK = False
    Config.USE_LESION_SPARSE_TASK = bool(experiment_cfg["use_sparse"])
    Config.USE_LESION_SYS_TASK = bool(experiment_cfg["use_sys"])
    Config.USE_GLAND_TASK = False

    Config.LESION_DENSE_LOSS_WEIGHT = 0.0
    Config.LESION_SPARSE_LOSS_WEIGHT = float(experiment_cfg["sparse_weight"])
    Config.LESION_SYS_LOSS_WEIGHT = float(experiment_cfg["sys_weight"])

    Config.USE_EM_WEIGHTING = False
    Config.USE_CURRICULUM = bool(experiment_cfg["use_curriculum"])
    Config.USE_OUTSIDE_GLAND_PENALTY = False
    Config.USE_PATIENT_RISK_LOSS = False

    Config.LESION_DENSE_START_EPOCH = 1
    Config.LESION_SPARSE_START_EPOCH = int(experiment_cfg["sparse_start"])
    Config.LESION_SYS_START_EPOCH = int(experiment_cfg["sys_start"])
    Config.OUTSIDE_GLAND_START_EPOCH = 1
    Config.PATIENT_RISK_START_EPOCH = 1

    Config.OUTSIDE_GLAND_LOSS_WEIGHT = 0.0
    Config.PATIENT_RISK_LOSS_WEIGHT = 0.0

    Config.BEST_MODEL_METRIC = str(experiment_cfg["best_metric"])

    if args.epochs is not None:
        Config.NUM_EPOCHS = int(args.epochs)
    if args.lr is not None:
        Config.LR = float(args.lr)
    if args.pos_weight is not None:
        Config.POS_WEIGHT_VAL = float(args.pos_weight)
    if args.sys_pos_weight is not None:
        Config.SYS_POS_WEIGHT_VAL = float(args.sys_pos_weight)
    if args.dropout_rate is not None:
        Config.DROPOUT_RATE = float(args.dropout_rate)

    Config.EXPERIMENT_TAG = f"{experiment_cfg['base_tag']}_{experiment_cfg['suffix']}"

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
        "DROPOUT_RATE",
        "BEST_MODEL_METRIC",
        "TRAIN_DATASET_TASK",
        "VAL_DATASET_TASK",
        "TEST_DATASET_TASK",
        "USE_LESION_DENSE_TASK",
        "USE_LESION_SPARSE_TASK",
        "USE_LESION_SYS_TASK",
        "LESION_DENSE_LOSS_WEIGHT",
        "LESION_SPARSE_LOSS_WEIGHT",
        "LESION_SYS_LOSS_WEIGHT",
        "USE_CURRICULUM",
        "LESION_SPARSE_START_EPOCH",
        "LESION_SYS_START_EPOCH",
        "USE_OUTSIDE_GLAND_PENALTY",
        "USE_PATIENT_RISK_LOSS",
        "VALIDATION_COMPUTE_OPERATING_METRICS",
        "VALIDATION_COMPUTE_FROC_METRICS",
        "FINAL_TEST_COMPUTE_OPERATING_METRICS",
        "FINAL_TEST_COMPUTE_FROC_METRICS",
        "FINAL_TEST_CHECKPOINT_EPOCHS",
        "FINAL_TEST_INCLUDE_BEST",
        "FINAL_TEST_INCLUDE_LAST",
        "FIXED_LOSS_WEIGHTS",
    ]
    cfg = EXPERIMENTS[experiment]
    print(f"Selected {experiment.upper()} experiment: {cfg['description']}")
    print("Final test: internal + external datasets after every training run")
    for key in keys:
        print(f"{key:<32}: {getattr(Config, key, None)}")


def main() -> None:
    args = _parse_args()
    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = args.dataset_root

    from config import Config

    experiment_cfg = EXPERIMENTS[args.experiment]
    _refresh_paths(
        Config,
        experiment_cfg,
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
