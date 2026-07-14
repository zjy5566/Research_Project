#!/usr/bin/env python3
"""Run RA+TBx (N2) and RA+SBx (N3) mixed-supervision experiments."""

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
    "n2": {
        "mode": "N2_PUB_TCIA_TBX_ONLY",
        "description": "RA dense masks + TBx-confirmed target ROIs",
        "train_csv": "N2_PUB_TCIA_TBx_only_train.csv",
        "use_sparse": True,
        "use_sys": False,
        "best_metric": "ra_tbx_auprc_composite",
        "tag": "N2_RA_TBx",
    },
    "n3": {
        "mode": "N3_PUB_TCIA_SBX_ONLY",
        "description": "RA dense masks + SBx region labels",
        "train_csv": "N3_PUB_TCIA_SBx_only_train.csv",
        "use_sparse": False,
        "use_sys": True,
        "best_metric": "ra_sbx_auprc_composite",
        "tag": "N3_RA_SBx",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N2 or N3 experiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--base-dir", default=os.environ.get("RP_BASE_DIR"))
    parser.add_argument("--dataset-root", default=os.environ.get("RP_DATASET_ROOT"))
    parser.add_argument("--exp-dir", default=os.environ.get("RP_EXP_DIR"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--biopsy-weight", type=float, default=1.0)
    parser.add_argument("--outside-gland-weight", type=float, default=0.05)
    parser.add_argument("--patient-risk-weight", type=float, default=0.05)
    parser.add_argument("--disable-outside-gland", action="store_true")
    parser.add_argument("--disable-patient-risk", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--sys-pos-weight", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a matching experiment already completed both final tests.",
    )
    return parser.parse_args()


def _number(value: float) -> str:
    return f"{float(value):g}"


def _resolved_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(EXPERIMENTS[args.experiment])
    outside_weight = 0.0 if args.disable_outside_gland else args.outside_gland_weight
    patient_weight = 0.0 if args.disable_patient_risk else args.patient_risk_weight
    biopsy_name = "TBx" if cfg["use_sparse"] else "SBx"
    parts = [
        cfg["tag"],
        f"FixedW_Dense{_number(args.dense_weight)}_{biopsy_name}{_number(args.biopsy_weight)}",
        "NoCurr",
        f"LR{_number(args.lr)}",
    ]
    parts.append(
        f"OutGlandW{_number(outside_weight)}" if outside_weight > 0 else "NoOutGland"
    )
    parts.append(
        f"PatientRiskW{_number(patient_weight)}" if patient_weight > 0 else "NoPatientRisk"
    )
    cfg.update(
        {
            "lr": float(args.lr),
            "dense_weight": float(args.dense_weight),
            "biopsy_weight": float(args.biopsy_weight),
            "outside_weight": float(outside_weight),
            "patient_weight": float(patient_weight),
            "experiment_tag": "_".join(parts),
        }
    )
    return cfg


def _resolved_exp_dir(args: argparse.Namespace) -> str:
    if args.exp_dir:
        return os.path.abspath(args.exp_dir)
    if args.base_dir:
        return os.path.join(os.path.abspath(args.base_dir), "Experiments")
    return "/raid/candi/jiayi/RP/Experiments"


def _completed_run(exp_dir: str, experiment_tag: str) -> Optional[str]:
    root = Path(exp_dir)
    if not root.is_dir():
        return None
    for run_dir in sorted(root.glob(f"*{experiment_tag}*"), reverse=True):
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
    experiment: Dict[str, Any],
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

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, experiment["train_csv"])
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
    Config.FIXED_LOSS_WEIGHTS = {
        "grade_tbx": 0.0,
        "grade_sbx": 0.0,
        "lesion_dense": float(Config.LESION_DENSE_LOSS_WEIGHT),
        "lesion_sparse": float(Config.LESION_SPARSE_LOSS_WEIGHT),
        "lesion_sys": float(Config.LESION_SYS_LOSS_WEIGHT),
        "lesion_outside_gland": float(Config.OUTSIDE_GLAND_LOSS_WEIGHT),
        "lesion_patient": float(Config.PATIENT_RISK_LOSS_WEIGHT),
        "gland": 0.0,
    }


def _apply_experiment(Config: Any, args: argparse.Namespace, experiment: Dict[str, Any]) -> None:
    Config.EXPERIMENT_MODE = experiment["mode"]
    Config.EXPERIMENT_TAG = experiment["experiment_tag"]

    Config.TASK = "mixed"
    Config.DATASET_TASK = "mixed"
    Config.TRAIN_DATASET_TASK = "mixed"
    Config.VAL_DATASET_TASK = "mixed"
    Config.TEST_DATASET_TASK = "mixed"

    Config.USE_GRADE_TBX_TASK = False
    Config.USE_GRADE_SBX_TASK = False
    Config.USE_GLAND_TASK = False
    Config.USE_LESION_DENSE_TASK = True
    Config.USE_LESION_SPARSE_TASK = bool(experiment["use_sparse"])
    Config.USE_LESION_SYS_TASK = bool(experiment["use_sys"])

    Config.LESION_DENSE_LOSS_WEIGHT = experiment["dense_weight"]
    Config.LESION_SPARSE_LOSS_WEIGHT = (
        experiment["biopsy_weight"] if experiment["use_sparse"] else 0.0
    )
    Config.LESION_SYS_LOSS_WEIGHT = (
        experiment["biopsy_weight"] if experiment["use_sys"] else 0.0
    )
    Config.USE_EM_WEIGHTING = False
    Config.USE_CURRICULUM = False
    Config.LESION_DENSE_START_EPOCH = 1
    Config.LESION_SPARSE_START_EPOCH = 1
    Config.LESION_SYS_START_EPOCH = 1

    Config.USE_OUTSIDE_GLAND_PENALTY = experiment["outside_weight"] > 0
    Config.OUTSIDE_GLAND_LOSS_WEIGHT = experiment["outside_weight"]
    Config.OUTSIDE_GLAND_START_EPOCH = 1
    Config.USE_PATIENT_RISK_LOSS = experiment["patient_weight"] > 0
    Config.PATIENT_RISK_LOSS_WEIGHT = experiment["patient_weight"]
    Config.PATIENT_RISK_START_EPOCH = 1
    Config.PATIENT_RISK_POOLING = "lme"
    Config.PATIENT_RISK_LME_R = 8.0
    Config.PATIENT_RISK_USE_GLAND_MASK = False

    Config.LR = experiment["lr"]
    Config.BEST_MODEL_METRIC = str(experiment["best_metric"])
    if "bacc" in Config.BEST_MODEL_METRIC.lower():
        raise ValueError("N2/N3 checkpoint selection must not use BACC.")

    if args.epochs is not None:
        Config.NUM_EPOCHS = int(args.epochs)
    if args.pos_weight is not None:
        Config.POS_WEIGHT_VAL = float(args.pos_weight)
    if args.sys_pos_weight is not None:
        Config.SYS_POS_WEIGHT_VAL = float(args.sys_pos_weight)
    if args.dropout_rate is not None:
        Config.DROPOUT_RATE = float(args.dropout_rate)

    _refresh_fixed_loss_weights(Config)


def _print_resolved_config(Config: Any, experiment: Dict[str, Any]) -> None:
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
        "BEST_MODEL_METRIC",
        "USE_LESION_DENSE_TASK",
        "USE_LESION_SPARSE_TASK",
        "USE_LESION_SYS_TASK",
        "LESION_DENSE_LOSS_WEIGHT",
        "LESION_SPARSE_LOSS_WEIGHT",
        "LESION_SYS_LOSS_WEIGHT",
        "USE_EM_WEIGHTING",
        "USE_CURRICULUM",
        "USE_OUTSIDE_GLAND_PENALTY",
        "OUTSIDE_GLAND_LOSS_WEIGHT",
        "USE_PATIENT_RISK_LOSS",
        "PATIENT_RISK_LOSS_WEIGHT",
        "PATIENT_RISK_POOLING",
        "PATIENT_RISK_LME_R",
        "VALIDATION_COMPUTE_FROC_METRICS",
        "FINAL_TEST_COMPUTE_FROC_METRICS",
        "FINAL_TEST_CHECKPOINT_EPOCHS",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected experiment: {experiment['description']}")
    print("Training: fixed weights, all active supervision starts at epoch 1")
    print("Final test: selected checkpoints on internal + external datasets")
    for key in keys:
        print(f"{key:<32}: {getattr(Config, key, None)}")


def main() -> None:
    args = _parse_args()
    experiment = _resolved_experiment(args)
    if not args.force:
        completed = _completed_run(_resolved_exp_dir(args), experiment["experiment_tag"])
        if completed:
            print(f"Skipping completed {args.experiment.upper()} experiment: {completed}")
            return

    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = args.dataset_root

    from config import Config

    _refresh_paths(
        Config,
        experiment,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    _apply_experiment(Config, args, experiment)
    _print_resolved_config(Config, experiment)

    if args.dry_run:
        return

    import train

    train.main()


if __name__ == "__main__":
    main()
