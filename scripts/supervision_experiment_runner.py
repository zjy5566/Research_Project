#!/usr/bin/env python3
"""Shared runner utilities for the redesigned B/N supervision experiments.

The B/N scripts deliberately separate optimisation hyperparameters from
supervision membership:

  D = dense radiologist annotation (RA)
  T = targeted-biopsy ROI supervision (TBx)
  S = systematic-biopsy region supervision (SBx)
  P = biopsy-confirmed patient-level supervision

Outside-gland suppression is disabled in the main B/N matrix. It is an anatomy
ablation, not one of the four cancer-supervision levels above.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@dataclass(frozen=True)
class ExperimentSpec:
    """One fully specified B/N experiment and its inherited best parameters."""

    key: str
    mode: str
    description: str
    parameter_source: str
    train_csv: str
    train_dataset_task: str = "mixed"
    use_dense: bool = False
    use_tbx: bool = False
    use_sbx: bool = False
    use_patient: bool = False
    dense_weight: float = 0.0
    tbx_weight: float = 0.0
    sbx_weight: float = 0.0
    patient_weight: float = 0.0
    use_curriculum: bool = False
    dense_start: int = 1
    tbx_start: int = 1
    sbx_start: int = 1
    patient_start: int = 1
    lr: float = 1e-4
    pos_weight: float = 2.0
    sys_pos_weight: float = 1.0
    best_metric: str = "tbx_roi_auprc"

    @property
    def supervision_code(self) -> str:
        active = []
        if self.use_dense:
            active.append("D")
        if self.use_tbx:
            active.append("T")
        if self.use_sbx:
            active.append("S")
        if self.use_patient:
            active.append("P")
        return "".join(active) or "None"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add safe runtime overrides shared by both family runners."""

    parser.add_argument("--base-dir", default=os.environ.get("RP_BASE_DIR"))
    parser.add_argument("--dataset-root", default=os.environ.get("RP_DATASET_ROOT"))
    parser.add_argument("--exp-dir", default=os.environ.get("RP_EXP_DIR"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--sys-pos-weight", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without starting training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if an identical tag already completed internal/external tests.",
    )


def _number(value: float) -> str:
    return f"{float(value):g}"


def _resolved_value(cli_value: Optional[float], default: float) -> float:
    return float(default if cli_value is None else cli_value)


def _resolved_exp_dir(args: argparse.Namespace) -> str:
    if args.exp_dir:
        return os.path.abspath(args.exp_dir)
    if args.base_dir:
        return os.path.join(os.path.abspath(args.base_dir), "Experiments")
    return "/raid/candi/jiayi/RP/Experiments"


def _dry_run_config(args: argparse.Namespace) -> SimpleNamespace:
    """Build the small Config surface needed for dependency-free dry runs."""

    base_dir = os.path.abspath(args.base_dir) if args.base_dir else "/raid/candi/jiayi/RP"
    dataset_root = (
        os.path.abspath(args.dataset_root)
        if args.dataset_root
        else os.path.join(base_dir, "data")
    )
    unified = os.path.join(dataset_root, "Unified_Dataset")
    split_dir = os.path.join(unified, "splits")
    return SimpleNamespace(
        BASE_DIR=base_dir,
        DATASET_ROOT=dataset_root,
        UNIFIED_DATA_DIR=unified,
        SPLIT_DIR=split_dir,
        COMMON_INTERNAL_VAL_CSV=os.path.join(split_dir, "common_internal_evaluation.csv"),
        COMMON_INTERNAL_TEST_CSV=os.path.join(split_dir, "common_internal_test.csv"),
        COMMON_EXTERNAL_TEST_CSV=os.path.join(split_dir, "N4_mixed_PROMIS_external_val.csv"),
        EXP_DIR=(os.path.abspath(args.exp_dir) if args.exp_dir else os.path.join(base_dir, "Experiments")),
        NUM_EPOCHS=150,
        SEED=42,
        DROPOUT_RATE=0.2,
    )


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
    spec: ExperimentSpec,
    *,
    base_dir: Optional[str],
    dataset_root: Optional[str],
    exp_dir: Optional[str],
) -> None:
    if base_dir:
        Config.BASE_DIR = os.path.abspath(base_dir)
    if dataset_root:
        Config.DATASET_ROOT = os.path.abspath(dataset_root)
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

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, spec.train_csv)
    Config.VAL_CSV = Config.COMMON_INTERNAL_VAL_CSV
    Config.INTERNAL_TEST_CSV = Config.COMMON_INTERNAL_TEST_CSV
    Config.TEST_CSV = Config.COMMON_EXTERNAL_TEST_CSV
    Config.COMMON_FINAL_TEST_DATASETS = (
        ("internal", Config.INTERNAL_TEST_CSV),
        ("external", Config.TEST_CSV),
    )
    Config.FINAL_TEST_DATASETS = Config.COMMON_FINAL_TEST_DATASETS

    if exp_dir:
        Config.EXP_DIR = os.path.abspath(exp_dir)
    elif base_dir:
        Config.EXP_DIR = os.path.join(Config.BASE_DIR, "Experiments")


def _experiment_tag(
    spec: ExperimentSpec,
    *,
    epochs: int,
    seed: int,
    lr: float,
    pos_weight: float,
    sys_pos_weight: float,
) -> str:
    curriculum = "Curr" if spec.use_curriculum else "NoCurr"
    return "_".join(
        [
            spec.key.upper(),
            f"Sup{spec.supervision_code}",
            "BestPrior",
            f"D{_number(spec.dense_weight)}",
            f"T{_number(spec.tbx_weight)}",
            f"S{_number(spec.sbx_weight)}",
            f"P{_number(spec.patient_weight)}",
            "A0",
            curriculum,
            f"LR{_number(lr)}",
            f"PosW{_number(pos_weight)}",
            f"SysPosW{_number(sys_pos_weight)}",
            f"Seed{seed}",
            f"E{epochs}",
        ]
    )


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


def apply_experiment(Config: Any, spec: ExperimentSpec, args: argparse.Namespace) -> None:
    """Resolve one spec into Config without inheriting mode-dependent auxiliaries."""

    epochs = int(Config.NUM_EPOCHS if args.epochs is None else args.epochs)
    seed = int(Config.SEED if args.seed is None else args.seed)
    lr = _resolved_value(args.lr, spec.lr)
    pos_weight = _resolved_value(args.pos_weight, spec.pos_weight)
    sys_pos_weight = _resolved_value(args.sys_pos_weight, spec.sys_pos_weight)

    Config.EXPERIMENT_MODE = spec.mode
    Config.EXPERIMENT_TAG = _experiment_tag(
        spec,
        epochs=epochs,
        seed=seed,
        lr=lr,
        pos_weight=pos_weight,
        sys_pos_weight=sys_pos_weight,
    )

    Config.TASK = "mixed"
    Config.DATASET_TASK = "mixed"
    Config.TRAIN_DATASET_TASK = spec.train_dataset_task
    Config.VAL_DATASET_TASK = "mixed"
    Config.TEST_DATASET_TASK = "mixed"

    Config.USE_GRADE_TBX_TASK = False
    Config.USE_GRADE_SBX_TASK = False
    Config.USE_GLAND_TASK = False
    Config.USE_LESION_DENSE_TASK = bool(spec.use_dense)
    Config.USE_LESION_SPARSE_TASK = bool(spec.use_tbx)
    Config.USE_LESION_SYS_TASK = bool(spec.use_sbx)

    Config.LESION_DENSE_LOSS_WEIGHT = float(spec.dense_weight)
    Config.LESION_SPARSE_LOSS_WEIGHT = float(spec.tbx_weight)
    Config.LESION_SYS_LOSS_WEIGHT = float(spec.sbx_weight)
    Config.USE_EM_WEIGHTING = False
    Config.USE_CURRICULUM = bool(spec.use_curriculum)
    Config.LESION_DENSE_START_EPOCH = int(spec.dense_start)
    Config.LESION_SPARSE_START_EPOCH = int(spec.tbx_start)
    Config.LESION_SYS_START_EPOCH = int(spec.sbx_start)

    # A is excluded from the main B/N matrix. P is activated only by the spec.
    Config.USE_OUTSIDE_GLAND_PENALTY = False
    Config.OUTSIDE_GLAND_LOSS_WEIGHT = 0.0
    Config.OUTSIDE_GLAND_START_EPOCH = 1
    Config.USE_PATIENT_RISK_LOSS = bool(spec.use_patient)
    Config.PATIENT_RISK_LOSS_WEIGHT = float(spec.patient_weight)
    Config.PATIENT_RISK_START_EPOCH = int(spec.patient_start)

    # Train and evaluate the same full-crop, logit-LME patient score. No
    # ground-truth gland mask is required at deployment.
    Config.PATIENT_RISK_POOLING = "lme"
    Config.PATIENT_RISK_LME_R = 8.0
    Config.PATIENT_RISK_USE_GLAND_MASK = False
    Config.SEG_PATIENT_POOLING = "logit_lme"
    Config.SEG_RISK_LME_R = 8.0
    Config.SEG_EVAL_USE_GLAND_MASK = False
    Config.USE_RA_LESION_PRESENCE_AS_PATIENT_LABEL = False

    Config.LR = lr
    Config.POS_WEIGHT_VAL = pos_weight
    Config.SYS_POS_WEIGHT_VAL = sys_pos_weight
    Config.NUM_EPOCHS = epochs
    Config.SEED = seed
    Config.BEST_MODEL_METRIC = str(spec.best_metric)
    Config.WEIGHT_DECAY = 1e-4
    Config.BATCH_SIZE = 4
    Config.DROPOUT_RATE = (
        float(Config.DROPOUT_RATE)
        if args.dropout_rate is None
        else float(args.dropout_rate)
    )
    Config.USE_AUGMENTATION = True
    Config.USE_TBX_POSITIVE_ONLY_LOSS = False
    Config.USE_SYS_CLASS_BALANCED_BCE = True
    Config.SYS_FOCAL_ALPHA = 0.75
    Config.SYS_FOCAL_GAMMA = 2.0
    Config.MASK_TARGET_IN_SYS = bool(spec.use_tbx and spec.use_sbx)

    _refresh_fixed_loss_weights(Config)
    validate_resolved_config(Config, spec)


def validate_resolved_config(Config: Any, spec: ExperimentSpec) -> None:
    active = (
        bool(Config.USE_LESION_DENSE_TASK),
        bool(Config.USE_LESION_SPARSE_TASK),
        bool(Config.USE_LESION_SYS_TASK),
        bool(Config.USE_PATIENT_RISK_LOSS),
    )
    expected = (spec.use_dense, spec.use_tbx, spec.use_sbx, spec.use_patient)
    if active != expected:
        raise ValueError(f"Resolved supervision {active} does not match {spec.key} {expected}.")
    if not any(active):
        raise ValueError(f"{spec.key} has no active supervision.")
    if bool(Config.USE_OUTSIDE_GLAND_PENALTY) or float(Config.OUTSIDE_GLAND_LOSS_WEIGHT) != 0.0:
        raise ValueError("Outside-gland supervision must be disabled in the main B/N matrix.")
    if spec.use_patient and float(Config.PATIENT_RISK_LOSS_WEIGHT) <= 0.0:
        raise ValueError(f"{spec.key} enables P but has a non-positive patient weight.")
    if not spec.use_patient and float(Config.PATIENT_RISK_LOSS_WEIGHT) != 0.0:
        raise ValueError(f"{spec.key} disables P but retained a patient weight.")
    if str(Config.BEST_MODEL_METRIC) != "tbx_roi_auprc":
        raise ValueError("All redesigned B/N experiments must use common TBx ROI AUPRC selection.")


def print_resolved_config(Config: Any, spec: ExperimentSpec) -> None:
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
        "SEED",
        "LR",
        "WEIGHT_DECAY",
        "BATCH_SIZE",
        "DROPOUT_RATE",
        "POS_WEIGHT_VAL",
        "SYS_POS_WEIGHT_VAL",
        "BEST_MODEL_METRIC",
        "USE_LESION_DENSE_TASK",
        "USE_LESION_SPARSE_TASK",
        "USE_LESION_SYS_TASK",
        "USE_PATIENT_RISK_LOSS",
        "USE_OUTSIDE_GLAND_PENALTY",
        "LESION_DENSE_LOSS_WEIGHT",
        "LESION_SPARSE_LOSS_WEIGHT",
        "LESION_SYS_LOSS_WEIGHT",
        "PATIENT_RISK_LOSS_WEIGHT",
        "USE_CURRICULUM",
        "LESION_DENSE_START_EPOCH",
        "LESION_SPARSE_START_EPOCH",
        "LESION_SYS_START_EPOCH",
        "PATIENT_RISK_START_EPOCH",
        "PATIENT_RISK_POOLING",
        "SEG_PATIENT_POOLING",
        "SEG_EVAL_USE_GLAND_MASK",
        "MASK_TARGET_IN_SYS",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected {spec.key.upper()}: {spec.description}")
    print(f"Supervision: {spec.supervision_code}; anatomy A: disabled")
    print(f"Inherited parameter source: {spec.parameter_source}")
    print("Checkpoint selection: common validation TBx ROI AUPRC")
    print("Patient label: biopsy/explicit csPCa only; RA lesion-presence fallback disabled")
    for key in keys:
        print(f"{key:<36}: {getattr(Config, key, None)}")


def execute_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> None:
    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = os.path.abspath(args.dataset_root)

    if args.dry_run:
        # Configuration inspection should work on login/local machines that do
        # not have the CUDA/PyTorch training environment installed.
        Config = _dry_run_config(args)
    else:
        # Import after the dataset-root environment override so config.py derives
        # its initial paths from the requested training environment.
        from config import Config

    _refresh_paths(
        Config,
        spec,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    apply_experiment(Config, spec, args)
    print_resolved_config(Config, spec)

    if args.dry_run:
        return

    if not args.force:
        completed = _completed_run(_resolved_exp_dir(args), Config.EXPERIMENT_TAG)
        if completed:
            print(f"Skipping completed {spec.key.upper()} experiment: {completed}")
            return

    import train

    train.main()
