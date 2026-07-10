#!/usr/bin/env python3
"""Run B1 generalization ablations without editing config.py.

This wrapper is intended for the remote training environment. It imports the
existing Config, applies one named ablation, refreshes dependent loss weights,
then calls train.main().
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "G1": {
        "suffix": "G1_LR5e-5",
        "description": "LR 1e-4 -> 5e-5, otherwise unchanged",
        "overrides": {"LR": 5e-5},
    },
    "G2": {
        "suffix": "G2_WD1e-4",
        "description": "Set Adam weight_decay=1e-4",
        "overrides": {"WEIGHT_DECAY": 1e-4},
    },
    "G3": {
        "suffix": "G3_PosW1",
        "description": "POS_WEIGHT_VAL 2 -> 1",
        "overrides": {"POS_WEIGHT_VAL": 1.0},
    },
    "G4": {
        "suffix": "G4_PatientRiskStart30",
        "description": "Delay PatientRisk branch to epoch 30",
        "overrides": {
            "USE_CURRICULUM": True,
            "USE_PATIENT_RISK_LOSS": True,
            "PATIENT_RISK_LOSS_WEIGHT": 0.05,
            "PATIENT_RISK_START_EPOCH": 30,
        },
    },
    "G4_off": {
        "suffix": "G4off_NoPatientRisk",
        "description": "Disable PatientRisk branch",
        "overrides": {
            "USE_PATIENT_RISK_LOSS": False,
            "PATIENT_RISK_LOSS_WEIGHT": 0.0,
            "PATIENT_RISK_START_EPOCH": 1,
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one B1 generalization ablation by overriding Config at startup."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(ABLATIONS),
        required=True,
        help="Ablation to run. Use G1/G2/G3/G4 for the main plan; G4_off is optional.",
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("RP_BASE_DIR"),
        help="Optional project base dir on the training machine. Defaults to Config.BASE_DIR.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("RP_DATASET_ROOT"),
        help="Optional dataset root. Also forwarded through RP_DATASET_ROOT before importing Config.",
    )
    parser.add_argument(
        "--exp-dir",
        default=os.environ.get("RP_EXP_DIR"),
        help="Optional output experiment directory. Defaults to Config.EXP_DIR.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional NUM_EPOCHS override for quick checks or shorter ablations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved overrides and exit before importing train.py.",
    )
    return parser.parse_args()


def _refresh_paths(
    Config: Any,
    *,
    base_dir: Optional[str],
    dataset_root: Optional[str],
    exp_dir: Optional[str],
) -> None:
    # Config derives several CSV paths from BASE_DIR/DATASET_ROOT at import
    # time, so CLI path overrides must refresh every dependent path together.
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

        if getattr(Config, "EXPERIMENT_MODE", "") == "B1_TCIA_TBX_BASELINE":
            Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, "B1_TCIA_TBx_baseline_train.csv")
            Config.VAL_CSV = Config.COMMON_INTERNAL_VAL_CSV
            Config.TEST_CSV = os.path.join(Config.SPLIT_DIR, "B1_PROMIS_external_val.csv")

    if exp_dir:
        Config.EXP_DIR = exp_dir
    elif base_dir:
        Config.EXP_DIR = os.path.join(Config.BASE_DIR, "Experiments")


def _refresh_fixed_loss_weights(Config: Any) -> None:
    # Ablations mutate individual weights after Config has built the dictionary;
    # rebuild the dictionary so MixedSupervisionLoss sees the resolved values.
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


def _apply_ablation(Config: Any, experiment: str, *, epochs: Optional[int]) -> None:
    spec = ABLATIONS[experiment]
    for key, value in spec["overrides"].items():
        setattr(Config, key, value)
    if epochs is not None:
        Config.NUM_EPOCHS = int(epochs)

    base_tag = str(getattr(Config, "EXPERIMENT_TAG", "B1_TCIA_TBxROI"))
    suffix = spec["suffix"]
    if suffix not in base_tag:
        Config.EXPERIMENT_TAG = f"{base_tag}_{suffix}"

    _refresh_fixed_loss_weights(Config)


def _print_resolved_config(Config: Any, experiment: str) -> None:
    keys = [
        "EXPERIMENT_MODE",
        "EXPERIMENT_TAG",
        "TRAIN_CSV",
        "VAL_CSV",
        "TEST_CSV",
        "EXP_DIR",
        "NUM_EPOCHS",
        "LR",
        "WEIGHT_DECAY",
        "POS_WEIGHT_VAL",
        "USE_PATIENT_RISK_LOSS",
        "PATIENT_RISK_LOSS_WEIGHT",
        "PATIENT_RISK_START_EPOCH",
        "USE_OUTSIDE_GLAND_PENALTY",
        "OUTSIDE_GLAND_LOSS_WEIGHT",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected ablation: {experiment} - {ABLATIONS[experiment]['description']}")
    for key in keys:
        print(f"{key:<32}: {getattr(Config, key, None)}")


def main() -> None:
    args = _parse_args()
    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = args.dataset_root

    from config import Config

    _refresh_paths(
        Config,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    _apply_ablation(Config, args.experiment, epochs=args.epochs)
    _print_resolved_config(Config, args.experiment)

    if args.dry_run:
        return

    import train

    train.main()


if __name__ == "__main__":
    main()
