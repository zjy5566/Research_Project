#!/usr/bin/env python3
"""Run one redesigned B-family experiment (B0--B4) from scripts/.

B experiments do not use dense RA supervision. Each configuration inherits the
best available optimisation parameters from the closest completed experiment,
while P remains an explicit supervision level and A remains disabled.
"""

from __future__ import annotations

import argparse
from typing import Dict

from supervision_experiment_runner import (
    ExperimentSpec,
    add_common_arguments,
    execute_experiment,
)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "b0": ExperimentSpec(
        key="b0",
        mode="B0_PATIENT_ONLY",
        description="Biopsy-confirmed patient-level supervision only",
        parameter_source="B3 default optimiser + prior PatientRiskW0.05",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_patient=True,
        patient_weight=0.05,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "b1": ExperimentSpec(
        key="b1",
        mode="B1_TCIA_TBX_BASELINE",
        description="TBx target-ROI supervision only",
        parameter_source="B1 LR5e-5 + PosW1 selected run",
        train_csv="B1_TCIA_TBx_baseline_train.csv",
        use_tbx=True,
        tbx_weight=1.0,
        lr=5e-5,
        pos_weight=1.0,
        sys_pos_weight=1.0,
    ),
    "b2": ExperimentSpec(
        key="b2",
        mode="B2_TCIA_SBX_ONLY",
        description="SBx region-level supervision only",
        parameter_source="B2 default selected run",
        train_csv="B2_TCIA_SBx_only_train.csv",
        use_sbx=True,
        sbx_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "b3": ExperimentSpec(
        key="b3",
        mode="B3_TCIA_TBX_SBX",
        description="TBx target-ROI + SBx region supervision",
        parameter_source="B3 default selected run",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_tbx=True,
        use_sbx=True,
        tbx_weight=1.0,
        sbx_weight=1.0,
        use_curriculum=True,
        tbx_start=1,
        sbx_start=10,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "b4": ExperimentSpec(
        key="b4",
        mode="B4_TCIA_TBX_SBX_PATIENT",
        description="TBx + SBx + biopsy-confirmed patient-level supervision",
        parameter_source="B3 default selected run + prior PatientRiskW0.05",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_tbx=True,
        use_sbx=True,
        use_patient=True,
        tbx_weight=1.0,
        sbx_weight=1.0,
        patient_weight=0.05,
        use_curriculum=True,
        tbx_start=1,
        sbx_start=10,
        patient_start=1,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one B0--B4 experiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
