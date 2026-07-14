#!/usr/bin/env python3
"""Run one redesigned N-family experiment (N1--N5) from scripts/.

N experiments are anchored by dense RA supervision. N5 is the only N-family
main experiment that adds the explicit patient-level supervision pathway.
Outside-gland suppression is reserved for a separate anatomy ablation.
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
    "n1": ExperimentSpec(
        key="n1",
        mode="N1_RADIOLOGIST_ONLY",
        description="Dense RA voxel-level supervision only",
        parameter_source="N1 default selected run",
        train_csv="N1_radiologist_only_train.csv",
        train_dataset_task="radiologist_only",
        use_dense=True,
        dense_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "n2": ExperimentSpec(
        key="n2",
        mode="N2_PUB_TCIA_TBX_ONLY",
        description="Dense RA + TBx target-ROI supervision",
        parameter_source="completed N2 RA+TBx run, excluding its P/A supervision",
        train_csv="N2_PUB_TCIA_TBx_only_train.csv",
        use_dense=True,
        use_tbx=True,
        dense_weight=1.0,
        tbx_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "n3": ExperimentSpec(
        key="n3",
        mode="N3_PUB_TCIA_SBX_ONLY",
        description="Dense RA + SBx region supervision",
        parameter_source="latest N3 RA+SBx run, excluding its P/A supervision",
        train_csv="N3_PUB_TCIA_SBx_only_train.csv",
        use_dense=True,
        use_sbx=True,
        dense_weight=1.0,
        sbx_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "n4": ExperimentSpec(
        key="n4",
        mode="N4_MIXED_CLEAN",
        description="Dense RA + TBx + SBx supervision",
        parameter_source="evidence-guided N4 weights from N2/N3: D/T/S=1/0.05/0.25",
        train_csv="N4_mixed_PUB_TCIA_train.csv",
        use_dense=True,
        use_tbx=True,
        use_sbx=True,
        dense_weight=1.0,
        tbx_weight=0.05,
        sbx_weight=0.25,
        use_curriculum=True,
        dense_start=1,
        tbx_start=15,
        sbx_start=15,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
    "n5": ExperimentSpec(
        key="n5",
        mode="N5_MIXED_PATIENT",
        description="Dense RA + TBx + SBx + biopsy-confirmed patient supervision",
        parameter_source="N4 evidence-guided weights + prior PatientRiskW0.05",
        train_csv="N4_mixed_PUB_TCIA_train.csv",
        use_dense=True,
        use_tbx=True,
        use_sbx=True,
        use_patient=True,
        dense_weight=1.0,
        tbx_weight=0.05,
        sbx_weight=0.25,
        patient_weight=0.05,
        use_curriculum=True,
        dense_start=1,
        tbx_start=15,
        sbx_start=15,
        patient_start=1,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N1--N5 experiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
