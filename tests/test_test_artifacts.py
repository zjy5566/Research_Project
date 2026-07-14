import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test import TestArtifactExporter
from utils import SegRiskMapEvaluator


def test_test_artifact_exporter_writes_every_sample_and_region(tmp_path):
    lesion_probs = torch.tensor(
        [
            [[[[0.90, 0.10, 0.70, 0.20]]]],
            [[[[0.80, 0.70, 0.10, 0.20]]]],
        ],
        dtype=torch.float32,
    )
    lesion_mask = torch.tensor(
        [
            [[[[1.0, 0.0, 0.0, 0.0]]]],
            [[[[0.0, 0.0, 0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )
    zones_mask = torch.tensor(
        [
            [[[[0.0, 0.0, 0.0, 0.0]]]],
            [[[[1.0, 1.0, 2.0, 2.0]]]],
        ],
        dtype=torch.float32,
    )
    sys_labels = torch.full((2, 20), -1, dtype=torch.long)
    sys_labels[1, 0] = 3
    sys_labels[1, 1] = 0
    batch = {
        "pid": ["PUB_case", "PROMIS_case"],
        "source": ["PUB", "PROMIS"],
        "input": torch.zeros((2, 3, 1, 1, 4), dtype=torch.float32),
        "lesion_mask": lesion_mask,
        "target_mask": torch.zeros_like(lesion_mask),
        "zones_mask": zones_mask,
        "gland_mask": torch.ones_like(lesion_mask),
        "sys_labels": sys_labels,
        "has_lesion": torch.tensor([1.0, 0.0]),
        "has_target": torch.tensor([0.0, 0.0]),
        "has_sys": torch.tensor([0.0, 1.0]),
        "has_gland": torch.tensor([1.0, 1.0]),
    }
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        patient_pooling="max",
        region_pooling="max",
        max_zones=2,
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    evaluator.update_from_batch(lesion_probs, batch)

    exporter = TestArtifactExporter(
        str(tmp_path),
        dataset_label="external",
        dataset_csv="test.csv",
        checkpoint_label="best",
        checkpoint_path="best_checkpoint.pth",
        checkpoint_epoch=7,
        visualization_policy="none",
    )
    exporter.update(batch, lesion_probs, evaluator)
    sample_df = exporter.finalize()

    sample_csv = tmp_path / "per_sample_metrics.csv"
    region_csv = tmp_path / "per_region_metrics.csv"
    assert sample_csv.exists()
    assert region_csv.exists()
    assert len(sample_df) == 2
    assert list(sample_df["patient_id"]) == ["PUB_case", "PROMIS_case"]

    pub = sample_df.loc[sample_df["patient_id"] == "PUB_case"].iloc[0]
    assert pub["lesion_tp"] == 1
    assert pub["lesion_fp"] == 1
    assert abs(pub["lesion_dice"] - 2.0 / 3.0) < 1e-8
    assert pub["patient_confusion"] == ""

    region_df = pd.read_csv(region_csv)
    assert len(region_df) == 2
    assert set(region_df["region_confusion"]) == {"TP", "TN"}
    assert region_df["region_correct"].sum() == 2


def test_test_artifact_exporter_saves_selected_visualization(tmp_path):
    lesion_probs = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)
    lesion_probs[0, 0, 2, 4, 4] = 0.9
    lesion_mask = torch.zeros_like(lesion_probs)
    lesion_mask[0, 0, 2, 4, 4] = 1.0
    batch = {
        "pid": ["PUB_visual"],
        "source": ["PUB"],
        "input": torch.zeros((1, 3, 4, 8, 8), dtype=torch.float32),
        "lesion_mask": lesion_mask,
        "target_mask": torch.zeros_like(lesion_mask),
        "zones_mask": torch.zeros_like(lesion_mask),
        "gland_mask": torch.ones_like(lesion_mask),
        "sys_labels": torch.full((1, 20), -1, dtype=torch.long),
        "has_lesion": torch.tensor([1.0]),
        "has_target": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
        "has_gland": torch.tensor([1.0]),
    }
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        patient_pooling="max",
        region_pooling="max",
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    evaluator.update_from_batch(lesion_probs, batch)
    exporter = TestArtifactExporter(
        str(tmp_path),
        visualization_policy="representative",
        max_visualizations=1,
    )
    exporter.update(batch, lesion_probs, evaluator)
    sample_df = exporter.finalize()

    relative_path = sample_df.iloc[0]["visualization_path"]
    assert relative_path
    assert (tmp_path / relative_path).exists()
    assert sample_df.iloc[0]["visualization_reason"] == "representative"
