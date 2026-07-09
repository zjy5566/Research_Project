import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Loss_function import MixedSupervisionLoss
from utils import (
    LesionMILEvaluator,
    MetricTracker,
    SegRiskMapEvaluator,
    compute_dice_per_case,
    compute_topk_dice_per_case,
)


def test_tbx_pos_neg_bce_uses_sampled_roi_and_ignores_unsampled():
    criterion = MixedSupervisionLoss(
        positive_threshold=3,
        pos_weight_val=1.0,
        use_tbx_positive_only_loss=False,
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 1.0,
            "lesion_sys": 0.0,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": True,
            "lesion_sys": False,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 1.0, -1.0, 2.0]]]]])
    target_mask = torch.tensor([[[[[0.0, 1.0, 3.0, 2.0]]]]])
    batch = {
        "target_mask": target_mask,
        "has_target": torch.tensor([1.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)

    valid_logits = torch.tensor([1.0, -1.0, 2.0])
    valid_targets = torch.tensor([0.0, 1.0, 0.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
    )

    torch.testing.assert_close(loss_dict["loss_lesion_sparse"], expected)
    torch.testing.assert_close(loss_dict["total_loss"], expected)

    counts = loss_dict["loss_counts"]
    assert counts["lesion_sparse_voxels"] == 3
    assert counts["lesion_sparse_positive_voxels"] == 1
    assert counts["lesion_sparse_negative_voxels"] == 2
    torch.testing.assert_close(
        torch.tensor(counts["tbx_pos_prob_mean"]),
        torch.sigmoid(torch.tensor(-1.0)),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_prob_mean"]),
        torch.sigmoid(torch.tensor([1.0, 2.0])).mean(),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_1mp_mean"]),
        (1.0 - torch.sigmoid(torch.tensor([1.0, 2.0]))).mean(),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_pos_bce"]),
        torch.nn.functional.softplus(torch.tensor(1.0)),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_bce"]),
        torch.nn.functional.softplus(torch.tensor([1.0, 2.0])).mean(),
    )

    tracker = MetricTracker()
    tracker.update_losses(loss_dict)
    assert tracker.tbx_pos_prob_mean.count == 1
    assert tracker.tbx_neg_prob_mean.count == 2
    torch.testing.assert_close(
        torch.tensor(tracker.tbx_pos_bce.avg),
        torch.nn.functional.softplus(torch.tensor(1.0)),
    )


def test_pub_dense_cases_do_not_enter_patient_metrics():
    evaluator = LesionMILEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        invalid_sys_label=-1,
    )

    lesion_probs = torch.tensor(
        [
            [[[[0.9]]]],
            [[[[0.8]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_lesion": torch.tensor([1.0, 0.0]),
        "has_target": torch.tensor([0.0, 1.0]),
        "has_sys": torch.tensor([0.0, 0.0]),
        "lesion_mask": torch.tensor([[[[[1.0]]]], [[[[0.0]]]]]),
        "target_mask": torch.tensor([[[[[0.0]]]], [[[[3.0]]]]]),
    }

    evaluator.update_from_batch(lesion_probs, batch)
    metrics = evaluator.compute_metrics()

    assert metrics["patient_n"] == 1
    assert evaluator.patient_true == [1]
    assert abs(evaluator.patient_score[0] - 0.8) < 1e-6


def test_seg_risk_map_metrics_use_seg_patient_gt_and_sbx_region_gt():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        top_percent=50.0,
        max_zones=2,
        invalid_sys_label=-1,
    )

    lesion_probs = torch.tensor(
        [
            [[[[0.9, 0.8, 0.1, 0.1]]]],
            [[[[0.7, 0.1, 0.1, 0.1]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_lesion": torch.tensor([1.0, 1.0]),
        "lesion_mask": torch.tensor(
            [
                [[[[1.0, 0.0, 0.0, 0.0]]]],
                [[[[0.0, 0.0, 0.0, 0.0]]]],
            ],
            dtype=torch.float32,
        ),
        "gland_mask": torch.ones_like(lesion_probs),
        "zones_mask": torch.tensor(
            [
                [[[[1.0, 1.0, 2.0, 2.0]]]],
                [[[[1.0, 1.0, 2.0, 2.0]]]],
            ],
            dtype=torch.float32,
        ),
        "sys_labels": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    evaluator.update_from_batch(lesion_probs, batch)
    metrics = evaluator.compute_metrics()

    assert evaluator.patient_true == [1, 0]
    assert abs(evaluator.patient_score[0] - 0.85) < 1e-6
    assert metrics["patient_n"] == 2
    assert metrics["patient_sens"] > 0.99
    assert metrics["patient_spec"] > 0.99
    assert metrics["region_n"] == 4
    assert evaluator.region_true == [1, 0, 0, 0]

    sbx_only_evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        top_percent=50.0,
        max_zones=2,
        invalid_sys_label=-1,
    )
    sbx_only_batch = {
        "has_lesion": torch.tensor([0.0, 0.0]),
        "has_target": torch.tensor([0.0, 0.0]),
        "zones_mask": batch["zones_mask"],
        "sys_labels": batch["sys_labels"],
    }
    sbx_only_evaluator.update_from_batch(lesion_probs, sbx_only_batch)
    assert sbx_only_evaluator.patient_true == [1, 0]
    assert sbx_only_evaluator.region_true == [1, 0, 0, 0]


def test_outside_gland_penalty_uses_only_outside_gland_voxels():
    criterion = MixedSupervisionLoss(
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 0.0,
            "lesion_sys": 0.0,
            "lesion_outside_gland": 0.5,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "lesion_outside_gland": True,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 1.0, -1.0, 2.0]]]]])
    gland_mask = torch.tensor([[[[[0.0, 1.0, 0.0, 1.0]]]]])
    batch = {
        "gland_mask": gland_mask,
        "has_gland": torch.tensor([1.0]),
        "has_target": torch.tensor([0.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)
    expected_raw = torch.nn.functional.softplus(torch.tensor([0.0, -1.0])).mean()

    torch.testing.assert_close(loss_dict["loss_lesion_outside_gland"], expected_raw)
    torch.testing.assert_close(loss_dict["total_loss"], expected_raw * 0.5)
    counts = loss_dict["loss_counts"]
    assert counts["lesion_outside_gland_cases"] == 1
    assert counts["lesion_outside_gland_voxels"] == 2
    torch.testing.assert_close(
        torch.tensor(counts["outside_gland_prob_mean"]),
        torch.sigmoid(torch.tensor([0.0, -1.0])).mean(),
    )


def test_patient_risk_loss_pools_risk_map_inside_gland():
    criterion = MixedSupervisionLoss(
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 0.0,
            "lesion_sys": 0.0,
            "lesion_outside_gland": 0.0,
            "lesion_patient": 0.5,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "lesion_outside_gland": False,
            "lesion_patient": True,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 2.0, -1.0, 1.0]]]]])
    gland_mask = torch.tensor([[[[[0.0, 1.0, 0.0, 1.0]]]]])
    batch = {
        "gland_mask": gland_mask,
        "has_gland": torch.tensor([1.0]),
        "has_cls": torch.tensor([1.0]),
        "cls_cspc_label": torch.tensor([1]),
        "has_target": torch.tensor([0.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)
    inside_logits = torch.tensor([2.0, 1.0])
    r = torch.tensor(8.0)
    pooled_logit = torch.logsumexp(inside_logits * r, dim=0) / r - torch.log(
        torch.tensor(float(inside_logits.numel()))
    ) / r
    expected_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        pooled_logit.reshape(1),
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(loss_dict["loss_lesion_patient"], expected_raw)
    torch.testing.assert_close(loss_dict["total_loss"], expected_raw * 0.5)
    counts = loss_dict["loss_counts"]
    assert counts["lesion_patient_cases"] == 1
    assert counts["lesion_patient_positive_cases"] == 1
    assert counts["lesion_patient_negative_cases"] == 0
    torch.testing.assert_close(
        torch.tensor(counts["patient_risk_prob_mean"]),
        torch.sigmoid(pooled_logit),
    )


def test_target_cspca_dice_uses_only_positive_target_cases():
    tracker = MetricTracker()
    positive_threshold = 3
    prob_threshold = 0.5

    lesion_probs = torch.tensor(
        [
            [[[[0.9, 0.1, 0.8, 0.1]]]],
            [[[[0.9, 0.1, 0.1, 0.1]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_target": torch.tensor([1.0, 1.0]),
        "target_mask": torch.tensor(
            [
                [[[[3.0, 0.0, 0.0, 0.0]]]],
                [[[[1.0, 0.0, 0.0, 0.0]]]],
            ],
            dtype=torch.float32,
        ),
    }

    target_cspca = (batch["target_mask"] >= positive_threshold).float()
    positive_target_cases = (batch["has_target"] > 0) & target_cspca.reshape(target_cspca.size(0), -1).any(dim=1)
    pred_bin = (lesion_probs[positive_target_cases] >= prob_threshold).float()
    tracker.update_target_cspca_dice_values(
        compute_dice_per_case(pred_bin, target_cspca[positive_target_cases])
    )

    assert tracker.target_cspca_dice_n == 1
    expected = (2.0 + 1e-5) / (3.0 + 1e-5)
    assert abs(tracker.target_cspca_dice.avg - expected) < 1e-6


def test_tbx_roi_metrics_report_sensitivity_at_fixed_fpr():
    tracker = MetricTracker()

    tracker.update_tbx_roi_samples(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )
    tracker.finalize_tbx_roi_metrics(threshold=0.5)

    assert tracker.tbx_roi_n == 4
    assert tracker.tbx_roi_fixed_spec_target == 0.95
    assert tracker.tbx_roi_actual_fpr_at_fixed_spec <= 0.05 + 1e-8
    assert tracker.tbx_roi_sens_at_fixed_spec == 1.0
    assert tracker.tbx_roi_auc == 1.0


def test_target_cspca_aux_dice_reports_swept_threshold_and_topk_upper_bound():
    tracker = MetricTracker()
    lesion_probs = torch.tensor([[[[[0.90, 0.80, 0.70, 0.10]]]]], dtype=torch.float32)
    target = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]], dtype=torch.float32)

    tracker.update_target_cspca_aux_dice(lesion_probs, target)
    tracker.finalize_target_cspca_aux_dice()

    assert tracker.target_cspca_best_threshold_dice == 1.0
    assert tracker.target_cspca_best_threshold > 0.5
    assert tracker.target_cspca_topk_dice.avg == 1.0
    assert compute_topk_dice_per_case(lesion_probs, target, mode="target_volume")[0] == 1.0


if __name__ == "__main__":
    test_tbx_pos_neg_bce_uses_sampled_roi_and_ignores_unsampled()
    test_pub_dense_cases_do_not_enter_patient_metrics()
    test_seg_risk_map_metrics_use_seg_patient_gt_and_sbx_region_gt()
    test_outside_gland_penalty_uses_only_outside_gland_voxels()
    test_patient_risk_loss_pools_risk_map_inside_gland()
    test_target_cspca_dice_uses_only_positive_target_cases()
    test_tbx_roi_metrics_report_sensitivity_at_fixed_fpr()
    test_target_cspca_aux_dice_reports_swept_threshold_and_topk_upper_bound()
