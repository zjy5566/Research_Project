"""
Test/inference script for the revised lesion-segmentation + MIL setting.

This version removes grade/gland evaluation and reports:
  - voxel-level lesion metrics when dense lesion masks are available, e.g. PUB
  - patient-level metrics derived from segmentation risk maps
  - region-level metrics from risk maps and RA-lesion-to-zone IoU labels
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import ProstateUnifiedDataset
from Loss_function import MixedSupervisionLoss

try:
    from model import ProstateSegMILNet as ModelClass
except ImportError:  # pragma: no cover - transition compatibility
    from model import ProstateMixedSupervisionNet as ModelClass

import utils


VISUALIZATION_POLICIES = ("none", "errors", "representative", "all")


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(Config, name, default)


def get_dataset_task() -> str:
    return _cfg(
        "TEST_DATASET_TASK",
        _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed"))),
    )


def build_dataset(csv_path: str):
    task = get_dataset_task()
    try:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
            task=task,
        )
    except TypeError:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
        )


def build_model(device: torch.device):
    try:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            max_zones=_cfg("MAX_ZONES", 20),
            base_channels=_cfg("BASE_CHANNELS", 32),
            dropout_rate=_cfg("DROPOUT_RATE", 0.0),
            mil_pooling=_cfg("MIL_POOLING", "lme"),
            lme_r=_cfg("LME_R", 8.0),
            return_dict=True,
        )
    except TypeError:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            num_grade_classes=_cfg("NUM_CLASSES", 7),
            max_zones=_cfg("MAX_ZONES", 20),
        )
    return model.to(device)


def build_criterion(device: torch.device):
    """Build the same loss module used by training for test-loss reporting."""
    positive_threshold = _cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1))
    kwargs = {
        "positive_threshold": positive_threshold,
        "invalid_sys_label": _cfg("INVALID_SYS_LABEL", -1),
        "pos_weight_val": _cfg("POS_WEIGHT_VAL", 2.0),
        "sys_pos_weight_val": _cfg("SYS_POS_WEIGHT_VAL", _cfg("POS_WEIGHT_VAL", 2.0)),
        "sys_focal_alpha": _cfg("SYS_FOCAL_ALPHA", 0.75),
        "sys_focal_gamma": _cfg("SYS_FOCAL_GAMMA", 2.0),
        "use_sys_class_balanced_bce": _cfg("USE_SYS_CLASS_BALANCED_BCE", True),
        "use_tbx_positive_only_loss": _cfg("USE_TBX_POSITIVE_ONLY_LOSS", False),
        "return_dict": True,
    }
    try:
        criterion = MixedSupervisionLoss(**kwargs)
    except TypeError:  # pragma: no cover - compatibility with older loss class
        criterion = MixedSupervisionLoss(
            csPCa_threshold=positive_threshold,
            invalid_sys_label=_cfg("INVALID_SYS_LABEL", -1),
            pos_weight_val=_cfg("POS_WEIGHT_VAL", 2.0),
        )
    return criterion.to(device)


def load_model_weights(model, model_path: str, device: torch.device):
    """Load a checkpoint or a plain state_dict.

    The function first tries exact loading. If the architecture has been renamed
    during migration, it falls back to loading only matching keys.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Remove DataParallel prefix if present.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    try:
        model.load_state_dict(cleaned, strict=True)
        print("Loaded model weights with strict=True")
    except RuntimeError as err:
        print(f"Strict loading failed: {err}")
        model_state = model.state_dict()
        matched = {
            k: v for k, v in cleaned.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(matched)}/{len(model_state)} matching tensors with strict=False")

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": cleaned}


def unpack_model_output(raw_outputs):
    if hasattr(utils, "unpack_model_output"):
        return utils.unpack_model_output(raw_outputs)
    if isinstance(raw_outputs, dict):
        return raw_outputs
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) >= 5:
        return {
            "lesion_logits": raw_outputs[2],
            "region_logits": raw_outputs[3],
            "region_valid_mask": None,
        }
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 3:
        return {
            "lesion_logits": raw_outputs[0],
            "region_logits": raw_outputs[1],
            "region_valid_mask": raw_outputs[2],
        }
    raise TypeError("Unsupported model output format.")


def infer_dataset_type(batch: Dict, b: int) -> str:
    if hasattr(utils, "infer_dataset_type"):
        return utils.infer_dataset_type(batch, b)
    pid = str(batch.get("pid", [""])[b]) if "pid" in batch else ""
    if pid.startswith("PUB"):
        return "PUB"
    if pid.startswith("TCIA"):
        return "TCIA"
    if pid.startswith("PROMIS"):
        return "PROMIS"
    if batch.get("has_lesion", torch.zeros(1))[b].item() > 0:
        return "PUB"
    if batch.get("has_target", torch.zeros(1))[b].item() > 0:
        return "TCIA"
    if batch.get("has_sys", torch.zeros(1))[b].item() > 0:
        return "PROMIS"
    return "OTHER"


def compute_patient_label(batch: Dict, b: int, positive_threshold: int, invalid_sys_label: int) -> int:
    """Biopsy-based patient label; PUB dense masks are lesion-Dice labels only."""
    has_target = batch.get("has_target", torch.zeros(1))[b].item() > 0
    has_sys = batch.get("has_sys", torch.zeros(1))[b].item() > 0
    if not (has_target or has_sys):
        return invalid_sys_label

    label = 0

    if has_target:
        if batch["target_mask"][b].max().item() >= positive_threshold:
            label = 1

    if has_sys:
        labels = batch["sys_labels"][b]
        valid = labels != invalid_sys_label
        if valid.any() and labels[valid].max().item() >= positive_threshold:
            label = 1

    return int(label)


def compute_patient_score(lesion_prob: torch.Tensor, gland_mask: torch.Tensor | None = None) -> float:
    """Patient-level score as max lesion probability, restricted to gland when possible."""
    if gland_mask is not None and gland_mask.max().item() > 0:
        values = lesion_prob[gland_mask > 0]
        if values.numel() > 0:
            return float(values.max().detach().cpu().item())
    return float(lesion_prob.max().detach().cpu().item())


def make_sys_label_volume(zones_mask: np.ndarray, sys_labels: np.ndarray, invalid_sys_label: int) -> np.ndarray:
    out = np.zeros_like(zones_mask, dtype=np.float32)
    for z_idx in range(1, min(20, len(sys_labels)) + 1):
        if int(sys_labels[z_idx - 1]) != invalid_sys_label:
            out[zones_mask == z_idx] = float(sys_labels[z_idx - 1])
    return out


def choose_visual_slice(lesion_prob: np.ndarray, lesion_gt: np.ndarray, target_gt: np.ndarray, sys_gt: np.ndarray) -> int:
    if lesion_gt.sum() > 0:
        return int(np.argmax(lesion_gt.sum(axis=(1, 2))))
    if target_gt.sum() > 0:
        return int(np.argmax(target_gt.sum(axis=(1, 2))))
    if sys_gt.sum() > 0:
        return int(np.argmax(sys_gt.sum(axis=(1, 2))))
    if lesion_prob.sum() > 0:
        return int(np.argmax(lesion_prob.sum(axis=(1, 2))))
    return int(lesion_prob.shape[0] // 2)


def save_seg_mil_vis(
    img_vol: np.ndarray,
    lesion_gt: np.ndarray,
    target_gt: np.ndarray,
    zones_mask: np.ndarray,
    sys_labels: np.ndarray,
    lesion_prob: np.ndarray,
    region_label_map: np.ndarray | None,
    region_pred_map: np.ndarray | None,
    pid: str,
    save_path: str,
):
    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    sys_gt = make_sys_label_volume(zones_mask, sys_labels, invalid_sys_label)
    sys_pos = (sys_gt >= positive_threshold).astype(np.float32)

    z = choose_visual_slice(lesion_prob, lesion_gt, target_gt, sys_gt)
    s_img = img_vol[z]
    s_prob = lesion_prob[z]
    s_lesion = lesion_gt[z]
    s_target = target_gt[z]
    s_zones = zones_mask[z]
    s_sys_pos = sys_pos[z]
    s_region_label = region_label_map[z] if region_label_map is not None else None
    s_region_pred = region_pred_map[z] if region_pred_map is not None else None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Patient: {pid} | Slice: {z}/{img_vol.shape[0]}", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(s_img, cmap="gray")
    axes[0, 0].set_title("T2 MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(s_img, cmap="gray")
    risk_masked = np.ma.masked_where(s_prob < 0.1, s_prob)
    im_risk = axes[0, 1].imshow(risk_masked, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted lesion probability")
    axes[0, 1].axis("off")
    fig.colorbar(im_risk, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[0, 2].imshow(np.ma.masked_where(s_lesion == 0, s_lesion), cmap="autumn", alpha=0.55)
    axes[0, 2].set_title("GT dense lesion mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(s_img, cmap="gray")
    if s_target.sum() > 0:
        target_pos = (s_target >= positive_threshold).astype(np.float32)
        axes[1, 0].imshow(np.ma.masked_where(target_pos == 0, target_pos), cmap="autumn", alpha=0.55)
    axes[1, 0].set_title("TBx-positive target ROI voxels")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(s_img, cmap="gray")
    if s_zones.max() > 0:
        axes[1, 1].imshow(np.ma.masked_where(s_zones == 0, s_zones), cmap="tab20", alpha=0.18)
    if s_region_label is not None and np.max(s_region_label) > 0:
        axes[1, 1].imshow(np.ma.masked_where(s_region_label == 0, s_region_label), cmap="Greens", alpha=0.42)
    if s_region_pred is not None and np.max(s_region_pred) > 0:
        axes[1, 1].imshow(np.ma.masked_where(s_region_pred == 0, s_region_pred), cmap="Reds", alpha=0.42)
    elif s_sys_pos.sum() > 0:
        axes[1, 1].contour(s_sys_pos, levels=[0.5], linewidths=1.5)
    axes[1, 1].set_title("Regions: label green / pred red")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[1, 2].contour(s_lesion, levels=[0.5], linewidths=2)
    if s_prob.max() > 0:
        axes[1, 2].contour((s_prob >= 0.5).astype(np.float32), levels=[0.5], linewidths=2, linestyles="dashed")
    axes[1, 2].set_title("Contours: GT solid / Pred dashed")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


class TestArtifactExporter:
    """Collect per-case/per-region metrics and selected QA visualisations.

    The exporter is deliberately independent from the aggregate metric tracker:
    aggregate AUC/FROC values remain dataset-level statistics, while this class
    writes only metrics that are well-defined for one test case or one region.
    """

    __test__ = False  # Prevent pytest from treating this helper as a test class.
    REGION_COLUMNS = (
        "sample_index",
        "patient_id",
        "source",
        "dataset_label",
        "dataset_csv",
        "checkpoint_label",
        "checkpoint_epoch",
        "checkpoint_path",
        "zone_id",
        "region_label",
        "region_score",
        "region_pred",
        "region_correct",
        "region_confusion",
    )

    def __init__(
        self,
        output_dir: str,
        *,
        dataset_label: str = "external",
        dataset_csv: str = "",
        checkpoint_label: str = "best",
        checkpoint_path: str = "",
        checkpoint_epoch: int = 0,
        visualization_policy: str = "representative",
        max_visualizations: int = 12,
        low_dice_threshold: float = 0.5,
    ):
        policy = str(visualization_policy).lower()
        if policy not in VISUALIZATION_POLICIES:
            raise ValueError(
                f"Unknown visualization policy {visualization_policy!r}; "
                f"choose from {VISUALIZATION_POLICIES}."
            )

        self.output_dir = os.path.abspath(output_dir)
        self.visualization_dir = os.path.join(self.output_dir, "visualizations")
        self.dataset_label = str(dataset_label)
        self.dataset_csv = str(dataset_csv)
        self.checkpoint_label = str(checkpoint_label)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_epoch = int(checkpoint_epoch)
        self.visualization_policy = policy
        self.max_visualizations = max(0, int(max_visualizations))
        self.low_dice_threshold = float(low_dice_threshold)
        self.sample_rows = []
        self.region_rows = []
        self.saved_visualizations = 0
        self.saved_representative_buckets = set()

    @staticmethod
    def _flag(batch: Dict, key: str, b: int) -> bool:
        if key not in batch:
            return False
        value = batch[key][b]
        if torch.is_tensor(value):
            return bool(value.item() > 0)
        return bool(value)

    @staticmethod
    def _binary_metrics(prefix: str, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred = pred.bool().reshape(-1)
        target = target.bool().reshape(-1)
        tp = int(torch.logical_and(pred, target).sum().item())
        fp = int(torch.logical_and(pred, ~target).sum().item())
        fn = int(torch.logical_and(~pred, target).sum().item())
        tn = int(torch.logical_and(~pred, ~target).sum().item())
        pred_pos = tp + fp
        gt_pos = tp + fn
        dice_denom = 2 * tp + fp + fn
        dice = 1.0 if dice_denom == 0 else (2.0 * tp) / dice_denom
        sensitivity = float("nan") if gt_pos == 0 else tp / gt_pos
        specificity = float("nan") if (tn + fp) == 0 else tn / (tn + fp)
        return {
            f"{prefix}_num_voxels": int(target.numel()),
            f"{prefix}_gt_positive_voxels": gt_pos,
            f"{prefix}_pred_positive_voxels": pred_pos,
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_tn": tn,
            f"{prefix}_dice": float(dice),
            f"{prefix}_f1": float(dice),
            f"{prefix}_sensitivity": float(sensitivity),
            f"{prefix}_specificity": float(specificity),
        }

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        return float("nan") if denominator == 0 else float(numerator / denominator)

    def _visualization_reason(self, row: Dict[str, Any]) -> str:
        reasons = []
        if row.get("patient_correct") == 0:
            reasons.append("patient_error")
        if int(row.get("region_fp", 0)) + int(row.get("region_fn", 0)) > 0:
            reasons.append("region_error")

        lesion_dice = row.get("lesion_dice", np.nan)
        if np.isfinite(lesion_dice) and lesion_dice < self.low_dice_threshold:
            reasons.append("low_lesion_dice")
        target_dice = row.get("target_cspca_dice", np.nan)
        if np.isfinite(target_dice) and target_dice < self.low_dice_threshold:
            reasons.append("low_target_dice")
        return "+".join(reasons)

    def _should_visualize(self, row: Dict[str, Any]) -> Tuple[bool, str, Optional[Tuple[str, str]]]:
        if self.visualization_policy == "none":
            return False, "", None
        if self.max_visualizations > 0 and self.saved_visualizations >= self.max_visualizations:
            return False, "", None

        error_reason = self._visualization_reason(row)
        if self.visualization_policy == "errors":
            return bool(error_reason), error_reason, None
        if self.visualization_policy == "all":
            return True, error_reason or "all", None

        label = row.get("patient_label", np.nan)
        label_bucket = "unlabelled" if not np.isfinite(label) else f"patient_{int(label)}"
        bucket = (str(row.get("source", "OTHER")), label_bucket)
        if error_reason:
            return True, error_reason, bucket
        if bucket not in self.saved_representative_buckets:
            return True, "representative", bucket
        return False, "", bucket

    def _save_visualization(
        self,
        *,
        batch: Dict,
        b: int,
        lesion_prob: torch.Tensor,
        region_label_map: Optional[np.ndarray],
        region_pred_map: Optional[np.ndarray],
        row: Dict[str, Any],
        reason: str,
        bucket: Optional[Tuple[str, str]],
    ) -> None:
        os.makedirs(self.visualization_dir, exist_ok=True)
        patient_id = str(row["patient_id"])
        filename = (
            f"{int(row['sample_index']):04d}_{utils.safe_vis_filename(row['source'])}_"
            f"{utils.safe_vis_filename(patient_id)}.png"
        )
        save_path = os.path.join(self.visualization_dir, filename)
        empty_like = torch.zeros_like(lesion_prob)
        sys_labels = batch.get(
            "sys_labels",
            torch.full(
                (batch["input"].size(0), int(_cfg("MAX_ZONES", 20))),
                int(_cfg("INVALID_SYS_LABEL", -1)),
                device=lesion_prob.device,
            ),
        )
        save_seg_mil_vis(
            img_vol=batch["input"][b, 0].detach().cpu().numpy(),
            lesion_gt=utils.mask_for_visualisation(batch, "lesion_mask", b, empty_like),
            target_gt=utils.mask_for_visualisation(batch, "target_mask", b, empty_like),
            zones_mask=utils.mask_for_visualisation(batch, "zones_mask", b, empty_like),
            sys_labels=sys_labels[b].detach().cpu().numpy(),
            lesion_prob=lesion_prob.detach().cpu().numpy(),
            region_label_map=region_label_map,
            region_pred_map=region_pred_map,
            pid=patient_id,
            save_path=save_path,
        )
        row["visualization_path"] = os.path.relpath(save_path, self.output_dir)
        row["visualization_reason"] = reason
        self.saved_visualizations += 1
        if bucket is not None:
            self.saved_representative_buckets.add(bucket)

    def update(self, batch: Dict, lesion_probs: torch.Tensor, seg_evaluator) -> None:
        """Add every case in one inference batch to the export tables."""
        prob_threshold = float(seg_evaluator.prob_threshold)
        positive_threshold = int(seg_evaluator.positive_threshold)
        device = lesion_probs.device

        for b in range(lesion_probs.size(0)):
            patient_id = str(batch.get("pid", [f"case_{len(self.sample_rows) + 1}"])[b])
            source = infer_dataset_type(batch, b)
            lesion_prob = lesion_probs[b, 0]
            pred_binary = lesion_prob >= prob_threshold
            has_lesion = self._flag(batch, "has_lesion", b)
            has_target = self._flag(batch, "has_target", b)
            has_sys = self._flag(batch, "has_sys", b)
            has_gland = self._flag(batch, "has_gland", b)

            row: Dict[str, Any] = {
                "sample_index": len(self.sample_rows) + 1,
                "patient_id": patient_id,
                "source": source,
                "dataset_label": self.dataset_label,
                "dataset_csv": self.dataset_csv,
                "checkpoint_label": self.checkpoint_label,
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_path": self.checkpoint_path,
                "probability_threshold": prob_threshold,
                "positive_label_threshold": positive_threshold,
                "has_lesion": int(has_lesion),
                "has_target": int(has_target),
                "has_sys": int(has_sys),
                "has_gland": int(has_gland),
                "risk_min": float(lesion_prob.min().item()),
                "risk_mean": float(lesion_prob.mean().item()),
                "risk_max": float(lesion_prob.max().item()),
                "risk_std": float(lesion_prob.float().std(unbiased=False).item()),
                "pred_positive_voxels": int(pred_binary.sum().item()),
                "pred_positive_fraction": float(pred_binary.float().mean().item()),
                "lesion_dice": np.nan,
                "target_cspca_dice": np.nan,
                "patient_label": np.nan,
                "patient_score": np.nan,
                "patient_pred": np.nan,
                "patient_correct": np.nan,
                "patient_confusion": "",
                "region_n": 0,
                "region_positive_gt": 0,
                "region_positive_pred": 0,
                "region_tp": 0,
                "region_fp": 0,
                "region_fn": 0,
                "region_tn": 0,
                "region_sensitivity": np.nan,
                "region_specificity": np.nan,
                "region_bacc": np.nan,
                "region_f1": np.nan,
                "case_has_error": 0,
                "visualization_path": "",
                "visualization_reason": "",
                "visualization_error": "",
            }

            if has_lesion and "lesion_mask" in batch:
                lesion_gt = batch["lesion_mask"][b, 0] > 0
                row.update(self._binary_metrics("lesion", pred_binary, lesion_gt))

            if has_target and "target_mask" in batch:
                target_mask = batch["target_mask"][b, 0]
                sampled_roi = target_mask > 0
                target_gt = target_mask >= positive_threshold
                row["tbx_sampled_voxels"] = int(sampled_roi.sum().item())
                if sampled_roi.any():
                    roi_metrics = self._binary_metrics(
                        "tbx_roi",
                        pred_binary[sampled_roi],
                        target_gt[sampled_roi],
                    )
                    row.update(roi_metrics)
                    if int(target_gt[sampled_roi].sum().item()) > 0:
                        row["target_cspca_dice"] = roi_metrics["tbx_roi_dice"]
            else:
                row["tbx_sampled_voxels"] = 0

            patient_label = seg_evaluator._patient_label(batch, b, device)
            if patient_label is not None:
                patient_mask = seg_evaluator._patient_score_mask(batch, b, device, lesion_prob)
                patient_score = utils.masked_probability_pool(
                    lesion_prob,
                    patient_mask,
                    mode=seg_evaluator.patient_pooling,
                    top_percent=seg_evaluator.top_percent,
                    lme_r=seg_evaluator.lme_r,
                )
                if patient_score is not None:
                    patient_pred = int(patient_score >= prob_threshold)
                    row["patient_label"] = int(patient_label)
                    row["patient_score"] = float(patient_score)
                    row["patient_pred"] = patient_pred
                    row["patient_correct"] = int(patient_pred == int(patient_label))
                    row["patient_confusion"] = (
                        "TP" if patient_label == 1 and patient_pred == 1
                        else "FN" if patient_label == 1
                        else "FP" if patient_pred == 1
                        else "TN"
                    )

            region_info = seg_evaluator.case_region_info(lesion_prob, batch, b, device)
            region_label_map = None
            region_pred_map = None
            if region_info is not None:
                region_label_map = region_info["label_map"].detach().cpu().numpy()
                region_pred_map = region_info["pred_map"].detach().cpu().numpy()
                zone_ids = sorted(region_info["zone_score"])
                for zone_id in zone_ids:
                    y_true = int(region_info["zone_true"][zone_id])
                    y_score = float(region_info["zone_score"][zone_id])
                    y_pred = int(region_info["zone_pred"][zone_id])
                    self.region_rows.append(
                        {
                            "sample_index": row["sample_index"],
                            "patient_id": patient_id,
                            "source": source,
                            "dataset_label": self.dataset_label,
                            "dataset_csv": self.dataset_csv,
                            "checkpoint_label": self.checkpoint_label,
                            "checkpoint_epoch": self.checkpoint_epoch,
                            "checkpoint_path": self.checkpoint_path,
                            "zone_id": int(zone_id),
                            "region_label": y_true,
                            "region_score": y_score,
                            "region_pred": y_pred,
                            "region_correct": int(y_true == y_pred),
                            "region_confusion": (
                                "TP" if y_true == 1 and y_pred == 1
                                else "FN" if y_true == 1
                                else "FP" if y_pred == 1
                                else "TN"
                            ),
                        }
                    )

                y_true = np.asarray([region_info["zone_true"][z] for z in zone_ids], dtype=np.int64)
                y_pred = np.asarray([region_info["zone_pred"][z] for z in zone_ids], dtype=np.int64)
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                sens = self._safe_ratio(tp, tp + fn)
                spec = self._safe_ratio(tn, tn + fp)
                finite = [value for value in (sens, spec) if np.isfinite(value)]
                row.update(
                    {
                        "region_n": len(zone_ids),
                        "region_positive_gt": int(y_true.sum()),
                        "region_positive_pred": int(y_pred.sum()),
                        "region_tp": tp,
                        "region_fp": fp,
                        "region_fn": fn,
                        "region_tn": tn,
                        "region_sensitivity": sens,
                        "region_specificity": spec,
                        "region_bacc": float(np.mean(finite)) if finite else np.nan,
                        "region_f1": self._safe_ratio(2 * tp, 2 * tp + fp + fn),
                    }
                )

            error_reason = self._visualization_reason(row)
            row["case_has_error"] = int(bool(error_reason))
            should_save, reason, bucket = self._should_visualize(row)
            if should_save:
                try:
                    self._save_visualization(
                        batch=batch,
                        b=b,
                        lesion_prob=lesion_prob,
                        region_label_map=region_label_map,
                        region_pred_map=region_pred_map,
                        row=row,
                        reason=reason,
                        bucket=bucket,
                    )
                except Exception as exc:  # Keep metrics even if rendering fails.
                    row["visualization_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"Warning: failed to save test visualization for {patient_id}: {exc}")

            self.sample_rows.append(row)

    def finalize(self) -> pd.DataFrame:
        os.makedirs(self.output_dir, exist_ok=True)
        sample_path = os.path.join(self.output_dir, "per_sample_metrics.csv")
        region_path = os.path.join(self.output_dir, "per_region_metrics.csv")
        sample_df = pd.DataFrame(self.sample_rows)
        region_df = pd.DataFrame(self.region_rows, columns=self.REGION_COLUMNS)
        sample_df.to_csv(sample_path, index=False)
        region_df.to_csv(region_path, index=False)
        print(f"Per-sample test metrics saved to: {sample_path}")
        print(f"Per-region test metrics saved to: {region_path}")
        if self.saved_visualizations:
            print(
                f"Saved {self.saved_visualizations} selected test visualizations to: "
                f"{self.visualization_dir}"
            )
        return sample_df


def get_test_dir() -> str:
    return _cfg("TEST_DIR", os.path.join(_cfg("BASE_DIR", "."), "test"))


def get_test_csv() -> str:
    if _cfg("TEST_CSV", None) is not None:
        return _cfg("TEST_CSV")
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    for name in [
        "N4_mixed_PROMIS_external_val.csv",
        "task_cls_external_val.csv",
        "external_val.csv",
        "test.csv",
    ]:
        candidate = os.path.join(split_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(split_dir, "N4_mixed_PROMIS_external_val.csv")


def get_model_path(test_dir: str) -> str:
    if _cfg("TEST_MODEL_PATH", None) is not None:
        return _cfg("TEST_MODEL_PATH")
    for name in ["best_checkpoint.pth", "best_model.pth"]:
        candidate = os.path.join(test_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model file not found. Check TEST_MODEL_PATH or {test_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpoint on one test CSV and export aggregate, per-case, "
            "per-region, and selected visual QA results."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Directory containing best_checkpoint.pth/best_model.pth.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path; overrides checkpoint discovery in --experiment-dir.",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Explicit test split CSV; defaults to Config.TEST_CSV.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("RP_DATASET_ROOT"),
        help="Dataset root containing Unified_Dataset (same meaning as RP_DATASET_ROOT).",
    )
    parser.add_argument(
        "--unified-data-dir",
        default=None,
        help="Explicit Unified_Dataset directory; overrides --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Default: <experiment-dir>/test_artifacts/<dataset>/<checkpoint>.",
    )
    parser.add_argument("--dataset-label", default="external")
    parser.add_argument("--checkpoint-label", default="best")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Inference device. Defaults to Config.DEVICE.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--save-images",
        choices=VISUALIZATION_POLICIES,
        default=None,
        help=(
            "none, errors only, representative errors/positive/negative cases, or all. "
            "Default: Config.TEST_VIS_POLICY (representative)."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images for this run; 0 means unlimited.",
    )
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    configured_test_name = os.path.basename(
        str(_cfg("TEST_CSV", "B1_PROMIS_external_val.csv"))
    )
    if args.experiment_dir:
        Config.TEST_DIR = os.path.abspath(args.experiment_dir)
    if args.checkpoint:
        Config.TEST_MODEL_PATH = os.path.abspath(args.checkpoint)
        if not args.experiment_dir:
            Config.TEST_DIR = os.path.dirname(Config.TEST_MODEL_PATH)
    if args.dataset_root:
        dataset_root = os.path.abspath(args.dataset_root)
        os.environ["RP_DATASET_ROOT"] = dataset_root
        Config.DATASET_ROOT = dataset_root
        Config.UNIFIED_DATA_DIR = os.path.join(dataset_root, "Unified_Dataset")
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        if not args.test_csv:
            Config.TEST_CSV = os.path.join(Config.SPLIT_DIR, configured_test_name)
    if args.unified_data_dir:
        Config.UNIFIED_DATA_DIR = os.path.abspath(args.unified_data_dir)
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        if not args.test_csv:
            Config.TEST_CSV = os.path.join(Config.SPLIT_DIR, configured_test_name)
    if args.test_csv:
        Config.TEST_CSV = os.path.abspath(args.test_csv)
    if args.device:
        Config.DEVICE = args.device
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        Config.BATCH_SIZE = int(args.batch_size)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative.")
        Config.NUM_WORKERS = int(args.num_workers)


def _resolve_device() -> torch.device:
    requested = str(_cfg("DEVICE", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable in this PyTorch build.")
    return torch.device(requested)


def _preflight_paths(model_path: str, test_csv: str) -> None:
    missing = []
    if not os.path.isfile(model_path):
        missing.append(f"checkpoint: {model_path}")
    if not os.path.isfile(test_csv):
        missing.append(f"test CSV: {test_csv}")
    unified_dir = str(_cfg("UNIFIED_DATA_DIR", ""))
    if not os.path.isdir(unified_dir):
        missing.append(f"Unified_Dataset: {unified_dir}")
    if missing:
        detail = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Test preflight failed; the following inputs are unavailable:\n"
            f"  - {detail}\n"
            "Pass --dataset-root/--unified-data-dir, --test-csv, and --checkpoint as needed."
        )


def _summary_row(
    tracker,
    *,
    dataset_label: str,
    test_csv: str,
    checkpoint_label: str,
    checkpoint_epoch: int,
    checkpoint_path: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "test_dataset_label": dataset_label,
        "checkpoint_label": checkpoint_label,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": checkpoint_path,
        "test_csv": test_csv,
    }
    for key, value in tracker.get_val_dict().items():
        test_key = key.replace("val_", "test_", 1) if key.startswith("val_") else f"test_{key}"
        row[test_key] = value
    return row


def main() -> None:
    args = _parse_args()
    _apply_cli_overrides(args)
    if hasattr(Config, "set_seed"):
        Config.set_seed()

    experiment_dir = get_test_dir()
    model_path = os.path.abspath(get_model_path(experiment_dir))
    test_csv = os.path.abspath(get_test_csv())
    _preflight_paths(model_path, test_csv)
    device = _resolve_device()

    checkpoint_label = str(args.checkpoint_label)
    dataset_label = str(args.dataset_label)
    output_dir = args.output_dir or os.path.join(
        experiment_dir,
        str(_cfg("TEST_ARTIFACT_SUBDIR", "test_artifacts")),
        utils.safe_vis_filename(dataset_label),
        utils.safe_vis_filename(checkpoint_label),
    )
    output_dir = os.path.abspath(output_dir)

    print(f"[Experiment]  {_cfg('EXPERIMENT_MODE', 'unknown')}")
    print(f"[Checkpoint]  {model_path}")
    print(f"[Test CSV]    {test_csv}")
    print(f"[Data root]   {_cfg('UNIFIED_DATA_DIR', '')}")
    print(f"[Device]      {device}")
    print(f"[Output]      {output_dir}")

    test_loader = DataLoader(
        build_dataset(test_csv),
        batch_size=int(_cfg("BATCH_SIZE", 1)),
        shuffle=False,
        num_workers=int(_cfg("NUM_WORKERS", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(device)
    criterion = build_criterion(device)
    checkpoint = load_model_weights(model, model_path, device)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if checkpoint.get("criterion_state_dict") is not None:
        try:
            criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
        except TypeError:  # pragma: no cover - older torch compatibility
            criterion.load_state_dict(checkpoint["criterion_state_dict"])
    if hasattr(criterion, "set_epoch") and checkpoint_epoch > 0:
        criterion.set_epoch(checkpoint_epoch)

    visualization_policy = args.save_images or str(
        _cfg("TEST_VIS_POLICY", "representative")
    )
    max_visualizations = (
        int(args.max_images)
        if args.max_images is not None
        else int(_cfg("TEST_VIS_MAX_PER_RUN", 12))
    )
    exporter = TestArtifactExporter(
        output_dir,
        dataset_label=dataset_label,
        dataset_csv=test_csv,
        checkpoint_label=checkpoint_label,
        checkpoint_path=model_path,
        checkpoint_epoch=checkpoint_epoch,
        visualization_policy=visualization_policy,
        max_visualizations=max_visualizations,
        low_dice_threshold=float(_cfg("TEST_VIS_LOW_DICE_THRESHOLD", 0.5)),
    )

    tracker = utils.validate(
        model,
        test_loader,
        criterion,
        device,
        checkpoint_epoch,
        save_dir="",
        compute_operating_metrics=bool(_cfg("FINAL_TEST_COMPUTE_OPERATING_METRICS", True)),
        compute_froc_metrics=bool(_cfg("FINAL_TEST_COMPUTE_FROC_METRICS", True)),
        sample_exporter=exporter,
    )
    summary = _summary_row(
        tracker,
        dataset_label=dataset_label,
        test_csv=test_csv,
        checkpoint_label=checkpoint_label,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_path=model_path,
    )
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"Test [{dataset_label}/{checkpoint_label}] | {tracker.print_val_summary()}")
    print(f"Aggregate test metrics saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
