"""
Utility functions for the revised lesion-segmentation + MIL setting.

This version matches the new project setup after 2026-06-10:
  - main voxel-level task: lesion segmentation
  - weak supervision: TCIA TBx-confirmed target lesion ROIs and SBx/PROMIS region labels
  - no grade-prediction metrics, no gland-segmentation metrics

Expected model output from the revised model:
    outputs["lesion_logits"]      : (B, 1, D, H, W)
    outputs["region_logits"]      : (B, max_zones, 1) or (B, max_zones), optional
    outputs["region_valid_mask"]  : (B, max_zones), optional

Expected loss output from the revised loss:
    loss_dict["total_loss"]
    loss_dict["loss_lesion_total"]
    loss_dict["loss_lesion_dense"]
    loss_dict["loss_lesion_sparse"]
    loss_dict["loss_lesion_sys"]
    loss_dict["em_weights"]
    loss_dict["active_tasks"]
    loss_dict["curriculum_status"]
"""

from __future__ import annotations

import os
import random
from typing import Dict, Mapping, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

from config import Config


# -----------------------------------------------------------------------------
# Basic metric helpers
# -----------------------------------------------------------------------------

def _cfg(name: str, default):
    return getattr(Config, name, default)


def tensor_to_float(value) -> float:
    """Convert tensor / numpy scalar / python number to a safe float."""
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Batch-mean Dice for binary masks."""
    pred = pred.float().contiguous().view(pred.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.mean().detach().cpu().item())


def compute_dice_per_case(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> np.ndarray:
    """Per-case Dice values for reporting mean +/- SD."""
    pred = pred.float().contiguous().view(pred.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.detach().cpu().numpy().astype(np.float64)


def compute_topk_dice_per_case(
    prob: torch.Tensor,
    target: torch.Tensor,
    mode: str = "target_volume",
    top_percent: float = 1.0,
    smooth: float = 1e-5,
) -> np.ndarray:
    """Per-case Dice from top-scoring voxels.

    mode="target_volume" uses the ground-truth positive voxel count as k, so it
    is an optimistic localisation upper bound rather than a deployable metric.
    mode="percent" uses a fixed percentage of all voxels.
    """
    prob = prob.float().contiguous().view(prob.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    values = []
    for case_idx in range(prob.shape[0]):
        target_flat = target[case_idx]
        num_voxels = int(target_flat.numel())
        if mode == "target_volume":
            k = int(target_flat.sum().detach().cpu().item())
        elif mode == "percent":
            k = int(np.ceil(num_voxels * max(float(top_percent), 0.0) / 100.0))
        else:
            raise ValueError(f"Unknown top-k Dice mode: {mode}")
        if k <= 0 or num_voxels <= 0:
            continue
        k = min(k, num_voxels)
        top_idx = torch.topk(prob[case_idx], k=k, largest=True, sorted=False).indices
        pred_flat = torch.zeros_like(target_flat)
        pred_flat[top_idx] = 1.0
        intersection = (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        values.append(float(dice.detach().cpu().item()))
    return np.asarray(values, dtype=np.float64)


def summarise_values(values) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def configured_target_dice_thresholds() -> Tuple[float, ...]:
    thresholds = _cfg("TARGET_DICE_SWEEP_THRESHOLDS", None)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)
    if isinstance(thresholds, str):
        thresholds = [float(item.strip()) for item in thresholds.split(",") if item.strip()]
    thresholds = tuple(
        sorted({round(float(th), 6) for th in thresholds if 0.0 <= float(th) <= 1.0})
    )
    return thresholds or (0.5,)


def compute_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    if targets_np.sum() == 0 and preds_np.sum() == 0:
        return 1.0
    return float(f1_score(targets_np, preds_np, zero_division=0))


def compute_sens(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np, labels=[0, 1]).ravel()
    return float(tp / (tp + fn + 1e-7))


def compute_spec(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np, labels=[0, 1]).ravel()
    return float(tn / (tn + fp + 1e-7))


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def safe_auprc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def operating_point_metrics(
    y_true,
    y_score,
    fixed_specificity: float = 0.95,
    fixed_sensitivity: float = 0.90,
) -> Dict[str, float]:
    """Return ROC operating-point metrics.

    Reports sensitivity at a minimum fixed specificity and specificity at a
    minimum fixed sensitivity. Thresholds are selected from sklearn's ROC curve.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "fixed_spec_target": float(fixed_specificity),
            "sens_at_fixed_spec": 0.0,
            "actual_spec_at_fixed_spec": 0.0,
            "actual_fpr_at_fixed_spec": 0.0,
            "threshold_at_fixed_spec": float("nan"),
            "fixed_sens_target": float(fixed_sensitivity),
            "spec_at_fixed_sens": 0.0,
            "actual_sens_at_fixed_sens": 0.0,
            "threshold_at_fixed_sens": float("nan"),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    specificity = 1.0 - fpr

    spec_candidates = np.where(specificity >= fixed_specificity)[0]
    if spec_candidates.size > 0:
        idx_spec = spec_candidates[np.argmax(tpr[spec_candidates])]
    else:
        idx_spec = int(np.argmax(specificity))

    sens_candidates = np.where(tpr >= fixed_sensitivity)[0]
    if sens_candidates.size > 0:
        idx_sens = sens_candidates[np.argmax(specificity[sens_candidates])]
    else:
        idx_sens = int(np.argmax(tpr))

    return {
        "fixed_spec_target": float(fixed_specificity),
        "sens_at_fixed_spec": float(tpr[idx_spec]),
        "actual_spec_at_fixed_spec": float(specificity[idx_spec]),
        "actual_fpr_at_fixed_spec": float(1.0 - specificity[idx_spec]),
        "threshold_at_fixed_spec": float(thresholds[idx_spec]),
        "fixed_sens_target": float(fixed_sensitivity),
        "spec_at_fixed_sens": float(specificity[idx_sens]),
        "actual_sens_at_fixed_sens": float(tpr[idx_sens]),
        "threshold_at_fixed_sens": float(thresholds[idx_sens]),
    }


# -----------------------------------------------------------------------------
# Model/loss compatibility helpers
# -----------------------------------------------------------------------------

def move_batch_to_device(batch: Mapping, device: torch.device) -> Dict:
    """Move tensor values in a batch dictionary to device; leave pid/list/string values unchanged."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def unpack_model_output(outputs) -> Dict[str, Optional[torch.Tensor]]:
    """Normalise model output to a dictionary.

    Preferred new output is already a dict. A legacy 5-tuple is also accepted:
        grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred
    In that case only lesion_pred and sys_lesion_preds are kept.
    """
    if isinstance(outputs, dict):
        return {
            "lesion_logits": outputs.get("lesion_logits"),
            "region_logits": outputs.get("region_logits"),
            "region_valid_mask": outputs.get("region_valid_mask"),
        }

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            lesion_logits, region_logits, region_valid_mask = outputs
            return {
                "lesion_logits": lesion_logits,
                "region_logits": region_logits,
                "region_valid_mask": region_valid_mask,
            }
        if len(outputs) >= 5:
            # Legacy model: (grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred)
            return {
                "lesion_logits": outputs[2],
                "region_logits": outputs[3],
                "region_valid_mask": None,
            }

    raise TypeError(
        "Unsupported model output. Expected dict, compact 3-tuple, or legacy 5-tuple."
    )


def normalise_loss_output(loss_output) -> Dict[str, object]:
    """Normalise loss output to a dictionary for clean logging.

    Preferred new output is a dict. Compact tuple from Loss_function_seg_mil.py is also accepted:
        total, lesion_total, dense, sparse, sys, em_weights, active_tasks, curriculum_status
    """
    if isinstance(loss_output, dict):
        return {
            "total_loss": loss_output.get("total_loss", 0.0),
            "loss_lesion_total": loss_output.get("loss_lesion_total", 0.0),
            "loss_lesion_dense": loss_output.get("loss_lesion_dense", 0.0),
            "loss_lesion_sparse": loss_output.get("loss_lesion_sparse", 0.0),
            "loss_lesion_sys": loss_output.get("loss_lesion_sys", 0.0),
            "em_weights": loss_output.get("em_weights", {}),
            "active_tasks": loss_output.get("active_tasks", {}),
            "curriculum_status": loss_output.get("curriculum_status", {}),
            "loss_counts": loss_output.get("loss_counts", {}),
        }

    if isinstance(loss_output, (tuple, list)):
        # New compact tuple.
        if len(loss_output) == 8:
            total, lesion_total, dense, sparse, sys, em_weights, active_tasks, curriculum_status = loss_output
            return {
                "total_loss": total,
                "loss_lesion_total": lesion_total,
                "loss_lesion_dense": dense,
                "loss_lesion_sparse": sparse,
                "loss_lesion_sys": sys,
                "em_weights": em_weights,
                "active_tasks": active_tasks,
                "curriculum_status": curriculum_status,
                "loss_counts": {},
            }

        # Legacy 12-tuple from the old multi-task loss. Grade/gland values are ignored.
        if len(loss_output) >= 12:
            return {
                "total_loss": loss_output[0],
                "loss_lesion_total": loss_output[4],
                "loss_lesion_dense": loss_output[5],
                "loss_lesion_sparse": loss_output[6],
                "loss_lesion_sys": loss_output[7],
                "em_weights": loss_output[9],
                "active_tasks": loss_output[10],
                "curriculum_status": loss_output[11],
                "loss_counts": {},
            }

    raise ValueError(f"Unexpected loss output format: {type(loss_output)}")


def call_criterion(criterion, outputs: Dict[str, torch.Tensor], batch: Mapping):
    """Call either the new criterion(outputs, batch) or a legacy criterion signature."""
    try:
        return criterion(outputs, batch)
    except TypeError:
        # Compatibility with old MixedSupervisionLoss signature.
        lesion_logits = outputs["lesion_logits"]
        region_logits = outputs.get("region_logits")
        return criterion(
            None,
            None,
            lesion_logits,
            region_logits,
            None,
            batch.get("target_mask"),
            batch.get("sys_labels"),
            batch.get("lesion_mask"),
            batch.get("gland_mask"),
            batch.get("has_target"),
            batch.get("has_sys"),
            batch.get("has_lesion"),
            batch.get("has_gland", torch.zeros_like(batch.get("has_lesion"))),
        )


# -----------------------------------------------------------------------------
# Region / patient-level evaluator for MIL segmentation
# -----------------------------------------------------------------------------

class LesionMILEvaluator:
    """Patient-level and region-level cancer/csPCa evaluation from lesion probabilities.

    The evaluator can use either:
      - region_probs from model MIL pooling; or
      - voxel-level lesion probability map pooled manually inside zones.
    """

    def __init__(
        self,
        prob_threshold: float = 0.5,
        positive_threshold: int = 1,
        invalid_sys_label: int = -1,
        fixed_specificity: Optional[float] = None,
        fixed_sensitivity: Optional[float] = None,
    ):
        self.prob_threshold = float(prob_threshold)
        self.positive_threshold = int(positive_threshold)
        self.invalid_sys_label = int(invalid_sys_label)
        if fixed_specificity is None:
            fixed_specificity = _cfg("FIXED_SPECIFICITY_TARGET", 0.95)
        if fixed_sensitivity is None:
            fixed_sensitivity = _cfg("FIXED_SENSITIVITY_TARGET", 0.90)
        self.fixed_specificity = float(fixed_specificity)
        self.fixed_sensitivity = float(fixed_sensitivity)

        self.patient_true = []
        self.patient_score = []
        self.region_true = []
        self.region_score = []

    def update_from_batch(
        self,
        lesion_probs: torch.Tensor,
        batch: Mapping,
        region_logits: Optional[torch.Tensor] = None,
        region_valid_mask: Optional[torch.Tensor] = None,
    ):
        """Update evaluator from a batch.

        lesion_probs: (B, 1, D, H, W)
        region_logits: optional (B, Z, 1) or (B, Z)
        """
        B = lesion_probs.size(0)
        device = lesion_probs.device

        if region_logits is not None:
            region_probs = torch.sigmoid(region_logits)
            if region_probs.dim() == 3 and region_probs.size(-1) == 1:
                region_probs = region_probs.squeeze(-1)
        else:
            region_probs = None

        for b in range(B):
            has_sys = bool(batch.get("has_sys", torch.zeros(B))[b].item() > 0)
            has_target = bool(batch.get("has_target", torch.zeros(B))[b].item() > 0)
            # Patient-level GT is biopsy-based. PUB dense lesion masks are used
            # for lesion Dice only and must not enter patient BAcc/AUC.
            if not (has_sys or has_target):
                continue

            # Patient-level GT: positive if any available biopsy supervision is positive.
            patient_gt = 0
            if has_sys and "sys_labels" in batch:
                labels = batch["sys_labels"][b].to(device)
                valid = labels != self.invalid_sys_label
                if valid.any() and labels[valid].max().item() >= self.positive_threshold:
                    patient_gt = 1
            if has_target and "target_mask" in batch:
                target_mask = batch["target_mask"][b].to(device)
                if target_mask.max().item() >= self.positive_threshold:
                    patient_gt = 1

            patient_score = self._patient_score(lesion_probs[b, 0], batch, b, device)
            self.patient_true.append(patient_gt)
            self.patient_score.append(patient_score)

            # Region-level GT/pred only exists for systematic biopsy samples.
            if has_sys and "sys_labels" in batch:
                labels = batch["sys_labels"][b].to(device)
                for z_idx in range(labels.numel()):
                    z_label = int(labels[z_idx].item())
                    if z_label == self.invalid_sys_label:
                        continue

                    if region_valid_mask is not None:
                        if not bool(region_valid_mask[b, z_idx].item()):
                            continue

                    y_true = int(z_label >= self.positive_threshold)

                    if region_probs is not None and z_idx < region_probs.shape[1]:
                        y_score = float(region_probs[b, z_idx].detach().cpu().item())
                    else:
                        # Fallback: max pooling inside the zone mask.
                        zones_mask = batch["zones_mask"][b, 0].to(device)
                        voxels = zones_mask.round().long() == (z_idx + 1)
                        if not voxels.any():
                            continue
                        y_score = float(lesion_probs[b, 0][voxels].max().detach().cpu().item())

                    self.region_true.append(y_true)
                    self.region_score.append(y_score)

    def _patient_score(self, lesion_prob_3d: torch.Tensor, batch: Mapping, b: int, device: torch.device) -> float:
        """Patient-level score: max lesion probability inside gland if available, otherwise whole image."""
        if "gland_mask" in batch and batch["gland_mask"][b].numel() > 0:
            gland_mask = batch["gland_mask"][b, 0].to(device) > 0
            if gland_mask.any():
                return float(lesion_prob_3d[gland_mask].max().detach().cpu().item())
        return float(lesion_prob_3d.max().detach().cpu().item())

    def _binary_metrics(self, y_true, y_score, threshold: float) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(np.int64)
        y_score = np.asarray(y_score).astype(np.float32)
        if len(y_true) == 0:
            return {
                "sens": 0.0,
                "spec": 0.0,
                "bacc": 0.0,
                "auc": 0.0,
                "auprc": 0.0,
                "n": 0,
                **operating_point_metrics(
                    y_true,
                    y_score,
                    fixed_specificity=self.fixed_specificity,
                    fixed_sensitivity=self.fixed_sensitivity,
                ),
            }
        y_pred = (y_score >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        return {
            "sens": float(sens),
            "spec": float(spec),
            "bacc": float((sens + spec) / 2.0),
            "auc": safe_auc(y_true, y_score),
            "auprc": safe_auprc(y_true, y_score),
            "n": int(len(y_true)),
            **operating_point_metrics(
                y_true,
                y_score,
                fixed_specificity=self.fixed_specificity,
                fixed_sensitivity=self.fixed_sensitivity,
            ),
        }

    def compute_metrics(self) -> Dict[str, float]:
        patient = self._binary_metrics(self.patient_true, self.patient_score, self.prob_threshold)
        region = self._binary_metrics(self.region_true, self.region_score, self.prob_threshold)
        return {
            "patient_sens": patient["sens"],
            "patient_spec": patient["spec"],
            "patient_bacc": patient["bacc"],
            "patient_auc": patient["auc"],
            "patient_auprc": patient["auprc"],
            "patient_n": patient["n"],
            "patient_fixed_spec_target": patient["fixed_spec_target"],
            "patient_sens_at_fixed_spec": patient["sens_at_fixed_spec"],
            "patient_actual_spec_at_fixed_spec": patient["actual_spec_at_fixed_spec"],
            "patient_actual_fpr_at_fixed_spec": patient["actual_fpr_at_fixed_spec"],
            "patient_threshold_at_fixed_spec": patient["threshold_at_fixed_spec"],
            "patient_fixed_sens_target": patient["fixed_sens_target"],
            "patient_spec_at_fixed_sens": patient["spec_at_fixed_sens"],
            "patient_actual_sens_at_fixed_sens": patient["actual_sens_at_fixed_sens"],
            "patient_threshold_at_fixed_sens": patient["threshold_at_fixed_sens"],
            "region_sens": region["sens"],
            "region_spec": region["spec"],
            "region_bacc": region["bacc"],
            "region_auc": region["auc"],
            "region_auprc": region["auprc"],
            "region_n": region["n"],
            "region_fixed_spec_target": region["fixed_spec_target"],
            "region_sens_at_fixed_spec": region["sens_at_fixed_spec"],
            "region_actual_spec_at_fixed_spec": region["actual_spec_at_fixed_spec"],
            "region_actual_fpr_at_fixed_spec": region["actual_fpr_at_fixed_spec"],
            "region_threshold_at_fixed_spec": region["threshold_at_fixed_spec"],
            "region_fixed_sens_target": region["fixed_sens_target"],
            "region_spec_at_fixed_sens": region["spec_at_fixed_sens"],
            "region_actual_sens_at_fixed_sens": region["actual_sens_at_fixed_sens"],
            "region_threshold_at_fixed_sens": region["threshold_at_fixed_sens"],
        }


# Backward-compatible alias for older imports.
BalancedAccuracyEvaluator = LesionMILEvaluator


# -----------------------------------------------------------------------------
# Metric tracker
# -----------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        if val is None:
            return
        val = tensor_to_float(val)
        if not np.isnan(val) and not np.isinf(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """Tracks only segmentation/MIL losses and metrics."""

    def __init__(self):
        self.loss_total = AverageMeter()
        self.loss_lesion = AverageMeter()
        self.loss_lesion_dense = AverageMeter()
        self.loss_lesion_sparse = AverageMeter()
        self.loss_lesion_sys = AverageMeter()

        self.lesion_dice = AverageMeter()
        self.lesion_dice_values = []
        self.lesion_dice_std = 0.0
        self.lesion_dice_n = 0
        self.lesion_f1 = AverageMeter()
        self.lesion_sens = AverageMeter()
        self.lesion_spec = AverageMeter()
        self.target_cspca_dice = AverageMeter()
        self.target_cspca_dice_values = []
        self.target_cspca_dice_std = 0.0
        self.target_cspca_dice_n = 0
        self.target_cspca_dice_sweep_thresholds = configured_target_dice_thresholds()
        self.target_cspca_dice_sweep_values = {
            threshold: [] for threshold in self.target_cspca_dice_sweep_thresholds
        }
        self.target_cspca_best_threshold_dice = 0.0
        self.target_cspca_best_threshold_dice_std = 0.0
        self.target_cspca_best_threshold_dice_n = 0
        self.target_cspca_best_threshold = float("nan")
        self.target_cspca_topk_dice = AverageMeter()
        self.target_cspca_topk_dice_values = []
        self.target_cspca_topk_dice_std = 0.0
        self.target_cspca_topk_dice_n = 0
        self.target_cspca_top_percent = float(_cfg("TARGET_DICE_TOP_PERCENT", 1.0))
        self.target_cspca_top_percent_dice = AverageMeter()
        self.target_cspca_top_percent_dice_values = []
        self.target_cspca_top_percent_dice_std = 0.0
        self.target_cspca_top_percent_dice_n = 0
        self.tbx_roi_true = []
        self.tbx_roi_score = []
        self.tbx_roi_bacc = 0.0
        self.tbx_roi_sens = 0.0
        self.tbx_roi_spec = 0.0
        self.tbx_roi_auc = 0.0
        self.tbx_roi_auprc = 0.0
        self.tbx_roi_n = 0
        self.tbx_roi_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.tbx_roi_sens_at_fixed_spec = 0.0
        self.tbx_roi_actual_spec_at_fixed_spec = 0.0
        self.tbx_roi_actual_fpr_at_fixed_spec = 0.0
        self.tbx_roi_threshold_at_fixed_spec = float("nan")

        self.patient_bacc = 0.0
        self.patient_sens = 0.0
        self.patient_spec = 0.0
        self.patient_auc = 0.0
        self.patient_auprc = 0.0
        self.patient_n = 0
        self.patient_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.patient_sens_at_fixed_spec = 0.0
        self.patient_actual_spec_at_fixed_spec = 0.0
        self.patient_actual_fpr_at_fixed_spec = 0.0
        self.patient_threshold_at_fixed_spec = float("nan")
        self.patient_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.patient_spec_at_fixed_sens = 0.0
        self.patient_actual_sens_at_fixed_sens = 0.0
        self.patient_threshold_at_fixed_sens = float("nan")

        self.region_bacc = 0.0
        self.region_sens = 0.0
        self.region_spec = 0.0
        self.region_auc = 0.0
        self.region_auprc = 0.0
        self.region_n = 0
        self.region_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.region_sens_at_fixed_spec = 0.0
        self.region_actual_spec_at_fixed_spec = 0.0
        self.region_actual_fpr_at_fixed_spec = 0.0
        self.region_threshold_at_fixed_spec = float("nan")
        self.region_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.region_spec_at_fixed_sens = 0.0
        self.region_actual_sens_at_fixed_sens = 0.0
        self.region_threshold_at_fixed_sens = float("nan")

        self.em_w_lesion_dense = AverageMeter()
        self.em_w_lesion_sparse = AverageMeter()
        self.em_w_lesion_sys = AverageMeter()

        self.active_lesion_dense = AverageMeter()
        self.active_lesion_sparse = AverageMeter()
        self.active_lesion_sys = AverageMeter()

        self.loss_num_batches = 0
        self.loss_num_cases = 0
        self.loss_dense_cases = 0
        self.loss_sparse_cases = 0
        self.loss_sparse_has_target_cases = 0
        self.loss_sparse_sampled_cases = 0
        self.loss_sparse_positive_cases = 0
        self.loss_sparse_negative_cases = 0
        self.loss_sparse_voxels = 0
        self.loss_sparse_positive_voxels = 0
        self.loss_sparse_negative_voxels = 0
        self.loss_sys_cases = 0
        self.loss_sys_regions = 0
        self.tbx_pos_prob_mean = AverageMeter()
        self.tbx_neg_prob_mean = AverageMeter()
        self.tbx_neg_1mp_mean = AverageMeter()
        self.tbx_pos_bce = AverageMeter()
        self.tbx_neg_bce = AverageMeter()

    def update_losses(self, *args, em_weights=None, active_tasks=None, **kwargs):
        """Update loss meters from either a loss_dict or legacy positional args.

        Preferred:
            tracker.update_losses(loss_dict)

        Also accepts old call style:
            update_losses(total, g_tot, g_tbx, g_sbx, l_tot, l_dense, l_sparse, l_sys, gl, ...)
        Grade/gland values are ignored.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            loss_dict = normalise_loss_output(args[0])
            total = loss_dict["total_loss"]
            l_tot = loss_dict["loss_lesion_total"]
            l_dense = loss_dict["loss_lesion_dense"]
            l_sparse = loss_dict["loss_lesion_sparse"]
            l_sys = loss_dict["loss_lesion_sys"]
            em_weights = loss_dict.get("em_weights", em_weights)
            active_tasks = loss_dict.get("active_tasks", active_tasks)
            loss_counts = loss_dict.get("loss_counts", {})
        elif len(args) >= 9:
            # Legacy multi-task order. Ignore grade/gland.
            total = args[0]
            l_tot = args[4]
            l_dense = args[5]
            l_sparse = args[6]
            l_sys = args[7]
            loss_counts = {}
        elif len(args) >= 5:
            # Compact new order.
            total, l_tot, l_dense, l_sparse, l_sys = args[:5]
            loss_counts = {}
        else:
            total = kwargs.get("total", kwargs.get("total_loss", 0.0))
            l_tot = kwargs.get("loss_lesion_total", 0.0)
            l_dense = kwargs.get("loss_lesion_dense", 0.0)
            l_sparse = kwargs.get("loss_lesion_sparse", 0.0)
            l_sys = kwargs.get("loss_lesion_sys", 0.0)
            loss_counts = kwargs.get("loss_counts", {})
        loss_counts = loss_counts or {}

        if active_tasks is not None:
            dense_active = float(active_tasks.get("lesion_dense", 0.0)) > 0
            sparse_active = float(active_tasks.get("lesion_sparse", 0.0)) > 0
            sys_active = float(active_tasks.get("lesion_sys", 0.0)) > 0
        else:
            dense_active = sparse_active = sys_active = True

        batch_n = int(loss_counts.get("batch_size", 1) or 1)
        dense_n = int(loss_counts.get("lesion_dense_cases", 0) or 0)
        sparse_case_n = int(loss_counts.get("lesion_sparse_cases", 0) or 0)
        sparse_has_target_n = int(loss_counts.get("lesion_sparse_has_target_cases", sparse_case_n) or 0)
        sparse_sampled_n = int(loss_counts.get("lesion_sparse_sampled_cases", sparse_case_n) or 0)
        sparse_positive_n = int(loss_counts.get("lesion_sparse_positive_cases", 0) or 0)
        sparse_negative_n = int(loss_counts.get("lesion_sparse_negative_cases", 0) or 0)
        sparse_voxel_n = int(loss_counts.get("lesion_sparse_voxels", 0) or 0)
        sparse_positive_voxel_n = int(loss_counts.get("lesion_sparse_positive_voxels", 0) or 0)
        sparse_negative_voxel_n = int(loss_counts.get("lesion_sparse_negative_voxels", 0) or 0)
        sys_case_n = int(loss_counts.get("lesion_sys_cases", 0) or 0)
        sys_region_n = int(loss_counts.get("lesion_sys_regions", 0) or 0)
        tbx_pos_prob_mean = loss_counts.get("tbx_pos_prob_mean", None)
        tbx_neg_prob_mean = loss_counts.get("tbx_neg_prob_mean", None)
        tbx_neg_1mp_mean = loss_counts.get("tbx_neg_1mp_mean", None)
        tbx_pos_bce = loss_counts.get("tbx_pos_bce", None)
        tbx_neg_bce = loss_counts.get("tbx_neg_bce", None)

        self.loss_num_batches += 1
        self.loss_num_cases += batch_n
        self.loss_dense_cases += dense_n
        self.loss_sparse_cases += sparse_case_n
        self.loss_sparse_has_target_cases += sparse_has_target_n
        self.loss_sparse_sampled_cases += sparse_sampled_n
        self.loss_sparse_positive_cases += sparse_positive_n
        self.loss_sparse_negative_cases += sparse_negative_n
        self.loss_sparse_voxels += sparse_voxel_n
        self.loss_sparse_positive_voxels += sparse_positive_voxel_n
        self.loss_sparse_negative_voxels += sparse_negative_voxel_n
        self.loss_sys_cases += sys_case_n
        self.loss_sys_regions += sys_region_n

        self.loss_total.update(total, n=batch_n)
        self.loss_lesion.update(l_tot, n=batch_n)
        if dense_active:
            self.loss_lesion_dense.update(l_dense, n=max(dense_n, 1))
        if sparse_active:
            self.loss_lesion_sparse.update(l_sparse, n=max(sparse_voxel_n, sparse_case_n, 1))
        if sys_active:
            self.loss_lesion_sys.update(l_sys, n=max(sys_region_n, sys_case_n, 1))

        if sparse_positive_voxel_n > 0:
            self.tbx_pos_prob_mean.update(tbx_pos_prob_mean, n=sparse_positive_voxel_n)
            self.tbx_pos_bce.update(tbx_pos_bce, n=sparse_positive_voxel_n)
        if sparse_negative_voxel_n > 0:
            self.tbx_neg_prob_mean.update(tbx_neg_prob_mean, n=sparse_negative_voxel_n)
            self.tbx_neg_1mp_mean.update(tbx_neg_1mp_mean, n=sparse_negative_voxel_n)
            self.tbx_neg_bce.update(tbx_neg_bce, n=sparse_negative_voxel_n)

        if em_weights is not None:
            self.em_w_lesion_dense.update(em_weights.get("lesion_dense", 1.0))
            self.em_w_lesion_sparse.update(em_weights.get("lesion_sparse", 1.0))
            self.em_w_lesion_sys.update(em_weights.get("lesion_sys", 1.0))

        if active_tasks is not None:
            self.active_lesion_dense.update(active_tasks.get("lesion_dense", 0.0))
            self.active_lesion_sparse.update(active_tasks.get("lesion_sparse", 0.0))
            self.active_lesion_sys.update(active_tasks.get("lesion_sys", 0.0))

    def update_lesion_dice_values(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.lesion_dice_values.extend(values.tolist())
        summary = summarise_values(self.lesion_dice_values)
        self.lesion_dice.avg = summary["mean"]
        self.lesion_dice.sum = summary["mean"] * summary["n"]
        self.lesion_dice.count = summary["n"]
        self.lesion_dice.val = float(values[-1])
        self.lesion_dice_std = summary["std"]
        self.lesion_dice_n = summary["n"]

    def update_target_cspca_dice_values(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.target_cspca_dice_values.extend(values.tolist())
        summary = summarise_values(self.target_cspca_dice_values)
        self.target_cspca_dice.avg = summary["mean"]
        self.target_cspca_dice.sum = summary["mean"] * summary["n"]
        self.target_cspca_dice.count = summary["n"]
        self.target_cspca_dice.val = float(values[-1])
        self.target_cspca_dice_std = summary["std"]
        self.target_cspca_dice_n = summary["n"]

    def update_target_cspca_aux_dice(self, probs: torch.Tensor, target: torch.Tensor):
        if probs.numel() == 0 or target.numel() == 0:
            return

        for threshold in self.target_cspca_dice_sweep_thresholds:
            values = compute_dice_per_case((probs >= threshold).float(), target)
            values = values[np.isfinite(values)]
            if values.size > 0:
                self.target_cspca_dice_sweep_values[threshold].extend(values.tolist())

        topk_values = compute_topk_dice_per_case(probs, target, mode="target_volume")
        self._update_value_summary(
            topk_values,
            self.target_cspca_topk_dice_values,
            self.target_cspca_topk_dice,
            "target_cspca_topk_dice_std",
            "target_cspca_topk_dice_n",
        )

        top_percent_values = compute_topk_dice_per_case(
            probs,
            target,
            mode="percent",
            top_percent=self.target_cspca_top_percent,
        )
        self._update_value_summary(
            top_percent_values,
            self.target_cspca_top_percent_dice_values,
            self.target_cspca_top_percent_dice,
            "target_cspca_top_percent_dice_std",
            "target_cspca_top_percent_dice_n",
        )

    def finalize_target_cspca_aux_dice(self):
        best_threshold = float("nan")
        best_summary = {"mean": 0.0, "std": 0.0, "n": 0}
        for threshold, values in self.target_cspca_dice_sweep_values.items():
            summary = summarise_values(values)
            if summary["n"] == 0:
                continue
            if summary["mean"] > best_summary["mean"]:
                best_threshold = float(threshold)
                best_summary = summary

        self.target_cspca_best_threshold = best_threshold
        self.target_cspca_best_threshold_dice = best_summary["mean"]
        self.target_cspca_best_threshold_dice_std = best_summary["std"]
        self.target_cspca_best_threshold_dice_n = best_summary["n"]

    def _update_value_summary(self, values, store, meter, std_attr: str, n_attr: str):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        store.extend(values.tolist())
        summary = summarise_values(store)
        meter.avg = summary["mean"]
        meter.sum = summary["mean"] * summary["n"]
        meter.count = summary["n"]
        meter.val = float(values[-1])
        setattr(self, std_attr, summary["std"])
        setattr(self, n_attr, summary["n"])

    def update_tbx_roi_samples(self, y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y_score)
        if valid.size == 0 or not valid.any():
            return
        self.tbx_roi_true.extend(y_true[valid].tolist())
        self.tbx_roi_score.extend(y_score[valid].tolist())

    def finalize_tbx_roi_metrics(self, threshold: float):
        y_true = np.asarray(self.tbx_roi_true, dtype=np.int64)
        y_score = np.asarray(self.tbx_roi_score, dtype=np.float32)
        self.tbx_roi_n = int(len(y_true))
        if self.tbx_roi_n == 0:
            return

        y_pred = (y_score >= float(threshold)).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        op = operating_point_metrics(
            y_true,
            y_score,
            fixed_specificity=self.tbx_roi_fixed_spec_target,
            fixed_sensitivity=float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90)),
        )

        self.tbx_roi_sens = float(sens)
        self.tbx_roi_spec = float(spec)
        self.tbx_roi_bacc = float((sens + spec) / 2.0)
        self.tbx_roi_auc = safe_auc(y_true, y_score)
        self.tbx_roi_auprc = safe_auprc(y_true, y_score)
        self.tbx_roi_fixed_spec_target = op["fixed_spec_target"]
        self.tbx_roi_sens_at_fixed_spec = op["sens_at_fixed_spec"]
        self.tbx_roi_actual_spec_at_fixed_spec = op["actual_spec_at_fixed_spec"]
        self.tbx_roi_actual_fpr_at_fixed_spec = op["actual_fpr_at_fixed_spec"]
        self.tbx_roi_threshold_at_fixed_spec = op["threshold_at_fixed_spec"]

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    def print_train_summary(self) -> str:
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"L_Les: {self.loss_lesion.avg:.4f} "
            f"(Dense {self.loss_lesion_dense.avg:.4f}, "
            f"Sparse {self.loss_lesion_sparse.avg:.4f}, "
            f"Sys {self.loss_lesion_sys.avg:.4f}) | "
            f"TBx p+: {self.tbx_pos_prob_mean.avg:.4f}, "
            f"p-: {self.tbx_neg_prob_mean.avg:.4f}, "
            f"1-p-: {self.tbx_neg_1mp_mean.avg:.4f}"
        )

    def print_val_summary(self) -> str:
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"Les-Dice: {self.lesion_dice.avg:.4f}+/-{self.lesion_dice_std:.4f} (n={self.lesion_dice_n}) | "
            f"Target-csPCa Dice@0.5: {self.target_cspca_dice.avg:.4f}+/-{self.target_cspca_dice_std:.4f} "
            f"(n={self.target_cspca_dice_n}) | "
            f"BestThrDice: {self.target_cspca_best_threshold_dice:.4f}@{self.target_cspca_best_threshold:.2f} | "
            f"TopKDice: {self.target_cspca_topk_dice.avg:.4f} | "
            f"TBx p+: {self.tbx_pos_prob_mean.avg:.4f}, "
            f"p-: {self.tbx_neg_prob_mean.avg:.4f}, "
            f"TBx ROI AUC/AUPRC: {self.tbx_roi_auc:.4f}/{self.tbx_roi_auprc:.4f} | "
            f"TBx ROI Sens@FPR{1.0 - self.tbx_roi_fixed_spec_target:.2f}: "
            f"{self.tbx_roi_sens_at_fixed_spec:.4f} | "
            f"Pat Sens@Spec{self.patient_fixed_spec_target:.2f}: {self.patient_sens_at_fixed_spec:.4f} | "
            f"Pat Spec@Sens{self.patient_fixed_sens_target:.2f}: {self.patient_spec_at_fixed_sens:.4f} | "
            f"Region Sens@Spec{self.region_fixed_spec_target:.2f}: {self.region_sens_at_fixed_spec:.4f}"
        )

    def get_train_dict(self) -> Dict[str, float]:
        return {
            "train_loss_total": self.loss_total.avg,
            "train_loss_lesion": self.loss_lesion.avg,
            "train_loss_lesion_dense": self.loss_lesion_dense.avg,
            "train_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "train_loss_lesion_sys": self.loss_lesion_sys.avg,
            "em_w_lesion_dense": self.em_w_lesion_dense.avg,
            "em_w_lesion_sparse": self.em_w_lesion_sparse.avg,
            "em_w_lesion_sys": self.em_w_lesion_sys.avg,
            "active_lesion_dense": self.active_lesion_dense.avg,
            "active_lesion_sparse": self.active_lesion_sparse.avg,
            "active_lesion_sys": self.active_lesion_sys.avg,
            "active_lesion_dense_batch_rate": self.active_lesion_dense.avg,
            "active_lesion_sparse_batch_rate": self.active_lesion_sparse.avg,
            "active_lesion_sys_batch_rate": self.active_lesion_sys.avg,
            "train_loss_num_batches": self.loss_num_batches,
            "train_loss_num_cases": self.loss_num_cases,
            "train_loss_dense_cases": self.loss_dense_cases,
            "train_loss_sparse_cases": self.loss_sparse_cases,
            "train_loss_sparse_has_target_cases": self.loss_sparse_has_target_cases,
            "train_loss_sparse_sampled_cases": self.loss_sparse_sampled_cases,
            "train_loss_sparse_positive_cases": self.loss_sparse_positive_cases,
            "train_loss_sparse_negative_cases": self.loss_sparse_negative_cases,
            "train_loss_sparse_positive_case_rate": self._ratio(
                self.loss_sparse_positive_cases, self.loss_sparse_has_target_cases
            ),
            "train_loss_sparse_negative_case_rate": self._ratio(
                self.loss_sparse_negative_cases, self.loss_sparse_has_target_cases
            ),
            "train_loss_sparse_voxels": self.loss_sparse_voxels,
            "train_loss_sparse_positive_voxels": self.loss_sparse_positive_voxels,
            "train_loss_sparse_negative_voxels": self.loss_sparse_negative_voxels,
            "train_tbx_pos_prob_mean": self.tbx_pos_prob_mean.avg,
            "train_tbx_neg_prob_mean": self.tbx_neg_prob_mean.avg,
            "train_tbx_neg_1mp_mean": self.tbx_neg_1mp_mean.avg,
            "train_tbx_pos_bce": self.tbx_pos_bce.avg,
            "train_tbx_neg_bce": self.tbx_neg_bce.avg,
            "train_loss_sys_cases": self.loss_sys_cases,
            "train_loss_sys_regions": self.loss_sys_regions,
        }

    def get_val_dict(self) -> Dict[str, float]:
        return {
            "val_loss_total": self.loss_total.avg,
            "val_loss_lesion": self.loss_lesion.avg,
            "val_loss_lesion_dense": self.loss_lesion_dense.avg,
            "val_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "val_loss_lesion_sys": self.loss_lesion_sys.avg,
            "val_lesion_dice": self.lesion_dice.avg,
            "val_lesion_dice_mean": self.lesion_dice.avg,
            "val_lesion_dice_std": self.lesion_dice_std,
            "val_lesion_dice_n": self.lesion_dice_n,
            "val_lesion_f1": self.lesion_f1.avg,
            "val_lesion_sens": self.lesion_sens.avg,
            "val_lesion_spec": self.lesion_spec.avg,
            "val_target_cspca_dice": self.target_cspca_dice.avg,
            "val_target_cspca_dice_at_prob_threshold": self.target_cspca_dice.avg,
            "val_target_cspca_dice_mean": self.target_cspca_dice.avg,
            "val_target_cspca_dice_std": self.target_cspca_dice_std,
            "val_target_cspca_dice_n": self.target_cspca_dice_n,
            "val_target_cspca_best_threshold_dice": self.target_cspca_best_threshold_dice,
            "val_target_cspca_best_threshold_dice_mean": self.target_cspca_best_threshold_dice,
            "val_target_cspca_best_threshold_dice_std": self.target_cspca_best_threshold_dice_std,
            "val_target_cspca_best_threshold_dice_n": self.target_cspca_best_threshold_dice_n,
            "val_target_cspca_best_threshold": self.target_cspca_best_threshold,
            "val_target_cspca_topk_dice": self.target_cspca_topk_dice.avg,
            "val_target_cspca_topk_dice_mean": self.target_cspca_topk_dice.avg,
            "val_target_cspca_topk_dice_std": self.target_cspca_topk_dice_std,
            "val_target_cspca_topk_dice_n": self.target_cspca_topk_dice_n,
            "val_target_cspca_top_percent": self.target_cspca_top_percent,
            "val_target_cspca_top_percent_dice": self.target_cspca_top_percent_dice.avg,
            "val_target_cspca_top_percent_dice_mean": self.target_cspca_top_percent_dice.avg,
            "val_target_cspca_top_percent_dice_std": self.target_cspca_top_percent_dice_std,
            "val_target_cspca_top_percent_dice_n": self.target_cspca_top_percent_dice_n,
            "val_tbx_roi_bacc": self.tbx_roi_bacc,
            "val_tbx_roi_sens": self.tbx_roi_sens,
            "val_tbx_roi_spec": self.tbx_roi_spec,
            "val_tbx_roi_auc": self.tbx_roi_auc,
            "val_tbx_roi_auprc": self.tbx_roi_auprc,
            "val_tbx_roi_n": self.tbx_roi_n,
            "val_tbx_roi_fixed_spec_target": self.tbx_roi_fixed_spec_target,
            "val_tbx_roi_sens_at_fixed_spec": self.tbx_roi_sens_at_fixed_spec,
            "val_tbx_roi_actual_spec_at_fixed_spec": self.tbx_roi_actual_spec_at_fixed_spec,
            "val_tbx_roi_actual_fpr_at_fixed_spec": self.tbx_roi_actual_fpr_at_fixed_spec,
            "val_tbx_roi_threshold_at_fixed_spec": self.tbx_roi_threshold_at_fixed_spec,
            "val_patient_bacc": self.patient_bacc,
            "val_patient_sens": self.patient_sens,
            "val_patient_spec": self.patient_spec,
            "val_patient_auc": self.patient_auc,
            "val_patient_auprc": self.patient_auprc,
            "val_patient_n": self.patient_n,
            "val_patient_fixed_spec_target": self.patient_fixed_spec_target,
            "val_patient_sens_at_fixed_spec": self.patient_sens_at_fixed_spec,
            "val_patient_actual_spec_at_fixed_spec": self.patient_actual_spec_at_fixed_spec,
            "val_patient_actual_fpr_at_fixed_spec": self.patient_actual_fpr_at_fixed_spec,
            "val_patient_threshold_at_fixed_spec": self.patient_threshold_at_fixed_spec,
            "val_patient_fixed_sens_target": self.patient_fixed_sens_target,
            "val_patient_spec_at_fixed_sens": self.patient_spec_at_fixed_sens,
            "val_patient_actual_sens_at_fixed_sens": self.patient_actual_sens_at_fixed_sens,
            "val_patient_threshold_at_fixed_sens": self.patient_threshold_at_fixed_sens,
            "val_region_bacc": self.region_bacc,
            "val_region_sens": self.region_sens,
            "val_region_spec": self.region_spec,
            "val_region_auc": self.region_auc,
            "val_region_auprc": self.region_auprc,
            "val_region_n": self.region_n,
            "val_region_fixed_spec_target": self.region_fixed_spec_target,
            "val_region_sens_at_fixed_spec": self.region_sens_at_fixed_spec,
            "val_region_actual_spec_at_fixed_spec": self.region_actual_spec_at_fixed_spec,
            "val_region_actual_fpr_at_fixed_spec": self.region_actual_fpr_at_fixed_spec,
            "val_region_threshold_at_fixed_spec": self.region_threshold_at_fixed_spec,
            "val_region_fixed_sens_target": self.region_fixed_sens_target,
            "val_region_spec_at_fixed_sens": self.region_spec_at_fixed_sens,
            "val_region_actual_sens_at_fixed_sens": self.region_actual_sens_at_fixed_sens,
            "val_region_threshold_at_fixed_sens": self.region_threshold_at_fixed_sens,
            "val_loss_num_batches": self.loss_num_batches,
            "val_loss_num_cases": self.loss_num_cases,
            "val_loss_dense_cases": self.loss_dense_cases,
            "val_loss_sparse_cases": self.loss_sparse_cases,
            "val_loss_sparse_has_target_cases": self.loss_sparse_has_target_cases,
            "val_loss_sparse_sampled_cases": self.loss_sparse_sampled_cases,
            "val_loss_sparse_positive_cases": self.loss_sparse_positive_cases,
            "val_loss_sparse_negative_cases": self.loss_sparse_negative_cases,
            "val_loss_sparse_positive_case_rate": self._ratio(
                self.loss_sparse_positive_cases, self.loss_sparse_has_target_cases
            ),
            "val_loss_sparse_negative_case_rate": self._ratio(
                self.loss_sparse_negative_cases, self.loss_sparse_has_target_cases
            ),
            "val_loss_sparse_voxels": self.loss_sparse_voxels,
            "val_loss_sparse_positive_voxels": self.loss_sparse_positive_voxels,
            "val_loss_sparse_negative_voxels": self.loss_sparse_negative_voxels,
            "val_tbx_pos_prob_mean": self.tbx_pos_prob_mean.avg,
            "val_tbx_neg_prob_mean": self.tbx_neg_prob_mean.avg,
            "val_tbx_neg_1mp_mean": self.tbx_neg_1mp_mean.avg,
            "val_tbx_pos_bce": self.tbx_pos_bce.avg,
            "val_tbx_neg_bce": self.tbx_neg_bce.avg,
            "val_loss_sys_cases": self.loss_sys_cases,
            "val_loss_sys_regions": self.loss_sys_regions,
        }


# -----------------------------------------------------------------------------
# Validation loop
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, save_dir):
    """Validate the segmentation + MIL model.

    This function accepts the new dict-output model and loss. It also tolerates
    the old 5-tuple model output for easier transition, but only lesion-related
    outputs are used.
    """
    model.eval()
    tracker = MetricTracker()

    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    prob_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))

    mil_evaluator = LesionMILEvaluator(
        prob_threshold=prob_threshold,
        positive_threshold=positive_threshold,
        invalid_sys_label=invalid_sys_label,
    )

    vis_dir = os.path.join(save_dir, _cfg("VIS_SUBDIR", "visualizations"), f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)

    saved_counts = {"PUB": 0, "TCIA": 0, "PROMIS": 0, "OTHER": 0}
    max_saves_per_type = int(_cfg("MAX_VAL_VIS_PER_TYPE", 2))
    plot_prob = float(_cfg("VAL_VIS_PROB", 0.15))

    for batch in tqdm(loader, desc="Validation"):
        batch = move_batch_to_device(batch, device)
        imgs = batch["input"]
        zones_mask = batch.get("zones_mask", None)

        raw_outputs = model(imgs, zones_mask)
        outputs = unpack_model_output(raw_outputs)
        lesion_logits = outputs["lesion_logits"]
        if lesion_logits is None:
            raise ValueError("Model output does not contain lesion logits.")

        loss_output = call_criterion(criterion, outputs, batch)
        loss_dict = normalise_loss_output(loss_output)
        tracker.update_losses(loss_dict)

        lesion_probs = torch.sigmoid(lesion_logits)

        # Dense segmentation metrics only for PUB/radiologist-annotated cases.
        if "has_lesion" in batch and batch["has_lesion"].sum() > 0:
            idx = batch["has_lesion"] > 0
            pred_bin = (lesion_probs[idx] >= prob_threshold).float()
            target = batch["lesion_mask"][idx].float()
            tracker.update_lesion_dice_values(compute_dice_per_case(pred_bin, target))
            tracker.lesion_f1.update(compute_f1(pred_bin, target))
            tracker.lesion_sens.update(compute_sens(pred_bin, target))
            tracker.lesion_spec.update(compute_spec(pred_bin, target))

        # B-series csPCa localisation metric on biopsy-confirmed target ROIs.
        if "has_target" in batch and "target_mask" in batch and batch["has_target"].sum() > 0:
            target_cspca = (batch["target_mask"] >= positive_threshold).float()
            positive_target_cases = (batch["has_target"] > 0) & target_cspca.reshape(target_cspca.size(0), -1).any(dim=1)
            if positive_target_cases.any():
                positive_probs = lesion_probs[positive_target_cases]
                positive_target = target_cspca[positive_target_cases]
                pred_bin = (positive_probs >= prob_threshold).float()
                tracker.update_target_cspca_dice_values(
                    compute_dice_per_case(pred_bin, positive_target)
                )
                tracker.update_target_cspca_aux_dice(positive_probs, positive_target)

            sampled_tbx_roi = (batch["has_target"] > 0).view(-1, 1, 1, 1, 1) & (batch["target_mask"] > 0)
            if sampled_tbx_roi.any():
                tracker.update_tbx_roi_samples(
                    target_cspca[sampled_tbx_roi].detach().cpu().numpy(),
                    lesion_probs[sampled_tbx_roi].detach().cpu().numpy(),
                )

        # Patient-level and region-level MIL metrics.
        mil_evaluator.update_from_batch(
            lesion_probs=lesion_probs,
            batch=batch,
            region_logits=outputs.get("region_logits"),
            region_valid_mask=outputs.get("region_valid_mask"),
        )

        # # Optional visualization.
        # for b in range(imgs.size(0)):
        #     d_type = infer_dataset_type(batch, b)
        #     if saved_counts.get(d_type, 0) >= max_saves_per_type:
        #         continue
        #     if random.random() >= plot_prob:
        #         continue

        #     gt_dict = {
        #         "type": d_type,
        #         "lesion_mask": batch.get("lesion_mask", torch.zeros_like(lesion_probs))[b, 0].detach().cpu().numpy(),
        #         "target_mask": batch.get("target_mask", torch.zeros_like(lesion_probs))[b, 0].detach().cpu().numpy(),
        #         "zones_mask": batch.get("zones_mask", torch.zeros_like(lesion_probs))[b, 0].detach().cpu().numpy(),
        #         "sys_labels": batch.get("sys_labels", torch.empty(0, device=device))[b].detach().cpu().numpy()
        #             if "sys_labels" in batch else np.asarray([]),
        #     }
        #     pid = batch["pid"][b] if "pid" in batch else f"case_{b}"
        #     vis_filename = f"{d_type}_{pid}.png"
        #     visualize_predictions(
        #         input_tensor=imgs[b],
        #         risk_map=lesion_probs[b],
        #         gt_dict=gt_dict,
        #         save_path=os.path.join(vis_dir, vis_filename),
        #         patient_id=pid,
        #     )
        #     saved_counts[d_type] = saved_counts.get(d_type, 0) + 1

    mil_metrics = mil_evaluator.compute_metrics()
    tracker.finalize_target_cspca_aux_dice()
    tracker.finalize_tbx_roi_metrics(prob_threshold)

    tracker.patient_sens = mil_metrics["patient_sens"]
    tracker.patient_spec = mil_metrics["patient_spec"]
    tracker.patient_bacc = mil_metrics["patient_bacc"]
    tracker.patient_auc = mil_metrics["patient_auc"]
    tracker.patient_auprc = mil_metrics["patient_auprc"]
    tracker.patient_n = mil_metrics["patient_n"]
    tracker.patient_fixed_spec_target = mil_metrics["patient_fixed_spec_target"]
    tracker.patient_sens_at_fixed_spec = mil_metrics["patient_sens_at_fixed_spec"]
    tracker.patient_actual_spec_at_fixed_spec = mil_metrics["patient_actual_spec_at_fixed_spec"]
    tracker.patient_actual_fpr_at_fixed_spec = mil_metrics["patient_actual_fpr_at_fixed_spec"]
    tracker.patient_threshold_at_fixed_spec = mil_metrics["patient_threshold_at_fixed_spec"]
    tracker.patient_fixed_sens_target = mil_metrics["patient_fixed_sens_target"]
    tracker.patient_spec_at_fixed_sens = mil_metrics["patient_spec_at_fixed_sens"]
    tracker.patient_actual_sens_at_fixed_sens = mil_metrics["patient_actual_sens_at_fixed_sens"]
    tracker.patient_threshold_at_fixed_sens = mil_metrics["patient_threshold_at_fixed_sens"]

    tracker.region_sens = mil_metrics["region_sens"]
    tracker.region_spec = mil_metrics["region_spec"]
    tracker.region_bacc = mil_metrics["region_bacc"]
    tracker.region_auc = mil_metrics["region_auc"]
    tracker.region_auprc = mil_metrics["region_auprc"]
    tracker.region_n = mil_metrics["region_n"]
    tracker.region_fixed_spec_target = mil_metrics["region_fixed_spec_target"]
    tracker.region_sens_at_fixed_spec = mil_metrics["region_sens_at_fixed_spec"]
    tracker.region_actual_spec_at_fixed_spec = mil_metrics["region_actual_spec_at_fixed_spec"]
    tracker.region_actual_fpr_at_fixed_spec = mil_metrics["region_actual_fpr_at_fixed_spec"]
    tracker.region_threshold_at_fixed_spec = mil_metrics["region_threshold_at_fixed_spec"]
    tracker.region_fixed_sens_target = mil_metrics["region_fixed_sens_target"]
    tracker.region_spec_at_fixed_sens = mil_metrics["region_spec_at_fixed_sens"]
    tracker.region_actual_sens_at_fixed_sens = mil_metrics["region_actual_sens_at_fixed_sens"]
    tracker.region_threshold_at_fixed_sens = mil_metrics["region_threshold_at_fixed_sens"]

    return tracker


def infer_dataset_type(batch: Mapping, b: int) -> str:
    """Infer dataset type for logging/visualisation."""
    if "source" in batch:
        source = batch["source"][b]
        if isinstance(source, str):
            return source

    pid = str(batch.get("pid", [""])[b]) if "pid" in batch else ""
    if pid.startswith("PUB"):
        return "PUB"
    if pid.startswith("TCIA"):
        return "TCIA"
    if pid.startswith("PROMIS"):
        return "PROMIS"

    if "has_lesion" in batch and batch["has_lesion"][b].item() > 0:
        return "PUB"
    if "has_target" in batch and batch["has_target"][b].item() > 0:
        return "TCIA"
    if "has_sys" in batch and batch["has_sys"][b].item() > 0:
        return "PROMIS"
    return "OTHER"


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_loss_curves(log_path: str, save_path: str):
    """Plot lesion-related training/validation losses and EM weights."""
    try:
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        ax1 = axes[0]
        for col, label in [
            ("train_loss_total", "Train Total"),
            ("val_loss_total", "Val Total"),
            ("train_loss_lesion", "Train Lesion Total"),
            ("train_loss_lesion_dense", "Train Dense"),
            ("train_loss_lesion_sparse", "Train TBx ROI"),
            ("train_loss_lesion_sys", "Train Sys MIL"),
            ("val_loss_lesion", "Val Lesion Total"),
            ("val_loss_lesion_dense", "Val Dense"),
            ("val_loss_lesion_sparse", "Val TBx ROI"),
            ("val_loss_lesion_sys", "Val Sys MIL"),
        ]:
            if col in df.columns:
                ax1.plot(df["epoch"], df[col], label=label, linewidth=2 if "total" in col else 1.2)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Segmentation + MIL Loss Curves")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = axes[1]
        for col, label in [
            ("em_w_lesion_dense", "Dense Weight"),
            ("em_w_lesion_sparse", "TBx ROI Weight"),
            ("em_w_lesion_sys", "Sys MIL Weight"),
        ]:
            if col in df.columns:
                ax2.plot(df["epoch"], df[col], label=label)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learned multiplier exp(-log_var)")
        ax2.set_title("EM / Uncertainty Weights")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")


def visualize_predictions(input_tensor, risk_map, gt_dict, save_path: str, patient_id: str):
    """Visualise lesion risk map and available ground truth.

    Removed the old grade-map row. The figure now shows:
      row 1: T2 + predicted lesion risk
      row 2: available ground truth / biopsy supervision
      row 3: systematic zones if available
    """
    t2 = input_tensor[0].detach().cpu().numpy()
    risk = risk_map[0].detach().cpu().numpy()

    mid = t2.shape[0] // 2
    slices = [max(0, mid - 5), mid, min(t2.shape[0] - 1, mid + 5)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f"Patient: {patient_id} | Dataset Type: {gt_dict['type']}", fontsize=16, y=0.98)

    for i, s_idx in enumerate(slices):
        # Row 1: predicted lesion risk map.
        axes[0, i].imshow(t2[s_idx], cmap="gray")
        risk_overlay = np.ma.masked_where(risk[s_idx] < 0.2, risk[s_idx])
        im1 = axes[0, i].imshow(risk_overlay, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Prediction: Lesion Risk (Slice {s_idx})")
        axes[0, i].axis("off")
        if i == 2:
            fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Row 2: available supervision.
        axes[1, i].imshow(t2[s_idx], cmap="gray")
        gt_slice, title, cmap, vmin, vmax = _build_gt_slice(gt_dict, s_idx)
        if gt_slice is not None:
            gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
            im2 = axes[1, i].imshow(gt_overlay, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
            if i == 2:
                fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        axes[1, i].set_title(title)
        axes[1, i].axis("off")

        # Row 3: systematic zones, useful for SBx/PROMIS interpretation.
        axes[2, i].imshow(t2[s_idx], cmap="gray")
        z_slice = gt_dict.get("zones_mask", None)
        if z_slice is not None and np.max(z_slice) > 0:
            zone_overlay = np.ma.masked_where(z_slice[s_idx] == 0, z_slice[s_idx])
            im3 = axes[2, i].imshow(zone_overlay, cmap="tab20", alpha=0.35)
            if i == 2:
                fig.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
            axes[2, i].set_title(f"Systematic Zones (Slice {s_idx})")
        else:
            axes[2, i].set_title(f"No systematic zones (Slice {s_idx})")
        axes[2, i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _build_gt_slice(gt_dict: Mapping, s_idx: int):
    d_type = gt_dict.get("type", "OTHER")
    invalid = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))

    if d_type == "PUB":
        lesion = gt_dict.get("lesion_mask")
        if lesion is not None:
            return lesion[s_idx], f"GT: Radiologist Lesion Mask (Slice {s_idx})", "autumn", 0, 1

    if d_type == "TCIA":
        target = gt_dict.get("target_mask")
        if target is not None and np.max(target) > 0:
            binary_target = (target[s_idx] >= positive_threshold).astype(np.float32)
            if binary_target.max() == 0:
                # Show TBx-confirmed target ROI even when benign-labelled.
                binary_target = (target[s_idx] > 0).astype(np.float32)
            return binary_target, f"GT: TBx-confirmed Target ROI (Slice {s_idx})", "autumn", 0, 1

    if d_type in {"TCIA", "PROMIS"}:
        zones = gt_dict.get("zones_mask")
        sys_labels = gt_dict.get("sys_labels", np.asarray([]))
        if zones is not None and len(sys_labels) > 0:
            z_slice = zones[s_idx]
            gt_slice = np.zeros_like(z_slice, dtype=np.float32)
            for z_idx in range(1, min(len(sys_labels), 20) + 1):
                label = sys_labels[z_idx - 1]
                if label != invalid:
                    gt_slice[z_slice == z_idx] = float(label >= positive_threshold)
            return gt_slice, f"GT: SBx Positive Regions (Slice {s_idx})", "autumn", 0, 1

    return None, f"GT: No dense supervision (Slice {s_idx})", "autumn", 0, 1
