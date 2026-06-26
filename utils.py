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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score
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

    def __init__(self, prob_threshold: float = 0.5, positive_threshold: int = 1, invalid_sys_label: int = -1):
        self.prob_threshold = float(prob_threshold)
        self.positive_threshold = int(positive_threshold)
        self.invalid_sys_label = int(invalid_sys_label)

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
            has_lesion = bool(batch.get("has_lesion", torch.zeros(B))[b].item() > 0)

            # Patient-level GT: positive if any available supervision is positive.
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
            if has_lesion and "lesion_mask" in batch:
                lesion_mask = batch["lesion_mask"][b].to(device)
                if lesion_mask.max().item() > 0:
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

    @staticmethod
    def _binary_metrics(y_true, y_score, threshold: float) -> Dict[str, float]:
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
            "region_sens": region["sens"],
            "region_spec": region["spec"],
            "region_bacc": region["bacc"],
            "region_auc": region["auc"],
            "region_auprc": region["auprc"],
            "region_n": region["n"],
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
        self.lesion_f1 = AverageMeter()
        self.lesion_sens = AverageMeter()
        self.lesion_spec = AverageMeter()

        self.patient_bacc = 0.0
        self.patient_sens = 0.0
        self.patient_spec = 0.0
        self.patient_auc = 0.0
        self.patient_auprc = 0.0
        self.patient_n = 0

        self.region_bacc = 0.0
        self.region_sens = 0.0
        self.region_spec = 0.0
        self.region_auc = 0.0
        self.region_auprc = 0.0
        self.region_n = 0

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
        self.loss_sparse_voxels = 0
        self.loss_sys_cases = 0
        self.loss_sys_regions = 0

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
        sparse_voxel_n = int(loss_counts.get("lesion_sparse_voxels", 0) or 0)
        sys_case_n = int(loss_counts.get("lesion_sys_cases", 0) or 0)
        sys_region_n = int(loss_counts.get("lesion_sys_regions", 0) or 0)

        self.loss_num_batches += 1
        self.loss_num_cases += batch_n
        self.loss_dense_cases += dense_n
        self.loss_sparse_cases += sparse_case_n
        self.loss_sparse_voxels += sparse_voxel_n
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

        if em_weights is not None:
            self.em_w_lesion_dense.update(em_weights.get("lesion_dense", 1.0))
            self.em_w_lesion_sparse.update(em_weights.get("lesion_sparse", 1.0))
            self.em_w_lesion_sys.update(em_weights.get("lesion_sys", 1.0))

        if active_tasks is not None:
            self.active_lesion_dense.update(active_tasks.get("lesion_dense", 0.0))
            self.active_lesion_sparse.update(active_tasks.get("lesion_sparse", 0.0))
            self.active_lesion_sys.update(active_tasks.get("lesion_sys", 0.0))

    def print_train_summary(self) -> str:
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"L_Les: {self.loss_lesion.avg:.4f} "
            f"(Dense {self.loss_lesion_dense.avg:.4f}, "
            f"Sparse {self.loss_lesion_sparse.avg:.4f}, "
            f"Sys {self.loss_lesion_sys.avg:.4f})"
        )

    def print_val_summary(self) -> str:
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"Les-Dice: {self.lesion_dice.avg:.4f} | "
            f"Les-F1: {self.lesion_f1.avg:.4f} | "
            f"Patient-BAcc: {self.patient_bacc:.4f} | "
            f"Region-BAcc: {self.region_bacc:.4f}"
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
            "train_loss_num_batches": self.loss_num_batches,
            "train_loss_num_cases": self.loss_num_cases,
            "train_loss_dense_cases": self.loss_dense_cases,
            "train_loss_sparse_cases": self.loss_sparse_cases,
            "train_loss_sparse_voxels": self.loss_sparse_voxels,
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
            "val_lesion_f1": self.lesion_f1.avg,
            "val_lesion_sens": self.lesion_sens.avg,
            "val_lesion_spec": self.lesion_spec.avg,
            "val_patient_bacc": self.patient_bacc,
            "val_patient_sens": self.patient_sens,
            "val_patient_spec": self.patient_spec,
            "val_patient_auc": self.patient_auc,
            "val_patient_auprc": self.patient_auprc,
            "val_patient_n": self.patient_n,
            "val_region_bacc": self.region_bacc,
            "val_region_sens": self.region_sens,
            "val_region_spec": self.region_spec,
            "val_region_auc": self.region_auc,
            "val_region_auprc": self.region_auprc,
            "val_region_n": self.region_n,
            "val_loss_num_batches": self.loss_num_batches,
            "val_loss_num_cases": self.loss_num_cases,
            "val_loss_dense_cases": self.loss_dense_cases,
            "val_loss_sparse_cases": self.loss_sparse_cases,
            "val_loss_sparse_voxels": self.loss_sparse_voxels,
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
            tracker.lesion_dice.update(compute_dice(pred_bin, target))
            tracker.lesion_f1.update(compute_f1(pred_bin, target))
            tracker.lesion_sens.update(compute_sens(pred_bin, target))
            tracker.lesion_spec.update(compute_spec(pred_bin, target))

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
    tracker.patient_sens = mil_metrics["patient_sens"]
    tracker.patient_spec = mil_metrics["patient_spec"]
    tracker.patient_bacc = mil_metrics["patient_bacc"]
    tracker.patient_auc = mil_metrics["patient_auc"]
    tracker.patient_auprc = mil_metrics["patient_auprc"]
    tracker.patient_n = mil_metrics["patient_n"]

    tracker.region_sens = mil_metrics["region_sens"]
    tracker.region_spec = mil_metrics["region_spec"]
    tracker.region_bacc = mil_metrics["region_bacc"]
    tracker.region_auc = mil_metrics["region_auc"]
    tracker.region_auprc = mil_metrics["region_auprc"]
    tracker.region_n = mil_metrics["region_n"]

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
