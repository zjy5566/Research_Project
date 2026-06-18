"""
Loss functions for the new segmentation + MIL setting.

This file removes the old grade/gland branches and keeps only lesion-related
supervision:
  1) lesion_dense  : dense radiologist lesion-mask supervision, e.g. PUB
  2) lesion_sparse : sparse TBx needle-track supervision, e.g. TCIA target biopsy
  3) lesion_sys    : region-level MIL supervision from SBx zones, e.g. TCIA/PROMIS

The optional EM/uncertainty weighting, log-var clamp, and curriculum gates are
kept because they may still be useful for balancing dense and weak supervision.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from config import Config
except Exception:  # pragma: no cover - allows standalone unit testing
    Config = None


TASK_KEYS: Tuple[str, ...] = (
    "lesion_dense",
    "lesion_sparse",
    "lesion_sys",
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _cfg(name: str, default):
    """Safely read a value from Config when it exists."""
    return getattr(Config, name, default) if Config is not None else default


def _default_task_switches_from_config() -> Dict[str, bool]:
    """Read lesion-task switches from Config; enable all branches by default."""
    return {
        "lesion_dense": bool(_cfg("USE_LESION_DENSE_TASK", True)),
        "lesion_sparse": bool(_cfg("USE_LESION_SPARSE_TASK", True)),
        "lesion_sys": bool(_cfg("USE_LESION_SYS_TASK", True)),
    }


def _default_branch_start_epochs_from_config() -> Dict[str, int]:
    """Read curriculum start epochs from Config; start all branches at epoch 1."""
    return {
        "lesion_dense": int(_cfg("LESION_DENSE_START_EPOCH", 1)),
        "lesion_sparse": int(_cfg("LESION_SPARSE_START_EPOCH", 1)),
        "lesion_sys": int(_cfg("LESION_SYS_START_EPOCH", 1)),
    }


# -----------------------------------------------------------------------------
# Basic losses
# -----------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Binary Dice loss for dense voxel-level lesion masks."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.float().reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss for sparse/region labels with class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------------------------------------------------------
# Main mixed-supervision segmentation loss
# -----------------------------------------------------------------------------

class MixedSupervisionLoss(nn.Module):
    """
    Lesion segmentation loss with MIL weak supervision.

    Expected model output dictionary from the revised model.py:
        outputs["lesion_logits"]      : (B, 1, D, H, W)
        outputs["region_logits"]      : (B, max_zones, 1) or (B, max_zones)
        outputs["region_valid_mask"]  : (B, max_zones), optional

    Expected batch dictionary from the revised dataset.py:
        batch["lesion_mask"] : dense lesion mask for PUB cases
        batch["target_mask"] : TBx needle-track labels for TCIA cases
        batch["sys_labels"]  : SBx zone labels; invalid_sys_label means unsampled
        batch["has_lesion"]  : 1 if dense lesion supervision exists
        batch["has_target"]  : 1 if TBx supervision exists
        batch["has_sys"]     : 1 if SBx supervision exists

    Label convention:
        sys_labels == invalid_sys_label : invalid / unsampled / no supervision
        sys_labels == 0                 : valid benign / negative region
        sys_labels >= positive_threshold: positive cancer/csPCa region

    EM weighting, when enabled:
        weighted_loss_i = loss_i * exp(-s_i) + s_i
    """

    def __init__(
        self,
        positive_threshold: Optional[int] = None,
        pos_weight_val: float = 2.0,
        sys_pos_weight_val: Optional[float] = None,
        sys_focal_alpha: Optional[float] = None,
        sys_focal_gamma: Optional[float] = None,
        use_sys_class_balanced_bce: Optional[bool] = None,
        invalid_sys_label: Optional[int] = None,
        use_em_weighting: Optional[bool] = None,
        fixed_loss_weights: Optional[Dict[str, float]] = None,
        task_switches: Optional[Dict[str, bool]] = None,
        use_logvar_clamp: Optional[bool] = None,
        logvar_min: Optional[float] = None,
        logvar_max: Optional[float] = None,
        use_curriculum: Optional[bool] = None,
        branch_start_epochs: Optional[Dict[str, int]] = None,
        tbx_positive_soft_label: Optional[float] = None,
        tbx_negative_soft_label: Optional[float] = None,
        return_dict: bool = True,
    ):
        super().__init__()

        # For cancer/no-cancer region supervision, set LESION_POSITIVE_THRESHOLD = 1.
        # For csPCa supervision, set it to Config.CSPC_THRESHOLD.
        if positive_threshold is None:
            positive_threshold = _cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1))
        if invalid_sys_label is None:
            invalid_sys_label = _cfg("INVALID_SYS_LABEL", -1)
        if use_em_weighting is None:
            use_em_weighting = _cfg("USE_EM_WEIGHTING", True)
        if use_logvar_clamp is None:
            use_logvar_clamp = _cfg("USE_LOGVAR_CLAMP", False)
        if logvar_min is None:
            logvar_min = _cfg("LOGVAR_MIN", -3.0)
        if logvar_max is None:
            logvar_max = _cfg("LOGVAR_MAX", 3.0)
        if use_curriculum is None:
            use_curriculum = _cfg("USE_CURRICULUM", False)
        if tbx_positive_soft_label is None:
            tbx_positive_soft_label = _cfg("TBX_POSITIVE_SOFT_LABEL", 1.0)
        if tbx_negative_soft_label is None:
            tbx_negative_soft_label = _cfg("TBX_NEGATIVE_SOFT_LABEL", 0.0)
        if sys_pos_weight_val is None:
            sys_pos_weight_val = _cfg("SYS_POS_WEIGHT_VAL", pos_weight_val)
        if sys_focal_alpha is None:
            sys_focal_alpha = _cfg("SYS_FOCAL_ALPHA", 0.75)
        if sys_focal_gamma is None:
            sys_focal_gamma = _cfg("SYS_FOCAL_GAMMA", 2.0)
        if use_sys_class_balanced_bce is None:
            use_sys_class_balanced_bce = _cfg("USE_SYS_CLASS_BALANCED_BCE", True)

        self.positive_threshold = int(positive_threshold)
        self.invalid_sys_label = int(invalid_sys_label)
        self.use_em_weighting = bool(use_em_weighting)
        self.use_logvar_clamp = bool(use_logvar_clamp)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.use_curriculum = bool(use_curriculum)
        self.return_dict = bool(return_dict)
        self.current_epoch = 1
        self.tbx_positive_soft_label = float(tbx_positive_soft_label)
        self.tbx_negative_soft_label = float(tbx_negative_soft_label)
        self.use_sys_class_balanced_bce = bool(use_sys_class_balanced_bce)

        default_fixed_loss_weights = {
            "lesion_dense": 1.0,
            "lesion_sparse": 1.0,
            "lesion_sys": 1.0,
        }
        cfg_fixed_loss_weights = _cfg("FIXED_LOSS_WEIGHTS", None)
        if fixed_loss_weights is None and cfg_fixed_loss_weights is not None:
            fixed_loss_weights = cfg_fixed_loss_weights
        merged_weights = default_fixed_loss_weights.copy()
        if fixed_loss_weights is not None:
            # Ignore old grade/gland keys if they still exist in Config.FIXED_LOSS_WEIGHTS.
            merged_weights.update({k: float(v) for k, v in fixed_loss_weights.items() if k in TASK_KEYS})
        self.fixed_loss_weights = merged_weights

        switches = _default_task_switches_from_config()
        if task_switches is not None:
            switches.update({k: bool(v) for k, v in task_switches.items() if k in TASK_KEYS})
        self.task_switches = switches

        starts = _default_branch_start_epochs_from_config()
        if branch_start_epochs is not None:
            starts.update({k: int(v) for k, v in branch_start_epochs.items() if k in TASK_KEYS})
        self.branch_start_epochs = starts

        # Keep EM/log-var parameters only for lesion-related branches.
        self.log_vars = nn.ParameterDict({key: nn.Parameter(torch.zeros(1)) for key in TASK_KEYS})

        self.register_buffer("pos_weight", torch.tensor([pos_weight_val], dtype=torch.float32))
        self.register_buffer("sys_pos_weight", torch.tensor([sys_pos_weight_val], dtype=torch.float32))
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.sys_focal_loss = FocalLoss(alpha=sys_focal_alpha, gamma=sys_focal_gamma)

    # ------------------------------------------------------------------
    # Task gates and EM weighting
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        """Call once per epoch if curriculum learning is enabled."""
        self.current_epoch = int(epoch)

    def is_enabled_by_switch(self, key: str) -> bool:
        return bool(self.task_switches.get(key, True))

    def is_started_by_curriculum(self, key: str) -> bool:
        if not self.use_curriculum:
            return True
        return self.current_epoch >= int(self.branch_start_epochs.get(key, 1))

    def is_enabled(self, key: str) -> bool:
        return self.is_enabled_by_switch(key) and self.is_started_by_curriculum(key)

    def _get_log_var_for_loss(self, key: str) -> torch.Tensor:
        s = self.log_vars[key]
        if self.use_logvar_clamp:
            s = torch.clamp(s, min=self.logvar_min, max=self.logvar_max)
        return s

    def _weighted(self, loss: torch.Tensor, key: str) -> torch.Tensor:
        if self.use_em_weighting:
            s = self._get_log_var_for_loss(key)
            return loss * torch.exp(-s) + s
        return loss * float(self.fixed_loss_weights.get(key, 1.0))

    @staticmethod
    def _zero(device: torch.device) -> torch.Tensor:
        return torch.tensor(0.0, device=device)

    def _class_balanced_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Average positive and negative region BCE separately before combining."""
        targets = targets.float()
        per_region_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.sys_pos_weight.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        positive_mask = targets > 0.5
        negative_mask = ~positive_mask

        terms = []
        if positive_mask.any():
            terms.append(per_region_loss[positive_mask].mean())
        if negative_mask.any():
            terms.append(per_region_loss[negative_mask].mean())
        if not terms:
            return self._zero(logits.device)
        return torch.stack(terms).mean()

    @staticmethod
    def _infer_device(*tensors: Optional[torch.Tensor]) -> torch.device:
        for tensor in tensors:
            if tensor is not None:
                return tensor.device
        return torch.device("cpu")

    def get_current_weights(self) -> Dict[str, float]:
        """Displayed branch weights. Disabled/not-yet-started branches report 0."""
        weights: Dict[str, float] = {}
        for key in TASK_KEYS:
            if not self.is_enabled(key):
                weights[key] = 0.0
            elif self.use_em_weighting:
                s = self.log_vars[key].detach()
                if self.use_logvar_clamp:
                    s = torch.clamp(s, min=self.logvar_min, max=self.logvar_max)
                weights[key] = torch.exp(-s).item()
            else:
                weights[key] = float(self.fixed_loss_weights.get(key, 1.0))
        return weights

    def get_curriculum_status(self) -> Dict[str, float]:
        return {key: float(self.is_enabled(key)) for key in TASK_KEYS}

    # ------------------------------------------------------------------
    # Individual lesion losses
    # ------------------------------------------------------------------

    def _dense_lesion_loss(
        self,
        lesion_logits: torch.Tensor,
        lesion_mask: torch.Tensor,
        has_lesion: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """PUB/radiologist dense lesion mask loss."""
        valid_batch = has_lesion > 0
        if not (self.is_enabled("lesion_dense") and valid_batch.any()):
            return self._zero(lesion_logits.device), False

        pred = lesion_logits[valid_batch]
        target = lesion_mask[valid_batch].float()
        loss = self.bce_loss(pred, target) + self.dice_loss(pred, target)
        return loss, True

    def _sparse_tbx_loss(
        self,
        lesion_logits: torch.Tensor,
        target_mask: torch.Tensor,
        has_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """TCIA TBx needle-track sparse voxel loss."""
        valid_batch = has_target > 0
        if not (self.is_enabled("lesion_sparse") and valid_batch.any()):
            return self._zero(lesion_logits.device), False

        pred = lesion_logits[valid_batch]
        target_mask = target_mask[valid_batch]

        valid_voxels = target_mask > 0
        if not valid_voxels.any():
            return self._zero(lesion_logits.device), False

        # Convert biopsy grade/score on the sampled track into a binary lesion label.
        target = (target_mask >= self.positive_threshold).float()
        pred_valid = pred[valid_voxels]
        target_valid = target[valid_voxels]
        target_valid = torch.where(
            target_valid > 0,
            torch.full_like(target_valid, self.tbx_positive_soft_label),
            torch.full_like(target_valid, self.tbx_negative_soft_label),
        )

        loss = self.bce_loss(pred_valid, target_valid) + self.focal_loss(pred_valid, target_valid)
        return loss, True

    def _region_mil_loss(
        self,
        region_logits: torch.Tensor,
        sys_labels: torch.Tensor,
        has_sys: torch.Tensor,
        region_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """TCIA/PROMIS SBx region-level MIL loss from pooled lesion logits."""
        valid_batch = has_sys > 0
        if not (self.is_enabled("lesion_sys") and valid_batch.any()):
            return self._zero(region_logits.device), False

        logits = region_logits[valid_batch]
        labels = sys_labels[valid_batch]

        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        labels = labels[:, : logits.size(1)]

        valid_regions = labels != self.invalid_sys_label
        if region_valid_mask is not None:
            rmask = region_valid_mask[valid_batch].bool()[:, : logits.size(1)]
            valid_regions = valid_regions & rmask

        if not valid_regions.any():
            return self._zero(region_logits.device), False

        target = (labels >= self.positive_threshold).float()
        pred_valid = logits[valid_regions]
        target_valid = target[valid_regions]

        if self.use_sys_class_balanced_bce:
            bce_loss = self._class_balanced_bce_loss(pred_valid, target_valid)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_valid,
                target_valid,
                pos_weight=self.sys_pos_weight.to(device=pred_valid.device, dtype=pred_valid.dtype),
            )
        loss = bce_loss + self.sys_focal_loss(pred_valid, target_valid)
        return loss, True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        *,
        lesion_logits: Optional[torch.Tensor] = None,
        region_logits: Optional[torch.Tensor] = None,
        region_valid_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        sys_labels: Optional[torch.Tensor] = None,
        lesion_mask: Optional[torch.Tensor] = None,
        has_target: Optional[torch.Tensor] = None,
        has_sys: Optional[torch.Tensor] = None,
        has_lesion: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, object], Tuple[torch.Tensor, ...]]:
        """
        Preferred usage:
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["total_loss"]

        Direct usage is also supported through keyword arguments.
        """
        if outputs is not None:
            lesion_logits = outputs.get("lesion_logits", lesion_logits)
            region_logits = outputs.get("region_logits", region_logits)
            region_valid_mask = outputs.get("region_valid_mask", region_valid_mask)

        if batch is not None:
            target_mask = batch.get("target_mask", target_mask)
            sys_labels = batch.get("sys_labels", sys_labels)
            lesion_mask = batch.get("lesion_mask", lesion_mask)
            has_target = batch.get("has_target", has_target)
            has_sys = batch.get("has_sys", has_sys)
            has_lesion = batch.get("has_lesion", has_lesion)

        device = self._infer_device(lesion_logits, region_logits, lesion_mask, target_mask, sys_labels)

        if lesion_logits is None:
            raise ValueError("lesion_logits is required. Pass outputs['lesion_logits'] or lesion_logits=...")

        batch_size = lesion_logits.size(0)
        if has_lesion is None:
            has_lesion = torch.zeros(batch_size, device=device)
        if has_target is None:
            has_target = torch.zeros(batch_size, device=device)
        if has_sys is None:
            has_sys = torch.zeros(batch_size, device=device)

        has_lesion = has_lesion.to(device).bool()
        has_target = has_target.to(device).bool()
        has_sys = has_sys.to(device).bool()

        raw_losses: Dict[str, torch.Tensor] = {
            "lesion_dense": self._zero(device),
            "lesion_sparse": self._zero(device),
            "lesion_sys": self._zero(device),
        }
        active_tasks: Dict[str, float] = {key: 0.0 for key in TASK_KEYS}
        loss_counts: Dict[str, int] = {
            "batch_size": int(batch_size),
            "lesion_dense_cases": 0,
            "lesion_sparse_cases": 0,
            "lesion_sparse_voxels": 0,
            "lesion_sys_cases": 0,
            "lesion_sys_regions": 0,
        }

        if lesion_mask is not None:
            raw_losses["lesion_dense"], active = self._dense_lesion_loss(
                lesion_logits=lesion_logits,
                lesion_mask=lesion_mask.to(device),
                has_lesion=has_lesion,
            )
            active_tasks["lesion_dense"] = float(active)
            if active:
                loss_counts["lesion_dense_cases"] = int(has_lesion.sum().detach().cpu().item())

        if target_mask is not None:
            target_mask_device = target_mask.to(device)
            raw_losses["lesion_sparse"], active = self._sparse_tbx_loss(
                lesion_logits=lesion_logits,
                target_mask=target_mask_device,
                has_target=has_target,
            )
            active_tasks["lesion_sparse"] = float(active)
            if active:
                valid_target_mask = target_mask_device[has_target] > 0
                loss_counts["lesion_sparse_cases"] = int(has_target.sum().detach().cpu().item())
                loss_counts["lesion_sparse_voxels"] = int(valid_target_mask.sum().detach().cpu().item())

        if region_logits is not None and sys_labels is not None:
            region_logits_device = region_logits.to(device)
            sys_labels_device = sys_labels.to(device)
            region_valid_mask_device = region_valid_mask.to(device) if region_valid_mask is not None else None
            raw_losses["lesion_sys"], active = self._region_mil_loss(
                region_logits=region_logits_device,
                sys_labels=sys_labels_device,
                has_sys=has_sys,
                region_valid_mask=region_valid_mask_device,
            )
            active_tasks["lesion_sys"] = float(active)
            if active:
                logits_for_count = region_logits_device[has_sys]
                if logits_for_count.dim() == 3 and logits_for_count.size(-1) == 1:
                    logits_for_count = logits_for_count.squeeze(-1)
                labels_for_count = sys_labels_device[has_sys][:, : logits_for_count.size(1)]
                valid_regions = labels_for_count != self.invalid_sys_label
                if region_valid_mask_device is not None:
                    rmask = region_valid_mask_device[has_sys].bool()[:, : logits_for_count.size(1)]
                    valid_regions = valid_regions & rmask
                loss_counts["lesion_sys_cases"] = int(has_sys.sum().detach().cpu().item())
                loss_counts["lesion_sys_regions"] = int(valid_regions.sum().detach().cpu().item())

        weighted_terms = []
        for key in TASK_KEYS:
            if active_tasks[key] > 0 and self.is_enabled(key):
                weighted_terms.append(self._weighted(raw_losses[key], key))

        total_loss = (
            torch.stack([term.reshape(()) for term in weighted_terms]).sum()
            if weighted_terms
            else self._zero(device)
        )
        lesion_loss_total = raw_losses["lesion_dense"] + raw_losses["lesion_sparse"] + raw_losses["lesion_sys"]

        result = {
            "total_loss": total_loss,
            "loss_lesion_total": lesion_loss_total,
            "loss_lesion_dense": raw_losses["lesion_dense"],
            "loss_lesion_sparse": raw_losses["lesion_sparse"],
            "loss_lesion_sys": raw_losses["lesion_sys"],
            "em_weights": self.get_current_weights(),
            "active_tasks": active_tasks,
            "curriculum_status": self.get_curriculum_status(),
            "loss_counts": loss_counts,
        }

        if self.return_dict:
            return result

        # Optional compact tuple for simple legacy-style logging.
        return (
            result["total_loss"],
            result["loss_lesion_total"],
            result["loss_lesion_dense"],
            result["loss_lesion_sparse"],
            result["loss_lesion_sys"],
            result["em_weights"],
            result["active_tasks"],
            result["curriculum_status"],
        )


# Clear alias for the new setting. Existing code can still import MixedSupervisionLoss.
SegmentationMILLoss = MixedSupervisionLoss
