import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MixedSupervisionLoss(nn.Module):
    """
    Mixed-supervision loss with optional uncertainty-based dynamic weighting.

    If use_em_weighting=True:
        L_i * exp(-s_i) + s_i

    If use_em_weighting=False:
        fixed_weight_i * L_i

    Critical implementation detail:
    A supervision branch is added to the total loss only when that branch is active
    in the current batch. This prevents absent tasks from optimizing their log_var term alone.

    Label conventions:
      - sys_labels == invalid_sys_label: unsampled / no supervision
      - sys_labels == 0: valid negative systematic biopsy region
      - sys_labels >= csPCa_threshold: clinically significant PCa
    """

    def __init__(
        self,
        csPCa_threshold: int = 3,
        pos_weight_val: float = 2.0,
        invalid_sys_label: int = -1,
        class_weights=None,
        use_em_weighting: bool = True,
        fixed_loss_weights=None,
    ):
        super().__init__()
        self.csPCa_threshold = int(csPCa_threshold)
        self.invalid_sys_label = int(invalid_sys_label)
        self.use_em_weighting = bool(use_em_weighting)

        default_fixed_loss_weights = {
            "grade_tbx": 1.0,
            "grade_sbx": 1.0,
            "lesion_dense": 1.0,
            "lesion_sparse": 1.0,
            "lesion_sys": 1.0,
            "gland": 1.0,
        }

        if fixed_loss_weights is None:
            fixed_loss_weights = default_fixed_loss_weights
        else:
            merged_weights = default_fixed_loss_weights.copy()
            merged_weights.update(fixed_loss_weights)
            fixed_loss_weights = merged_weights

        self.fixed_loss_weights = fixed_loss_weights

        self.log_vars = nn.ParameterDict(
            {
                "grade_tbx": nn.Parameter(torch.zeros(1)),
                "grade_sbx": nn.Parameter(torch.zeros(1)),
                "lesion_dense": nn.Parameter(torch.zeros(1)),
                "lesion_sparse": nn.Parameter(torch.zeros(1)),
                "lesion_sys": nn.Parameter(torch.zeros(1)),
                "gland": nn.Parameter(torch.zeros(1)),
            }
        )

        self.register_buffer("pos_weight", torch.tensor([pos_weight_val], dtype=torch.float32))
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.lesion_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        if class_weights is None:
            class_weights = [0.1, 0.5, 2.0, 2.0, 3.0, 3.0, 3.0]
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=self.invalid_sys_label)

    def _weighted(self, loss, key: str):
        """
        EM weighting:
            L_i * exp(-s_i) + s_i

        Fixed weighting:
            w_i * L_i
        """
        if self.use_em_weighting:
            return loss * torch.exp(-self.log_vars[key]) + self.log_vars[key]

        weight = float(self.fixed_loss_weights.get(key, 1.0))
        return loss * weight

    def _zero(self, device):
        return torch.tensor(0.0, device=device)

    def forward(
        self,
        grade_preds,
        sys_grade_preds,
        lesion_pred,
        sys_lesion_preds,
        gland_pred,
        target_mask,
        sys_labels,
        lesion_mask,
        gland_mask,
        has_target,
        has_sys,
        has_lesion,
        has_gland,
    ):
        device = lesion_pred.device if lesion_pred is not None else grade_preds.device

        loss_grade_target = self._zero(device)
        loss_grade_sys = self._zero(device)
        loss_lesion_dense = self._zero(device)
        loss_lesion_sparse = self._zero(device)
        loss_lesion_sys = self._zero(device)
        loss_gland = self._zero(device)

        active = {
            "grade_tbx": False,
            "grade_sbx": False,
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "gland": False,
        }

        valid_target_batch = has_target > 0
        valid_sys_batch = has_sys > 0
        valid_lesion_batch = has_lesion > 0
        valid_gland_batch = has_gland > 0

        # 1A. Targeted-biopsy voxel-level ISUP grade supervision.
        if grade_preds is not None and valid_target_batch.any():
            g_p_t = grade_preds[valid_target_batch]
            mask_t = target_mask[valid_target_batch]
            if mask_t.dim() == 5 and mask_t.shape[1] == 1:
                mask_t = mask_t.squeeze(1)
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                preds_valid = g_p_t.permute(0, 2, 3, 4, 1)[valid_pixels]
                labels_valid = mask_t[valid_pixels].long()
                loss_grade_target = self.ce_loss(preds_valid, labels_valid)
                active["grade_tbx"] = True

        # 1B. Systematic-biopsy region-level ISUP grade supervision.
        if sys_grade_preds is not None and valid_sys_batch.any():
            s_g_p = sys_grade_preds[valid_sys_batch]
            s_labels = sys_labels[valid_sys_batch]
            s_g_p_flat = s_g_p.reshape(-1, s_g_p.size(-1))
            s_labels_flat = s_labels.reshape(-1)
            valid_zones = s_labels_flat != self.invalid_sys_label
            if valid_zones.any():
                loss_grade_sys = self.ce_loss(s_g_p_flat[valid_zones], s_labels_flat[valid_zones].long())
                active["grade_sbx"] = True

        # 2A. Dense voxel-level lesion mask supervision.
        if lesion_pred is not None and valid_lesion_batch.any():
            pred_l = lesion_pred[valid_lesion_batch]
            mask_l = lesion_mask[valid_lesion_batch].float()
            loss_lesion_dense = self.lesion_bce_loss(pred_l, mask_l) + self.lesion_focal_loss(pred_l, mask_l)
            active["lesion_dense"] = True

        # 2B. Targeted-biopsy sparse csPCa supervision on needle-track voxels only.
        if lesion_pred is not None and valid_target_batch.any():
            pred_t = lesion_pred[valid_target_batch]
            mask_t = target_mask[valid_target_batch]
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                pred_valid = pred_t[valid_pixels]
                label_valid = target_lesion_label[valid_pixels]
                loss_lesion_sparse = self.lesion_bce_loss(pred_valid, label_valid) + self.lesion_focal_loss(pred_valid, label_valid)
                active["lesion_sparse"] = True

        # 2C. Systematic-biopsy region-level csPCa weak supervision.
        if sys_lesion_preds is not None and valid_sys_batch.any():
            s_l_p = sys_lesion_preds[valid_sys_batch]
            s_labels = sys_labels[valid_sys_batch]
            s_l_p_flat = s_l_p.reshape(-1)
            s_labels_flat = s_labels.reshape(-1)
            valid_zones = s_labels_flat != self.invalid_sys_label
            if valid_zones.any():
                sys_lesion_label = (s_labels_flat[valid_zones] >= self.csPCa_threshold).float()
                pred_valid = s_l_p_flat[valid_zones]
                loss_lesion_sys = self.lesion_bce_loss(pred_valid, sys_lesion_label) + self.lesion_focal_loss(pred_valid, sys_lesion_label)
                active["lesion_sys"] = True

        # 3. Dense gland segmentation supervision.
        if gland_pred is not None and valid_gland_batch.any():
            g_pred_valid = gland_pred[valid_gland_batch]
            g_mask_valid = gland_mask[valid_gland_batch].float()
            loss_gland = self.gland_bce_loss(g_pred_valid, g_mask_valid) + self.dice_loss(g_pred_valid, g_mask_valid)
            active["gland"] = True

        weighted_terms = []
        if active["grade_tbx"]:
            weighted_terms.append(self._weighted(loss_grade_target, "grade_tbx"))
        if active["grade_sbx"]:
            weighted_terms.append(self._weighted(loss_grade_sys, "grade_sbx"))
        if active["lesion_dense"]:
            weighted_terms.append(self._weighted(loss_lesion_dense, "lesion_dense"))
        if active["lesion_sparse"]:
            weighted_terms.append(self._weighted(loss_lesion_sparse, "lesion_sparse"))
        if active["lesion_sys"]:
            weighted_terms.append(self._weighted(loss_lesion_sys, "lesion_sys"))
        if active["gland"]:
            weighted_terms.append(self._weighted(loss_gland, "gland"))

        if weighted_terms:
            total_loss = torch.stack([term.reshape(()) for term in weighted_terms]).sum()
        else:
            # Keeps graph valid in the unlikely event of a completely unsupervised batch.
            total_loss = self._zero(device)

        loss_grade_total = loss_grade_target + loss_grade_sys
        loss_lesion_total = loss_lesion_dense + loss_lesion_sparse + loss_lesion_sys

        if self.use_em_weighting:
            em_weights = {k: torch.exp(-v.detach()).item() for k, v in self.log_vars.items()}
        else:
            em_weights = {k: float(self.fixed_loss_weights.get(k, 1.0)) for k in self.log_vars.keys()}

        active_tasks = {k: float(v) for k, v in active.items()}

        return (
            total_loss,
            loss_grade_total,
            loss_grade_target,
            loss_grade_sys,
            loss_lesion_total,
            loss_lesion_dense,
            loss_lesion_sparse,
            loss_lesion_sys,
            loss_gland,
            em_weights,
            active_tasks,
        )