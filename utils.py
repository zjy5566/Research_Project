import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from tqdm import tqdm

from config import Config


def compute_dice(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


def compute_f1(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    if targets.sum() == 0 and preds.sum() == 0:
        return 1.0
    return f1_score(targets, preds, zero_division=0)


def compute_sens(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return tp / (tp + fn + 1e-7)


def compute_kappa(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    if len(np.unique(targets)) <= 1:
        return 0.0
    try:
        return cohen_kappa_score(targets, preds, weights="quadratic", labels=np.arange(7))
    except Exception:
        return 0.0


class BalancedAccuracyEvaluator:
    """
    Patient/gland-level and systematic region-level csPCa balanced accuracy.

    Label convention:
      - invalid_sys_label: unsampled / no supervision
      - 0: valid negative region
      - >= cs_pca_threshold: csPCa positive
    """

    def __init__(self, prob_threshold=0.5, cs_pca_threshold=3, invalid_sys_label=-1):
        self.prob_threshold = prob_threshold
        self.cs_pca_threshold = cs_pca_threshold
        self.invalid_sys_label = invalid_sys_label
        self.gland_tp = self.gland_tn = self.gland_fp = self.gland_fn = 0
        self.region_tp = self.region_tn = self.region_fp = self.region_fn = 0

    def update(self, pred_prob_3d, gland_mask, zones_mask, sys_labels, lesion_mask, target_mask, has_sys, has_lesion, has_target):
        patient_gt = 0
        if has_sys and sys_labels is not None:
            valid = sys_labels != self.invalid_sys_label
            if valid.any() and sys_labels[valid].max() >= self.cs_pca_threshold:
                patient_gt = 1
        if has_target and target_mask is not None:
            if target_mask.max() >= self.cs_pca_threshold:
                patient_gt = 1
        if has_lesion and lesion_mask is not None:
            if lesion_mask.max() > 0:
                patient_gt = 1

        valid_gland_probs = pred_prob_3d[gland_mask > 0]
        if len(valid_gland_probs) > 0:
            patient_pred = int(valid_gland_probs.max().item() >= self.prob_threshold)
            self._update_gland_counts(patient_gt, patient_pred)

        if has_sys and zones_mask is not None and sys_labels is not None:
            unique_zones = torch.unique(zones_mask)
            for z_idx in unique_zones:
                z_int = int(z_idx.item())
                if z_int <= 0:
                    continue
                if z_int - 1 >= len(sys_labels):
                    continue
                z_label = int(sys_labels[z_int - 1].item())
                if z_label == self.invalid_sys_label:
                    continue
                zone_gt = int(z_label >= self.cs_pca_threshold)
                zone_probs = pred_prob_3d[zones_mask == z_idx]
                if len(zone_probs) > 0:
                    zone_pred = int(zone_probs.max().item() >= self.prob_threshold)
                    self._update_region_counts(zone_gt, zone_pred)

    def _update_gland_counts(self, y_true, y_pred):
        if y_true == 1 and y_pred == 1:
            self.gland_tp += 1
        elif y_true == 0 and y_pred == 0:
            self.gland_tn += 1
        elif y_true == 0 and y_pred == 1:
            self.gland_fp += 1
        elif y_true == 1 and y_pred == 0:
            self.gland_fn += 1

    def _update_region_counts(self, y_true, y_pred):
        if y_true == 1 and y_pred == 1:
            self.region_tp += 1
        elif y_true == 0 and y_pred == 0:
            self.region_tn += 1
        elif y_true == 0 and y_pred == 1:
            self.region_fp += 1
        elif y_true == 1 and y_pred == 0:
            self.region_fn += 1

    def compute_metrics(self):
        g_tpr = self.gland_tp / (self.gland_tp + self.gland_fn + 1e-8)
        g_tnr = self.gland_tn / (self.gland_tn + self.gland_fp + 1e-8)
        r_tpr = self.region_tp / (self.region_tp + self.region_fn + 1e-8)
        r_tnr = self.region_tn / (self.region_tn + self.region_fp + 1e-8)
        return {
            "gland_sens": g_tpr,
            "gland_spec": g_tnr,
            "gland_bacc": (g_tpr + g_tnr) / 2.0,
            "region_sens": r_tpr,
            "region_spec": r_tnr,
            "region_bacc": (r_tpr + r_tnr) / 2.0,
        }


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if val is None:
            return
        if not np.isnan(val) and not np.isinf(val):
            self.val = float(val)
            self.sum += float(val) * n
            self.count += n
            self.avg = self.sum / self.count


class MetricTracker:
    def __init__(self):
        self.loss_total = AverageMeter()
        self.loss_grade_total = AverageMeter()
        self.loss_grade_tbx = AverageMeter()
        self.loss_grade_sbx = AverageMeter()
        self.loss_lesion = AverageMeter()
        self.loss_lesion_dense = AverageMeter()
        self.loss_lesion_sparse = AverageMeter()
        self.loss_lesion_sys = AverageMeter()
        self.loss_gland = AverageMeter()

        self.lesion_dice = AverageMeter()
        self.lesion_f1 = AverageMeter()
        self.lesion_sens = AverageMeter()
        self.gland_dice = AverageMeter()
        self.grade_kappa = AverageMeter()

        self.gland_bacc = 0.0
        self.gland_tpr = 0.0
        self.gland_tnr = 0.0
        self.region_bacc = 0.0
        self.region_tpr = 0.0
        self.region_tnr = 0.0

        self.em_w_grade_tbx = AverageMeter()
        self.em_w_grade_sbx = AverageMeter()
        self.em_w_lesion_dense = AverageMeter()
        self.em_w_lesion_sparse = AverageMeter()
        self.em_w_lesion_sys = AverageMeter()
        self.em_w_gland = AverageMeter()

        self.active_grade_tbx = AverageMeter()
        self.active_grade_sbx = AverageMeter()
        self.active_lesion_dense = AverageMeter()
        self.active_lesion_sparse = AverageMeter()
        self.active_lesion_sys = AverageMeter()
        self.active_gland = AverageMeter()

    def update_losses(self, total, g_tot, g_tbx, g_sbx, l_tot, l_dense, l_sparse, l_sys, gl, em_weights=None, active_tasks=None):
        self.loss_total.update(total)
        self.loss_grade_total.update(g_tot)
        self.loss_grade_tbx.update(g_tbx)
        self.loss_grade_sbx.update(g_sbx)
        self.loss_lesion.update(l_tot)
        self.loss_lesion_dense.update(l_dense)
        self.loss_lesion_sparse.update(l_sparse)
        self.loss_lesion_sys.update(l_sys)
        self.loss_gland.update(gl)

        if em_weights is not None:
            self.em_w_grade_tbx.update(em_weights.get("grade_tbx", 1.0))
            self.em_w_grade_sbx.update(em_weights.get("grade_sbx", 1.0))
            self.em_w_lesion_dense.update(em_weights.get("lesion_dense", 1.0))
            self.em_w_lesion_sparse.update(em_weights.get("lesion_sparse", 1.0))
            self.em_w_lesion_sys.update(em_weights.get("lesion_sys", 1.0))
            self.em_w_gland.update(em_weights.get("gland", 1.0))

        if active_tasks is not None:
            self.active_grade_tbx.update(active_tasks.get("grade_tbx", 0.0))
            self.active_grade_sbx.update(active_tasks.get("grade_sbx", 0.0))
            self.active_lesion_dense.update(active_tasks.get("lesion_dense", 0.0))
            self.active_lesion_sparse.update(active_tasks.get("lesion_sparse", 0.0))
            self.active_lesion_sys.update(active_tasks.get("lesion_sys", 0.0))
            self.active_gland.update(active_tasks.get("gland", 0.0))

    def print_train_summary(self):
        return (
            f"Loss: {self.loss_total.avg:.4f} | L_Grad: {self.loss_grade_total.avg:.4f} | "
            f"L_Les: {self.loss_lesion.avg:.4f} | L_Glan: {self.loss_gland.avg:.4f}"
        )

    def print_val_summary(self):
        return (
            f"Loss: {self.loss_total.avg:.4f} | Les-Dice: {self.lesion_dice.avg:.4f} | "
            f"Grade-Kap: {self.grade_kappa.avg:.4f} | Glan-BAcc: {self.gland_bacc:.4f} | "
            f"Reg-BAcc: {self.region_bacc:.4f}"
        )

    def get_train_dict(self):
        return {
            "train_loss_total": self.loss_total.avg,
            "train_loss_grade": self.loss_grade_total.avg,
            "train_loss_grade_tbx": self.loss_grade_tbx.avg,
            "train_loss_grade_sbx": self.loss_grade_sbx.avg,
            "train_loss_lesion": self.loss_lesion.avg,
            "train_loss_lesion_dense": self.loss_lesion_dense.avg,
            "train_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "train_loss_lesion_sys": self.loss_lesion_sys.avg,
            "train_loss_gland": self.loss_gland.avg,
            "em_w_grade_tbx": self.em_w_grade_tbx.avg,
            "em_w_grade_sbx": self.em_w_grade_sbx.avg,
            "em_w_lesion_dense": self.em_w_lesion_dense.avg,
            "em_w_lesion_sparse": self.em_w_lesion_sparse.avg,
            "em_w_lesion_sys": self.em_w_lesion_sys.avg,
            "em_w_gland": self.em_w_gland.avg,
            "active_grade_tbx": self.active_grade_tbx.avg,
            "active_grade_sbx": self.active_grade_sbx.avg,
            "active_lesion_dense": self.active_lesion_dense.avg,
            "active_lesion_sparse": self.active_lesion_sparse.avg,
            "active_lesion_sys": self.active_lesion_sys.avg,
            "active_gland": self.active_gland.avg,
        }

    def get_val_dict(self):
        return {
            "val_loss_total": self.loss_total.avg,
            "val_loss_grade": self.loss_grade_total.avg,
            "val_loss_grade_tbx": self.loss_grade_tbx.avg,
            "val_loss_grade_sbx": self.loss_grade_sbx.avg,
            "val_loss_lesion": self.loss_lesion.avg,
            "val_loss_lesion_dense": self.loss_lesion_dense.avg,
            "val_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "val_loss_lesion_sys": self.loss_lesion_sys.avg,
            "val_loss_gland": self.loss_gland.avg,
            "val_lesion_dice": self.lesion_dice.avg,
            "val_lesion_f1": self.lesion_f1.avg,
            "val_lesion_sens": self.lesion_sens.avg,
            "val_gland_dice": self.gland_dice.avg,
            "val_grade_kappa": self.grade_kappa.avg,
            "val_gland_bacc": self.gland_bacc,
            "val_gland_sens": self.gland_tpr,
            "val_gland_spec": self.gland_tnr,
            "val_region_bacc": self.region_bacc,
            "val_region_sens": self.region_tpr,
            "val_region_spec": self.region_tnr,
        }


def unpack_loss_output(loss_output):
    """Accepts both old 10-item and new 11-item loss returns."""
    if len(loss_output) == 11:
        return loss_output
    if len(loss_output) == 10:
        return (*loss_output, None)
    raise ValueError(f"Unexpected criterion return length: {len(loss_output)}")


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, save_dir):
    model.eval()
    tracker = MetricTracker()
    invalid_sys_label = int(getattr(Config, "INVALID_SYS_LABEL", -1))
    balanced_evaluator = BalancedAccuracyEvaluator(
        prob_threshold=0.5,
        cs_pca_threshold=int(getattr(Config, "CSPC_THRESHOLD", 3)),
        invalid_sys_label=invalid_sys_label,
    )

    vis_dir = os.path.join(save_dir, Config.VIS_SUBDIR, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)

    saved_counts = {"PUB": 0, "TCIA": 0, "PROMIS": 0}
    max_saves_per_type = 2
    plot_prob = 0.15

    for batch in tqdm(loader, desc="Validation"):
        imgs = batch["input"].to(device)
        z_mask = batch["zones_mask"].to(device)
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)

        loss_output = criterion(
            g_p,
            s_g_p,
            l_p,
            s_l_p,
            gl_p,
            batch["target_mask"].to(device),
            batch["sys_labels"].to(device),
            batch["lesion_mask"].to(device),
            batch["gland_mask"].to(device),
            batch["has_target"].to(device),
            batch["has_sys"].to(device),
            batch["has_lesion"].to(device),
            batch["has_gland"].to(device),
        )
        total_loss, l_grad_tot, l_grad_tbx, l_grad_sbx, l_les_tot, l_les_dense, l_les_sparse, l_les_sys, l_gland, em_weights, active_tasks = unpack_loss_output(loss_output)

        tracker.update_losses(
            total_loss.item(),
            l_grad_tot.item(),
            l_grad_tbx.item(),
            l_grad_sbx.item(),
            l_les_tot.item(),
            l_les_dense.item(),
            l_les_sparse.item(),
            l_les_sys.item(),
            l_gland.item(),
            em_weights=em_weights,
            active_tasks=active_tasks,
        )

        if batch["has_gland"].sum() > 0:
            idx = batch["has_gland"] > 0
            g_bin = (torch.sigmoid(gl_p[idx]) > 0.5).float()
            tracker.gland_dice.update(compute_dice(g_bin, batch["gland_mask"][idx].to(device)))

        if batch["has_lesion"].sum() > 0:
            idx = batch["has_lesion"] > 0
            lp, lt = torch.sigmoid(l_p[idx]), batch["lesion_mask"][idx].to(device)
            lb = (lp > 0.5).float()
            tracker.lesion_dice.update(compute_dice(lb, lt))
            tracker.lesion_f1.update(compute_f1(lb, lt))
            tracker.lesion_sens.update(compute_sens(lb, lt))

        if batch["has_sys"].sum() > 0 and s_g_p is not None:
            idx = batch["has_sys"] > 0
            sys_pred_flat = torch.argmax(s_g_p[idx], dim=-1).flatten()
            sys_true_flat = batch["sys_labels"][idx].flatten().to(device)
            valid_mask = sys_true_flat != invalid_sys_label
            if valid_mask.sum() > 0:
                tracker.grade_kappa.update(compute_kappa(sys_pred_flat[valid_mask], sys_true_flat[valid_mask]))

        r_probs = torch.sigmoid(l_p)
        for b in range(imgs.size(0)):
            has_s = batch["has_sys"][b].item() > 0
            has_l = batch["has_lesion"][b].item() > 0
            has_t = batch["has_target"][b].item() > 0
            balanced_evaluator.update(
                pred_prob_3d=r_probs[b, 0],
                gland_mask=batch["gland_mask"][b, 0].to(device),
                zones_mask=batch["zones_mask"][b, 0].to(device) if has_s else None,
                sys_labels=batch["sys_labels"][b].to(device) if has_s else None,
                lesion_mask=batch["lesion_mask"][b, 0].to(device) if has_l else None,
                target_mask=batch["target_mask"][b, 0].to(device) if has_t else None,
                has_sys=has_s,
                has_lesion=has_l,
                has_target=has_t,
            )

        g_preds = torch.argmax(g_p, dim=1, keepdim=True)
        for b in range(imgs.size(0)):
            if batch["has_lesion"][b] > 0:
                d_type = "PUB"
            elif batch["has_target"][b] > 0:
                d_type = "TCIA"
            elif batch["has_sys"][b] > 0:
                d_type = "PROMIS"
            else:
                continue

            if saved_counts[d_type] >= max_saves_per_type:
                continue
            if random.random() < plot_prob:
                gt_dict = {
                    "type": d_type,
                    "lesion_mask": batch["lesion_mask"][b][0].cpu().numpy(),
                    "target_mask": batch["target_mask"][b][0].cpu().numpy(),
                    "zones_mask": batch["zones_mask"][b][0].cpu().numpy(),
                    "sys_labels": batch["sys_labels"][b].cpu().numpy(),
                }
                vis_filename = f"{d_type}_{batch['pid'][b]}.png"
                visualize_predictions(imgs[b], r_probs[b], g_preds[b], gt_dict, os.path.join(vis_dir, vis_filename), batch["pid"][b])
                saved_counts[d_type] += 1

    bacc_results = balanced_evaluator.compute_metrics()
    tracker.gland_tpr = bacc_results["gland_sens"]
    tracker.gland_tnr = bacc_results["gland_spec"]
    tracker.gland_bacc = bacc_results["gland_bacc"]
    tracker.region_tpr = bacc_results["region_sens"]
    tracker.region_tnr = bacc_results["region_spec"]
    tracker.region_bacc = bacc_results["region_bacc"]
    return tracker


def plot_loss_curves(log_path, save_path):
    try:
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))

        ax1 = axes[0]
        if "train_loss_total" in df.columns:
            ax1.plot(df["epoch"], df["train_loss_total"], label="Total Loss", linewidth=2)
        for col, label in [
            ("train_loss_grade", "Grade Total"),
            ("train_loss_grade_tbx", "Grade TBx"),
            ("train_loss_grade_sbx", "Grade SBx"),
            ("train_loss_lesion", "Lesion Total"),
            ("train_loss_lesion_dense", "Lesion Dense"),
            ("train_loss_lesion_sparse", "Lesion Sparse"),
            ("train_loss_lesion_sys", "Lesion Sys"),
            ("train_loss_gland", "Gland"),
        ]:
            if col in df.columns:
                ax1.plot(df["epoch"], df[col], label=label, alpha=0.8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Multi-task Training Loss Curves")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = axes[1]
        for col, label in [
            ("em_w_grade_tbx", "Grade TBx Weight"),
            ("em_w_grade_sbx", "Grade SBx Weight"),
            ("em_w_lesion_dense", "Lesion Dense Weight"),
            ("em_w_lesion_sparse", "Lesion Sparse Weight"),
            ("em_w_lesion_sys", "Lesion Sys Weight"),
            ("em_w_gland", "Gland Weight"),
        ]:
            if col in df.columns:
                ax2.plot(df["epoch"], df[col], label=label)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learned Multiplier exp(-log_var)")
        ax2.set_title("Dynamic Uncertainty Weights")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")


def visualize_predictions(input_tensor, risk_map, grade_map, gt_dict, save_path, patient_id):
    t2 = input_tensor[0].detach().cpu().numpy()
    risk = risk_map[0].detach().cpu().numpy()
    grade = grade_map[0].detach().cpu().numpy()

    mid = t2.shape[0] // 2
    slices = [max(0, mid - 5), mid, min(t2.shape[0] - 1, mid + 5)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Patient: {patient_id} | Dataset Type: {gt_dict['type']}", fontsize=18, y=0.98)
    grade_cmap = plt.get_cmap("jet", 7)

    for i, s_idx in enumerate(slices):
        axes[0, i].imshow(t2[s_idx], cmap="gray")
        rmask = np.ma.masked_where(risk[s_idx] < 0.2, risk[s_idx])
        im1 = axes[0, i].imshow(rmask, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred: Risk Map (Slice {s_idx})")
        axes[0, i].axis("off")
        if i == 2:
            fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        axes[1, i].imshow(t2[s_idx], cmap="gray")
        gmask = np.ma.masked_where(grade[s_idx] == 0, grade[s_idx])
        im2 = axes[1, i].imshow(gmask, cmap=grade_cmap, alpha=0.4, vmin=0, vmax=6)
        axes[1, i].set_title(f"Pred: Grade Map (Slice {s_idx})")
        axes[1, i].axis("off")
        if i == 2:
            cbar2 = fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.set_ticks(np.arange(7))
            cbar2.set_ticklabels(["BG", "Ben", "IS1", "IS2", "IS3", "IS4", "IS5"])

        axes[2, i].imshow(t2[s_idx], cmap="gray")
        gt_slice = None
        cmap_gt, vmin_gt, vmax_gt = grade_cmap, 0, 6
        if gt_dict["type"] == "PUB":
            gt_slice = gt_dict["lesion_mask"][s_idx]
            cmap_gt, vmin_gt, vmax_gt = "autumn", 0, 1
            axes[2, i].set_title(f"GT: PUB Lesion Mask (Slice {s_idx})")
        elif gt_dict["type"] == "TCIA":
            gt_slice = gt_dict["target_mask"][s_idx]
            axes[2, i].set_title(f"GT: TCIA Biopsy Target (Slice {s_idx})")
        elif gt_dict["type"] == "PROMIS":
            z_slice = gt_dict["zones_mask"][s_idx]
            sys_labels = gt_dict["sys_labels"]
            gt_slice = np.zeros_like(z_slice)
            for z_idx in range(1, min(20, len(sys_labels)) + 1):
                if sys_labels[z_idx - 1] != getattr(Config, "INVALID_SYS_LABEL", -1):
                    gt_slice[z_slice == z_idx] = sys_labels[z_idx - 1]
            axes[2, i].set_title(f"GT: PROMIS Zone Grades (Slice {s_idx})")

        if gt_slice is not None:
            gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
            im3 = axes[2, i].imshow(gt_mask, cmap=cmap_gt, alpha=0.5, vmin=vmin_gt, vmax=vmax_gt)
            if i == 2:
                cbar3 = fig.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
                if gt_dict["type"] != "PUB":
                    cbar3.set_ticks(np.arange(7))
                    cbar3.set_ticklabels(["BG", "Ben", "IS1", "IS2", "IS3", "IS4", "IS5"])
        axes[2, i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
