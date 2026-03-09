import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 import plt 之前
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff

# ==========================================
# 1. 密集分割指标 (Segmentation Metrics)
# ==========================================
def compute_dice(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()

def compute_iou(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()

# ==========================================
# 2. 分类与临床指标 (Classification Metrics)
# ==========================================
def compute_auc(probs, targets):
    probs = probs.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    try:
        if len(np.unique(targets)) > 1:
            return roc_auc_score(targets, probs)
        return np.nan
    except:
        return np.nan

def compute_sens_spec(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    return sensitivity, specificity

def compute_kappa(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    if len(np.unique(targets)) <= 1:
        return 0.0
    try:
        # labels=np.arange(7) 保证 7x7 混淆矩阵，解决只有单标签时的报错
        return cohen_kappa_score(targets, preds, weights='quadratic', labels=np.arange(7))
    except:
        return 0.0

# ==========================================
# 3. 可视化函数 (修复了标题和颜色栏)
# ==========================================
def plot_loss_curves(log_path, save_path):
    try:
        df = pd.read_csv(log_path)
        plt.figure(figsize=(10, 6))
        
        # 匹配 train.py 中由 get_dict(prefix='train_') 生成的列名
        plt.plot(df['epoch'], df['train_loss_total'], label='Total Loss', color='black', lw=2)
        
        if 'train_loss_grade' in df.columns:
            plt.plot(df['epoch'], df['train_loss_grade'], '--', label='Grade Loss', alpha=0.7)
        if 'train_loss_lesion' in df.columns:
            plt.plot(df['epoch'], df['train_loss_lesion'], '--', label='Lesion Loss', alpha=0.7)
        if 'train_loss_gland' in df.columns:
            plt.plot(df['epoch'], df['train_loss_gland'], '--', label='Gland Loss', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Multi-Task Training Loss Curves')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")
def visualize_predictions(input_tensor, risk_map, grade_map, gt_dict, save_path, patient_id):
    """
    将 Risk Map, Grade Map, 以及 Ground Truth 叠加到 T2 图像上。
    :param gt_dict: 包含当前样本来源和真实标签的字典
    """
    t2 = input_tensor[0].cpu().numpy()
    risk = risk_map[0].cpu().numpy()
    grade = grade_map[0].cpu().numpy()
    
    mid = t2.shape[0] // 2
    slices = [mid - 5, mid, mid + 5]
    
    # 扩大画布为 3 行 (Risk Pred, Grade Pred, Ground Truth)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Patient: {patient_id} | Dataset Type: {gt_dict['type']}", fontsize=18, y=0.98)
    
    # 颜色映射
    grade_cmap = plt.get_cmap('jet', 7) 

    for i, s_idx in enumerate(slices):
        # ---------------------------------------------------------
        # Row 1: 预测的 Risk Map (热图)
        # ---------------------------------------------------------
        axes[0, i].imshow(t2[s_idx], cmap='gray')
        risk_slice = risk[s_idx]
        rmask = np.ma.masked_where(risk_slice < 0.2, risk_slice)
        im1 = axes[0, i].imshow(rmask, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred: Risk Map (Slice {s_idx})")
        axes[0, i].axis('off')
        if i == 2: fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # ---------------------------------------------------------
        # Row 2: 预测的 Grade Map (分级)
        # ---------------------------------------------------------
        axes[1, i].imshow(t2[s_idx], cmap='gray')
        grade_slice = grade[s_idx]
        gmask = np.ma.masked_where(grade_slice == 0, grade_slice)
        im2 = axes[1, i].imshow(gmask, cmap=grade_cmap, alpha=0.4, vmin=0, vmax=6)
        axes[1, i].set_title(f"Pred: Grade Map (Slice {s_idx})")
        axes[1, i].axis('off')
        if i == 2:
            cbar2 = fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            cbar2.set_ticks(np.arange(7))
            cbar2.set_ticklabels(['BG', 'Ben', 'IS1', 'IS2', 'IS3', 'IS4', 'IS5'])

        # ---------------------------------------------------------
        # Row 3: 真实的 Ground Truth (根据数据集类型动态绘制)
        # ---------------------------------------------------------
        axes[2, i].imshow(t2[s_idx], cmap='gray')
        
        gt_slice = None
        cmap_gt, vmin_gt, vmax_gt = grade_cmap, 0, 6
        
        if gt_dict['type'] == 'PUB':
            # PUB: 只有二分类的病灶 Mask
            gt_slice = gt_dict['lesion_mask'][s_idx]
            cmap_gt = 'autumn' # 用黄红色调表示病灶掩膜
            vmin_gt, vmax_gt = 0, 1
            axes[2, i].set_title(f"GT: PUB Lesion Mask (Slice {s_idx})")
            
        elif gt_dict['type'] == 'TCIA':
            # TCIA: 靶向针道 (自带 0-6 标签)
            gt_slice = gt_dict['target_mask'][s_idx]
            axes[2, i].set_title(f"GT: TCIA Biopsy Target (Slice {s_idx})")
            
        elif gt_dict['type'] == 'PROMIS':
            # PROMIS: 区域分级。需要将 1D 的 sys_labels 填入 zones_mask 中
            z_slice = gt_dict['zones_mask'][s_idx]
            sys_labels = gt_dict['sys_labels']
            gt_slice = np.zeros_like(z_slice)
            for z_idx in range(1, 21): # 遍历 20 个分区
                gt_slice[z_slice == z_idx] = sys_labels[z_idx - 1]
            axes[2, i].set_title(f"GT: PROMIS Zone Grades (Slice {s_idx})")
        
        if gt_slice is not None:
            # 半透叠加 GT
            gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
            im3 = axes[2, i].imshow(gt_mask, cmap=cmap_gt, alpha=0.5, vmin=vmin_gt, vmax=vmax_gt)
            
        axes[2, i].axis('off')
        if i == 2 and gt_slice is not None:
            cbar3 = fig.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
            if gt_dict['type'] != 'PUB':
                cbar3.set_ticks(np.arange(7))
                cbar3.set_ticklabels(['BG', 'Ben', 'IS1', 'IS2', 'IS3', 'IS4', 'IS5'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
# ==========================================
# 4. 实验追踪器 (修复了 get_dict 包含子 Loss)
# ==========================================
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        if not np.isnan(val) and not np.isinf(val):
            self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

class MetricTracker:
    def __init__(self):
        self.loss_total = AverageMeter()
        self.loss_grade = AverageMeter()
        self.loss_lesion = AverageMeter()
        self.loss_gland = AverageMeter()
        
        self.grade_kappa = AverageMeter()
        self.lesion_dice = AverageMeter()
        self.lesion_auc = AverageMeter()
        self.lesion_sens = AverageMeter()
        self.lesion_spec = AverageMeter()
        self.gland_dice = AverageMeter()

    def update_losses(self, total, g, l, gl):
        self.loss_total.update(total)
        self.loss_grade.update(g)
        self.loss_lesion.update(l)
        self.loss_gland.update(gl)

    def print_summary(self):
        return (f"Loss: {self.loss_total.avg:.4f} | Kappa: {self.grade_kappa.avg:.4f} | "
                f"L-Dice: {self.lesion_dice.avg:.4f} | L-Sens: {self.lesion_sens.avg:.4f}")

    def get_dict(self, prefix=''):
        """将所有子 Loss 及其它指标暴露出来给 CSV"""
        return {
            f'{prefix}loss_total': self.loss_total.avg,
            f'{prefix}loss_grade': self.loss_grade.avg,
            f'{prefix}loss_lesion': self.loss_lesion.avg,
            f'{prefix}loss_gland': self.loss_gland.avg,
            f'{prefix}grade_kappa': self.grade_kappa.avg,
            f'{prefix}lesion_dice': self.lesion_dice.avg,
            f'{prefix}lesion_auc': self.lesion_auc.avg,
            f'{prefix}lesion_sens': self.lesion_sens.avg,
            f'{prefix}lesion_spec': self.lesion_spec.avg,
            f'{prefix}gland_dice': self.gland_dice.avg
        }