import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from tqdm import tqdm
from config import Config

# ==========================================
# 1. 指标计算核心函数
# ==========================================
def compute_dice(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()

def compute_f1(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    if targets.sum() == 0 and preds.sum() == 0:
        return 1.0
    return f1_score(targets, preds, zero_division=0)

# [新增] 计算 Sensitivity (召回率)
def compute_sens(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-7)
    return sensitivity

def compute_kappa(preds, targets):
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    if len(np.unique(targets)) <= 1:
        return 0.0
    try:
        return cohen_kappa_score(targets, preds, weights='quadratic', labels=np.arange(7))
    except:
        return 0.0

# ==========================================
# 2. 实验追踪器 (MetricTracker)
# ==========================================
class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        if not np.isnan(val) and not np.isinf(val):
            self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

class MetricTracker:
    def __init__(self):
        # 损失 (Train 和 Val 都需要)
        self.loss_total = AverageMeter()
        self.loss_grade = AverageMeter()
        self.loss_lesion = AverageMeter()
        
        # [修改] 新增三个细分的 lesion loss
        self.loss_lesion_dense = AverageMeter()
        self.loss_lesion_sparse = AverageMeter()
        self.loss_lesion_sys = AverageMeter()
        
        self.loss_gland = AverageMeter()
        
        # 性能指标 (只给 Val 算)
        self.lesion_dice = AverageMeter()
        self.lesion_f1 = AverageMeter()
        self.lesion_sens = AverageMeter() # [新增]
        self.gland_dice = AverageMeter()
        self.grade_kappa = AverageMeter()

    # [修改] 接收三个细分的 lesion loss
    def update_losses(self, total, g, l, l_dense, l_sparse, l_sys, gl):
        self.loss_total.update(total)
        self.loss_grade.update(g)
        self.loss_lesion.update(l)
        self.loss_lesion_dense.update(l_dense)
        self.loss_lesion_sparse.update(l_sparse)
        self.loss_lesion_sys.update(l_sys)
        self.loss_gland.update(gl)

    def print_train_summary(self):
        return (f"Loss: {self.loss_total.avg:.4f} | L_Grad: {self.loss_grade.avg:.4f} | "
                f"L_Les: {self.loss_lesion.avg:.4f} | L_Glan: {self.loss_gland.avg:.4f}")

    def print_val_summary(self):
        return (f"Loss: {self.loss_total.avg:.4f} | Les-Dice(PUB): {self.lesion_dice.avg:.4f} | "
                f"Les-Sens: {self.lesion_sens.avg:.4f} | Glan-Dice: {self.gland_dice.avg:.4f} | Grade-Kap(TC/PR): {self.grade_kappa.avg:.4f}")

    def get_train_dict(self):
        """训练集只返回 Loss"""
        return {
            'train_loss_total': self.loss_total.avg,
            'train_loss_grade': self.loss_grade.avg,
            'train_loss_lesion': self.loss_lesion.avg,
            # [修改] 暴露子分支 Loss
            'train_loss_lesion_dense': self.loss_lesion_dense.avg,
            'train_loss_lesion_sparse': self.loss_lesion_sparse.avg,
            'train_loss_lesion_sys': self.loss_lesion_sys.avg,
            'train_loss_gland': self.loss_gland.avg
        }

    def get_val_dict(self):
        """验证集返回全面指标"""
        return {
            'val_loss_total': self.loss_total.avg,
            'val_loss_grade': self.loss_grade.avg,
            'val_loss_lesion': self.loss_lesion.avg,
            # [修改] 暴露子分支 Loss
            'val_loss_lesion_dense': self.loss_lesion_dense.avg,
            'val_loss_lesion_sparse': self.loss_lesion_sparse.avg,
            'val_loss_lesion_sys': self.loss_lesion_sys.avg,
            'val_loss_gland': self.loss_gland.avg,
            'val_lesion_dice': self.lesion_dice.avg,
            'val_lesion_f1': self.lesion_f1.avg,
            'val_lesion_sens': self.lesion_sens.avg, # [新增]
            'val_gland_dice': self.gland_dice.avg,
            'val_grade_kappa': self.grade_kappa.avg
        }

# ==========================================
# 3. 集成验证流程 (Validation Routine)
# ==========================================
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, save_dir):
    model.eval()
    tracker = MetricTracker()
    vis_dir = os.path.join(save_dir, Config.VIS_SUBDIR, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)

    # 随机抽样配置
    saved_counts = {'PUB': 0, 'TCIA': 0, 'PROMIS': 0}
    max_saves_per_type = 2 
    plot_prob = 0.15 

    for batch in tqdm(loader, desc="Validation"):
        imgs = batch['input'].to(device)
        z_mask = batch['zones_mask'].to(device)
        
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)

        # ---------------------------
        # 计算 Loss
        # ---------------------------
        # [修改] 接收新增的三个 lesion 子 loss
        total_loss, l_grad, l_sys, l_les, l_les_dense, l_les_sparse, l_les_sys, l_gland = criterion(
            g_p, s_g_p, l_p, s_l_p, gl_p,
            batch['target_mask'].to(device), batch['sys_labels'].to(device),
            batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
            batch['has_target'].to(device), batch['has_sys'].to(device),
            batch['has_lesion'].to(device), batch['has_gland'].to(device)
        )
        
        # [修改] 传递子 loss 给 tracker
        tracker.update_losses(
            total_loss.item(), 
            (l_grad + l_sys).item(), 
            l_les.item(), 
            l_les_dense.item(), 
            l_les_sparse.item(), 
            l_les_sys.item(), 
            l_gland.item()
        )

        # ---------------------------
        # 计算评价指标 (分源独立计算)
        # ---------------------------
        # 1. Gland (所有数据集只要有标注就能算)
        if batch['has_gland'].sum() > 0:
            idx = batch['has_gland'] > 0
            g_bin = (torch.sigmoid(gl_p[idx]) > 0.5).float()
            tracker.gland_dice.update(compute_dice(g_bin, batch['gland_mask'][idx].to(device)))
        
        # 2. Lesion Dice, F1 & Sens (极其严格：只能用 PUB 算)
        if batch['has_lesion'].sum() > 0:
            idx = batch['has_lesion'] > 0
            lp, lt = torch.sigmoid(l_p[idx]), batch['lesion_mask'][idx].to(device)
            lb = (lp > 0.5).float()
            tracker.lesion_dice.update(compute_dice(lb, lt))
            tracker.lesion_f1.update(compute_f1(lb, lt))
            tracker.lesion_sens.update(compute_sens(lb, lt)) # [新增]
            
        # 3. Grade Kappa (系统活检 TCIA, PROMIS 算)
        if batch['has_sys'].sum() > 0:
            idx = batch['has_sys'] > 0
            sys_pred_flat = torch.argmax(s_g_p[idx], dim=-1).flatten()
            sys_true_flat = batch['sys_labels'][idx].flatten().to(device)
            valid_mask = sys_true_flat > 0
            if valid_mask.sum() > 0:
                tracker.grade_kappa.update(compute_kappa(sys_pred_flat[valid_mask], sys_true_flat[valid_mask]))

        # ---------------------------
        # 智能随机可视化抽样
        # ---------------------------
        r_probs = torch.sigmoid(l_p)
        g_preds = torch.argmax(g_p, dim=1, keepdim=True)

        for b in range(imgs.size(0)):
            if batch['has_lesion'][b] > 0: d_type = 'PUB'
            elif batch['has_target'][b] > 0: d_type = 'TCIA'
            elif batch['has_sys'][b] > 0: d_type = 'PROMIS'
            else: continue

            if saved_counts[d_type] >= max_saves_per_type:
                continue
                
            if random.random() < plot_prob:
                gt_dict = {
                    'type': d_type,
                    'lesion_mask': batch['lesion_mask'][b][0].cpu().numpy(),
                    'target_mask': batch['target_mask'][b][0].cpu().numpy(),
                    'zones_mask': batch['zones_mask'][b][0].cpu().numpy(),
                    'sys_labels': batch['sys_labels'][b].cpu().numpy()
                }
                
                vis_filename = f"{d_type}_{batch['pid'][b]}.png"
                visualize_predictions(
                    imgs[b], r_probs[b], g_preds[b], gt_dict,
                    os.path.join(vis_dir, vis_filename), batch['pid'][b]
                )
                saved_counts[d_type] += 1

    return tracker


# ==========================================
# 4. 图表工具
# ==========================================
def plot_loss_curves(log_path, save_path):
    try:
        df = pd.read_csv(log_path)
        plt.figure(figsize=(12, 8)) # 稍微放大画板容纳更多曲线
        
        plt.plot(df['epoch'], df['train_loss_total'], label='Total Loss', color='black', lw=2)
        
        if 'train_loss_grade' in df.columns:
            plt.plot(df['epoch'], df['train_loss_grade'], '--', label='Grade Loss', alpha=0.7)
        if 'train_loss_gland' in df.columns:
            plt.plot(df['epoch'], df['train_loss_gland'], '--', label='Gland Loss', alpha=0.7)
            
        # [修改] 绘制整体的 Lesion Loss 和 细分的 Lesion Loss
        if 'train_loss_lesion' in df.columns:
            plt.plot(df['epoch'], df['train_loss_lesion'], '-.', label='Lesion Total', alpha=0.9, lw=2)
        if 'train_loss_lesion_dense' in df.columns:
            plt.plot(df['epoch'], df['train_loss_lesion_dense'], ':', label='Lesion Dense (PUB)', alpha=0.7)
        if 'train_loss_lesion_sparse' in df.columns:
            plt.plot(df['epoch'], df['train_loss_lesion_sparse'], ':', label='Lesion Sparse (TCIA)', alpha=0.7)
        if 'train_loss_lesion_sys' in df.columns:
            plt.plot(df['epoch'], df['train_loss_lesion_sys'], ':', label='Lesion Sys (PROMIS)', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Multi-Task Training Loss Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放到外侧避免遮挡
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")

def visualize_predictions(input_tensor, risk_map, grade_map, gt_dict, save_path, patient_id):
    t2 = input_tensor[0].cpu().numpy()
    risk = risk_map[0].cpu().numpy()
    grade = grade_map[0].cpu().numpy()
    
    mid = t2.shape[0] // 2
    slices = [max(0, mid - 5), mid, min(t2.shape[0] - 1, mid + 5)]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Patient: {patient_id} | Dataset Type: {gt_dict['type']}", fontsize=18, y=0.98)
    
    grade_cmap = plt.get_cmap('jet', 7) 

    for i, s_idx in enumerate(slices):
        # Row 1: Risk Map
        axes[0, i].imshow(t2[s_idx], cmap='gray')
        risk_slice = risk[s_idx]
        rmask = np.ma.masked_where(risk_slice < 0.2, risk_slice)
        im1 = axes[0, i].imshow(rmask, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Pred: Risk Map (Slice {s_idx})")
        axes[0, i].axis('off')
        if i == 2: fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Row 2: Grade Map
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

        # Row 3: Ground Truth
        axes[2, i].imshow(t2[s_idx], cmap='gray')
        
        gt_slice = None
        cmap_gt, vmin_gt, vmax_gt = grade_cmap, 0, 6
        
        if gt_dict['type'] == 'PUB':
            gt_slice = gt_dict['lesion_mask'][s_idx]
            cmap_gt = 'autumn' 
            vmin_gt, vmax_gt = 0, 1
            axes[2, i].set_title(f"GT: PUB Lesion Mask (Slice {s_idx})")
            
        elif gt_dict['type'] == 'TCIA':
            gt_slice = gt_dict['target_mask'][s_idx]
            axes[2, i].set_title(f"GT: TCIA Biopsy Target (Slice {s_idx})")
            
        elif gt_dict['type'] == 'PROMIS':
            z_slice = gt_dict['zones_mask'][s_idx]
            sys_labels = gt_dict['sys_labels']
            gt_slice = np.zeros_like(z_slice)
            for z_idx in range(1, 21): 
                gt_slice[z_slice == z_idx] = sys_labels[z_idx - 1]
            axes[2, i].set_title(f"GT: PROMIS Zone Grades (Slice {s_idx})")
        
        if gt_slice is not None:
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