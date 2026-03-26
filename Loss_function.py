import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """标准的二分类 Dice Loss"""
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)                 
        probs = probs.view(probs.size(0), -1)         
        targets = targets.view(targets.size(0), -1)   
        
        intersection = (probs * targets).sum(dim=1)   
        union = probs.sum(dim=1) + targets.sum(dim=1) 
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth) 
        return 1.0 - dice.mean()                      

# ===================================================================
# [核心组件]：Focal Loss (针对微小目标和极度不平衡的 Hard 样本挖掘)
# ===================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,         # 主任务 (靶向 ISUP 分级)
                 lambda_tb=1.0,            # 预留参数
                 lambda_sys=0.5,           # 弱监督分级 (系统区域池化)
                 lambda_lesion=1.0,        # 辅任务A (寻找病灶整体权重)
                 lambda_gland=0.2,         # 辅任务B (腺体轮廓)
                 lesion_w_dense=1.0,       # 辅任务A内部：密集强监督 (PUB)
                 lesion_w_sparse=1.0,      # 辅任务A内部：稀疏强监督 (靶向)
                 lesion_w_regional=1.0,    # 辅任务A内部：系统区域弱监督 (PROMIS)
                 csPCa_threshold=3,        # 临床显著性前列腺癌的 ISUP 阈值
                 lesion_w_small=1.0,
                 pos_weight_val=2.0):      # [建议]: 提高 Dense 正样本权重，帮助网络画图
        
        super(MixedSupervisionLoss, self).__init__()
        
        self.lambda_grade = lambda_grade
        self.lambda_sys = lambda_sys
        self.lambda_lesion = lambda_lesion
        self.lambda_gland = lambda_gland
        
        # 动态课程学习权重（这三个属性在 train.py 中会被动态修改）
        self.l_w_dense = lesion_w_dense
        self.l_w_sparse = lesion_w_sparse
        self.l_w_regional = lesion_w_regional
        
        self.csPCa_threshold = csPCa_threshold
        
        # --- Loss 实例化 ---
        self.pos_weight = torch.tensor([pos_weight_val])
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.lesion_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        # [新增] 多分类交叉熵，用于 ISUP 级别预测 (忽略值为 -1 的非合法区域)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, grade_preds, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = lesion_pred.device if lesion_pred is not None else grade_preds.device
        
        # 设备同步
        if self.lesion_bce_loss.pos_weight.device != device:
            self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)

        # 1. 初始化所有 Loss
        loss_grade_target = torch.tensor(0.0, device=device)
        loss_grade_sys = torch.tensor(0.0, device=device)
        
        loss_lesion_dense = torch.tensor(0.0, device=device)
        loss_lesion_sparse = torch.tensor(0.0, device=device)
        loss_lesion_sys = torch.tensor(0.0, device=device)
        loss_gland = torch.tensor(0.0, device=device)
        
        # 有效数据标记
        valid_target = has_target > 0
        valid_sys = has_sys > 0
        valid_lesion = has_lesion > 0
        valid_gland = has_gland > 0

        # ===================================================================
        # [任务 1]：Grade ISUP 分级任务 (多分类)
        # ===================================================================
        # 1A. 靶向针道密集分类强监督 (TCIA)
        if grade_preds is not None and valid_target.any():
            g_p_t = grade_preds[valid_target]       # [N, C, D, H, W]
            mask_t = target_mask[valid_target]      # [N, D, H, W]
            
            # 只选出有针道的地方 (ISUP > 0) 进行分类监督
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                # 维度转换: [N, C, D, H, W] -> [N, D, H, W, C] 方便按照 bool 掩膜提取
                g_p_t_permuted = g_p_t.permute(0, 2, 3, 4, 1)
                
                preds_valid_grade = g_p_t_permuted[valid_pixels] # [num_pixels, C]
                labels_valid_grade = mask_t[valid_pixels].long() # [num_pixels]
                
                loss_grade_target = self.ce_loss(preds_valid_grade, labels_valid_grade)

        # 1B. 系统分区域池化弱监督分类 (PROMIS/TCIA)
        if sys_grade_preds is not None and valid_sys.any():
            s_g_p = sys_grade_preds[valid_sys]      # [N, num_zones, C]
            s_labels = sys_labels[valid_sys]        # [N, num_zones]
            
            # 打平批次和区域维度
            s_g_p_flat = s_g_p.view(-1, s_g_p.size(-1)) # [N * num_zones, C]
            s_labels_flat = s_labels.view(-1)           # [N * num_zones]
            
            # 过滤掉标签为负数（未穿刺）的区域
            valid_zones = s_labels_flat >= 0
            if valid_zones.any():
                loss_grade_sys = self.ce_loss(s_g_p_flat[valid_zones], s_labels_flat[valid_zones].long())

        # ===================================================================
        # [任务 2]：Lesion 临床显著性病灶检测 (二分类)
        # ===================================================================
        # 2A. PUB 数据集 -> 密集强监督
        if lesion_pred is not None and valid_lesion.any():
            pred_l = lesion_pred[valid_lesion]
            mask_l = lesion_mask[valid_lesion].float()
            
            loss_bce_dense = self.lesion_bce_loss(pred_l, mask_l)
            loss_focal_dense = self.lesion_focal_loss(pred_l, mask_l)
            # 建议: 如果后续想恢复形状，可以在这里加回 Dice Loss
            loss_lesion_dense = loss_bce_dense + loss_focal_dense

        # 2B. TCIA 数据集 -> 靶向针道稀疏强监督
        if lesion_pred is not None and valid_target.any():
            pred_t = lesion_pred[valid_target]
            mask_t = target_mask[valid_target]
            
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                
                pred_valid = pred_t[valid_pixels]
                label_valid = target_lesion_label[valid_pixels]
                
                loss_bce_sparse = self.lesion_bce_loss(pred_valid, label_valid)
                loss_focal_sparse = self.lesion_focal_loss(pred_valid, label_valid)
                loss_lesion_sparse = loss_bce_sparse + loss_focal_sparse

        # 2C. PROMIS 数据集 -> 系统分区域池化弱监督
        if sys_lesion_preds is not None and valid_sys.any():
            s_l_p = sys_lesion_preds[valid_sys]
            s_labels = sys_labels[valid_sys]
            
            s_l_p_flat = s_l_p.view(-1)
            s_labels_flat = s_labels.view(-1)
            
            valid_zones = s_labels_flat >= 0
            if valid_zones.any():
                sys_lesion_label = (s_labels_flat >= self.csPCa_threshold).float()
                
                pred_valid = s_l_p_flat[valid_zones]
                label_valid = sys_lesion_label[valid_zones]
                
                loss_bce_sys = self.lesion_bce_loss(pred_valid, label_valid)
                loss_focal_sys = self.lesion_focal_loss(pred_valid, label_valid)
                loss_lesion_sys = loss_bce_sys + loss_focal_sys

        # ===================================================================
        # [任务 3]：Gland 腺体密集强监督
        # ===================================================================
        if gland_pred is not None and valid_gland.any():
            g_pred_valid = gland_pred[valid_gland]
            g_mask_valid = gland_mask[valid_gland].float()
            
            loss_g_bce = self.gland_bce_loss(g_pred_valid, g_mask_valid)
            loss_g_dice = self.dice_loss(g_pred_valid, g_mask_valid)
            loss_gland = loss_g_bce + loss_g_dice

        # ===================================================================
        # 汇总返回
        # ===================================================================
        # Lesion 内部的动态权重融合
        loss_lesion_total = (self.l_w_dense * loss_lesion_dense + 
                             self.l_w_sparse * loss_lesion_sparse + 
                             self.l_w_regional * loss_lesion_sys)

        # 四大总任务的最终融合
        total_loss = (self.lambda_grade * loss_grade_target) + \
                     (self.lambda_sys * loss_grade_sys) + \
                     (self.lambda_lesion * loss_lesion_total) + \
                     (self.lambda_gland * loss_gland)
                      
        # 返回结构与 train.py 严格对齐
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_total, loss_lesion_dense, loss_lesion_sparse, loss_lesion_sys, loss_gland