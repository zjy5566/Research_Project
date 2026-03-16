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
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,         
                 lambda_sys=0.5,           
                 lambda_lesion=1.0,        
                 lambda_gland=0.2,         
                 lesion_w_dense=1.0,       # PUB 数据集权重
                 lesion_w_sparse=1.0,      # TCIA 靶向权重
                 lesion_w_regional=0.2,    # PROMIS 系统权重 (存在假阴性，作降权处理)
                 csPCa_threshold=3,        # ISUP阈值
                 lesion_pos_weight=40.0):  # 针对极小病灶将权重提高
        """
        阶段 2 调试版 (Phase 2 Debugging Version)
        开启三大独立 Lesion 监督分支：
        1. PUB (Dense): 密集掩膜 BCE + Focal + (Dice)
        2. TCIA (Sparse): 针道掩膜 BCE + Focal
        3. PROMIS (Regional): 区域池化 BCE + Focal
        """
        super(MixedSupervisionLoss, self).__init__()
        
        self.lambda_grade = lambda_grade
        self.lambda_sys = lambda_sys
        self.lambda_lesion = lambda_lesion
        self.lambda_gland = lambda_gland
        
        self.l_w_dense = lesion_w_dense
        self.l_w_sparse = lesion_w_sparse
        self.l_w_regional = lesion_w_regional
        
        self.csPCa_threshold = csPCa_threshold
        
        # Lesion 分割组件 (BCE + Focal Loss 治漏诊组合)
        self.lesion_pos_weight = torch.tensor([lesion_pos_weight])
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.lesion_pos_weight)
        self.lesion_focal_loss = FocalLoss(alpha=0.5, gamma=2.0)
        
        # Gland 腺体分割 (普通 BCE + Dice Loss 稳定组合)
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, 
                grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = target_mask.device
        
        # 确保动态权重在正确的 GPU 设备上
        if self.lesion_bce_loss.pos_weight.device != device:
            self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)
            
        # ===================================================================
        # [安全截断] 防崩溃机制
        # ===================================================================
        dummy_sum = 0.0
        if lesion_pred is not None:
            dummy_sum = dummy_sum + lesion_pred.sum()
        if sys_lesion_preds is not None:
            dummy_sum = dummy_sum + sys_lesion_preds.sum()
        if gland_pred is not None:
            dummy_sum = dummy_sum + gland_pred.sum()
            
        if isinstance(dummy_sum, torch.Tensor):
            zero_loss = dummy_sum * 0.0
        else:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Grade 相关的 Loss 依然用 zero_loss 占位 (当前专注 Lesion)
        loss_grade_target = zero_loss
        loss_grade_sys = zero_loss
        
        # 初始化各大 Loss 容器
        loss_lesion_dense = zero_loss
        loss_lesion_sparse = zero_loss
        loss_lesion_sys = zero_loss
        loss_gland = zero_loss

        # 区分当前 Batch 中的有效数据集类型
        valid_lesion = has_lesion > 0  
        valid_target = has_target > 0  
        valid_sys = has_sys > 0        
        valid_gland = has_gland > 0    

        # ===================================================================
        # [任务 1A]：PUB 数据集 -> Lesion 密集强监督
        # ===================================================================
        if lesion_pred is not None and valid_lesion.any():
            l_pred_valid = lesion_pred[valid_lesion]                       
            l_mask_valid = lesion_mask[valid_lesion].float()               
            
            loss_bce = self.lesion_bce_loss(l_pred_valid, l_mask_valid)
            loss_focal = self.lesion_focal_loss(l_pred_valid, l_mask_valid)
            loss_lesion_dense = loss_bce + loss_focal

        # ===================================================================
        # [任务 1B]：TCIA 数据集 -> 靶向针道稀疏强监督
        # ===================================================================
        if lesion_pred is not None and valid_target.any():
            pred_t = lesion_pred[valid_target]
            mask_t = target_mask[valid_target]
            
            # 仅筛选出针道经过的像素点 (>0)
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                # 动态二值化：将针道里的 ISUP 转化为是否有病灶 (0/1)
                target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                
                # 提取有效的一维张量计算 Loss
                pred_valid = pred_t[valid_pixels]
                label_valid = target_lesion_label[valid_pixels]
                
                loss_bce_sparse = self.lesion_bce_loss(pred_valid, label_valid)
                loss_focal_sparse = self.lesion_focal_loss(pred_valid, label_valid)
                loss_lesion_sparse = loss_bce_sparse + loss_focal_sparse

        # ===================================================================
        # [任务 1C]：PROMIS 数据集 -> 系统活检区域弱监督
        # ===================================================================
        if sys_lesion_preds is not None and valid_sys.any():
            sys_labels_flat = sys_labels.view(-1).long()
            valid_zones = sys_labels_flat > 0 # 剔除未穿刺的背景区(0)
            
            if valid_zones.any():
                sys_lesion_flat = sys_lesion_preds.view(-1)
                sys_lesion_label = (sys_labels_flat >= self.csPCa_threshold).float()
                
                pred_valid = sys_lesion_flat[valid_zones]
                label_valid = sys_lesion_label[valid_zones]
                
                loss_bce_sys = self.lesion_bce_loss(pred_valid, label_valid)
                loss_focal_sys = self.lesion_focal_loss(pred_valid, label_valid)
                loss_lesion_sys = loss_bce_sys + loss_focal_sys

        # ===================================================================
        # [任务 2]：Gland 腺体密集强监督
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
        # 内部融合三条路线的 Lesion Loss
        loss_lesion_total = (self.l_w_dense * loss_lesion_dense + 
                             self.l_w_sparse * loss_lesion_sparse + 
                             self.l_w_regional * loss_lesion_sys)

        # 最终加权汇总
        total_loss = (self.lambda_lesion * loss_lesion_total) + \
                     (self.lambda_gland * loss_gland)
                      
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_total,loss_lesion_dense, loss_lesion_sparse, loss_lesion_sys, loss_gland