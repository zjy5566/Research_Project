import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        标准的二分类 Dice Loss
        :param smooth: 平滑系数，防止分母为 0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 网络的输出是没有经过激活的 logits，需要先 Sigmoid
        probs = torch.sigmoid(logits)
        
        # 展平空间维度 (保持 Batch 维度独立计算)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        # 返回 1 - dice 作为 Loss
        return 1.0 - dice.mean()


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,   # 主任务 (靶向 ISUP 分级)
                 lambda_sys=0.5,     # 弱监督 (系统区域池化)
                 lambda_lesion=1.0,  # 辅任务A (寻找病灶，BCE+Dice)
                 lambda_gland=0.2,   # 辅任务B (腺体轮廓，极易收敛，权重设低)
                 csPCa_threshold=3): # ISUP >= 2 视为有病灶
        super(MixedSupervisionLoss, self).__init__()
        
        # 保存权重变量 (Latent variables)
        self.lambda_grade = lambda_grade
        self.lambda_sys = lambda_sys
        self.lambda_lesion = lambda_lesion
        self.lambda_gland = lambda_gland
        self.csPCa_threshold = csPCa_threshold
        
        # ==========================================
        # 损失函数组件实例化
        # ==========================================
        # 1. 忽略背景(0) 的多分类交叉熵
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        # 2. 二分类基础交叉熵
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 3. 经典的 Dice Loss
        self.dice_loss = DiceLoss()

    def forward(self, 
                grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = target_mask.device
        
        # 初始化各项 Loss，防止空 Batch 报错
        loss_grade_target = torch.tensor(0.0, device=device)
        loss_grade_sys = torch.tensor(0.0, device=device)
        loss_lesion_dense = torch.tensor(0.0, device=device)
        loss_lesion_sparse = torch.tensor(0.0, device=device)
        loss_lesion_sys = torch.tensor(0.0, device=device)
        loss_gland = torch.tensor(0.0, device=device)

        # -------------------------------------------------------------------
        # 1. Cancer Grade Map 监督 (分类任务: 0-6)
        # -------------------------------------------------------------------
        if grade_pred is not None and has_target.sum() > 0:
            t_mask_sq = target_mask.squeeze(1).long() 
            valid_t = has_target > 0
            if valid_t.sum() > 0:
                loss_grade_target = self.ce_loss(grade_pred[valid_t], t_mask_sq[valid_t])

        if sys_grade_preds is not None and has_sys.sum() > 0:
            sys_preds_flat = sys_grade_preds.view(-1, sys_grade_preds.size(-1))
            sys_labels_flat = sys_labels.view(-1).long()
            valid_s = has_sys > 0
            if valid_s.sum() > 0:
                loss_grade_sys = self.ce_loss(sys_preds_flat, sys_labels_flat)

        # -------------------------------------------------------------------
        # 2. Lesion Risk Map 监督 (二分类任务)
        # -------------------------------------------------------------------
        if lesion_pred is not None:
            # [2.A] 密集强监督 (PUB 数据集) -> 【BCE + Dice (1:1 经典比例)】
            valid_lesion = has_lesion > 0
            if valid_lesion.sum() > 0:
                l_pred_valid = lesion_pred[valid_lesion]
                l_mask_valid = lesion_mask[valid_lesion].float()
                
                loss_bce = self.bce_loss(l_pred_valid, l_mask_valid)
                loss_dice = self.dice_loss(l_pred_valid, l_mask_valid)
                loss_lesion_dense = loss_bce + loss_dice

            # [2.B] 稀疏靶向监督 (TCIA 针道) -> 【仅 BCE】 (极度稀疏的线算 Dice 会导致分母不稳)
            valid_target = has_target > 0
            if valid_target.sum() > 0:
                pred_t = lesion_pred[valid_target]
                mask_t = target_mask[valid_target]
                valid_pixels = mask_t > 0
                if valid_pixels.sum() > 0:
                    target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                    loss_lesion_sparse = self.bce_loss(pred_t[valid_pixels], target_lesion_label[valid_pixels])

            # [2.C] 区域弱监督 (系统活检) -> 【仅 BCE】
            valid_sys = has_sys > 0
            if sys_lesion_preds is not None and valid_sys.sum() > 0:
                sys_lesion_flat = sys_lesion_preds.view(-1)
                sys_labels_flat = sys_labels.view(-1)
                valid_zones = sys_labels_flat > 0
                if valid_zones.sum() > 0:
                    sys_lesion_label = (sys_labels_flat >= self.csPCa_threshold).float()
                    loss_lesion_sys = self.bce_loss(sys_lesion_flat[valid_zones], sys_lesion_label[valid_zones])
                    
        loss_lesion_total = loss_lesion_dense + loss_lesion_sparse + loss_lesion_sys

        # -------------------------------------------------------------------
        # 3. Gland Anatomy Map 监督 (二分类任务)
        # -------------------------------------------------------------------
        # [3] 密集前列腺分割 -> 【BCE + Dice (1:1 经典比例)】
        if gland_pred is not None and has_gland.sum() > 0:
            valid_gland = has_gland > 0
            if valid_gland.sum() > 0:
                g_pred_valid = gland_pred[valid_gland]
                g_mask_valid = gland_mask[valid_gland].float()
                
                loss_bce_g = self.bce_loss(g_pred_valid, g_mask_valid)
                loss_dice_g = self.dice_loss(g_pred_valid, g_mask_valid)
                loss_gland = loss_bce_g + loss_dice_g

        # -------------------------------------------------------------------
        # 4. 加权总和 (引入你要求的 Configurable Weights)
        # -------------------------------------------------------------------
        total_loss = (self.lambda_grade * loss_grade_target + 
                      self.lambda_sys * loss_grade_sys + 
                      self.lambda_lesion * loss_lesion_total + 
                      self.lambda_gland * loss_gland)
                      
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_total, loss_gland