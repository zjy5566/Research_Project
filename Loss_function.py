import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """标准的二分类 Dice Loss"""
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits 维度: (B, 1, D, H, W)
        probs = torch.sigmoid(logits)                 
        
        # 展平空间维度，保持 Batch 维度独立计算
        probs = probs.view(probs.size(0), -1)         
        targets = targets.view(targets.size(0), -1)   
        
        intersection = (probs * targets).sum(dim=1)   
        union = probs.sum(dim=1) + targets.sum(dim=1) 
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth) 
        return 1.0 - dice.mean()                      


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,         
                 lambda_sys=0.5,           
                 lambda_lesion=1.0,        
                 lambda_gland=0.2,         
                 lesion_w_dense=1.0,       
                 lesion_w_sparse=1.0,      
                 lesion_w_regional=0.2,    
                 csPCa_threshold=3,        
                 lesion_pos_weight=15.0):  
        """
        阶段 1.5 调试版 (Phase 1.5 Debugging Version)
        开启 PUB 数据集的：
        1. Lesion 密集分割 Loss (带正样本极高加权)
        2. Gland 腺体密集分割 Loss (普通加权)
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
        
        # ===================================================================
        # [核心组件 1]：Lesion 分割 (加权 BCE + Dice)
        # ===================================================================
        self.lesion_pos_weight = torch.tensor([lesion_pos_weight])
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.lesion_pos_weight)
        
        # ===================================================================
        # [核心组件 2]：Gland 分割 (普通 BCE + Dice)
        # 腺体足够大，普通 BCE 就能学得很好
        # ===================================================================
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_loss = DiceLoss()

    def forward(self, 
                grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = target_mask.device
        
        # 确保动态创建的权紧张量在正确的 GPU 设备上
        if self.lesion_bce_loss.pos_weight.device != device:
            self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)
            
        # ===================================================================
        # [安全截断]：创建带有梯度计算图的 0.0 占位符 (防崩溃神器升级版)
        # 把 lesion 和 gland 的预测都加进去，确保即使某一个没算，梯度也不会断
        # ===================================================================
        dummy_sum = 0.0
        if lesion_pred is not None:
            dummy_sum = dummy_sum + lesion_pred.sum()
        if gland_pred is not None:
            dummy_sum = dummy_sum + gland_pred.sum()
            
        if isinstance(dummy_sum, torch.Tensor):
            zero_loss = dummy_sum * 0.0
        else:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss_grade_target = zero_loss
        loss_grade_sys = zero_loss
        
        loss_lesion_dense = zero_loss
        loss_gland = zero_loss

        # 区分当前 Batch 中的有效数据集类型
        valid_lesion = has_lesion > 0  
        valid_gland = has_gland > 0    

        # ===================================================================
        # [任务 1]：PUB 数据集 -> Lesion 病灶发掘 (密集强监督)
        # ===================================================================
        if lesion_pred is not None and valid_lesion.any():
            l_pred_valid = lesion_pred[valid_lesion]                       
            l_mask_valid = lesion_mask[valid_lesion].float()               
            
            loss_bce = self.lesion_bce_loss(l_pred_valid, l_mask_valid)
            loss_dice = self.dice_loss(l_pred_valid, l_mask_valid)
            
            loss_lesion_dense = loss_bce + loss_dice

        # ===================================================================
        # [任务 2]：PUB 数据集 -> Gland 腺体分割 (密集强监督)
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
        # 使用预设的超参数权重将两者组合起来
        total_loss = (self.lambda_lesion * self.l_w_dense * loss_lesion_dense) + \
                     (self.lambda_gland * loss_gland)
                      
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_dense, loss_gland