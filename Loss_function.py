import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """标准的二分类 Dice Loss"""
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits 维度: (B, 1, D, H, W)
        probs = torch.sigmoid(logits)                 # probs 维度: (B, 1, D, H, W)
        
        # 展平空间维度，保持 Batch 维度独立计算
        probs = probs.view(probs.size(0), -1)         # probs 维度: (B, D*H*W)
        targets = targets.view(targets.size(0), -1)   # targets 维度: (B, D*H*W)
        
        intersection = (probs * targets).sum(dim=1)   # intersection 维度: (B,)
        union = probs.sum(dim=1) + targets.sum(dim=1) # union 维度: (B,)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth) # dice 维度: (B,)
        return 1.0 - dice.mean()                      # 标量 (Scalar)


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,         
                 lambda_sys=0.5,           
                 lambda_lesion=1.0,        
                 lambda_gland=0.2,         
                 lesion_w_dense=1.0,       
                 lesion_w_sparse=1.0,      
                 lesion_w_regional=0.2,    
                 csPCa_threshold=2):       
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
        # [优化 1]：引入原生类别权重解决长尾分布 (Long-tail Imbalance)
        # ===================================================================
        # isup_weights 维度: (7,)
        self.isup_weights = torch.tensor([0.0, 1.0, 1.0, 1.5, 2.0, 3.0, 3.0])
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.isup_weights, ignore_index=0)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, 
                grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        # -------------------------------------------------------------------
        # 【输入张量维度梳理】
        # grade_pred:       (B, 7, D, H, W)
        # sys_grade_preds:  (B, 20, 7)
        # lesion_pred:      (B, 1, D, H, W)
        # sys_lesion_preds: (B, 20) 或 (B, 20, 1)
        # gland_pred:       (B, 1, D, H, W)
        # target_mask:      (B, 1, D, H, W)
        # sys_labels:       (B, 20)
        # lesion_mask:      (B, 1, D, H, W)
        # gland_mask:       (B, 1, D, H, W)
        # has_*:            (B,) 的 float 张量
        # -------------------------------------------------------------------
        
        device = target_mask.device
        
        if self.ce_loss.weight.device != device:
            self.ce_loss.weight = self.ce_loss.weight.to(device)
            
        # 预先分配标量 Loss: ()
        loss_grade_target = torch.tensor(0.0, device=device)
        loss_grade_sys = torch.tensor(0.0, device=device)
        loss_lesion_dense = torch.tensor(0.0, device=device)
        loss_lesion_sparse = torch.tensor(0.0, device=device)
        loss_lesion_sys = torch.tensor(0.0, device=device)
        loss_gland = torch.tensor(0.0, device=device)

        # bool 掩膜，维度皆为 (B,)
        valid_lesion = has_lesion > 0  
        valid_target = has_target > 0  
        valid_sys = has_sys > 0        
        valid_gland = has_gland > 0    

        # ===================================================================
        # [优化 2]：Logit 先验叠加 (Logit Prior Conditioning)
        # ===================================================================
        if grade_pred is not None and lesion_pred is not None:
            grade_pred_cond = grade_pred.clone()                           # 维度: (B, 7, D, H, W)
            # 广播相加: grade_pred_cond[:, 2:] (B, 5, D, H, W) += lesion_pred (B, 1, D, H, W)
            grade_pred_cond[:, self.csPCa_threshold:, ...] += lesion_pred.detach() 
        else:
            grade_pred_cond = grade_pred

        if sys_grade_preds is not None and sys_lesion_preds is not None:
            sys_grade_preds_cond = sys_grade_preds.clone()                 # 维度: (B, 20, 7)
            
            # 【修复点】：强制转化为 (B, 20, 1)，无论输入是 (B, 20) 还是 (B, 20, 1)
            B = sys_lesion_preds.size(0)
            s_l_p_safe = sys_lesion_preds.view(B, -1, 1)                   # 维度: (B, 20, 1)
            
            # 广播相加: sys_grade_preds_cond[:, :, 2:] (B, 20, 5) += s_l_p_safe (B, 20, 1)
            sys_grade_preds_cond[:, :, self.csPCa_threshold:] += s_l_p_safe.detach()
        else:
            sys_grade_preds_cond = sys_grade_preds

        # -------------------------------------------------------------------
        # 路线 A: PUB 数据集特征 -> 密集强监督 
        # -------------------------------------------------------------------
        if lesion_pred is not None and valid_lesion.any():
            l_pred_valid = lesion_pred[valid_lesion]                       # 维度: (B_val, 1, D, H, W)
            l_mask_valid = lesion_mask[valid_lesion].float()               # 维度: (B_val, 1, D, H, W)
            loss_lesion_dense = self.bce_loss(l_pred_valid, l_mask_valid) + self.dice_loss(l_pred_valid, l_mask_valid)

        if gland_pred is not None and valid_gland.any():
            g_pred_valid = gland_pred[valid_gland]                         # 维度: (B_val, 1, D, H, W)
            g_mask_valid = gland_mask[valid_gland].float()                 # 维度: (B_val, 1, D, H, W)
            loss_gland = self.bce_loss(g_pred_valid, g_mask_valid) + self.dice_loss(g_pred_valid, g_mask_valid)

        # -------------------------------------------------------------------
        # 路线 B: TCIA 数据集特征 -> 靶向稀疏强监督
        # -------------------------------------------------------------------
        if valid_target.any():
            # 移除通道维，用于 CE Loss，维度: (B, D, H, W)
            t_mask_sq = target_mask.squeeze(1).long()                      
            
            if grade_pred_cond is not None:
                # 预测: (B_val, 7, D, H, W) vs 标签: (B_val, D, H, W)
                loss_grade_target = self.ce_loss(grade_pred_cond[valid_target], t_mask_sq[valid_target])
            
            if lesion_pred is not None:
                pred_t = lesion_pred[valid_target]                         # 维度: (B_val, 1, D, H, W)
                mask_t = target_mask[valid_target]                         # 维度: (B_val, 1, D, H, W)
                valid_pixels = mask_t > 0                                  # 维度: (B_val, 1, D, H, W) bool
                
                if valid_pixels.any():
                    # 标签二值化，维度: (B_val, 1, D, H, W)
                    target_lesion_label = (mask_t >= self.csPCa_threshold).float() 
                    
                    # 仅提取有效的像素点进行计算
                    # pred_t[valid_pixels] 的维度: (N_pixels,) 1D张量
                    # target_lesion_label[valid_pixels] 的维度: (N_pixels,) 1D张量
                    loss_lesion_sparse = self.bce_loss(pred_t[valid_pixels], target_lesion_label[valid_pixels])

        # -------------------------------------------------------------------
        # 路线 C: PROMIS / TCIA 特征 -> 系统活检区域弱监督
        # -------------------------------------------------------------------
        if valid_sys.any():
            sys_labels_flat = sys_labels.view(-1).long()                   # 维度: (B * 20,)
            valid_zones = sys_labels_flat > 0                              # 维度: (B * 20,) bool
            
            if sys_grade_preds_cond is not None and valid_zones.any():
                # 展平以便通过 CE Loss 计算
                sys_preds_flat = sys_grade_preds_cond.view(-1, sys_grade_preds_cond.size(-1)) # 维度: (B * 20, 7)
                
                # 预测: (B * 20, 7) vs 标签: (B * 20,)
                loss_grade_sys = self.ce_loss(sys_preds_flat, sys_labels_flat)

            if sys_lesion_preds is not None and valid_zones.any():
                sys_lesion_flat = sys_lesion_preds.view(-1)                # 维度: (B * 20,)
                sys_lesion_label = (sys_labels_flat >= self.csPCa_threshold).float() # 维度: (B * 20,)
                
                # 仅在有效区域上算 Loss
                # sys_lesion_flat[valid_zones] 维度: (N_zones,) 1D张量
                loss_lesion_sys = self.bce_loss(sys_lesion_flat[valid_zones], sys_lesion_label[valid_zones])
                    
        # ===================================================================
        # [步骤 4] 多源权重融合 (Weighted Fusion)
        # ===================================================================
        # 所有 loss_* 都是 () 形状的标量 (Scalar)
        loss_lesion_total = (self.l_w_dense * loss_lesion_dense + 
                             self.l_w_sparse * loss_lesion_sparse + 
                             self.l_w_regional * loss_lesion_sys)

        total_loss = (self.lambda_grade * loss_grade_target + 
                      self.lambda_sys * loss_grade_sys + 
                      self.lambda_lesion * loss_lesion_total + 
                      self.lambda_gland * loss_gland)
                      
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_total, loss_gland