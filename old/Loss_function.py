# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         """标准的二分类 Dice Loss"""
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)                 
#         probs = probs.view(probs.size(0), -1)         
#         targets = targets.view(targets.size(0), -1)   
        
#         intersection = (probs * targets).sum(dim=1)   
#         union = probs.sum(dim=1) + targets.sum(dim=1) 
        
#         dice = (2. * intersection + self.smooth) / (union + self.smooth) 
#         return 1.0 - dice.mean()                      

# # ===================================================================
# # [核心组件]：Focal Loss (针对微小目标和极度不平衡的 Hard 样本挖掘)
# # ===================================================================
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
#         probs = torch.sigmoid(logits)
        
#         p_t = probs * targets + (1 - probs) * (1 - targets)
#         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
#         focal_weight = alpha_t * (1 - p_t) ** self.gamma
#         loss = focal_weight * bce_loss
        
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


# class MixedSupervisionLoss(nn.Module):
#     def __init__(self, 
#                  lambda_grade=1.0,         # 主任务 (分级总体权重)
#                  lambda_lesion=1.0,        # 辅任务A (寻找病灶总体权重)
#                  lambda_gland=0.2,         # 辅任务B (腺体轮廓总体权重)
#                  grade_w_tbx=1.0,          # 分级内部：靶向强监督权重 (TCIA)
#                  grade_w_sbx=0.5,          # 分级内部：系统弱监督权重 (PROMIS/TCIA)
#                  lesion_w_dense=1.0,       # 病灶内部：密集强监督 (PUB)
#                  lesion_w_sparse=1.0,      # 病灶内部：稀疏强监督 (靶向)
#                  lesion_w_regional=1.0,    # 病灶内部：系统区域弱监督 (PROMIS)
#                  csPCa_threshold=3,        # 临床显著性前列腺癌的 ISUP 阈值
#                  pos_weight_val=2.0):      
#         super(MixedSupervisionLoss, self).__init__()
        
#         # 总权重
#         self.lambda_grade = lambda_grade
#         self.lambda_lesion = lambda_lesion
#         self.lambda_gland = lambda_gland
        
#         # Grade 内部子权重
#         self.g_w_tbx = grade_w_tbx
#         self.g_w_sbx = grade_w_sbx
        
#         # Lesion 内部子权重 (动态课程学习会在 train.py 修改这些)
#         self.l_w_dense = lesion_w_dense
#         self.l_w_sparse = lesion_w_sparse
#         self.l_w_regional = lesion_w_regional
        
#         self.csPCa_threshold = csPCa_threshold
        
#         # --- Loss 实例化 ---
#         self.pos_weight = torch.tensor([pos_weight_val])
#         self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) # 处理极度不平衡的lesion标签
#         self.lesion_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
#         self.gland_bce_loss = nn.BCEWithLogitsLoss()
#         self.dice_loss = DiceLoss()
        
#         # 多分类交叉熵，用于 ISUP 级别预测 (忽略值为 -1 的非合法区域)
#         # 类别 0 (健康) 权重极低，类别 1 (良性) 略高，ISUP 1-5 (类别 2-6) 权重极高
#         # 注意：你需要确保这个 tensor 放在正确的 device 上 (可以在 forward 里 to(device))
#         self.class_weights = torch.tensor([0.1, 0.5, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32)
        
#         # 传入 weight 参数
#         self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)


#     def forward(self, grade_preds, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
#                 target_mask, sys_labels, lesion_mask, gland_mask, 
#                 has_target, has_sys, has_lesion, has_gland):
        
#         device = lesion_pred.device if lesion_pred is not None else grade_preds.device
        
#         if self.lesion_bce_loss.pos_weight.device != device:
#             self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)

#         # 1. 初始化所有 Loss
#         loss_grade_target = torch.tensor(0.0, device=device)
#         loss_grade_sys = torch.tensor(0.0, device=device)
        
#         loss_lesion_dense = torch.tensor(0.0, device=device)
#         loss_lesion_sparse = torch.tensor(0.0, device=device)
#         loss_lesion_sys = torch.tensor(0.0, device=device)
#         loss_gland = torch.tensor(0.0, device=device)
        
#         valid_target = has_target > 0
#         valid_sys = has_sys > 0
#         valid_lesion = has_lesion > 0
#         valid_gland = has_gland > 0

#         # ===================================================================
#         # [任务 1]：Grade ISUP 分级任务 (多分类)
#         # ===================================================================
#         # 1A. 靶向针道密集分类强监督 (TBx - TCIA)
#         # ===================================================================
#         if grade_preds is not None and valid_target.any():
#             g_p_t = grade_preds[valid_target]       
#             mask_t = target_mask[valid_target]      
            
#             # 【修复点】：去掉 mask_t 多余的 Channel=1 维度
#             # 把它从 [B, 1, D, H, W] 变成 [B, D, H, W]
#             if mask_t.dim() == 5 and mask_t.shape[1] == 1:
#                 mask_t = mask_t.squeeze(1)
                
#             valid_pixels = mask_t > 0
#             if valid_pixels.any():
#                 g_p_t_permuted = g_p_t.permute(0, 2, 3, 4, 1) # [B, D, H, W, 7]
                
#                 # 现在 valid_pixels 是 [B, D, H, W]，刚好索引 permuted 的前4个维度
#                 preds_valid_grade = g_p_t_permuted[valid_pixels] # 结果形状: [N, 7]
#                 labels_valid_grade = mask_t[valid_pixels].long() # 结果形状: [N]
                
#                 loss_grade_target = self.ce_loss(preds_valid_grade, labels_valid_grade)

#         # 1B. 系统分区域池化弱监督分类 (SBx - PROMIS/TCIA)
#         if sys_grade_preds is not None and valid_sys.any():
#             s_g_p = sys_grade_preds[valid_sys]      
#             s_labels = sys_labels[valid_sys]        
#             s_g_p_flat = s_g_p.view(-1, s_g_p.size(-1)) 
#             s_labels_flat = s_labels.view(-1)           
#             valid_zones = s_labels_flat >= 0
#             if valid_zones.any():
#                 loss_grade_sys = self.ce_loss(s_g_p_flat[valid_zones], s_labels_flat[valid_zones].long())

#         # ===================================================================
#         # [任务 2]：Lesion 临床显著性病灶检测 (二分类)
#         # ===================================================================
#         if lesion_pred is not None and valid_lesion.any():
#             pred_l = lesion_pred[valid_lesion]
#             mask_l = lesion_mask[valid_lesion].float()
#             loss_lesion_dense = self.lesion_bce_loss(pred_l, mask_l) + self.lesion_focal_loss(pred_l, mask_l)

#         if lesion_pred is not None and valid_target.any():
#             pred_t = lesion_pred[valid_target]
#             mask_t = target_mask[valid_target]
#             valid_pixels = mask_t > 0
#             if valid_pixels.any():
#                 target_lesion_label = (mask_t >= self.csPCa_threshold).float()
#                 pred_valid = pred_t[valid_pixels]
#                 label_valid = target_lesion_label[valid_pixels]
#                 loss_lesion_sparse = self.lesion_bce_loss(pred_valid, label_valid) + self.lesion_focal_loss(pred_valid, label_valid)

#         if sys_lesion_preds is not None and valid_sys.any():
#             s_l_p = sys_lesion_preds[valid_sys]
#             s_labels = sys_labels[valid_sys]
#             s_l_p_flat = s_l_p.view(-1)
#             s_labels_flat = s_labels.view(-1)
#             valid_zones = s_labels_flat >= 0
#             if valid_zones.any():
#                 sys_lesion_label = (s_labels_flat >= self.csPCa_threshold).float()
#                 pred_valid = s_l_p_flat[valid_zones]
#                 label_valid = sys_lesion_label[valid_zones]
#                 loss_lesion_sys = self.lesion_bce_loss(pred_valid, label_valid) + self.lesion_focal_loss(pred_valid, label_valid)

#         # ===================================================================
#         # [任务 3]：Gland 腺体密集强监督
#         # ===================================================================
#         if gland_pred is not None and valid_gland.any():
#             g_pred_valid = gland_pred[valid_gland]
#             g_mask_valid = gland_mask[valid_gland].float()
#             loss_gland = self.gland_bce_loss(g_pred_valid, g_mask_valid) + self.dice_loss(g_pred_valid, g_mask_valid)

#         # ===================================================================
#         # 汇总返回 (包含全新的 Grade 子类拆分)
#         # ===================================================================
#         loss_grade_total = (self.g_w_tbx * loss_grade_target) + (self.g_w_sbx * loss_grade_sys)
        
#         loss_lesion_total = (self.l_w_dense * loss_lesion_dense + 
#                              self.l_w_sparse * loss_lesion_sparse + 
#                              self.l_w_regional * loss_lesion_sys)

#         total_loss = (self.lambda_grade * loss_grade_total) + \
#                      (self.lambda_lesion * loss_lesion_total) + \
#                      (self.lambda_gland * loss_gland)
                      
#         # 返回 9 个变量 (新增了 loss_grade_total 及其子项)
#         return total_loss, loss_grade_total, loss_grade_target, loss_grade_sys, loss_lesion_total, loss_lesion_dense, loss_lesion_sparse, loss_lesion_sys, loss_gland

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
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
    def __init__(self, csPCa_threshold=3, pos_weight_val=2.0):      
        super(MixedSupervisionLoss, self).__init__()
        
        self.csPCa_threshold = csPCa_threshold
        
        # ===================================================================
        # 【核心 EM 机制】：将任务权重注册为可学习的网络参数！
        # 初始值设为 0 (内部用 exp 运算，对应初始权重系数为 exp(0) = 1.0)
        # ===================================================================
        self.log_vars = nn.ParameterDict({
            'grade_tbx': nn.Parameter(torch.zeros(1)),
            'grade_sbx': nn.Parameter(torch.zeros(1)),
            'lesion_dense': nn.Parameter(torch.zeros(1)),
            'lesion_sparse': nn.Parameter(torch.zeros(1)),
            'lesion_sys': nn.Parameter(torch.zeros(1)),
            'gland': nn.Parameter(torch.zeros(1))
        })
        
        self.pos_weight = torch.tensor([pos_weight_val])
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.lesion_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        # 类别惩罚权重 (极度重要：维持分类任务不摆烂)
        self.class_weights = torch.tensor([0.1, 0.5, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)

    def forward(self, grade_preds, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = lesion_pred.device if lesion_pred is not None else grade_preds.device
        if self.lesion_bce_loss.pos_weight.device != device:
            self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)
        if self.ce_loss.weight.device != device:
            self.ce_loss.weight = self.ce_loss.weight.to(device)

        loss_grade_target = torch.tensor(0.0, device=device)
        loss_grade_sys = torch.tensor(0.0, device=device)
        loss_lesion_dense = torch.tensor(0.0, device=device)
        loss_lesion_sparse = torch.tensor(0.0, device=device)
        loss_lesion_sys = torch.tensor(0.0, device=device)
        loss_gland = torch.tensor(0.0, device=device)
        
        valid_target = has_target > 0
        valid_sys = has_sys > 0
        valid_lesion = has_lesion > 0
        valid_gland = has_gland > 0

        # [任务 1]：Grade ISUP
        if grade_preds is not None and valid_target.any():
            g_p_t = grade_preds[valid_target]       
            mask_t = target_mask[valid_target]      
            if mask_t.dim() == 5 and mask_t.shape[1] == 1:
                mask_t = mask_t.squeeze(1)
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                g_p_t_permuted = g_p_t.permute(0, 2, 3, 4, 1)
                preds_valid_grade = g_p_t_permuted[valid_pixels]
                labels_valid_grade = mask_t[valid_pixels].long()
                loss_grade_target = self.ce_loss(preds_valid_grade, labels_valid_grade)

        if sys_grade_preds is not None and valid_sys.any():
            s_g_p = sys_grade_preds[valid_sys]      
            s_labels = sys_labels[valid_sys]        
            s_g_p_flat = s_g_p.view(-1, s_g_p.size(-1)) 
            s_labels_flat = s_labels.view(-1)           
            valid_zones = s_labels_flat >= 0
            if valid_zones.any():
                loss_grade_sys = self.ce_loss(s_g_p_flat[valid_zones], s_labels_flat[valid_zones].long())

        # [任务 2]：Lesion
        if lesion_pred is not None and valid_lesion.any():
            pred_l = lesion_pred[valid_lesion]
            mask_l = lesion_mask[valid_lesion].float()
            loss_lesion_dense = self.lesion_bce_loss(pred_l, mask_l) + self.lesion_focal_loss(pred_l, mask_l)

        if lesion_pred is not None and valid_target.any():
            pred_t = lesion_pred[valid_target]
            mask_t = target_mask[valid_target]
            valid_pixels = mask_t > 0
            if valid_pixels.any():
                target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                pred_valid = pred_t[valid_pixels]
                label_valid = target_lesion_label[valid_pixels]
                loss_lesion_sparse = self.lesion_bce_loss(pred_valid, label_valid) + self.lesion_focal_loss(pred_valid, label_valid)

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
                loss_lesion_sys = self.lesion_bce_loss(pred_valid, label_valid) + self.lesion_focal_loss(pred_valid, label_valid)

        # [任务 3]：Gland
        if gland_pred is not None and valid_gland.any():
            g_pred_valid = gland_pred[valid_gland]
            g_mask_valid = gland_mask[valid_gland].float()
            loss_gland = self.gland_bce_loss(g_pred_valid, g_mask_valid) + self.dice_loss(g_pred_valid, g_mask_valid)

        # ===================================================================
        # 【EM 动态自适应加权计算】
        # 网络会自动拉低高方差(难学)任务的 exp(-log_var)，避免噪声梯度破坏模型
        # ===================================================================
        w_grade_tbx = loss_grade_target * torch.exp(-self.log_vars['grade_tbx']) + self.log_vars['grade_tbx']
        w_grade_sbx = loss_grade_sys * torch.exp(-self.log_vars['grade_sbx']) + self.log_vars['grade_sbx']
        
        w_les_dense = loss_lesion_dense * torch.exp(-self.log_vars['lesion_dense']) + self.log_vars['lesion_dense']
        w_les_sparse = loss_lesion_sparse * torch.exp(-self.log_vars['lesion_sparse']) + self.log_vars['lesion_sparse']
        w_les_sys = loss_lesion_sys * torch.exp(-self.log_vars['lesion_sys']) + self.log_vars['lesion_sys']
        
        w_gland = loss_gland * torch.exp(-self.log_vars['gland']) + self.log_vars['gland']

        total_loss = w_grade_tbx + w_grade_sbx + w_les_dense + w_les_sparse + w_les_sys + w_gland
        
        # 提取当前网络自己学到的“真实权重乘子”
        em_weights = {k: torch.exp(-v).item() for k, v in self.log_vars.items()}
                      
        return total_loss, loss_grade_target + loss_grade_sys, loss_grade_target, loss_grade_sys, \
               loss_lesion_dense + loss_lesion_sparse + loss_lesion_sys, loss_lesion_dense, loss_lesion_sparse, loss_lesion_sys, loss_gland, em_weights