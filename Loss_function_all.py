import torch
import torch.nn as nn

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


class MixedSupervisionLoss(nn.Module):
    def __init__(self, 
                 lambda_grade=1.0,         # 主任务 (靶向 ISUP 分级)
                 lambda_sys=0.5,           # 弱监督 (系统区域池化)
                 lambda_lesion=1.0,        # 辅任务A (寻找病灶整体权重)
                 lambda_gland=0.2,         # 辅任务B (腺体轮廓)
                 lesion_w_dense=1.0,       # 辅任务A内部：密集强监督 (PUB)
                 lesion_w_sparse=1.0,      # 辅任务A内部：稀疏强监督 (靶向)
                 lesion_w_regional=0.2,    # 辅任务A内部：区域弱监督 (系统)
                 csPCa_threshold=3,        # ISUP >= 2 视为有高危病灶 (如果你的标签是0-6, 这里可能要是2, 请根据实际情况确认)
                 lesion_pos_weight=15.0):  # [新增] 极小病灶的正样本惩罚权重
        super(MixedSupervisionLoss, self).__init__()
        
        self.lambda_grade = lambda_grade
        self.lambda_sys = lambda_sys
        self.lambda_lesion = lambda_lesion
        self.lambda_gland = lambda_gland
        
        self.l_w_dense = lesion_w_dense
        self.l_w_sparse = lesion_w_sparse
        self.l_w_regional = lesion_w_regional
        
        self.csPCa_threshold = csPCa_threshold
        
        # 损失函数组件
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        # ===================================================================
        # [修改点 1]：分离并加权 BCE Loss
        # 病灶极小，必须强惩罚漏诊；腺体很大，使用普通 BCE 即可。
        # ===================================================================
        self.lesion_pos_weight = torch.tensor([lesion_pos_weight])
        self.lesion_bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.lesion_pos_weight)
        
        self.gland_bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_loss = DiceLoss()

    def forward(self, 
                grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred, 
                target_mask, sys_labels, lesion_mask, gland_mask, 
                has_target, has_sys, has_lesion, has_gland):
        
        device = target_mask.device
        
        # ===================================================================
        # [修改点 2]：确保动态创建的权紧张量在正确的 GPU 设备上
        # ===================================================================
        if self.lesion_bce_loss.pos_weight.device != device:
            self.lesion_bce_loss.pos_weight = self.lesion_bce_loss.pos_weight.to(device)
            
        # ===================================================================
        # [步骤 1] 预先分配 Loss 容器，防止因当前 Batch 缺失某类数据导致报错
        # ===================================================================
        loss_grade_target = torch.tensor(0.0, device=device)
        loss_grade_sys = torch.tensor(0.0, device=device)
        loss_lesion_dense = torch.tensor(0.0, device=device)
        loss_lesion_sparse = torch.tensor(0.0, device=device)
        loss_lesion_sys = torch.tensor(0.0, device=device)
        loss_gland = torch.tensor(0.0, device=device)

        # ===================================================================
        # [步骤 2] 数据集路由判别 (Dataset Routing) - 核心优化点
        # 将 float 标识符转化为 boolean masks，明确当前 Batch 中包含哪些数据源
        # ===================================================================
        valid_lesion = has_lesion > 0  # True 代表属于 PUB 数据集 (拥有密集病灶标注)
        valid_target = has_target > 0  # True 代表属于 TCIA 数据集 (拥有靶向针道标注)
        valid_sys = has_sys > 0        # True 代表属于 PROMIS / TCIA (拥有系统活检区域标注)
        valid_gland = has_gland > 0    # True 代表拥有腺体解剖标注的数据集

        # ===================================================================
        # [步骤 3] 按数据集类型分发 Loss 计算 (Dataset-Centric Computation)
        # ===================================================================

        # -------------------------------------------------------------------
        # 路线 A: PUB 数据集特征 -> 密集强监督 (Dense Supervision)
        # 作用: 教会网络什么是肿瘤和腺体的完整 3D 形状与边缘。
        # -------------------------------------------------------------------
        # A.1 病灶密集分割 (Lesion)
        if lesion_pred is not None and valid_lesion.any():
            l_pred_valid = lesion_pred[valid_lesion]
            l_mask_valid = lesion_mask[valid_lesion].float()
            
            # 【修改点 3】：使用带强烈正样本惩罚的 lesion_bce_loss
            loss_lesion_dense = self.lesion_bce_loss(l_pred_valid, l_mask_valid) + \
                                self.dice_loss(l_pred_valid, l_mask_valid)

        # A.2 腺体密集分割 (Gland)
        if gland_pred is not None and valid_gland.any():
            g_pred_valid = gland_pred[valid_gland]
            g_mask_valid = gland_mask[valid_gland].float()
            
            # 【修改点 4】：腺体使用普通的 gland_bce_loss
            loss_gland = self.gland_bce_loss(g_pred_valid, g_mask_valid) + \
                         self.dice_loss(g_pred_valid, g_mask_valid)

        # -------------------------------------------------------------------
        # 路线 B: TCIA 数据集特征 -> 靶向稀疏强监督 (Sparse Supervision)
        # 作用: 仅在“确定扎出癌”的几根细线轨迹上，提供极高可信度的局部锚点，防止假阳性。
        # -------------------------------------------------------------------
        if valid_target.any():
            # 获取靶向掩膜，并将通道维度 squeeze 掉，用于 CE Loss
            t_mask_sq = target_mask.squeeze(1).long()
            
            # B.1 靶向 ISUP 分级 (主任务)
            if grade_pred is not None:
                # 巧妙利用 ignore_index=0，针道外的空白区域自动不产生任何梯度
                loss_grade_target = self.ce_loss(grade_pred[valid_target], t_mask_sq[valid_target])
            
            # B.2 靶向病灶发掘 (辅任务A)
            if lesion_pred is not None:
                pred_t = lesion_pred[valid_target]
                mask_t = target_mask[valid_target]
                
                # 仅筛选出针道经过的像素点 (>0)
                valid_pixels = mask_t > 0
                if valid_pixels.any():
                    # 动态二值化：将针道里的 ISUP 转化为是否有病灶 (0/1)
                    target_lesion_label = (mask_t >= self.csPCa_threshold).float()
                    # 【修改点 5】：仅在针道像素上计算带有强正样本惩罚的 BCE Loss
                    loss_lesion_sparse = self.lesion_bce_loss(pred_t[valid_pixels], target_lesion_label[valid_pixels])

        # -------------------------------------------------------------------
        # 路线 C: PROMIS / TCIA 特征 -> 系统活检区域弱监督 (Regional Weak Supervision)
        # 作用: 提供宏观约束 ("左叶有事，右叶没事")，但因为存在假阴性，整体作降权处理。
        # -------------------------------------------------------------------
        if valid_sys.any():
            # 展平区域预测 (B, 20, C) -> (B*20, C)
            sys_labels_flat = sys_labels.view(-1).long()
            valid_zones = sys_labels_flat > 0 # 剔除未穿刺的背景区(0)
            
            # C.1 区域 ISUP 分级 (主任务)
            if sys_grade_preds is not None and valid_zones.any():
                sys_preds_flat = sys_grade_preds.view(-1, sys_grade_preds.size(-1))
                loss_grade_sys = self.ce_loss(sys_preds_flat, sys_labels_flat)

            # C.2 区域病灶存在概率 (辅任务A)
            if sys_lesion_preds is not None and valid_zones.any():
                sys_lesion_flat = sys_lesion_preds.view(-1)
                # 动态生成区域二值标签
                sys_lesion_label = (sys_labels_flat >= self.csPCa_threshold).float()
                # 【修改点 6】：仅在实际穿刺了的区域上算带有强正样本惩罚的 BCE Loss
                loss_lesion_sys = self.lesion_bce_loss(sys_lesion_flat[valid_zones], sys_lesion_label[valid_zones])
                    
        # ===================================================================
        # [步骤 4] 多源权重融合 (Weighted Fusion)
        # ===================================================================
        # 融合寻找病灶(Lesion)的三路信号：形状(PUB) + 确信点(TCIA靶向) + 宏观面(PROMIS系统)
        loss_lesion_total = (self.l_w_dense * loss_lesion_dense + 
                             self.l_w_sparse * loss_lesion_sparse + 
                             self.l_w_regional * loss_lesion_sys)

        # 结合四大目标计算最终的总 Loss
        total_loss = (self.lambda_grade * loss_grade_target + 
                      self.lambda_sys * loss_grade_sys + 
                      self.lambda_lesion * loss_lesion_total + 
                      self.lambda_gland * loss_gland)
                      
        return total_loss, loss_grade_target, loss_grade_sys, loss_lesion_total, loss_gland