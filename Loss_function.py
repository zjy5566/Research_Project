import torch
import torch.nn as nn

class MixedSupervisionLoss(nn.Module):
    def __init__(self, lambda_target=1.0, lambda_sys=0.5, lambda_seg=1.0):
        super(MixedSupervisionLoss, self).__init__()
        self.lambda_target = lambda_target
        self.lambda_sys = lambda_sys
        self.lambda_seg = lambda_seg
        
        # ==========================================
        # 【核心魔法】：ignore_index=0
        # ==========================================
        # 只要 target_mask 的像素值为 0（代表背景、未穿刺区域），
        # CrossEntropyLoss 会直接在底层 C++ 代码中将其跳过，
        # 既不产生 Loss，也不会回传任何梯度，完美实现你的需求！
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        # 辅助病灶分割通常是一个二分类问题（0:无病灶, 1:有病灶）
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, target_pred, sys_preds, seg_pred, 
                target_mask, sys_labels, seg_label_mask, 
                has_target, has_sys, has_seg):
        """
        :param target_pred: (B, C, D, H, W) 靶向区域的体素级分类预测
        :param sys_preds:   (B, 12, C)      系统穿刺 12/20 分区的区域级分类预测
        :param seg_pred:    (B, 1, D, H, W) 辅助病灶分割预测
        :param target_mask: (B, 1, D, H, W) 靶向针道掩膜 (包含了 ISUP 等级 1-6，未穿刺为 0)
        :param sys_labels:  (B, 12)         系统穿刺分区标签 (未穿刺区域为 0)
        :param seg_label_mask: (B, 1, D, H, W) 公开数据集的病灶二值掩膜
        """
        
        # 动态获取 device，防止报错
        device = target_mask.device
        loss_target = torch.tensor(0.0, device=device)
        loss_sys = torch.tensor(0.0, device=device)
        loss_seg = torch.tensor(0.0, device=device)
        
        # ==========================================
        # 1. 靶向穿刺 Loss (仅计算穿刺针道内部的像素)
        # ==========================================
        if target_pred is not None and has_target.sum() > 0:
            # PyTorch 的多维 CE Loss 要求：
            # Input: (B, C, D, H, W)
            # Target: (B, D, H, W) 且数据类型为 long
            t_mask_sq = target_mask.squeeze(1).long()
            
            # 过滤出有靶向穿刺数据的 Batch 索引
            valid_t = has_target > 0
            
            if valid_t.sum() > 0:
                # 只有 valid_t 的样本参与计算
                # 穿刺针道以外的体素 (值为0) 会被 ignore_index=0 自动屏蔽！
                loss_target = self.ce_loss(target_pred[valid_t], t_mask_sq[valid_t])

        # ==========================================
        # 2. 系统穿刺 Loss (区域级别 MIL)
        # ==========================================
        if sys_preds is not None and has_sys.sum() > 0:
            # 将 Batch 和 Zone 维度展平，形状变为 (B * Num_Zones, C)
            sys_preds_flat = sys_preds.view(-1, sys_preds.size(-1))
            # 标签展平为 (B * Num_Zones,)
            sys_labels_flat = sys_labels.view(-1).long()
            
            valid_s = has_sys > 0
            if valid_s.sum() > 0:
                # 同样地，系统活检中没有结果的分区 (标签为 0) 会被自动忽略
                loss_sys = self.ce_loss(sys_preds_flat, sys_labels_flat)

        # ==========================================
        # 3. 辅助病灶分割 Loss
        # ==========================================
        if seg_pred is not None and has_seg.sum() > 0:
            valid_seg = has_seg > 0
            if valid_seg.sum() > 0:
                loss_seg = self.bce_loss(seg_pred[valid_seg], seg_label_mask[valid_seg].float())

        # ==========================================
        # 4. 加权总和
        # ==========================================
        total_loss = (self.lambda_target * loss_target + 
                      self.lambda_sys * loss_sys + 
                      self.lambda_seg * loss_seg)
                      
        return total_loss, loss_target, loss_sys, loss_seg