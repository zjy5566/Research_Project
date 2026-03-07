import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
import random

# 假设你的 config.py 与此文件在同一目录
from config import Config

class ProstateUnifiedDataset(Dataset):
    def __init__(self, csv_path, data_root, is_train=True):
        """
        统一的混合监督数据集加载器
        :param csv_path: 划分好的 CSV 索引表路径 (例如 train.csv)
        :param data_root: Unified_Dataset 根目录
        :param is_train: 是否为训练集 (控制是否应用数据增强)
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _apply_augmentations(self, input_tensor, masks_dict):
        """
        简单的 3D 空间数据增强 (随机水平/垂直翻转)
        注意：在医学图像中，上下翻转(Z轴)可能改变解剖结构(Base和Apex倒置)，
        因此我们通常只在横断面(X-Y平面，即轴 2 和 3)做随机翻转。
        """
        if random.random() > 0.5:
            # 水平翻转 (轴 3: Width/X)
            input_tensor = np.flip(input_tensor, axis=3).copy()
            for k in masks_dict:
                if masks_dict[k] is not None:
                    masks_dict[k] = np.flip(masks_dict[k], axis=3).copy()
                    
        if random.random() > 0.5:
            # 垂直翻转 (轴 2: Height/Y)
            input_tensor = np.flip(input_tensor, axis=2).copy()
            for k in masks_dict:
                if masks_dict[k] is not None:
                    masks_dict[k] = np.flip(masks_dict[k], axis=2).copy()
                    
        return input_tensor, masks_dict

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['patient_id']
        p_dir = os.path.join(self.data_root, pid)
        
        # ==========================================
        # 1. 读取基础图像 (所有病人必须有)
        # ==========================================
        # 形状: (3, 32, 64, 64) -> [T2, DWI, ADC]
        input_tensor = np.load(os.path.join(p_dir, 'input_tensor.npy')).astype(np.float32)

        # ==========================================
        # 2. 初始化容器与标志位
        # ==========================================
        # 空间维度
        D, H, W = Config.INPUT_SHAPE[1:] 
        
        # 默认使用全零填充 (Dummy Tensors)，确保 DataLoader 能成功 collate
        target_mask = np.zeros((1, D, H, W), dtype=np.float32)
        zones_mask = np.zeros((1, D, H, W), dtype=np.float32)
        seg_mask = np.zeros((1, D, H, W), dtype=np.float32)
        
        # 为了兼容 TCIA(12区) 和 PROMIS(20区)，我们统一将系统标签 pad 到长度 20
        # Loss 计算时会自动忽略值为 0 的背景/占位符
        sys_labels = np.zeros(20, dtype=np.int64) 
        
        has_target = float(row['has_target'])
        has_sys = float(row['has_sys_12'] or row['has_sys_20'])
        
        # 【修改点 1】: 根据新生成的 CSV，将 has_gland 改为了 has_lesion
        has_seg = float(row['has_lesion'])

        # ==========================================
        # 3. 按需加载具体标签数据
        # ==========================================
        masks_to_aug = {'target': None, 'zones': None, 'seg': None}

        # --- A. 强监督：靶向穿刺 (Target Biopsy) ---
        if has_target:
            t_img = sitk.ReadImage(os.path.join(p_dir, 'target_bx.nii.gz'))
            t_arr = sitk.GetArrayFromImage(t_img).astype(np.float32)
            masks_to_aug['target'] = np.expand_dims(t_arr, axis=0) # (1, D, H, W)

        # --- B. 弱监督：系统活检分区 (Systematic Zones) ---
        if has_sys:
            # 无论12区还是20区，都叫 zones_mask.nii.gz
            z_img = sitk.ReadImage(os.path.join(p_dir, 'zones_mask.nii.gz'))
            z_arr = sitk.GetArrayFromImage(z_img).astype(np.float32)
            masks_to_aug['zones'] = np.expand_dims(z_arr, axis=0)
            
            # 读取对应的分数并填入长度为 20 的数组前部
            if row['has_sys_12']:
                s_labels = np.load(os.path.join(p_dir, 'systematic_labels_12.npy'))
                sys_labels[:12] = s_labels
            elif row['has_sys_20']:
                s_labels = np.load(os.path.join(p_dir, 'systematic_labels_20.npy'))
                sys_labels[:20] = s_labels

        # --- C. 辅助监督：密集病灶分割 (Lesion Segmentation) ---
        if has_seg:
            # 【修改点 2】: 文件名由 gland_mask.npy 改为 lesion_mask.npy
            s_arr = np.load(os.path.join(p_dir, 'lesion_mask.npy')).astype(np.float32)
            masks_to_aug['seg'] = np.expand_dims(s_arr, axis=0)

        # ==========================================
        # 4. 数据增强 (仅在训练时)
        # ==========================================
        if self.is_train and getattr(Config, 'USE_AUGMENTATION', False):
            input_tensor, masks_to_aug = self._apply_augmentations(input_tensor, masks_to_aug)

        # 将增强后的数据取回
        if masks_to_aug['target'] is not None: target_mask = masks_to_aug['target']
        if masks_to_aug['zones'] is not None: zones_mask = masks_to_aug['zones']
        if masks_to_aug['seg'] is not None: seg_mask = masks_to_aug['seg']

        # ==========================================
        # 5. 特殊处理：防止信息泄露 (Rajagopal 等人的消融点)
        # ==========================================
        # 如果一个病人同时做过靶向和系统穿刺，系统盲穿往往有假阴性（没扎准）。
        # 为了防止系统标签的"假阴性"干扰靶向区域真实的"强阳性"，
        # 我们可以选择在计算系统 Loss 时，从 zones_mask 中挖掉 target_mask 所在的区域。
        if getattr(Config, 'MASK_TARGET_IN_SYS', False) and has_target and has_sys:
            # 把 target_mask > 0 的地方的 zones_mask 设为 0（背景）
            zones_mask[target_mask > 0] = 0

        # ==========================================
        # 6. 转为 PyTorch Tensors
        # ==========================================
        return {
            'pid': pid,
            'input': torch.from_numpy(input_tensor),         # (3, 32, 64, 64)
            'target_mask': torch.from_numpy(target_mask),    # (1, 32, 64, 64)
            'zones_mask': torch.from_numpy(zones_mask),      # (1, 32, 64, 64)
            'sys_labels': torch.from_numpy(sys_labels),      # (20,)
            'seg_mask': torch.from_numpy(seg_mask),          # (1, 32, 64, 64)
            
            # 以下标识符用于在 Loss 函数中进行动态路由
            'has_target': torch.tensor(has_target, dtype=torch.float32),
            'has_sys': torch.tensor(has_sys, dtype=torch.float32),
            'has_seg': torch.tensor(has_seg, dtype=torch.float32)
        }

# --- 本地测试代码 ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 模拟测试环境
    print("Testing ProstateUnifiedDataset...")
    test_csv = os.path.join(Config.SPLIT_DIR, 'train.csv')
    
    if os.path.exists(test_csv):
        dataset = ProstateUnifiedDataset(csv_path=test_csv, data_root=Config.UNIFIED_DATA_DIR, is_train=True)
        print(f"Dataset Size: {len(dataset)}")
        
        # 测试 DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        print("\nBatch Shapes:")
        print(f"  Inputs:      {batch['input'].shape}")
        print(f"  Target Mask: {batch['target_mask'].shape}")
        print(f"  Zones Mask:  {batch['zones_mask'].shape}")
        print(f"  Sys Labels:  {batch['sys_labels'].shape}")
        print(f"  Seg Mask:    {batch['seg_mask'].shape}")
        
        print("\nBatch Flags:")
        print(f"  Has Target:  {batch['has_target']}")
        print(f"  Has Sys:     {batch['has_sys']}")
        print(f"  Has Seg:     {batch['has_seg']}")
    else:
        print("CSV index not found. Please run the unified dataset script first.")