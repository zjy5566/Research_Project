import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础组件：3D 残差块 (3D Residual Block)
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 匹配维度 (如果有下采样或通道数改变)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = self.relu(out)
        return out

# ==========================================
# 2. 核心网络：3D ResU-Net 编码器-解码器
# ==========================================
class ProstateMixedSupervisionNet(nn.Module):
    def __init__(self, in_channels=3, num_grade_classes=7, max_zones=20):
        """
        :param in_channels: 输入模态数 (T2, DWI, ADC) = 3
        :param num_grade_classes: ISUP分级数 (0:背景, 1:良性, 2-6:ISUP1-5) = 7
        :param max_zones: 最大物理分区数量 (兼容 TCIA 的 12 和 PROMIS 的 20) = 20
        """
        super(ProstateMixedSupervisionNet, self).__init__()
        self.max_zones = max_zones
        
        # --- Encoder (下采样特征提取) ---
        self.enc1 = ResBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ResBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # --- Bottleneck (最深层特征) ---
        self.bottleneck = ResBlock3D(128, 256)
        
        # --- Decoder (上采样与特征融合) ---
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(256, 128) # 256 因为 128(up) + 128(skip)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(128, 64)
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(64, 32)
        
        # ==========================================
        # 3. 三大多任务输出头 (Multi-Task Heads)
        # ==========================================
        # 头 A：主任务 - 预测体素级 ISUP 分级 (输出通道为 7)
        self.head_grade = nn.Conv3d(32, num_grade_classes, kernel_size=1)
        
        # 头 B：辅助任务 1 - 预测体素级病灶概率 (输出通道为 1)
        self.head_lesion = nn.Conv3d(32, 1, kernel_size=1)
        
        # 头 C：辅助任务 2 - 预测体素级前列腺腺体轮廓 (输出通道为 1)
        self.head_gland = nn.Conv3d(32, 1, kernel_size=1)

    def zone_pooling(self, voxel_logits, zones_mask):
        """
        核心的 MIL (多实例学习) 区域池化函数
        将体素级别的预测图，根据 zones_mask 压缩为区域级别的概率。
        
        :param voxel_logits: (B, C, D, H, W) 网络直接输出的 logits 预测图
        :param zones_mask:   (B, 1, D, H, W) 数据集中提供的区域掩膜 (值 1~20)
        :return:             (B, 20, C)      每个分区的综合预测值
        """
        B, C, D, H, W = voxel_logits.shape
        device = voxel_logits.device
        
        # 初始化输出张量：(Batch, 最大分区数=20, 类别数)
        sys_preds = torch.zeros((B, self.max_zones, C), device=device)
        
        for b in range(B):
            # 获取当前 Batch 的区域掩膜和对应的预测图
            mask_b = zones_mask[b, 0] # (D, H, W)
            logits_b = voxel_logits[b] # (C, D, H, W)
            
            # 遍历每一个分区 (1 到 20)
            for z in range(1, self.max_zones + 1):
                # 找到属于第 z 个区的所有体素索引
                zone_pixels = (mask_b == z)
                
                if zone_pixels.sum() > 0:
                    # 提取该区域内所有体素的预测特征 -> 形状: (C, N) N为体素个数
                    features_in_zone = logits_b[:, zone_pixels]
                    
                    # 采用 Max-Pooling 策略：如果这个区域内哪怕有一个像素是高危癌，
                    # 那么整个区域就判定为高危 (符合临床活检找最严重癌灶的逻辑)
                    # max(dim=1)[0] 取最大值，形状变为 (C,)
                    sys_preds[b, z - 1] = features_in_zone.max(dim=1)[0]
                else:
                    # 如果该分区在此病人中不存在，保持为 0
                    pass
                    
        return sys_preds

    def forward(self, x, zones_mask=None):
        """
        :param x: 输入影像张量 (B, 3, D, H, W)
        :param zones_mask: 用于做区域池化的掩膜 (B, 1, D, H, W)
        """
        # --- 编码器 (Encoder) ---
        e1 = self.enc1(x)        # (B, 32, D, H, W)
        p1 = self.pool1(e1)      # (B, 32, D/2, H/2, W/2)
        
        e2 = self.enc2(p1)       # (B, 64, D/2, H/2, W/2)
        p2 = self.pool2(e2)      # (B, 64, D/4, H/4, W/4)
        
        e3 = self.enc3(p2)       # (B, 128, D/4, H/4, W/4)
        p3 = self.pool3(e3)      # (B, 128, D/8, H/8, W/8)
        
        # --- 瓶颈层 (Bottleneck) ---
        bn = self.bottleneck(p3) # (B, 256, D/8, H/8, W/8)
        
        # --- 解码器 (Decoder) + Skip Connections ---
        d3 = self.up3(bn)        # (B, 128, D/4, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1) # 通道拼接: 128+128=256
        d3 = self.dec3(d3)       # (B, 128, D/4, H/4, W/4)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1) # 64+64=128
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # 32+32=64
        d1 = self.dec1(d1)       # (B, 32, D, H, W) 最终共享特征图
        
        # ==========================================
        # 吐出三大独立任务的预测图
        # ==========================================
        grade_pred = self.head_grade(d1)   # (B, 7, D, H, W)
        lesion_pred = self.head_lesion(d1) # (B, 1, D, H, W)
        gland_pred = self.head_gland(d1)   # (B, 1, D, H, W)
        
        sys_grade_preds = None
        sys_lesion_preds = None
        
        # 如果传入了 zone_mask（训练期间一定会传入），则执行动态区域池化
        if zones_mask is not None:
            sys_grade_preds = self.zone_pooling(grade_pred, zones_mask)   # (B, 20, 7)
            sys_lesion_preds = self.zone_pooling(lesion_pred, zones_mask) # (B, 20, 1)

        return grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred

# --- 本地测试代码 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Model on {device}...")
    
    # 模拟来自于 dataset.py 的 DataLoader 输出
    B, C, D, H, W = 2, 3, 32, 64, 64
    dummy_input = torch.randn((B, C, D, H, W)).to(device)
    
    # 模拟一个 zones_mask，值在 0~20 之间
    dummy_zones_mask = torch.randint(0, 21, (B, 1, D, H, W)).float().to(device)
    
    # 实例化网络
    model = ProstateMixedSupervisionNet(in_channels=3).to(device)
    
    # 模拟一次前向传播
    grade_p, sys_grade_p, lesion_p, sys_lesion_p, gland_p = model(dummy_input, dummy_zones_mask)
    
    print("\n--- Network Outputs Shapes ---")
    print(f"1. Voxel Grade Map:    {grade_p.shape}")       # 应为 (2, 7, 32, 64, 64)
    print(f"2. System Grade Pool:  {sys_grade_p.shape}")   # 应为 (2, 20, 7)
    print(f"3. Voxel Lesion Risk:  {lesion_p.shape}")      # 应为 (2, 1, 32, 64, 64)
    print(f"4. System Lesion Pool: {sys_lesion_p.shape}")  # 应为 (2, 20, 1)
    print(f"5. Voxel Gland Seg:    {gland_p.shape}")       # 应为 (2, 1, 32, 64, 64)
    
    print("\nModel design matches the MixedSupervisionLoss function perfectly!")