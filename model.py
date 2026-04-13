import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 基础组件：3D 残差块 (3D Residual Block)
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        
        # 组数 (num_groups) 通常设为 8 或 16。必须保证 out_channels 能被 num_groups 整除。
        # 我们的通道数 [32, 64, 128, 256, 512] 全都可以被 8 整除。
        num_groups = 8
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # 匹配维度 (如果有下采样或通道数改变)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
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
# 2. 核心网络：3D ResU-Net (加深版 Deeper)
# ==========================================
class ProstateMixedSupervisionNet(nn.Module):
    def __init__(self, in_channels=3, num_grade_classes=7, max_zones=20):
        """
        :param in_channels: 输入模态数 (T2, DWI, ADC) = 3
        :param num_grade_classes: ISUP分级数 = 7
        :param max_zones: 最大物理分区数量 = 20
        """
        super(ProstateMixedSupervisionNet, self).__init__()
        self.max_zones = max_zones
        
        # --- Encoder (下采样特征提取 - 增加了一层深度) ---
        self.enc1 = ResBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ResBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = ResBlock3D(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # --- Bottleneck (最深层特征 - 感受野最大) ---
        self.bottleneck = ResBlock3D(256, 512)
        
        # --- Decoder (上采样与特征融合 - 增加了一层深度) ---
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(512, 256) 
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(256, 128) 
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(128, 64)  
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(64, 32)   
        
        # ==========================================
        # 3. 三大多任务输出头 (Multi-Task Heads)
        # ==========================================
        self.head_grade = nn.Conv3d(32, num_grade_classes, kernel_size=1)
        self.head_lesion = nn.Conv3d(32, 1, kernel_size=1)
        self.head_gland = nn.Conv3d(32, 1, kernel_size=1)

    # 【修改】：增加了 mode 参数，用来区分不同的池化策略
    def zone_pooling(self, voxel_logits, zones_mask, mode='lme'):
        B, C, D, H, W = voxel_logits.shape
        device = voxel_logits.device
        
        sys_preds = torch.zeros((B, self.max_zones, C), device=device)
        
        for b in range(B):
            mask_b = zones_mask[b, 0] # (D, H, W)
            logits_b = voxel_logits[b] # (C, D, H, W)
            
            for z in range(1, self.max_zones + 1):
                zone_pixels = (mask_b == z)
                if zone_pixels.sum() > 0:
                    features_in_zone = logits_b[:, zone_pixels] # 形状: (C, N)
                    
                    if mode == 'lme':
                        # LME (Log-Mean-Exp) 平滑池化，适用于 Lesion (二分类) 找异常高分
                        r = 8.0  
                        N = features_in_zone.shape[1]
                        lse = torch.logsumexp(features_in_zone * r, dim=1)
                        pooled = (lse - torch.log(torch.tensor(N, dtype=torch.float32, device=device))) / r
                    else:
                        # Mean Pooling 平均池化，适用于 Grade (多分类) 维持 Logits 相对分布
                        pooled = features_in_zone.mean(dim=1)
                        
                    sys_preds[b, z - 1] = pooled
                    
        return sys_preds

    def forward(self, x, zones_mask=None):
        # --- 编码器 (Encoder) ---
        e1 = self.enc1(x)       
        p1 = self.pool1(e1)     
        
        e2 = self.enc2(p1)      
        p2 = self.pool2(e2)     
        
        e3 = self.enc3(p2)      
        p3 = self.pool3(e3)     
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # --- 瓶颈层 (Bottleneck) ---
        b = self.bottleneck(p4) 
        
        # --- 解码器 (Decoder) ---
        d4 = self.up4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)        
        d3 = torch.cat((e3, d3), dim=1) 
        d3 = self.dec3(d3)       
        
        d2 = self.up2(d3)        
        d2 = torch.cat((e2, d2), dim=1) 
        d2 = self.dec2(d2)       
        
        d1 = self.up1(d2)        
        d1 = torch.cat((e1, d1), dim=1) 
        d1 = self.dec1(d1)       
        
        # --- 输出头 ---
        grade_pred = self.head_grade(d1)   
        lesion_pred = self.head_lesion(d1) 
        gland_pred = self.head_gland(d1)   
        
        sys_grade_preds = None
        sys_lesion_preds = None
        
        if zones_mask is not None:
            # 【核心修改】：Grade 用 mean，Lesion 用 lme
            sys_grade_preds = self.zone_pooling(grade_pred, zones_mask, mode='mean')   
            sys_lesion_preds = self.zone_pooling(lesion_pred, zones_mask, mode='lme') 

        return grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred