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
        # 【替换】BatchNorm3d -> GroupNorm
        self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 【替换】BatchNorm3d -> GroupNorm
        self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # 匹配维度 (如果有下采样或通道数改变)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # 【替换】BatchNorm3d -> GroupNorm
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
        
        # [新增] 第四层 Encoder
        self.enc4 = ResBlock3D(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # --- Bottleneck (最深层特征 - 感受野最大) ---
        # 此时的特征图尺寸为 (D/16, H/16, W/16) -> (2, 4, 4)
        self.bottleneck = ResBlock3D(256, 512)
        
        # --- Decoder (上采样与特征融合 - 增加了一层深度) ---
        # [新增] 第四层 Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(512, 256) # 512 因为 256(up) + 256(skip connection enc4)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(256, 128) # 256 因为 128(up) + 128(skip connection enc3)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(128, 64)  # 128 因为 64(up) + 64(skip connection enc2)
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(64, 32)   # 64 因为 32(up) + 32(skip connection enc1)
        
        # ==========================================
        # 3. 三大多任务输出头 (Multi-Task Heads)
        # ==========================================
        self.head_grade = nn.Conv3d(32, num_grade_classes, kernel_size=1)
        self.head_lesion = nn.Conv3d(32, 1, kernel_size=1)
        self.head_gland = nn.Conv3d(32, 1, kernel_size=1)

    def zone_pooling(self, voxel_logits, zones_mask):
        B, C, D, H, W = voxel_logits.shape
        device = voxel_logits.device
        
        sys_preds = torch.zeros((B, self.max_zones, C), device=device)
        
        for b in range(B):
            mask_b = zones_mask[b, 0] # (D, H, W)
            logits_b = voxel_logits[b] # (C, D, H, W)
            
            for z in range(1, self.max_zones + 1):
                zone_pixels = (mask_b == z)
                
                if zone_pixels.sum() > 0:
                    features_in_zone = logits_b[:, zone_pixels]
                    sys_preds[b, z - 1] = features_in_zone.max(dim=1)[0]
                    
        return sys_preds

    def forward(self, x, zones_mask=None):
        # --- 编码器 (Encoder) ---
        e1 = self.enc1(x)        # (B, 32, 32, 64, 64)
        p1 = self.pool1(e1)      # (B, 32, 16, 32, 32)
        
        e2 = self.enc2(p1)       # (B, 64, 16, 32, 32)
        p2 = self.pool2(e2)      # (B, 64, 8, 16, 16)
        
        e3 = self.enc3(p2)       # (B, 128, 8, 16, 16)
        p3 = self.pool3(e3)      # (B, 128, 4, 8, 8)
        
        # [新增] 第四层下采样
        e4 = self.enc4(p3)       # (B, 256, 4, 8, 8)
        p4 = self.pool4(e4)      # (B, 256, 2, 4, 4)
        
        # --- 瓶颈层 (Bottleneck) ---
        bn = self.bottleneck(p4) # (B, 512, 2, 4, 4)
        
        # --- 解码器 (Decoder) + Skip Connections ---
        # [新增] 第四层上采样
        d4 = self.up4(bn)        # (B, 256, 4, 8, 8)
        d4 = torch.cat([d4, e4], dim=1) # 256 + 256 = 512
        d4 = self.dec4(d4)       # (B, 256, 4, 8, 8)
        
        d3 = self.up3(d4)        # (B, 128, 8, 16, 16)
        d3 = torch.cat([d3, e3], dim=1) # 128 + 128 = 256
        d3 = self.dec3(d3)       # (B, 128, 8, 16, 16)
        
        d2 = self.up2(d3)        # (B, 64, 16, 32, 32)
        d2 = torch.cat([d2, e2], dim=1) # 64 + 64 = 128
        d2 = self.dec2(d2)       # (B, 64, 16, 32, 32)
        
        d1 = self.up1(d2)        # (B, 32, 32, 64, 64)
        d1 = torch.cat([d1, e1], dim=1) # 32 + 32 = 64
        d1 = self.dec1(d1)       # (B, 32, 32, 64, 64)
        
        # --- 输出头 ---
        grade_pred = self.head_grade(d1)   
        lesion_pred = self.head_lesion(d1) 
        gland_pred = self.head_gland(d1)   
        
        sys_grade_preds = None
        sys_lesion_preds = None
        
        if zones_mask is not None:
            sys_grade_preds = self.zone_pooling(grade_pred, zones_mask)   
            sys_lesion_preds = self.zone_pooling(lesion_pred, zones_mask) 

        return grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred

# --- 本地测试代码 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Deeper Model with GroupNorm on {device}...")
    
    B, C, D, H, W = 4, 3, 32, 64, 64
    dummy_input = torch.randn((B, C, D, H, W)).to(device)
    dummy_zones_mask = torch.randint(0, 21, (B, 1, D, H, W)).float().to(device)
    
    model = ProstateMixedSupervisionNet(in_channels=3).to(device)
    
    grade_p, sys_grade_p, lesion_p, sys_lesion_p, gland_p = model(dummy_input, dummy_zones_mask)
    
    print("\n--- Deeper Network Outputs Shapes ---")
    print(f"1. Voxel Grade Map:    {grade_p.shape}") 
    print(f"2. System Grade Pool:  {sys_grade_p.shape}")
    print(f"3. Voxel Lesion Risk:  {lesion_p.shape}")
    print(f"4. System Lesion Pool: {sys_lesion_p.shape}")
    print(f"5. Voxel Gland Seg:    {gland_p.shape}")