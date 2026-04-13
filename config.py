import os
import torch
from datetime import datetime
import random
import numpy as np

class Config:
    # ==========================================
    # 1. 路径配置 (Path Configurations)
    # ==========================================
    # 根目录
    BASE_DIR = r"/raid/candi/jiayi/RP"
    # BASE_DIR = r"F:\RP_dataset"
    
    # 统一数据集目录 (训练直接从这里读取)
    UNIFIED_DATA_DIR = os.path.join(BASE_DIR, 'data',"Unified_Dataset")
    SPLIT_DIR = os.path.join(UNIFIED_DATA_DIR, "splits")
    
    # 划分好的 CSV 索引表路径
    TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
    VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
    TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")
    
    # 实验结果保存路径 (权重、日志、TensorBoard等)
    EXP_DIR = os.path.join(BASE_DIR, "Experiments")

    # ==========================================
    # 2. 预处理与数据配置 (Data & Preprocessing)
    # ==========================================
    # 空间分辨率对齐 [X, Y, Z] (SimpleITK格式)
    TARGET_SPACING = [1.0, 1.0, 2.24]
    # 裁剪大小 [X, Y, Z] (SimpleITK格式)
    CROP_SIZE_SITK = [64, 64, 32] 
    # 模型输入张量维度 [Channels, Depth, Height, Width]
    INPUT_SHAPE = (3, 32, 64, 64) 
    
    # 穿刺针道膨胀半径 (像素)
    NEEDLE_RADIUS = 2

    # ==========================================
    # 3. 模型网络配置 (Model Architectures)
    # ==========================================
    IN_CHANNELS = 3    # T2, DWI, ADC
    NUM_CLASSES = 7    # 0:Background, 1:Benign, 2:ISUP1, 3:ISUP2, 4:ISUP3, 5:ISUP4, 6:ISUP5
    
    # 定义“临床显著性前列腺癌 (csPCa)”的阈值 (用于 Lesion 二分类)
    # 3 代表 ISUP 2 (即 Gleason 3+4=7 及以上视为有高危病灶)
    CSPC_THRESHOLD = 3

    # ==========================================
    # 4. 训练超参数 (Training Hyperparameters)
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42          # 全局随机种子，确保实验可复现
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 4
    NUM_WORKERS = 2    # DataLoader的线程数
    
    # 优化器参数
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 学习率衰减策略
    LR_SCHEDULER = "CosineAnnealing" # 可选: "StepLR", "CosineAnnealing"

    EARLY_STOP_PATIENCE = 300  # 如果连续 10 个 epoch 性能没有提升，则停止训练

   # ==========================================
    # Loss 权重配置
    # ==========================================
    # 1. 宏观三大任务总权重
    LAMBDA_GRADE = 1.0         # ISUP分级总任务权重
    LAMBDA_LESION = 1.0        # 病灶检测总任务权重
    LAMBDA_GLAND = 0.2         # 腺体分割总任务权重
    
    # 2. Grade 子任务权重细分 (新增)
    GRADE_W_TBX = 1.0          # 靶向穿刺强监督 (TCIA)
    GRADE_W_SBX = 0.5          # 系统穿刺弱监督 (PROMIS)
    
    # 3. Lesion 子任务权重细分
    LESION_W_DENSE = 1.0       # 密集掩膜强监督 (PUB)
    LESION_W_SPARSE = 1.0      # 靶向针道强监督 (TCIA)
    LESION_W_REGIONAL = 1.0    # 系统活检弱监督 (PROMIS)

    # (如果有 LAMBDA_TB 和 LAMBDA_SYS，现在可以删除了，因为它们被更清晰的子权重替代了)
    

    # [新增] 动态多阶段权重控制 (Curriculum Learning)
    USE_DYNAMIC_WEIGHTS = False         # 是否开启根据 Epoch 动态调节 Lesion 子权重
    DYNAMIC_WEIGHT_EPOCHS = [10, 30]   # 阶段切换的 Epoch 节点


    # 【辅任务B】 Gland Anatomy
    LAMBDA_GLAND = 0



    #针对极小病灶将权重
    LESION_W_SMALL = 5  # 小病灶权重 (根据实际情况调整，可能需要大于1以强调小病灶)
    
    # ==========================================
    # 6. 消融实验控制开关 (Ablation Study Flags)
    # ==========================================
    # 是否启用 3D 空间数据增强 (Data Augmentation)
    USE_AUGMENTATION = True
    
    # 是否在系统分区标签中屏蔽掉 Target 区域 (防止强弱监督信息冲突)
    MASK_TARGET_IN_SYS = True

    # ==========================================
    # 7. 可视化配置 (Visualization)
    # ==========================================
    # 在实验目录下新建文件夹保存预测图
    VIS_SUBDIR = "visualizations"

    # ==========================================
    # 辅助方法：打印当前配置 & 设置全局种子
    # ==========================================
    @classmethod
    def set_seed(cls):
        """固定全局随机种子，保证实验完全可复现"""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    @classmethod
    def show(cls):
        """打印当前配置，方便写入日志文件"""
        print("-" * 50)
        print("Experiment Configurations:")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("-" * 50)
        
    @classmethod
    def get_experiment_name(cls):
        """
        自动生成实验命名，例如：
        20241027_1530_G1.0_S0.5_L1.0_Gl0.2_LR0.0001
        """
        time_str = datetime.now().strftime("%Y%m%d_%H%M")
        name = (f"{time_str}_G{cls.LAMBDA_GRADE}_{cls.GRADE_W_TBX}_{cls.GRADE_W_SBX}"
                f"_L{cls.LAMBDA_LESION}_{cls.LESION_W_DENSE}_{cls.LESION_W_SPARSE}_{cls.LESION_W_REGIONAL}_Gl{cls.LAMBDA_GLAND}_LS{cls.LESION_W_SMALL}_LR{cls.LR}")
        if not cls.USE_AUGMENTATION:
            name += "_NoAug"
        return name