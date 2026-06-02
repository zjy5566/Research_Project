import os
import random
from datetime import datetime

import numpy as np
import torch


class Config:
    # ==========================================
    # 1. Path configurations
    # ==========================================
    BASE_DIR = r"/raid/candi/jiayi/RP"
    UNIFIED_DATA_DIR = os.path.join(BASE_DIR, "data", "Unified_Dataset")
    SPLIT_DIR = os.path.join(UNIFIED_DATA_DIR, "splits")

    TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
    VAL_CSV = os.path.join(SPLIT_DIR, "val.csv")
    TEST_CSV = os.path.join(SPLIT_DIR, "test.csv")

    EXP_DIR = os.path.join(BASE_DIR, "Experiments")

    # ==========================================
    # 2. Data and preprocessing
    # ==========================================
    TARGET_SPACING = [1.0, 1.0, 2.24]
    CROP_SIZE_SITK = [64, 64, 32]
    INPUT_SHAPE = (3, 32, 64, 64)
    NEEDLE_RADIUS = 2

    # ==========================================
    # 3. Model and labels
    # ==========================================
    IN_CHANNELS = 3
    NUM_CLASSES = 7

    # Label convention used by the current code:
    # 0: background / negative region, 1: benign, 2: ISUP1, 3: ISUP2, ..., 6: ISUP5.
    # Therefore csPCa is ISUP2+, so the threshold is 3.
    CSPC_THRESHOLD = 3

    # Critical convention for systematic biopsy labels:
    # -1 = invalid / unsampled / no supervision, 0 = valid negative, >=1 = valid label.
    INVALID_SYS_LABEL = -1

    # ==========================================
    # 4. Training hyperparameters
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    NUM_EPOCHS = 100
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER = "CosineAnnealing"
    EARLY_STOP_PATIENCE = 300

    # ==========================================
    # 5. Loss configuration
    # ==========================================
    # ==========================================
# 5. Loss configuration
# ==========================================
    USE_EM_WEIGHTING = False   # False = No EM weighting baseline

    # Fixed loss weights for No-EM ablation.
    # First baseline should use equal weights to test whether EM helps.
    FIXED_LOSS_WEIGHTS = {
        "grade_tbx": 1,
        "grade_sbx": 0,
        "lesion_dense": 1.0,
        "lesion_sparse": 1.0,
        "lesion_sys": 1.0,
        "gland": 1.0,
    }

    LESION_W_SMALL = 5

    # Best-model selection. Options:
    #   "lesion_dice", "clinical_bacc", "region_bacc", "gland_bacc", "grade_kappa", "composite"
    BEST_MODEL_METRIC = "composite"

    # ==========================================
    # 6. Ablation flags
    # ==========================================
    USE_AUGMENTATION = True
    MASK_TARGET_IN_SYS = True

    # ==========================================
    # 7. Visualization
    # ==========================================
    VIS_SUBDIR = "visualizations"

    @classmethod
    def set_seed(cls):
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
        print("-" * 50)
        print("Experiment Configurations:")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v) and not isinstance(v, classmethod):
                print(f"{k:<24}: {v}")
        print("-" * 50)

    @classmethod
    # def get_experiment_name(cls):
    #     time_str = datetime.now().strftime("%Y%m%d_%H%M")
    #     name = f"{time_str}_EM_Weighting_LR{cls.LR}_{cls.BEST_MODEL_METRIC}"
    #     if not getattr(cls, "USE_AUGMENTATION", True):
    #         name += "_NoAug"
    #     return name
    @classmethod
    def get_experiment_name(cls):
        time_str = datetime.now().strftime("%Y%m%d_%H%M")
        weighting_name = "EM_Weighting" if getattr(cls, "USE_EM_WEIGHTING", True) else "NoEM_FixedWeights"
        name = f"{time_str}_{weighting_name}_LR{cls.LR}_{cls.BEST_MODEL_METRIC}"
        if not getattr(cls, "USE_AUGMENTATION", True):
            name += "_NoAug"
        return name
