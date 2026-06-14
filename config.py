"""
Configuration for the revised RP pipeline after the 2026-06-10 project update.

Current main question:
    Does mixed supervision improve prostate cancer detection compared with
    strong supervision from radiologist annotations alone?

Main tasks in the current code:
    1) Lesion segmentation from dense radiologist annotation.
    2) Lesion/region MIL supervision from TBx/SBx labels.

This config intentionally removes grade/gland as active training tasks.
Keep grade-related constants only when they are needed to convert biopsy labels
into binary cancer / csPCa labels.
"""

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
    EXP_DIR = os.path.join(BASE_DIR, "Experiments")

    # ------------------------------------------------------------
    # Experiment mode
    # ------------------------------------------------------------
    # Options:
    #   "N1_RADIOLOGIST_ONLY"
    #       Strong-supervision baseline.
    #       Uses PUB radiologist lesion masks only.
    #
    #   "N4_MIXED"
    #       Main mixed-supervision experiment.
    #       Uses PUB dense lesion masks + TCIA TBx/SBx labels for training,
    #       and PROMIS SBx as external validation/test.
    #
    #   "CUSTOM"
    #       Manually set TRAIN_CSV / VAL_CSV / TEST_CSV below.
    EXPERIMENT_MODE = "N4_MIXED"

    if EXPERIMENT_MODE == "N1_RADIOLOGIST_ONLY":
        TRAIN_CSV = os.path.join(SPLIT_DIR, "N1_radiologist_only_train.csv")
        VAL_CSV = os.path.join(SPLIT_DIR, "N1_radiologist_only_internal_val.csv")
        # There is usually no true external dense-lesion test set yet.
        # Use this only if you later provide an external radiologist-mask CSV.
        TEST_CSV = os.path.join(SPLIT_DIR, "N1_radiologist_only_internal_val.csv")
        TASK = "radiologist_only"
        DATASET_TASK = "radiologist_only"
        EXPERIMENT_TAG = "N1_RadiologistOnly_Seg"
        BEST_MODEL_METRIC = "lesion_dice"

        USE_LESION_DENSE_TASK = True
        USE_LESION_SPARSE_TASK = False
        USE_LESION_SYS_TASK = False
        USE_EM_WEIGHTING = False
        USE_CURRICULUM = False

    elif EXPERIMENT_MODE == "N4_MIXED":
        TRAIN_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PUB_TCIA_train.csv")
        VAL_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PUB_TCIA_internal_val.csv")
        TEST_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PROMIS_external_val.csv")
        TASK = "mixed"
        DATASET_TASK = "mixed"
        EXPERIMENT_TAG = "N4_Mixed_PUB_TCIA_PROMISExternal"
        BEST_MODEL_METRIC = "composite"

        USE_LESION_DENSE_TASK = True
        USE_LESION_SPARSE_TASK = True
        USE_LESION_SYS_TASK = True
        USE_EM_WEIGHTING = True
        USE_CURRICULUM = True

    else:  # CUSTOM
        TRAIN_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PUB_TCIA_train.csv")
        VAL_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PUB_TCIA_internal_val.csv")
        TEST_CSV = os.path.join(SPLIT_DIR, "N4_mixed_PROMIS_external_val.csv")
        TASK = "mixed"
        DATASET_TASK = "mixed"
        EXPERIMENT_TAG = "Custom_SegMIL"
        BEST_MODEL_METRIC = "composite"

        USE_LESION_DENSE_TASK = True
        USE_LESION_SPARSE_TASK = True
        USE_LESION_SYS_TASK = True
        USE_EM_WEIGHTING = True
        USE_CURRICULUM = True

    # Optional test settings.
    # After training, set TEST_DIR to the experiment folder containing best_model.pth
    # or set TEST_MODEL_PATH directly to best_checkpoint.pth / best_model.pth.
    TEST_DIR = None
    TEST_MODEL_PATH = None

    # ==========================================
    # 2. Data and preprocessing
    # ==========================================
    TARGET_SPACING = [1.0, 1.0, 2.24]
    CROP_SIZE_SITK = [64, 64, 32]
    INPUT_SHAPE = (3, 32, 64, 64)
    NEEDLE_RADIUS = 2

    # ==========================================
    # 3. Model and label conventions
    # ==========================================
    IN_CHANNELS = 3
    BASE_CHANNELS = 32
    MAX_ZONES = 20

    # Kept for compatibility with old constructors; the current segmentation/MIL
    # model does not use a grade output head.
    NUM_CLASSES = 7

    # Biopsy label convention used in previous preprocessing:
    #   -1 = invalid / unsampled / no supervision
    #    0 = valid negative region
    #    1 = benign / non-significant label if present
    #    2 = ISUP1
    #    3 = ISUP2
    #    4 = ISUP3
    #    5 = ISUP4
    #    6 = ISUP5
    # If you define csPCa as ISUP2+, the threshold is 3 under this convention.
    INVALID_SYS_LABEL = -1
    CSPC_THRESHOLD = 3

    # This is the positive threshold used by the segmentation/MIL loss.
    # Use CSPC_THRESHOLD for csPCa detection. If you want any-cancer detection,
    # change this to 1 or 2 according to your actual biopsy-label convention.
    LESION_POSITIVE_THRESHOLD = CSPC_THRESHOLD

    # Prediction threshold for validation/test binary metrics.
    PRED_PROB_THRESHOLD = 0.5

    # MIL pooling from voxel lesion logits to region-level logits.
    # Options supported by the revised model: "lme", "max", "mean".
    MIL_POOLING = "lme"
    LME_R = 8.0

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
    GRAD_CLIP_NORM = 12.0

    # Class imbalance control for lesion/MIL BCE loss.
    POS_WEIGHT_VAL = 2.0

    # ==========================================
    # 5. Loss / EM weighting configuration
    # ==========================================
    # True  = dynamic uncertainty/EM weighting: L_i * exp(-s_i) + s_i
    # False = fixed scalar weights from FIXED_LOSS_WEIGHTS
    # The default value is set above by EXPERIMENT_MODE.

    EM_LR_MULTIPLIER = 10.0

    # Optional clamp for the learned log_var values.
    USE_LOGVAR_CLAMP = False
    LOGVAR_MIN = -3.0
    LOGVAR_MAX = 3.0

    # Fixed loss weights for No-EM ablation. Used only when USE_EM_WEIGHTING=False.
    FIXED_LOSS_WEIGHTS = {
        "lesion_dense": 1.0,
        "lesion_sparse": 1.0,
        "lesion_sys": 1.0,
    }

    # Curriculum learning for mixed supervision.
    # Recommended mixed-supervision schedule:
    #   epoch 1-9:   radiologist dense lesion supervision only
    #   epoch 10-29: dense + TBx sparse supervision
    #   epoch 30+:   dense + TBx sparse + SBx MIL supervision
    LESION_DENSE_START_EPOCH = 1
    LESION_SPARSE_START_EPOCH = 10
    LESION_SYS_START_EPOCH = 30

    # Compatibility flags for older code paths. They should stay False because
    # the current model/loss does not train grade or gland heads.
    USE_GRADE_TBX_TASK = False
    USE_GRADE_SBX_TASK = False
    USE_GLAND_TASK = False
    GRADE_TBX_START_EPOCH = 1
    GRADE_SBX_START_EPOCH = 1
    GLAND_START_EPOCH = 1

    # ==========================================
    # 6. Ablation and augmentation flags
    # ==========================================
    USE_AUGMENTATION = True

    # If both TBx and SBx supervision exist for one sample, remove TBx voxels
    # from SBx zones to reduce conflicting supervision.
    MASK_TARGET_IN_SYS = True

    # ==========================================
    # 7. Visualization and logging
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
            if k.startswith("__"):
                continue
            if callable(v) or isinstance(v, classmethod):
                continue
            print(f"{k:<32}: {v}")
        print("-" * 50)

    @classmethod
    def enabled_task_name(cls):
        enabled = []
        if getattr(cls, "USE_LESION_DENSE_TASK", False):
            enabled.append("LesDense")
        if getattr(cls, "USE_LESION_SPARSE_TASK", False):
            enabled.append("LesSparseTBx")
        if getattr(cls, "USE_LESION_SYS_TASK", False):
            enabled.append("LesSysMIL")
        return "_".join(enabled) if enabled else "NoTask"

    @classmethod
    def get_experiment_name(cls):
        time_str = datetime.now().strftime("%Y%m%d_%H%M")
        weighting_name = "EM" if getattr(cls, "USE_EM_WEIGHTING", True) else "FixedW"
        task_name = cls.enabled_task_name()
        clamp_name = "Clamp" if getattr(cls, "USE_LOGVAR_CLAMP", False) else "NoClamp"
        curriculum_name = "Curr" if getattr(cls, "USE_CURRICULUM", False) else "NoCurr"
        em_lr = getattr(cls, "EM_LR_MULTIPLIER", 1.0)
        tag = getattr(cls, "EXPERIMENT_TAG", "")

        parts = [time_str]
        if tag:
            parts.append(str(tag))
        parts.extend([
            weighting_name,
            task_name,
            clamp_name,
            curriculum_name,
            f"EMlrX{em_lr:g}",
            f"LR{cls.LR}",
            str(cls.BEST_MODEL_METRIC),
        ])
        name = "_".join(parts)
        if not getattr(cls, "USE_AUGMENTATION", True):
            name += "_NoAug"
        return name
