import os
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class ProstateUnifiedDataset(Dataset):
    """
    Unified mixed-supervision dataset loader for prostate mpMRI.

    Returned supervision types:
      - target_mask: voxel-level targeted-biopsy track labels, usually 0 outside track and 1-6 on track
      - zones_mask: anatomical systematic-biopsy zone ids, 0 for background
      - sys_labels: region labels for up to 20 zones; INVALID_SYS_LABEL means unsampled / no supervision
      - lesion_mask: dense binary lesion mask
      - gland_mask: dense binary prostate gland mask

    Important label convention:
      - sys_labels == INVALID_SYS_LABEL: invalid / not sampled / no supervision
      - sys_labels == 0: valid negative / benign systematic biopsy region
      - sys_labels >= CSPC_THRESHOLD: clinically significant PCa
    """

    def __init__(self, csv_path: str, data_root: str, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.is_train = is_train
        self.invalid_sys_label = int(getattr(Config, "INVALID_SYS_LABEL", -1))

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _as_bool(value) -> bool:
        if pd.isna(value):
            return False
        return bool(int(value))

    def _apply_augmentations(self, input_tensor: np.ndarray, masks_dict: Dict[str, Optional[np.ndarray]]):
        """
        Simple 3D spatial augmentation.
        Only flips in axial X-Y plane are used; Z flip is avoided because it can invert base/apex anatomy.
        """
        if random.random() > 0.5:
            input_tensor = np.flip(input_tensor, axis=3).copy()  # Width / X
            for key, mask in masks_dict.items():
                if mask is not None:
                    masks_dict[key] = np.flip(mask, axis=3).copy()

        if random.random() > 0.5:
            input_tensor = np.flip(input_tensor, axis=2).copy()  # Height / Y
            for key, mask in masks_dict.items():
                if mask is not None:
                    masks_dict[key] = np.flip(mask, axis=2).copy()

        return input_tensor, masks_dict

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        p_dir = os.path.join(self.data_root, pid)

        input_path = os.path.join(p_dir, "input_tensor.npy")
        input_tensor = np.load(input_path).astype(np.float32)  # (3, D, H, W)

        D, H, W = Config.INPUT_SHAPE[1:]

        target_mask = np.zeros((1, D, H, W), dtype=np.float32)
        zones_mask = np.zeros((1, D, H, W), dtype=np.float32)
        lesion_mask = np.zeros((1, D, H, W), dtype=np.float32)
        gland_mask = np.zeros((1, D, H, W), dtype=np.float32)

        # Critical fix: -1 means invalid / unsampled. 0 is kept as a valid negative label.
        sys_labels = np.full(20, self.invalid_sys_label, dtype=np.int64)

        has_target = self._as_bool(row.get("has_target", 0))
        has_sys_12 = self._as_bool(row.get("has_sys_12", 0))
        has_sys_20 = self._as_bool(row.get("has_sys_20", 0))
        has_sys = has_sys_12 or has_sys_20
        has_lesion = self._as_bool(row.get("has_lesion", 0))
        has_gland = self._as_bool(row.get("has_gland", 0))

        masks_to_aug = {"target": None, "zones": None, "lesion": None, "gland": None}

        # A. Targeted biopsy track labels.
        if has_target:
            t_path = os.path.join(p_dir, "target_bx.nii.gz")
            t_arr = sitk.GetArrayFromImage(sitk.ReadImage(t_path)).astype(np.float32)
            masks_to_aug["target"] = np.expand_dims(t_arr, axis=0)

        # B. Systematic biopsy zone labels.
        if has_sys:
            z_path = os.path.join(p_dir, "zones_mask.nii.gz")
            z_arr = sitk.GetArrayFromImage(sitk.ReadImage(z_path)).astype(np.float32)
            masks_to_aug["zones"] = np.expand_dims(z_arr, axis=0)

            if has_sys_12:
                s_labels = np.load(os.path.join(p_dir, "systematic_labels_12.npy")).astype(np.int64)
                sys_labels[: min(12, len(s_labels))] = s_labels[:12]
            elif has_sys_20:
                s_labels = np.load(os.path.join(p_dir, "systematic_labels_20.npy")).astype(np.int64)
                sys_labels[: min(20, len(s_labels))] = s_labels[:20]

        # C. Dense lesion mask.
        if has_lesion:
            l_arr = np.load(os.path.join(p_dir, "lesion_mask.npy")).astype(np.float32)
            masks_to_aug["lesion"] = np.expand_dims((l_arr > 0).astype(np.float32), axis=0)

        # D. Dense gland mask.
        if has_gland:
            g_path_nii = os.path.join(p_dir, "gland_mask.nii.gz")
            g_path_npy = os.path.join(p_dir, "gland_mask.npy")
            if os.path.exists(g_path_nii):
                g_arr = sitk.GetArrayFromImage(sitk.ReadImage(g_path_nii)).astype(np.float32)
            else:
                g_arr = np.load(g_path_npy).astype(np.float32)
            masks_to_aug["gland"] = np.expand_dims((g_arr > 0).astype(np.float32), axis=0)

        if self.is_train and getattr(Config, "USE_AUGMENTATION", False):
            input_tensor, masks_to_aug = self._apply_augmentations(input_tensor, masks_to_aug)

        if masks_to_aug["target"] is not None:
            target_mask = masks_to_aug["target"]
        if masks_to_aug["zones"] is not None:
            zones_mask = masks_to_aug["zones"]
        if masks_to_aug["lesion"] is not None:
            lesion_mask = masks_to_aug["lesion"]
        if masks_to_aug["gland"] is not None:
            gland_mask = masks_to_aug["gland"]

        # Prevent strong and weak supervision conflicts in the same voxels.
        if getattr(Config, "MASK_TARGET_IN_SYS", False) and has_target and has_sys:
            zones_mask[target_mask > 0] = 0

        return {
            "pid": pid,
            "input": torch.from_numpy(input_tensor),
            "target_mask": torch.from_numpy(target_mask),
            "zones_mask": torch.from_numpy(zones_mask),
            "sys_labels": torch.from_numpy(sys_labels),
            "lesion_mask": torch.from_numpy(lesion_mask),
            "gland_mask": torch.from_numpy(gland_mask),
            "has_target": torch.tensor(float(has_target), dtype=torch.float32),
            "has_sys": torch.tensor(float(has_sys), dtype=torch.float32),
            "has_lesion": torch.tensor(float(has_lesion), dtype=torch.float32),
            "has_gland": torch.tensor(float(has_gland), dtype=torch.float32),
        }


if __name__ == "__main__":
    print("Testing ProstateUnifiedDataset...")
    test_csv = os.path.join(Config.SPLIT_DIR, "train.csv")
    if os.path.exists(test_csv):
        dataset = ProstateUnifiedDataset(csv_path=test_csv, data_root=Config.UNIFIED_DATA_DIR, is_train=True)
        print(f"Dataset Size: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print("\nBatch Tensor Shapes:")
        for key in ["input", "target_mask", "zones_mask", "sys_labels", "lesion_mask", "gland_mask"]:
            print(f"  {key:<12}: {tuple(batch[key].shape)}")
        print("\nBatch Active Flags:")
        for key in ["has_target", "has_sys", "has_lesion", "has_gland"]:
            print(f"  {key:<12}: {batch[key]}")
        print(f"\nInvalid systematic label value: {getattr(Config, 'INVALID_SYS_LABEL', -1)}")
    else:
        print("CSV index not found. Please run the unified dataset script first.")
