import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RANDOM_STATE = 42


def _safe_train_val_split(df, val_size=0.2, random_state=RANDOM_STATE):
    """Split a single-source dataframe into train/internal_val safely."""
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if len(df) == 0:
        return df.copy(), df.copy()
    if len(df) < 5:
        # For very small sets, avoid sklearn errors. Keep at least one validation case.
        n_val = max(1, int(round(len(df) * val_size)))
        val_df = df.iloc[:n_val].copy()
        train_df = df.iloc[n_val:].copy()
        return train_df, val_df
    return train_test_split(df, test_size=val_size, random_state=random_state, shuffle=True)


def _write_split(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _add_task_columns(df):
    """Add task availability columns for cleaner experiment filtering."""
    df = df.copy()
    df["can_seg"] = df["has_lesion"].astype(int)
    df["can_tbx"] = df["has_target"].astype(int)
    df["can_sbx"] = ((df["has_sys_12"] == 1) | (df["has_sys_20"] == 1)).astype(int)
    df["can_cls"] = ((df["can_tbx"] == 1) | (df["can_sbx"] == 1)).astype(int)

    def supervision_type(row):
        if row["source"] == "PUB":
            return "radiologist_annotation"
        if row["source"] == "TCIA":
            if row["has_target"] and row["has_sys_12"]:
                return "tbx_and_sbx"
            if row["has_target"]:
                return "tbx_only"
            if row["has_sys_12"]:
                return "sbx_only"
        if row["source"] == "PROMIS":
            return "sbx_only"
        return "unknown"

    df["supervision_type"] = df.apply(supervision_type, axis=1)
    return df


def create_internal_external_splits(df, splits_dir, external_source="PROMIS", val_size=0.2, random_state=RANDOM_STATE):
    """
    New 2026-06-10 split logic:
    - Do NOT randomly mix all sources into train/val/test.
    - Use one source/domain as external validation.
    - Use remaining eligible sources for training and internal validation.

    Default setting:
    - PUB: radiologist annotation, used for lesion segmentation strong-supervision baseline.
    - TCIA: TBx/SBx, used for mixed-supervision training and internal classification validation.
    - PROMIS: SBx, held out as external validation for classification/generalisation.
    """
    os.makedirs(splits_dir, exist_ok=True)
    df = _add_task_columns(df)
    _write_split(df, os.path.join(splits_dir, "dataset_registry.csv"))

    # Source-specific pools
    pub_df = df[df["source"] == "PUB"].copy()
    tcia_df = df[df["source"] == "TCIA"].copy()
    promis_df = df[df["source"] == "PROMIS"].copy()

    # Whole-source external validation by default.
    external_df = df[df["source"] == external_source].copy()
    internal_pool_df = df[df["source"] != external_source].copy()

    # Split PUB for segmentation internal validation.
    pub_train_df, pub_internal_val_df = _safe_train_val_split(
        pub_df, val_size=val_size, random_state=random_state
    )

    # Split TCIA for biopsy/classification internal validation.
    tcia_train_df, tcia_internal_val_df = _safe_train_val_split(
        tcia_df, val_size=val_size, random_state=random_state
    )

    # If you later decide not to hold PROMIS entirely external, change external_source or pass a supervisor table.
    promis_external_df = promis_df.copy() if external_source == "PROMIS" else external_df.copy()

    # Core experiment splits for the revised RP.
    # N1: strong supervision baseline: Radiologist Annotation only.
    exp_n1_train = pub_train_df[pub_train_df["can_seg"] == 1].copy()
    exp_n1_internal_val = pub_internal_val_df[pub_internal_val_df["can_seg"] == 1].copy()
    exp_n1_external_val = pd.DataFrame(columns=df.columns)  # no external lesion masks currently available

    # N2/N4: mixed supervision: PUB lesion masks + TCIA TBx/SBx labels.
    # PROMIS is kept external by default to test generalisation of SBx-derived classification.
    exp_mixed_train = pd.concat(
        [
            pub_train_df[pub_train_df["can_seg"] == 1],
            tcia_train_df[tcia_train_df["can_cls"] == 1],
        ],
        ignore_index=True,
    )
    exp_mixed_internal_val = pd.concat(
        [
            pub_internal_val_df[pub_internal_val_df["can_seg"] == 1],
            tcia_internal_val_df[tcia_internal_val_df["can_cls"] == 1],
        ],
        ignore_index=True,
    )
    exp_mixed_external_val = promis_external_df[promis_external_df["can_cls"] == 1].copy()

    # Task-specific validation files. These are useful in the training/evaluation code.
    seg_train = pub_train_df[pub_train_df["can_seg"] == 1].copy()
    seg_internal_val = pub_internal_val_df[pub_internal_val_df["can_seg"] == 1].copy()
    seg_external_val = pd.DataFrame(columns=df.columns)  # update when an external lesion-mask source exists

    cls_train = tcia_train_df[tcia_train_df["can_cls"] == 1].copy()
    cls_internal_val = tcia_internal_val_df[tcia_internal_val_df["can_cls"] == 1].copy()
    cls_external_val = promis_external_df[promis_external_df["can_cls"] == 1].copy()

    # Save source-level pools.
    _write_split(internal_pool_df, os.path.join(splits_dir, "internal_pool.csv"))
    _write_split(external_df, os.path.join(splits_dir, "external_val.csv"))

    # Save experiment-level splits.
    _write_split(exp_n1_train, os.path.join(splits_dir, "N1_radiologist_only_train.csv"))
    _write_split(exp_n1_internal_val, os.path.join(splits_dir, "N1_radiologist_only_internal_val.csv"))
    _write_split(exp_n1_external_val, os.path.join(splits_dir, "N1_radiologist_only_external_val.csv"))

    _write_split(exp_mixed_train, os.path.join(splits_dir, "N4_mixed_PUB_TCIA_train.csv"))
    _write_split(exp_mixed_internal_val, os.path.join(splits_dir, "N4_mixed_PUB_TCIA_internal_val.csv"))
    _write_split(exp_mixed_external_val, os.path.join(splits_dir, "N4_mixed_PROMIS_external_val.csv"))

    # Save task-level splits.
    _write_split(seg_train, os.path.join(splits_dir, "task_seg_train.csv"))
    _write_split(seg_internal_val, os.path.join(splits_dir, "task_seg_internal_val.csv"))
    _write_split(seg_external_val, os.path.join(splits_dir, "task_seg_external_val.csv"))

    _write_split(cls_train, os.path.join(splits_dir, "task_cls_train.csv"))
    _write_split(cls_internal_val, os.path.join(splits_dir, "task_cls_internal_val.csv"))
    _write_split(cls_external_val, os.path.join(splits_dir, "task_cls_external_val.csv"))

    summary = []
    for name, split_df in {
        "registry": df,
        "PUB_train_for_seg": pub_train_df,
        "PUB_internal_val_for_seg": pub_internal_val_df,
        "TCIA_train_for_cls": tcia_train_df,
        "TCIA_internal_val_for_cls": tcia_internal_val_df,
        "external_val": external_df,
        "N1_train": exp_n1_train,
        "N1_internal_val": exp_n1_internal_val,
        "N4_mixed_train": exp_mixed_train,
        "N4_mixed_internal_val": exp_mixed_internal_val,
        "N4_mixed_external_val": exp_mixed_external_val,
    }.items():
        counts = split_df["source"].value_counts().to_dict() if len(split_df) else {}
        summary.append({"split": name, "n": len(split_df), "source_counts": counts})

    summary_df = pd.DataFrame(summary)
    _write_split(summary_df, os.path.join(splits_dir, "split_summary.csv"))

    print("\nNew internal/external splits created.")
    print(summary_df.to_string(index=False))
    print(f"\nSaved split CSVs to: {splits_dir}")


def create_unified_dataset(base_dir):
    # --- 1. Define paths ---
    src_pub = os.path.join(base_dir, "Dataset_prostate_MRI", "Dataset_prostate_MRI_dwi")
    src_promis = os.path.join(base_dir, "derived PROMIS data set", "Processed_PROMIS_dwi")
    src_tcia = os.path.join(base_dir, "Target biosy", "Processed_TCIA")

    dst_root = os.path.join(base_dir, "Unified_Dataset")
    os.makedirs(dst_root, exist_ok=True)
    registry_data = []

    # --- 2. TCIA: Target biopsy + possible systematic biopsy ---
    print("Processing TCIA Target Biopsy Dataset...")
    if os.path.exists(src_tcia):
        tcia_search_dir = src_tcia
        if os.path.exists(os.path.join(src_tcia, "Processed_PROMIS")):
            tcia_search_dir = os.path.join(src_tcia, "Processed_PROMIS")

        tcia_patients = [d for d in os.listdir(tcia_search_dir) if d.startswith("Prostate-MRI-US-Biopsy-")]
        for pid in tqdm(tcia_patients):
            src_p_dir = os.path.join(tcia_search_dir, pid)
            if not os.path.exists(os.path.join(src_p_dir, "input_tensor.npy")):
                continue

            target_path = os.path.join(src_p_dir, "target_bx_needle_crop.nii.gz")
            if not os.path.exists(target_path) and os.path.exists(os.path.join(src_p_dir, "tatarget_bx_needle_crop.nii.gz")):
                target_path = os.path.join(src_p_dir, "tatarget_bx_needle_crop.nii.gz")

            sys_mask_path = os.path.join(src_p_dir, "zones_mask_crop.nii.gz")
            sys_label_path = os.path.join(src_p_dir, "systematic_labels.npy")
            gland_path = os.path.join(src_p_dir, "gland_mask_crop.nii.gz")

            has_target = int(os.path.exists(target_path))
            has_sys_12 = int(os.path.exists(sys_mask_path) and os.path.exists(sys_label_path))
            has_gland = int(os.path.exists(gland_path))
            if has_target == 0 and has_sys_12 == 0:
                continue

            new_pid = f"TCIA_{pid.split('-')[-1]}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            shutil.copy2(os.path.join(src_p_dir, "input_tensor.npy"), os.path.join(dst_p_dir, "input_tensor.npy"))
            if has_sys_12:
                shutil.copy2(sys_mask_path, os.path.join(dst_p_dir, "zones_mask.nii.gz"))
                shutil.copy2(sys_label_path, os.path.join(dst_p_dir, "systematic_labels_12.npy"))
            if has_target:
                shutil.copy2(target_path, os.path.join(dst_p_dir, "target_bx.nii.gz"))
            if has_gland:
                shutil.copy2(gland_path, os.path.join(dst_p_dir, "gland_mask.nii.gz"))

            registry_data.append({
                "patient_id": new_pid,
                "source": "TCIA",
                "has_target": has_target,
                "has_sys_12": has_sys_12,
                "has_sys_20": 0,
                "has_lesion": 0,
                "has_gland": has_gland,
            })

    # --- 3. PROMIS: systematic biopsy only ---
    print("\nProcessing PROMIS Dataset...")
    if os.path.exists(src_promis):
        promis_patients = [d for d in os.listdir(src_promis) if d.startswith("P-")]
        for pid in tqdm(promis_patients):
            src_p_dir = os.path.join(src_promis, pid)
            req_files = ["input_tensor.npy", "zones_mask.nii.gz", "systematic_labels.npy"]
            if not all(os.path.exists(os.path.join(src_p_dir, f)) for f in req_files):
                continue

            has_gland = int(os.path.exists(os.path.join(src_p_dir, "gland_mask.nii.gz")))
            new_pid = f"PROMIS_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            shutil.copy2(os.path.join(src_p_dir, "input_tensor.npy"), os.path.join(dst_p_dir, "input_tensor.npy"))
            shutil.copy2(os.path.join(src_p_dir, "zones_mask.nii.gz"), os.path.join(dst_p_dir, "zones_mask.nii.gz"))
            shutil.copy2(os.path.join(src_p_dir, "systematic_labels.npy"), os.path.join(dst_p_dir, "systematic_labels_20.npy"))
            if has_gland:
                shutil.copy2(os.path.join(src_p_dir, "gland_mask.nii.gz"), os.path.join(dst_p_dir, "gland_mask.nii.gz"))

            registry_data.append({
                "patient_id": new_pid,
                "source": "PROMIS",
                "has_target": 0,
                "has_sys_12": 0,
                "has_sys_20": 1,
                "has_lesion": 0,
                "has_gland": has_gland,
            })

    # --- 4. PUB: radiologist lesion annotation ---
    print("\nProcessing PUB Radiologist Annotation Dataset...")
    if os.path.exists(src_pub):
        pub_files = os.listdir(src_pub)
        pub_ids = sorted(set(f.split("_")[0] for f in pub_files if f.endswith("_img.npy")))
        for pid in tqdm(pub_ids):
            src_img = os.path.join(src_pub, f"{pid}_img.npy")
            src_lab = os.path.join(src_pub, f"{pid}_lab.npy")
            src_zone = os.path.join(src_pub, f"{pid}_zone.npy")
            if not all(os.path.exists(f) for f in [src_img, src_lab, src_zone]):
                continue

            new_pid = f"PUB_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            shutil.copy2(src_img, os.path.join(dst_p_dir, "input_tensor.npy"))
            shutil.copy2(src_lab, os.path.join(dst_p_dir, "lesion_mask.npy"))
            shutil.copy2(src_zone, os.path.join(dst_p_dir, "gland_mask.npy"))

            registry_data.append({
                "patient_id": new_pid,
                "source": "PUB",
                "has_target": 0,
                "has_sys_12": 0,
                "has_sys_20": 0,
                "has_lesion": 1,
                "has_gland": 1,
            })

    if len(registry_data) == 0:
        print("Error: No valid data found. Please check source directories.")
        return

    df = pd.DataFrame(registry_data)
    splits_dir = os.path.join(dst_root, "splits")
    create_internal_external_splits(
        df,
        splits_dir=splits_dir,
        external_source="PROMIS",
        val_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"\nUnified dataset successfully created at: {dst_root}")


if __name__ == "__main__":
    BASE_DIR = r"F:\RP_dataset"
    create_unified_dataset(BASE_DIR)
