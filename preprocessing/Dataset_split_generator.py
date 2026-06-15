"""
Generate experiment split CSV files from an existing dataset registry.

This script does not copy, rename, or preprocess image files. It only reads an
existing registry CSV and creates reproducible train/internal-validation/external-
validation CSV files.

Default experiment design:
    N1: PUB radiologist annotations only.
    N2: PUB + TCIA TBx supervision only.
    N3: PUB + TCIA SBx supervision only.
    N4: PUB + TCIA TBx and SBx mixed supervision.

Training supervision differs across N1-N4, but all experiments use the same
internal evaluation cohort:
    - PUB internal-validation cases for lesion-level metrics;
    - TCIA internal-validation cases with full biopsy labels for patient-level
      and region-level metrics.

PROMIS is held out as the external validation source for every experiment.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.2


def _safe_train_val_split(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one source into train and internal validation sets safely."""
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if len(df) == 0:
        return df.copy(), df.copy()

    if len(df) < 5:
        # Avoid sklearn errors for very small datasets while keeping at least
        # one validation case whenever possible.
        n_val = max(1, int(round(len(df) * val_size)))
        n_val = min(n_val, len(df) - 1) if len(df) > 1 else 1
        val_df = df.iloc[:n_val].copy()
        train_df = df.iloc[n_val:].copy()
        return train_df, val_df

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _write_split(df: pd.DataFrame, path: str) -> None:
    """Write one split CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.reset_index(drop=True).to_csv(path, index=False)


def _prepare_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the registry and add supervision-availability columns."""
    df = df.copy()

    required_columns = {
        "patient_id",
        "source",
        "has_target",
        "has_sys_12",
        "has_sys_20",
        "has_lesion",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Registry CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    if "has_gland" not in df.columns:
        df["has_gland"] = 0

    df["source"] = df["source"].astype(str).str.upper().str.strip()

    binary_columns = [
        "has_target",
        "has_sys_12",
        "has_sys_20",
        "has_lesion",
        "has_gland",
    ]
    for column in binary_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["can_seg"] = df["has_lesion"].astype(int)
    df["can_tbx"] = df["has_target"].astype(int)
    df["can_sbx"] = (
        (df["has_sys_12"] == 1) | (df["has_sys_20"] == 1)
    ).astype(int)
    df["can_cls"] = (
        (df["can_tbx"] == 1) | (df["can_sbx"] == 1)
    ).astype(int)

    def supervision_type(row: pd.Series) -> str:
        if row["source"] == "PUB":
            return "radiologist_annotation"
        if row["source"] == "TCIA":
            if row["can_tbx"] and row["can_sbx"]:
                return "tbx_and_sbx"
            if row["can_tbx"]:
                return "tbx_only"
            if row["can_sbx"]:
                return "sbx_only"
        if row["source"] == "PROMIS" and row["can_sbx"]:
            return "sbx_only"
        return "unknown"

    df["supervision_type"] = df.apply(supervision_type, axis=1)
    return df


def _make_tbx_only_view(df: pd.DataFrame) -> pd.DataFrame:
    """Keep TBx-labelled TCIA cases and hide SBx supervision in the CSV.

    Cases that possess both TBx and SBx labels are retained, but their SBx
    availability flags are set to zero. Therefore N2 exposes only TBx
    supervision to the dataset/loss pipeline.
    """
    out = df[df["can_tbx"] == 1].copy()
    out["has_sys_12"] = 0
    out["has_sys_20"] = 0
    out["can_sbx"] = 0
    out["can_cls"] = out["can_tbx"]
    out["supervision_type"] = "tbx_only_for_experiment"
    return out


def _make_sbx_only_view(df: pd.DataFrame) -> pd.DataFrame:
    """Keep SBx-labelled TCIA cases and hide TBx supervision in the CSV.

    Cases that possess both TBx and SBx labels are retained, but their TBx
    availability flag is set to zero. Therefore N3 exposes only SBx
    supervision to the dataset/loss pipeline.
    """
    out = df[df["can_sbx"] == 1].copy()
    out["has_target"] = 0
    out["can_tbx"] = 0
    out["can_cls"] = out["can_sbx"]
    out["supervision_type"] = "sbx_only_for_experiment"
    return out


def _concat_pub_tcia(
    pub_df: pd.DataFrame,
    tcia_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine PUB dense cases and selected TCIA supervision cases."""
    return pd.concat(
        [
            pub_df[pub_df["can_seg"] == 1].copy(),
            tcia_df.copy(),
        ],
        ignore_index=True,
    )


def create_split_csvs(
    registry_csv: str,
    splits_dir: str,
    external_source: str = "PROMIS",
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """Create experiment CSV files from an existing registry CSV.

    PUB and TCIA are split once, and the same patient partitions are reused
    across N1-N4. Training CSVs expose only the supervision allowed by each
    experiment, whereas all N1-N4 internal-validation CSVs use the same common
    evaluation cohort with the original TCIA TBx/SBx flags preserved. This
    allows every experiment to report patient-level and region-level metrics
    on exactly the same cases. PROMIS is never included in training or internal
    validation.
    """
    if not os.path.exists(registry_csv):
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")

    os.makedirs(splits_dir, exist_ok=True)

    registry = _prepare_registry(pd.read_csv(registry_csv))
    _write_split(registry, os.path.join(splits_dir, "dataset_registry.csv"))

    pub_df = registry[registry["source"] == "PUB"].copy()
    tcia_df = registry[registry["source"] == "TCIA"].copy()
    external_df = registry[
        registry["source"] == external_source.upper()
    ].copy()

    pub_train, pub_internal_val = _safe_train_val_split(
        pub_df,
        val_size=val_size,
        random_state=random_state,
    )
    tcia_train, tcia_internal_val = _safe_train_val_split(
        tcia_df,
        val_size=val_size,
        random_state=random_state,
    )

    # External validation uses PROMIS systematic-biopsy supervision.
    promis_external = external_df[external_df["can_sbx"] == 1].copy()

    # ------------------------------------------------------------------
    # Common internal evaluation cohort for N1-N4
    # ------------------------------------------------------------------
    # Keep the original TCIA supervision flags in validation. Training views
    # may hide TBx or SBx labels, but validation must retain all available
    # biopsy labels so that every experiment can calculate:
    #   1) patient-level BACC on TCIA cases with can_cls == 1;
    #   2) region-level BACC on the subset with can_sbx == 1.
    # PUB validation cases are also included so lesion Dice can still be
    # measured on the same run. Metric code must use has_target/has_sys or
    # can_cls/can_sbx masks and must not treat PUB cases as negative biopsy
    # labels.
    tcia_common_internal_eval = tcia_internal_val[
        tcia_internal_val["can_cls"] == 1
    ].copy()
    common_internal_val = _concat_pub_tcia(
        pub_internal_val,
        tcia_common_internal_eval,
    )

    # N1: train with PUB radiologist annotations only.
    n1_train = pub_train[pub_train["can_seg"] == 1].copy()
    n1_internal_val = common_internal_val.copy()
    # Supports patient/region evaluation, not external lesion Dice.
    n1_external_val = promis_external.copy()

    # N2: train with PUB + TCIA TBx supervision only.
    # SBx labels are hidden only in the training view.
    tcia_tbx_train = _make_tbx_only_view(tcia_train)
    n2_train = _concat_pub_tcia(pub_train, tcia_tbx_train)
    n2_internal_val = common_internal_val.copy()
    n2_external_val = promis_external.copy()

    # N3: train with PUB + TCIA SBx supervision only.
    # TBx labels are hidden only in the training view.
    tcia_sbx_train = _make_sbx_only_view(tcia_train)
    n3_train = _concat_pub_tcia(pub_train, tcia_sbx_train)
    n3_internal_val = common_internal_val.copy()
    n3_external_val = promis_external.copy()

    # N4: train with PUB + all eligible TCIA biopsy supervision.
    tcia_mixed_train = tcia_train[tcia_train["can_cls"] == 1].copy()
    n4_train = _concat_pub_tcia(pub_train, tcia_mixed_train)
    n4_internal_val = common_internal_val.copy()
    n4_external_val = promis_external.copy()

    # Task-specific validation views remain available for diagnostic use, but
    # they are not the recommended model-selection CSVs for N1-N4.
    tcia_tbx_internal_val = _make_tbx_only_view(tcia_internal_val)
    tcia_sbx_internal_val = _make_sbx_only_view(tcia_internal_val)

    splits: Dict[str, pd.DataFrame] = {
        # Source-level reference files.
        "internal_pool.csv": registry[
            registry["source"] != external_source.upper()
        ].copy(),
        "external_val.csv": external_df.copy(),
        "common_internal_evaluation.csv": common_internal_val.copy(),
        "TCIA_common_internal_evaluation.csv": (
            tcia_common_internal_eval.copy()
        ),

        # N1.
        "N1_radiologist_only_train.csv": n1_train,
        "N1_radiologist_only_internal_val.csv": n1_internal_val,
        "N1_PROMIS_external_val.csv": n1_external_val,

        # N2: PUB + TCIA TBx only.
        "N2_PUB_TCIA_TBx_only_train.csv": n2_train,
        "N2_PUB_TCIA_TBx_only_internal_val.csv": n2_internal_val,
        "N2_PROMIS_external_val.csv": n2_external_val,

        # N3: PUB + TCIA SBx only.
        "N3_PUB_TCIA_SBx_only_train.csv": n3_train,
        "N3_PUB_TCIA_SBx_only_internal_val.csv": n3_internal_val,
        "N3_PROMIS_external_val.csv": n3_external_val,

        # N4: PUB + all eligible TCIA biopsy supervision.
        "N4_mixed_PUB_TCIA_train.csv": n4_train,
        "N4_mixed_PUB_TCIA_internal_val.csv": n4_internal_val,
        "N4_mixed_PROMIS_external_val.csv": n4_external_val,

        # Optional TCIA-only task files.
        "task_tbx_train.csv": tcia_tbx_train,
        "task_tbx_internal_val.csv": tcia_tbx_internal_val,
        "task_sbx_train.csv": tcia_sbx_train,
        "task_sbx_internal_val.csv": tcia_sbx_internal_val,
        "task_sbx_external_val.csv": promis_external.copy(),
    }

    for filename, split_df in splits.items():
        _write_split(split_df, os.path.join(splits_dir, filename))

    summary_rows = []
    for filename, split_df in splits.items():
        source_counts = (
            split_df["source"].value_counts().to_dict()
            if len(split_df) > 0 and "source" in split_df.columns
            else {}
        )
        summary_rows.append(
            {
                "file": filename,
                "n": len(split_df),
                "source_counts": source_counts,
                "n_seg": int(split_df["can_seg"].sum()) if len(split_df) else 0,
                "n_tbx": int(split_df["can_tbx"].sum()) if len(split_df) else 0,
                "n_sbx": int(split_df["can_sbx"].sum()) if len(split_df) else 0,
                # Patient metrics use all cases with at least one biopsy label.
                "n_patient_eval": (
                    int(split_df["can_cls"].sum()) if len(split_df) else 0
                ),
                # Region metrics require systematic-biopsy region labels.
                "n_region_eval": (
                    int(split_df["can_sbx"].sum()) if len(split_df) else 0
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    _write_split(summary_df, os.path.join(splits_dir, "split_summary.csv"))

    print("\nSplit CSV files created successfully.")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {splits_dir}")

    return splits


if __name__ == "__main__":
    # Update this path to match the existing Unified_Dataset location.
    UNIFIED_DATA_DIR = r"F:\RP_dataset\Unified_Dataset"
    REGISTRY_CSV = os.path.join(
        UNIFIED_DATA_DIR,
        "splits",
        "dataset_registry.csv",
    )
    SPLITS_DIR = os.path.join(UNIFIED_DATA_DIR, "splits")

    create_split_csvs(
        registry_csv=REGISTRY_CSV,
        splits_dir=SPLITS_DIR,
        external_source="PROMIS",
        val_size=0.2,
        random_state=RANDOM_STATE,
    )
