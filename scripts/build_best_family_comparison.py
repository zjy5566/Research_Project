from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE_ROOT / "result"
OUT = ROOT / "experiment_comparison"

FAMILY_ORDER = ["B1", "N1", "B2", "B3"]

METRICS = [
    {
        "category": "Segmentation",
        "label": "Target csPCa Dice",
        "column": "test_target_cspca_dice_mean",
        "std": "test_target_cspca_dice_std",
        "n": "test_target_cspca_dice_n",
    },
    {
        "category": "Segmentation",
        "label": "Target Dice (best threshold)",
        "column": "test_target_cspca_best_threshold_dice_mean",
        "std": "test_target_cspca_best_threshold_dice_std",
        "n": "test_target_cspca_best_threshold_dice_n",
    },
    {
        "category": "Segmentation",
        "label": "Target Dice (top-k)",
        "column": "test_target_cspca_topk_dice_mean",
        "std": "test_target_cspca_topk_dice_std",
        "n": "test_target_cspca_topk_dice_n",
    },
    {
        "category": "ROI",
        "label": "ROI AUROC",
        "column": "test_tbx_roi_auc",
        "n": "test_tbx_roi_n",
    },
    {
        "category": "ROI",
        "label": "ROI AUPRC",
        "column": "test_tbx_roi_auprc",
        "n": "test_tbx_roi_n",
    },
    {
        "category": "ROI",
        "label": "ROI sens @ spec=.95",
        "column": "test_tbx_roi_sens_at_fixed_spec",
        "n": "test_tbx_roi_n",
    },
    {
        "category": "ROI",
        "label": "ROI spec @ sens=.90",
        "column": "test_tbx_roi_spec_at_fixed_sens",
        "n": "test_tbx_roi_n",
    },
    {
        "category": "Patient",
        "label": "Patient AUROC",
        "column": "test_patient_auc",
        "n": "test_patient_n",
    },
    {
        "category": "Patient",
        "label": "Patient AUPRC",
        "column": "test_patient_auprc",
        "n": "test_patient_n",
    },
    {
        "category": "Patient",
        "label": "Patient sens @ spec=.95",
        "column": "test_patient_sens_at_fixed_spec",
        "n": "test_patient_n",
    },
    {
        "category": "Patient",
        "label": "Patient spec @ sens=.90",
        "column": "test_patient_spec_at_fixed_sens",
        "n": "test_patient_n",
    },
    {
        "category": "Region",
        "label": "Region AUROC",
        "column": "test_region_auc",
        "n": "test_region_n",
    },
    {
        "category": "Region",
        "label": "Region AUPRC",
        "column": "test_region_auprc",
        "n": "test_region_n",
    },
    {
        "category": "Region",
        "label": "Region sens @ spec=.95",
        "column": "test_region_sens_at_fixed_spec",
        "n": "test_region_n",
    },
    {
        "category": "Region",
        "label": "Region spec @ sens=.90",
        "column": "test_region_spec_at_fixed_sens",
        "n": "test_region_n",
    },
    {
        "category": "FROC",
        "label": "Target FROC @ 0.5 FP/p",
        "column": "test_target_cspca_sens_at_fp_per_patient_0p5",
        "n": "test_target_cspca_froc_num_gt",
    },
    {
        "category": "FROC",
        "label": "Target FROC @ 1.0 FP/p",
        "column": "test_target_cspca_sens_at_fp_per_patient_1p0",
        "n": "test_target_cspca_froc_num_gt",
    },
    {
        "category": "FROC",
        "label": "Target FROC @ 2.0 FP/p",
        "column": "test_target_cspca_sens_at_fp_per_patient_2p0",
        "n": "test_target_cspca_froc_num_gt",
    },
]

METRIC_BY_COLUMN = {metric["column"]: metric for metric in METRICS}

TABLE_SPECS = {
    "internal_localization": [
        "test_target_cspca_dice_mean",
        "test_target_cspca_best_threshold_dice_mean",
        "test_target_cspca_topk_dice_mean",
        "test_tbx_roi_auc",
        "test_tbx_roi_auprc",
        "test_tbx_roi_sens_at_fixed_spec",
        "test_tbx_roi_spec_at_fixed_sens",
        "test_target_cspca_sens_at_fp_per_patient_0p5",
        "test_target_cspca_sens_at_fp_per_patient_1p0",
        "test_target_cspca_sens_at_fp_per_patient_2p0",
    ],
    "internal_clinical": [
        "test_patient_auc",
        "test_patient_auprc",
        "test_patient_sens_at_fixed_spec",
        "test_patient_spec_at_fixed_sens",
        "test_region_auc",
        "test_region_auprc",
        "test_region_sens_at_fixed_spec",
        "test_region_spec_at_fixed_sens",
    ],
    "external_clinical": [
        "test_patient_auc",
        "test_patient_auprc",
        "test_patient_sens_at_fixed_spec",
        "test_patient_spec_at_fixed_sens",
        "test_region_auc",
        "test_region_auprc",
        "test_region_sens_at_fixed_spec",
        "test_region_spec_at_fixed_sens",
    ],
}


def finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def family_from_name(name: str) -> str | None:
    match = re.search(r"_(B1|N1|B2|B3)_", name)
    return match.group(1) if match else None


def variant_from_name(name: str, family: str) -> str:
    if family == "B1":
        if "LR5e-5_PosW1" in name:
            return "LR5e-5+PosW1"
        if "PosW1" in name:
            return "PosW1"
        if "LR5e-5" in name:
            return "LR5e-5"
        return "Default"
    if family == "N1":
        return "Dense"
    if family == "B2":
        return "SBx-only"
    if family == "B3":
        return "TBx+SBx"
    return name


def split_from_row(row: pd.Series) -> str:
    label = str(row.get("test_dataset_label", "")).strip().lower()
    if label in {"internal", "external"}:
        return label
    test_csv = str(row.get("test_csv", "")).lower()
    if "promis_external" in test_csv:
        return "external"
    if "internal_test" in test_csv:
        return "internal"
    return "other"


def load_best_checkpoint_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.glob("20*/test_log.csv")):
        directory = path.parent.name
        family = family_from_name(directory)
        if family is None:
            continue
        frame = pd.read_csv(path)
        best = frame[frame["checkpoint_label"].astype(str).str.lower().eq("best")]
        if best.empty:
            best = frame[pd.to_numeric(frame["is_best_checkpoint"], errors="coerce").fillna(0).eq(1)]
        for _, source in best.iterrows():
            record = source.to_dict()
            record.update(
                {
                    "family": family,
                    "variant": variant_from_name(directory, family),
                    "directory": directory,
                    "dataset_type": split_from_row(source),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def select_family_runs(best_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = best_rows[
        [
            "family",
            "variant",
            "directory",
            "best_model_metric_name",
            "checkpoint_best_metric_value",
            "checkpoint_epoch",
        ]
    ].drop_duplicates(subset=["directory"])
    selected = []
    for family in FAMILY_ORDER:
        candidates = runs[runs["family"].eq(family)].copy()
        if candidates.empty:
            continue
        candidates["selection_value"] = pd.to_numeric(
            candidates["checkpoint_best_metric_value"], errors="coerce"
        )
        idx = candidates["selection_value"].idxmax()
        selected.append(candidates.loc[idx])
    selected_runs = pd.DataFrame(selected).reset_index(drop=True)
    selected_rows = best_rows[best_rows["directory"].isin(selected_runs["directory"])].copy()
    return selected_rows, selected_runs


def choose_metric_winners(best_rows: pd.DataFrame) -> pd.DataFrame:
    winners = []
    for split in ["internal", "external"]:
        split_rows = best_rows[best_rows["dataset_type"].eq(split)]
        for family in FAMILY_ORDER:
            family_rows = split_rows[split_rows["family"].eq(family)]
            if family_rows.empty:
                continue
            for metric in METRICS:
                column = metric["column"]
                availability = metric["n"]
                values = pd.to_numeric(family_rows.get(column), errors="coerce")
                counts = pd.to_numeric(family_rows.get(availability), errors="coerce").fillna(0)
                valid = values.notna() & counts.gt(0)
                if not valid.any():
                    continue
                idx = values[valid].idxmax()
                source = family_rows.loc[idx]
                winners.append(
                    {
                        "dataset_type": split,
                        "family": family,
                        "category": metric["category"],
                        "metric": metric["label"],
                        "column": column,
                        "value": finite(source[column]),
                        "case_std": finite(source.get(metric.get("std"))) if metric.get("std") else math.nan,
                        "n": int(finite(source[availability])),
                        "variant": source["variant"],
                        "checkpoint_epoch": int(finite(source["checkpoint_epoch"])),
                        "directory": source["directory"],
                        "test_csv": Path(str(source.get("test_csv", ""))).name,
                    }
                )
    return pd.DataFrame(winners)


def winner_row(winners: pd.DataFrame, split: str, family: str, column: str) -> pd.Series | None:
    rows = winners[
        winners["dataset_type"].eq(split)
        & winners["family"].eq(family)
        & winners["column"].eq(column)
    ]
    return None if rows.empty else rows.iloc[0]


def format_cell(row: pd.Series | None, include_source: bool = False) -> str:
    if row is None:
        return "N/A"
    if math.isfinite(finite(row["case_std"])):
        value = f"{row['value']:.4f} ± {row['case_std']:.4f}"
    else:
        value = f"{row['value']:.4f}"
    if include_source:
        value += f" ({row['variant']}, e{int(row['checkpoint_epoch'])})"
    return value


def build_wide_table(
    winners: pd.DataFrame,
    split: str,
    families: list[str],
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for family in families:
        record = {"Experiment": family}
        for column in columns:
            winner = winner_row(winners, split, family, column)
            record[METRIC_BY_COLUMN[column]["label"]] = format_cell(
                winner,
                include_source=(family == "B1"),
            )
        rows.append(record)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, winners: pd.DataFrame, split: str, columns: list[str]) -> str:
    headers = list(frame.columns)
    maxima = {}
    for column in columns:
        values = []
        for family in frame["Experiment"]:
            row = winner_row(winners, split, family, column)
            if row is not None:
                values.append(row["value"])
        maxima[column] = max(values) if values else math.nan

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for _, source in frame.iterrows():
        cells = [source["Experiment"]]
        for column, label in zip(columns, headers[1:]):
            cell = source[label]
            row = winner_row(winners, split, source["Experiment"], column)
            if row is not None and np.isclose(row["value"], maxima[column]):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    winners: pd.DataFrame,
    selected_runs: pd.DataFrame,
    internal_localization: pd.DataFrame,
    internal_clinical: pd.DataFrame,
    external_clinical: pd.DataFrame,
) -> None:
    selected_run_rows = []
    for family in FAMILY_ORDER:
        rows = selected_runs[selected_runs["family"].eq(family)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        selected_run_rows.append(
            [
                family,
                str(row["variant"]),
                str(row["best_model_metric_name"]),
                f"{finite(row['checkpoint_best_metric_value']):.4f}",
                str(int(finite(row["checkpoint_epoch"]))),
            ]
        )
    selection_table = [
        "| Family | Selected run | Validation criterion | Best validation value | Checkpoint epoch |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    selection_table.extend("| " + " | ".join(row) + " |" for row in selected_run_rows)

    lines = [
        "# Best existing result per experiment family",
        "",
        "## Selection protocol",
        "",
        "- Only each run's validation-selected `best` checkpoint is eligible; fixed test checkpoints are not searched.",
        "- One complete run is selected per experiment family using that family's configured validation criterion. All metrics in a row therefore come from the same run; test metrics are not used to assemble an oracle row.",
        "- Dice is reported as case-level mean ± SD. Other metrics are point estimates because repeated seeds or patient-level confidence intervals are unavailable.",
        "- Bold denotes the largest value in that table. All listed metrics are higher-is-better.",
        "",
        "## Selected run per family",
        "",
        *selection_table,
        "",
        "The validation criteria differ across families, so the values in the selection table are not directly comparable. B1 is the only family with multiple candidate runs; N1/B2/B3 each have one available run.",
        "",
        "## PROMIS external clinical comparison",
        "",
        "This is the fairest direct comparison because B1/N1/B2/B3 all use a PROMIS external split. Dense-lesion, ROI, and FROC targets are unavailable on this split and are therefore omitted.",
        "",
        markdown_table(
            external_clinical,
            winners,
            "external",
            TABLE_SPECS["external_clinical"],
        ),
        "",
        "## Internal localization comparison",
        "",
        "N1/B2/B3 internal split filenames differ. Their target/ROI/region counts match, but patient counts do not; use this table as a parallel reference rather than a strictly paired comparison.",
        "",
        markdown_table(
            internal_localization,
            winners,
            "internal",
            TABLE_SPECS["internal_localization"],
        ),
        "",
        "## Internal clinical comparison",
        "",
        markdown_table(
            internal_clinical,
            winners,
            "internal",
            TABLE_SPECS["internal_clinical"],
        ),
        "",
        "## Main observations",
        "",
        "- B2 provides the strongest PROMIS patient AUROC/AUPRC and patient specificity at 90% sensitivity.",
        "- N1 remains strongest on PROMIS region AUROC/AUPRC and region sensitivity at 95% specificity; B3 narrowly leads region specificity at 90% sensitivity.",
        "- B3 is strongest for internal ROI AUROC/AUPRC and target FROC at 1–2 FP/p, while N1 remains strongest for Dice and internal patient AUROC/AUPRC.",
        "- B2/B3 have no valid dense-lesion Dice/FROC counts, so those zero-valued columns must not be interpreted as measured zero performance.",
        "",
    ]
    (OUT / "best_family_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_best_rows = load_best_checkpoint_rows()
    best_rows, selected_runs = select_family_runs(all_best_rows)
    winners = choose_metric_winners(best_rows)

    internal_families = [family for family in ["N1", "B2", "B3"] if family in set(winners["family"])]
    external_families = [family for family in FAMILY_ORDER if family in set(winners["family"])]

    internal_localization = build_wide_table(
        winners,
        "internal",
        internal_families,
        TABLE_SPECS["internal_localization"],
    )
    internal_clinical = build_wide_table(
        winners,
        "internal",
        internal_families,
        TABLE_SPECS["internal_clinical"],
    )
    external_clinical = build_wide_table(
        winners,
        "external",
        external_families,
        TABLE_SPECS["external_clinical"],
    )

    winners.to_csv(OUT / "metric_winners_long.csv", index=False)
    selected_runs.to_csv(OUT / "selected_family_runs.csv", index=False)
    internal_localization.to_csv(OUT / "internal_localization_best_by_family.csv", index=False)
    internal_clinical.to_csv(OUT / "internal_clinical_best_by_family.csv", index=False)
    external_clinical.to_csv(OUT / "external_clinical_best_by_family.csv", index=False)
    write_markdown(winners, selected_runs, internal_localization, internal_clinical, external_clinical)


if __name__ == "__main__":
    main()
