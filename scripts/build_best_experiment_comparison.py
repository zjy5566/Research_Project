from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = WORKSPACE_ROOT / "result"
OUT = RESULT_ROOT / "best_experiment_comparison"
FAMILY_ORDER = ["B1", "N1", "B2", "B3"]
DISPLAY_NAMES = {
    "B1": "TBx",
    "N1": "RA",
    "B2": "SBx",
    "B3": "TBx+SBx",
}
FAMILY_OVERRIDES = {"B1": "LR5e-5_PosW1"}

METRICS = {
    "lesion_dice_mean": "test_lesion_dice_mean",
    "lesion_dice_std": "test_lesion_dice_std",
    "lesion_dice_n": "test_lesion_dice_n",
    "target_dice_mean": "test_target_cspca_dice_mean",
    "target_dice_std": "test_target_cspca_dice_std",
    "target_dice_n": "test_target_cspca_dice_n",
    "target_best_dice_mean": "test_target_cspca_best_threshold_dice_mean",
    "target_best_dice_std": "test_target_cspca_best_threshold_dice_std",
    "target_best_dice_n": "test_target_cspca_best_threshold_dice_n",
    "target_topk_dice_mean": "test_target_cspca_topk_dice_mean",
    "target_topk_dice_std": "test_target_cspca_topk_dice_std",
    "target_topk_dice_n": "test_target_cspca_topk_dice_n",
    "roi_auroc": "test_tbx_roi_auc",
    "roi_auprc": "test_tbx_roi_auprc",
    "roi_sens_fixed_spec": "test_tbx_roi_sens_at_fixed_spec",
    "roi_spec_fixed_sens": "test_tbx_roi_spec_at_fixed_sens",
    "roi_n": "test_tbx_roi_n",
    "patient_auroc": "test_patient_auc",
    "patient_auprc": "test_patient_auprc",
    "patient_sens_fixed_spec": "test_patient_sens_at_fixed_spec",
    "patient_spec_fixed_sens": "test_patient_spec_at_fixed_sens",
    "patient_n": "test_patient_n",
    "region_auroc": "test_region_auc",
    "region_auprc": "test_region_auprc",
    "region_sens_fixed_spec": "test_region_sens_at_fixed_spec",
    "region_spec_fixed_sens": "test_region_spec_at_fixed_sens",
    "region_n": "test_region_n",
    "froc_0p5": "test_target_cspca_sens_at_fp_per_patient_0p5",
    "froc_1p0": "test_target_cspca_sens_at_fp_per_patient_1p0",
    "froc_2p0": "test_target_cspca_sens_at_fp_per_patient_2p0",
    "froc_num_gt": "test_target_cspca_froc_num_gt",
}


def finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def family_from_name(name: str) -> str | None:
    match = re.search(r"_(B1|B2|B3|N1)_", name)
    return match.group(1) if match else None


def split_from_row(row: pd.Series) -> str:
    label = str(row.get("test_dataset_label", "")).strip().lower()
    if label in {"internal", "external"}:
        return label
    path = str(row.get("test_csv", "")).lower()
    if "internal_test" in path:
        return "internal"
    if "promis_external" in path:
        return "external"
    return "other"


def load_candidates() -> list[dict]:
    candidates = []
    for test_path in sorted(RESULT_ROOT.glob("20*/test_log.csv")):
        family = family_from_name(test_path.parent.name)
        if family is None:
            continue
        test = pd.read_csv(test_path)
        best = test[test["checkpoint_label"].astype(str).str.lower().eq("best")].copy()
        if best.empty:
            continue
        candidates.append(
            {
                "family": family,
                "directory": test_path.parent,
                "test": test,
                "best": best,
                "selection_metric": str(best.iloc[0]["best_model_metric_name"]),
                "selection_score": finite(best.iloc[0]["checkpoint_best_metric_value"]),
                "best_epoch": int(best.iloc[0]["checkpoint_epoch"]),
            }
        )
    return candidates


def select_best_by_family(candidates: list[dict]) -> dict[str, dict]:
    selected = {}
    for family in FAMILY_ORDER:
        family_candidates = [item for item in candidates if item["family"] == family]
        if not family_candidates:
            continue
        override = FAMILY_OVERRIDES.get(family)
        if override is not None:
            matches = [item for item in family_candidates if override in item["directory"].name]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {family} override matching {override!r}, found {len(matches)}"
                )
            selected[family] = matches[0]
            continue
        metric_names = {item["selection_metric"] for item in family_candidates}
        if len(metric_names) != 1:
            raise RuntimeError(
                f"Cannot compare {family} candidates with different selection metrics: {metric_names}"
            )
        selected[family] = max(family_candidates, key=lambda item: item["selection_score"])
    return selected


def extract_best_rows(selected: dict[str, dict]) -> pd.DataFrame:
    records = []
    for family, experiment in selected.items():
        best_rows = []
        for split in ["internal", "external"]:
            artifact_path = (
                experiment["directory"]
                / "test_artifacts"
                / split
                / "best"
                / "summary_metrics.csv"
            )
            if artifact_path.exists():
                best_rows.append(pd.read_csv(artifact_path))
                continue
            legacy_rows = experiment["best"][
                experiment["best"].apply(split_from_row, axis=1).eq(split)
            ]
            if not legacy_rows.empty:
                best_rows.append(legacy_rows)
        if not best_rows:
            best_rows = [experiment["best"]]

        for _, row in pd.concat(best_rows, ignore_index=True).iterrows():
            record = {
                "family": family,
                "model": DISPLAY_NAMES[family],
                "directory": experiment["directory"].name,
                "split": split_from_row(row),
                "test_csv": Path(str(row.get("test_csv", ""))).name,
                "best_epoch": experiment["best_epoch"],
                "selection_metric": experiment["selection_metric"],
                "selection_score": experiment["selection_score"],
            }
            for alias, column in METRICS.items():
                record[alias] = finite(row.get(column))
            records.append(record)
    return pd.DataFrame(records)


def available(row: pd.Series, metric: str) -> bool:
    availability = {
        "lesion_dice_mean": "lesion_dice_n",
        "target_dice_mean": "target_dice_n",
        "target_best_dice_mean": "target_best_dice_n",
        "target_topk_dice_mean": "target_topk_dice_n",
        "roi_auroc": "roi_n",
        "roi_auprc": "roi_n",
        "roi_sens_fixed_spec": "roi_n",
        "roi_spec_fixed_sens": "roi_n",
        "patient_auroc": "patient_n",
        "patient_auprc": "patient_n",
        "patient_sens_fixed_spec": "patient_n",
        "patient_spec_fixed_sens": "patient_n",
        "region_auroc": "region_n",
        "region_auprc": "region_n",
        "region_sens_fixed_spec": "region_n",
        "region_spec_fixed_sens": "region_n",
        "froc_0p5": "froc_num_gt",
        "froc_1p0": "froc_num_gt",
        "froc_2p0": "froc_num_gt",
    }
    count_column = availability.get(metric)
    return math.isfinite(finite(row.get(metric))) and (
        count_column is None or finite(row.get(count_column)) > 0
    )


def fmt(value: float) -> str:
    return "N/A" if not math.isfinite(finite(value)) else f"{float(value):.4f}"


def dice_fmt(row: pd.Series, prefix: str) -> str:
    mean_key = f"{prefix}_mean"
    std_key = f"{prefix}_std"
    n_key = f"{prefix}_n"
    if not available(row, mean_key):
        return "N/A"
    return f"{fmt(row[mean_key])} ± {fmt(row[std_key])} (n={int(row[n_key])})"


def bold_best(values: list[tuple[str, float]]) -> dict[str, bool]:
    finite_values = [(name, value) for name, value in values if math.isfinite(finite(value))]
    if not finite_values:
        return {name: False for name, _ in values}
    maximum = max(value for _, value in finite_values)
    return {name: math.isclose(value, maximum, rel_tol=0.0, abs_tol=5e-5) for name, value in values}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_metric_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    by_family = {row["family"]: row for _, row in frame.iterrows()}
    rows = []
    for key, label in columns:
        values = []
        for family in FAMILY_ORDER:
            row = by_family.get(family)
            value = row[key] if row is not None and available(row, key) else math.nan
            values.append((family, value))
        best_flags = bold_best(values)
        cells = [label]
        for family, value in values:
            cell = fmt(value)
            if best_flags.get(family, False) and cell != "N/A":
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(cells)
    return markdown_table(["Metric"] + [DISPLAY_NAMES[family] for family in FAMILY_ORDER], rows)


def format_dice_table(frame: pd.DataFrame) -> str:
    columns = [
        ("lesion_dice", "Lesion Dice ↑"),
        ("target_dice", "Target csPCa Dice ↑"),
        ("target_best_dice", "Best-threshold Dice ↑"),
        ("target_topk_dice", "Top-k Dice ↑"),
    ]
    by_family = {row["family"]: row for _, row in frame.iterrows()}
    rows = []
    for prefix, label in columns:
        values = []
        for family in FAMILY_ORDER:
            row = by_family.get(family)
            value = (
                row[f"{prefix}_mean"]
                if row is not None and available(row, f"{prefix}_mean")
                else math.nan
            )
            values.append((family, value))
        best_flags = bold_best(values)
        cells = [label]
        for family, _ in values:
            row = by_family.get(family)
            cell = dice_fmt(row, prefix) if row is not None else "N/A"
            if best_flags.get(family, False) and cell != "N/A":
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(cells)
    return markdown_table(["Metric"] + [DISPLAY_NAMES[family] for family in FAMILY_ORDER], rows)


def formatted_comparison_frame(
    frame: pd.DataFrame, metrics: list[tuple[str, str, str]]
) -> pd.DataFrame:
    by_family = {row["family"]: row for _, row in frame.iterrows()}
    records = []
    for key, label, kind in metrics:
        record = {"Metric": label}
        for family in FAMILY_ORDER:
            row = by_family.get(family)
            if row is None:
                value = "N/A"
            elif kind == "dice":
                value = dice_fmt(row, key)
            else:
                value = fmt(row[key] if available(row, key) else math.nan)
            record[DISPLAY_NAMES[family]] = value
        records.append(record)
    return pd.DataFrame(
        records,
        columns=["Metric"] + [DISPLAY_NAMES[family] for family in FAMILY_ORDER],
    )


def selected_inventory(selected: dict[str, dict], rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for family in FAMILY_ORDER:
        if family not in selected:
            continue
        experiment = selected[family]
        family_rows = rows[rows["family"].eq(family)]
        records.append(
            {
                "family": family,
                "model": DISPLAY_NAMES[family],
                "directory": experiment["directory"].name,
                "selection_metric": experiment["selection_metric"],
                "selection_score": experiment["selection_score"],
                "best_epoch": experiment["best_epoch"],
                "internal_test": ", ".join(
                    family_rows.loc[family_rows["split"].eq("internal"), "test_csv"].tolist()
                ),
                "external_test": ", ".join(
                    family_rows.loc[family_rows["split"].eq("external"), "test_csv"].tolist()
                ),
            }
        )
    return pd.DataFrame(records)


def write_report(selected: dict[str, dict], rows: pd.DataFrame, inventory: pd.DataFrame) -> None:
    internal = rows[rows["split"].eq("internal")].copy()
    external = rows[rows["split"].eq("external")].copy()
    order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    internal = internal.sort_values("family", key=lambda series: series.map(order))
    external = external.sort_values("family", key=lambda series: series.map(order))

    roi_columns = [
        ("roi_auroc", "ROI AUROC ↑"),
        ("roi_auprc", "ROI AUPRC ↑"),
        ("roi_sens_fixed_spec", "Sens@Spec=.95 ↑"),
        ("roi_spec_fixed_sens", "Spec@Sens=.90 ↑"),
        ("froc_0p5", "FROC@0.5 FP/p ↑"),
        ("froc_1p0", "FROC@1 FP/p ↑"),
        ("froc_2p0", "FROC@2 FP/p ↑"),
    ]
    clinical_columns = [
        ("patient_auroc", "Patient AUROC ↑"),
        ("patient_auprc", "Patient AUPRC ↑"),
        ("patient_sens_fixed_spec", "Patient Sens@Spec=.95 ↑"),
        ("patient_spec_fixed_sens", "Patient Spec@Sens=.90 ↑"),
        ("region_auroc", "Region AUROC ↑"),
        ("region_auprc", "Region AUPRC ↑"),
        ("region_sens_fixed_spec", "Region Sens@Spec=.95 ↑"),
        ("region_spec_fixed_sens", "Region Spec@Sens=.90 ↑"),
    ]

    inventory_rows = []
    for _, row in inventory.iterrows():
        inventory_rows.append(
            [
                row["model"],
                row["selection_metric"],
                fmt(row["selection_score"]),
                str(int(row["best_epoch"])),
                row["internal_test"] or "N/A",
                row["external_test"] or "N/A",
            ]
        )

    report = [
        "# Best experiment comparison: internal and external test",
        "",
        "## Selection protocol",
        "",
        "One representative is retained per experiment family: TBx = B1 `LR5e-5_PosW1`, RA = N1 Dense, SBx = B2, and TBx+SBx = B3. B1 is explicitly fixed to the user-requested configuration; families without an override use the highest validation selection score. No family is reselected separately for individual test columns. All table values come from that experiment's validation-selected `best` checkpoint.",
        "",
        markdown_table(
            ["Model", "Validation selection metric", "Score", "Epoch", "Internal test", "External test"],
            inventory_rows,
        ),
        "",
        "TBx/B1 has four candidates, and `LR5e-5_PosW1` is retained by explicit user selection (validation `tbx_roi_auprc` = 0.6866, epoch 34). RA/N1, SBx/B2, and TBx+SBx/B3 each have one candidate.",
        "",
        "## Internal test",
        "",
        "Internal test split names differ across the selected experiments. Target/ROI/region counts are 48/41,978/1,032 for all four, but patient counts differ (TBx/RA: 143; SBx/TBx+SBx: 99). Lesion Dice is currently available for TBx and RA only (n=44) because the recorded SBx and TBx+SBx tests used the TCIA-only split. This is a missing cross-evaluation, not evidence that those models cannot produce dense segmentations: their existing best checkpoints should also be tested on the same common RA+TCIA internal split used by TBx before drawing a four-model Lesion Dice conclusion.",
        "",
        "### Segmentation",
        "",
        "Dice is reported as case-level mean ± SD. The SD describes variation between test cases, not variation between checkpoints or random seeds.",
        "",
        format_dice_table(internal),
        "",
        "### ROI localisation and FROC",
        "",
        format_metric_table(internal, roi_columns),
        "",
        "### Patient and region classification",
        "",
        format_metric_table(internal, clinical_columns),
        "",
        "Internal summary: RA leads the available Dice results, but the four-model Lesion Dice comparison remains incomplete until SBx and TBx+SBx are cross-tested on the common RA+TCIA split. RA still leads all four models for the currently available target-Dice variants. TBx+SBx leads ROI AUROC/AUPRC and sensitivity at 95% specificity; TBx leads ROI specificity at 90% sensitivity. RA has the best FROC at 0.5 FP/p, while TBx and TBx+SBx tie at 1–2 FP/p. At patient level, TBx leads AUROC and specificity at 90% sensitivity, whereas RA leads AUPRC and sensitivity at 95% specificity. At region level, TBx+SBx leads AUROC; TBx leads AUPRC and sensitivity at 95% specificity, while TBx and TBx+SBx tie for specificity at 90% sensitivity.",
        "",
        "## External PROMIS test",
        "",
        "All four selected experiments use a PROMIS external split. Dense-lesion, target-ROI, and FROC ground-truth counts are zero on this split, so these metrics are `N/A` rather than true zero performance. Only patient and region metrics are compared.",
        "",
        format_metric_table(external, clinical_columns),
        "",
        "External summary: SBx is strongest for patient AUROC/AUPRC and patient specificity at 90% sensitivity; RA and SBx tie for patient sensitivity at 95% specificity. RA leads region AUROC/AUPRC and region sensitivity at 95% specificity, while TBx+SBx narrowly leads region specificity at 90% sensitivity.",
        "",
        "## Reporting notes",
        "",
        "- `AUC` in the logs is ROC-AUC, i.e. AUROC.",
        "- Non-Dice metrics are point estimates because each experiment currently has only one training run. No checkpoint-derived ± values are used in these tables.",
        "- For paper-level stability, run at least 3 independent seeds and report mean ± SD across seeds; alternatively report patient-level 95% confidence intervals for AUROC/AUPRC/sensitivity/specificity/FROC.",
        "- Bold values indicate the highest available value within that table, not statistical significance.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    selected = select_best_by_family(candidates)
    rows = extract_best_rows(selected)
    inventory = selected_inventory(selected, rows)

    inventory.to_csv(OUT / "selected_experiments.csv", index=False)
    rows.to_csv(OUT / "best_checkpoint_metrics.csv", index=False)

    internal = rows[rows["split"].eq("internal")].copy()
    external = rows[rows["split"].eq("external")].copy()
    localisation_metrics = [
        ("lesion_dice", "Lesion Dice ↑", "dice"),
        ("target_dice", "Target csPCa Dice ↑", "dice"),
        ("target_best_dice", "Best-threshold Dice ↑", "dice"),
        ("target_topk_dice", "Top-k Dice ↑", "dice"),
        ("roi_auroc", "ROI AUROC ↑", "point"),
        ("roi_auprc", "ROI AUPRC ↑", "point"),
        ("roi_sens_fixed_spec", "Sens@Spec=.95 ↑", "point"),
        ("roi_spec_fixed_sens", "Spec@Sens=.90 ↑", "point"),
        ("froc_0p5", "FROC@0.5 FP/p ↑", "point"),
        ("froc_1p0", "FROC@1 FP/p ↑", "point"),
        ("froc_2p0", "FROC@2 FP/p ↑", "point"),
    ]
    clinical_metrics = [
        ("patient_auroc", "Patient AUROC ↑", "point"),
        ("patient_auprc", "Patient AUPRC ↑", "point"),
        ("patient_sens_fixed_spec", "Patient Sens@Spec=.95 ↑", "point"),
        ("patient_spec_fixed_sens", "Patient Spec@Sens=.90 ↑", "point"),
        ("region_auroc", "Region AUROC ↑", "point"),
        ("region_auprc", "Region AUPRC ↑", "point"),
        ("region_sens_fixed_spec", "Region Sens@Spec=.95 ↑", "point"),
        ("region_spec_fixed_sens", "Region Spec@Sens=.90 ↑", "point"),
    ]
    formatted_comparison_frame(internal, localisation_metrics).to_csv(
        OUT / "internal_localisation_comparison.csv", index=False
    )
    formatted_comparison_frame(internal, clinical_metrics).to_csv(
        OUT / "internal_clinical_comparison.csv", index=False
    )
    formatted_comparison_frame(external, clinical_metrics).to_csv(
        OUT / "external_clinical_comparison.csv", index=False
    )

    write_report(selected, rows, inventory)


if __name__ == "__main__":
    main()
