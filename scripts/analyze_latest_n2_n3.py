from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE_ROOT / "result"
OUT = ROOT / "latest_n2_n3_analysis"
PRIOR_INVENTORY = ROOT / "best_experiment_comparison" / "selected_experiments.csv"

MODEL_ORDER = ["TBx", "RA", "SBx", "TBx+SBx", "RA+TBx", "RA+SBx"]
LATEST_MODELS = ["RA+TBx", "RA+SBx"]
COLORS = {
    "TBx": "#4C78A8",
    "RA": "#7A5195",
    "SBx": "#F28E2B",
    "TBx+SBx": "#E15759",
    "RA+TBx": "#00897B",
    "RA+SBx": "#C44569",
}


SEGMENTATION_METRICS = [
    {
        "label": "Lesion Dice",
        "key": "test_lesion_dice_mean",
        "std": "test_lesion_dice_std",
        "count": "test_lesion_dice_n",
        "kind": "dice",
    },
    {
        "label": "Target csPCa Dice",
        "key": "test_target_cspca_dice_mean",
        "std": "test_target_cspca_dice_std",
        "count": "test_target_cspca_dice_n",
        "kind": "dice",
    },
    {
        "label": "Best-threshold target Dice",
        "key": "test_target_cspca_best_threshold_dice_mean",
        "std": "test_target_cspca_best_threshold_dice_std",
        "count": "test_target_cspca_best_threshold_dice_n",
        "kind": "dice",
    },
    {
        "label": "Top-k target Dice",
        "key": "test_target_cspca_topk_dice_mean",
        "std": "test_target_cspca_topk_dice_std",
        "count": "test_target_cspca_topk_dice_n",
        "kind": "dice",
    },
    {
        "label": "Lesion voxel Sens@Spec=.95",
        "key": "test_lesion_voxel_sens_at_fixed_spec",
        "count": "test_lesion_voxel_n",
        "kind": "point",
    },
    {
        "label": "Lesion voxel Spec@Sens=.90",
        "key": "test_lesion_voxel_spec_at_fixed_sens",
        "count": "test_lesion_voxel_n",
        "kind": "point",
    },
    {
        "label": "Target voxel Sens@Spec=.95",
        "key": "test_target_cspca_voxel_sens_at_fixed_spec",
        "count": "test_target_cspca_voxel_n",
        "kind": "point",
    },
    {
        "label": "Target voxel Spec@Sens=.90",
        "key": "test_target_cspca_voxel_spec_at_fixed_sens",
        "count": "test_target_cspca_voxel_n",
        "kind": "point",
    },
]

LOCALISATION_METRICS = [
    {
        "label": "ROI AUROC",
        "key": "test_tbx_roi_auc",
        "count": "test_tbx_roi_n",
        "kind": "point",
    },
    {
        "label": "ROI AUPRC",
        "key": "test_tbx_roi_auprc",
        "count": "test_tbx_roi_n",
        "kind": "point",
    },
    {
        "label": "ROI Sens@Spec=.95",
        "key": "test_tbx_roi_sens_at_fixed_spec",
        "count": "test_tbx_roi_n",
        "kind": "point",
    },
    {
        "label": "ROI Spec@Sens=.90",
        "key": "test_tbx_roi_spec_at_fixed_sens",
        "count": "test_tbx_roi_n",
        "kind": "point",
    },
    {
        "label": "RA dense FROC@0.5 FP/p",
        "key": "test_lesion_sens_at_fp_per_patient_0p5",
        "actual": "test_lesion_actual_fp_per_patient_0p5",
        "target": 0.5,
        "count": "test_lesion_froc_num_gt",
        "kind": "froc",
    },
    {
        "label": "RA dense FROC@1 FP/p",
        "key": "test_lesion_sens_at_fp_per_patient_1p0",
        "actual": "test_lesion_actual_fp_per_patient_1p0",
        "target": 1.0,
        "count": "test_lesion_froc_num_gt",
        "kind": "froc",
    },
    {
        "label": "RA dense FROC@2 FP/p",
        "key": "test_lesion_sens_at_fp_per_patient_2p0",
        "actual": "test_lesion_actual_fp_per_patient_2p0",
        "target": 2.0,
        "count": "test_lesion_froc_num_gt",
        "kind": "froc",
    },
    {
        "label": "TBx-target FROC@0.5 FP/p (auxiliary)",
        "key": "test_target_cspca_sens_at_fp_per_patient_0p5",
        "actual": "test_target_cspca_actual_fp_per_patient_0p5",
        "target": 0.5,
        "count": "test_target_cspca_froc_num_gt",
        "kind": "froc",
    },
    {
        "label": "TBx-target FROC@1 FP/p (auxiliary)",
        "key": "test_target_cspca_sens_at_fp_per_patient_1p0",
        "actual": "test_target_cspca_actual_fp_per_patient_1p0",
        "target": 1.0,
        "count": "test_target_cspca_froc_num_gt",
        "kind": "froc",
    },
    {
        "label": "TBx-target FROC@2 FP/p (auxiliary)",
        "key": "test_target_cspca_sens_at_fp_per_patient_2p0",
        "actual": "test_target_cspca_actual_fp_per_patient_2p0",
        "target": 2.0,
        "count": "test_target_cspca_froc_num_gt",
        "kind": "froc",
    },
]

CLINICAL_METRICS = [
    {
        "label": "Patient AUROC",
        "key": "test_patient_auc",
        "count": "test_patient_n",
        "kind": "point",
    },
    {
        "label": "Patient AUPRC",
        "key": "test_patient_auprc",
        "count": "test_patient_n",
        "kind": "point",
    },
    {
        "label": "Patient Sens@Spec=.95",
        "key": "test_patient_sens_at_fixed_spec",
        "count": "test_patient_n",
        "kind": "point",
    },
    {
        "label": "Patient Spec@Sens=.90",
        "key": "test_patient_spec_at_fixed_sens",
        "count": "test_patient_n",
        "kind": "point",
    },
    {
        "label": "Region AUROC",
        "key": "test_region_auc",
        "count": "test_region_n",
        "kind": "point",
    },
    {
        "label": "Region AUPRC",
        "key": "test_region_auprc",
        "count": "test_region_n",
        "kind": "point",
    },
    {
        "label": "Region Sens@Spec=.95",
        "key": "test_region_sens_at_fixed_spec",
        "count": "test_region_n",
        "kind": "point",
    },
    {
        "label": "Region Spec@Sens=.90",
        "key": "test_region_spec_at_fixed_sens",
        "count": "test_region_n",
        "kind": "point",
    },
]

VAL_PEAKS = [
    ("Train loss", "train_loss_total", "min", None),
    ("Validation loss", "val_loss_total", "min", None),
    ("Lesion Dice", "val_lesion_dice_mean", "max", "val_lesion_dice_n"),
    (
        "Target csPCa Dice",
        "val_target_cspca_dice_mean",
        "max",
        "val_target_cspca_dice_n",
    ),
    (
        "Best-threshold target Dice",
        "val_target_cspca_best_threshold_dice_mean",
        "max",
        "val_target_cspca_best_threshold_dice_n",
    ),
    ("Top-k target Dice", "val_target_cspca_topk_dice_mean", "max", "val_target_cspca_topk_dice_n"),
    ("ROI AUROC", "val_tbx_roi_auc", "max", "val_tbx_roi_n"),
    ("ROI AUPRC", "val_tbx_roi_auprc", "max", "val_tbx_roi_n"),
    ("Patient AUROC", "val_patient_auc", "max", "val_patient_n"),
    ("Patient AUPRC", "val_patient_auprc", "max", "val_patient_n"),
    ("Region AUROC", "val_region_auc", "max", "val_region_n"),
    ("Region AUPRC", "val_region_auprc", "max", "val_region_n"),
]


def number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value) -> str:
    value = number(value)
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def latest_experiment(token: str) -> Path:
    pattern = re.compile(r"^20\d{6}_\d{4}_")
    candidates = [
        path
        for path in ROOT.iterdir()
        if path.is_dir()
        and pattern.match(path.name)
        and f"_{token}_" in path.name
        and (path / "train_log.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found for {token}")
    return max(candidates, key=lambda path: path.name)


def split_from_row(row: pd.Series) -> str:
    label = str(row.get("test_dataset_label", "")).strip().lower()
    if label in {"internal", "external"}:
        return label
    test_csv = str(row.get("test_csv", "")).lower()
    if "internal_test" in test_csv:
        return "internal"
    if "promis_external" in test_csv:
        return "external"
    return "other"


def selection_series(model: str, frame: pd.DataFrame) -> pd.Series:
    lesion = pd.to_numeric(frame["val_lesion_dice_mean"], errors="coerce")
    if model == "RA+TBx":
        biopsy = pd.to_numeric(frame["val_tbx_roi_auprc"], errors="coerce")
    else:
        biopsy = pd.to_numeric(frame["val_region_auprc"], errors="coerce")
    return 0.5 * lesion + 0.5 * biopsy


def selected_epoch(model: str, frame: pd.DataFrame, test: pd.DataFrame | None) -> tuple[int, float, str]:
    if test is not None:
        best = test[test["checkpoint_label"].astype(str).str.lower().eq("best")]
        if not best.empty:
            return (
                int(best.iloc[0]["checkpoint_epoch"]),
                number(best.iloc[0].get("checkpoint_best_metric_value")),
                "complete",
            )
    scores = selection_series(model, frame)
    index = scores.idxmax()
    return int(frame.loc[index, "epoch"]), number(scores.loc[index]), "interim"


def load_latest() -> dict[str, dict]:
    experiments = {}
    for token, model in [("N2", "RA+TBx"), ("N3", "RA+SBx")]:
        directory = latest_experiment(token)
        train = pd.read_csv(directory / "train_log.csv")
        test_path = directory / "test_log.csv"
        test = pd.read_csv(test_path) if test_path.is_file() else None
        epoch, score, status = selected_epoch(model, train, test)
        experiments[model] = {
            "directory": directory,
            "train": train,
            "test": test,
            "selected_epoch": epoch,
            "selection_score": score,
            "status": status,
            "last_epoch": int(pd.to_numeric(train["epoch"], errors="coerce").max()),
        }
    return experiments


def best_rows_from_experiment(directory: Path) -> list[pd.Series]:
    rows: list[pd.Series] = []
    test_path = directory / "test_log.csv"
    test = pd.read_csv(test_path) if test_path.is_file() else None
    for split in ["internal", "external"]:
        artifact = directory / "test_artifacts" / split / "best" / "summary_metrics.csv"
        if artifact.is_file():
            artifact_rows = pd.read_csv(artifact)
            if not artifact_rows.empty:
                rows.append(artifact_rows.iloc[0])
            continue
        if test is None:
            continue
        best = test[test["checkpoint_label"].astype(str).str.lower().eq("best")].copy()
        best = best[best.apply(split_from_row, axis=1).eq(split)]
        if not best.empty:
            rows.append(best.iloc[0])
    return rows


def load_best_test_table(experiments: dict[str, dict]) -> pd.DataFrame:
    records: list[dict] = []
    inventory = pd.read_csv(PRIOR_INVENTORY)
    for _, item in inventory.iterrows():
        model = str(item["model"])
        directory = ROOT / str(item["directory"])
        for row in best_rows_from_experiment(directory):
            record = {"model": model, "status": "complete", "directory": directory.name}
            record.update(row.to_dict())
            record["split"] = split_from_row(row)
            records.append(record)
    for model in LATEST_MODELS:
        experiment = experiments[model]
        if experiment["test"] is None:
            continue
        best = experiment["test"][
            experiment["test"]["checkpoint_label"].astype(str).str.lower().eq("best")
        ]
        for _, row in best.iterrows():
            record = {
                "model": model,
                "status": experiment["status"],
                "directory": experiment["directory"].name,
            }
            record.update(row.to_dict())
            record["split"] = split_from_row(row)
            records.append(record)
    return pd.DataFrame(records)


def metric_value(row: pd.Series | None, spec: dict) -> float:
    if row is None:
        return math.nan
    count = spec.get("count")
    if count and number(row.get(count)) <= 0:
        return math.nan
    if spec.get("kind") == "froc":
        actual = number(row.get(spec["actual"]))
        if not math.isfinite(actual) or actual > float(spec["target"]) + 1e-8:
            return math.nan
    return number(row.get(spec["key"]))


def dice_text(row: pd.Series | None, spec: dict) -> str:
    value = metric_value(row, spec)
    if not math.isfinite(value) or row is None:
        return "N/A"
    std = number(row.get(spec["std"]))
    count = int(number(row.get(spec["count"])))
    return f"{value:.4f} \u00b1 {std:.4f} (n={count})"


def froc_text(row: pd.Series | None, spec: dict) -> str:
    if row is None or number(row.get(spec["count"])) <= 0:
        return "N/A"
    sensitivity = number(row.get(spec["key"]))
    actual = number(row.get(spec["actual"]))
    if not math.isfinite(sensitivity) or not math.isfinite(actual):
        return "N/A"
    status = "; target unmet" if actual > float(spec["target"]) + 1e-8 else ""
    return f"{sensitivity:.4f} (actual FP/p={actual:.4f}{status})"


def best_flags(values: dict[str, float]) -> dict[str, bool]:
    available = {key: value for key, value in values.items() if math.isfinite(value)}
    if not available:
        return {key: False for key in values}
    maximum = max(available.values())
    return {
        key: math.isfinite(value) and math.isclose(value, maximum, rel_tol=0.0, abs_tol=5e-5)
        for key, value in values.items()
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def comparison_table(best_test: pd.DataFrame, split: str, specs: list[dict]) -> tuple[str, pd.DataFrame]:
    split_rows = best_test[best_test["split"].eq(split)]
    by_model = {row["model"]: row for _, row in split_rows.iterrows()}
    markdown_rows = []
    csv_rows = []
    for spec in specs:
        values = {model: metric_value(by_model.get(model), spec) for model in MODEL_ORDER}
        flags = best_flags(values)
        markdown_cells = [spec["label"] + " \u2191"]
        csv_record = {"Metric": spec["label"] + " \u2191"}
        for model in MODEL_ORDER:
            row = by_model.get(model)
            if spec["kind"] == "dice":
                text = dice_text(row, spec)
            elif spec["kind"] == "froc":
                text = froc_text(row, spec)
            else:
                text = fmt(values[model])
            csv_record[model] = text
            positive_froc = spec["kind"] != "froc" or any(
                math.isfinite(value) and value > 0 for value in values.values()
            )
            if flags[model] and positive_froc and text != "N/A":
                text = f"**{text}**"
            markdown_cells.append(text)
        markdown_rows.append(markdown_cells)
        csv_rows.append(csv_record)
    headers = ["Metric"] + MODEL_ORDER
    return markdown_table(headers, markdown_rows), pd.DataFrame(csv_rows, columns=headers)


def load_validation_selected(experiments: dict[str, dict]) -> pd.DataFrame:
    records = []
    inventory = pd.read_csv(PRIOR_INVENTORY)
    for _, item in inventory.iterrows():
        directory = ROOT / str(item["directory"])
        train = pd.read_csv(directory / "train_log.csv")
        epoch = int(item["best_epoch"])
        selected = train[pd.to_numeric(train["epoch"], errors="coerce").eq(epoch)]
        if selected.empty:
            continue
        record = {"model": str(item["model"]), "status": "complete", "selected_epoch": epoch}
        record.update(selected.iloc[0].to_dict())
        records.append(record)
    for model, experiment in experiments.items():
        selected = experiment["train"][
            pd.to_numeric(experiment["train"]["epoch"], errors="coerce").eq(
                experiment["selected_epoch"]
            )
        ]
        if selected.empty:
            continue
        record = {
            "model": model,
            "status": experiment["status"],
            "selected_epoch": experiment["selected_epoch"],
            "selection_score": experiment["selection_score"],
        }
        record.update(selected.iloc[0].to_dict())
        records.append(record)
    return pd.DataFrame(records)


def validation_comparison_table(frame: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    specs = [
        ("Lesion Dice", "val_lesion_dice_mean", "val_lesion_dice_std", "val_lesion_dice_n", "dice"),
        (
            "Target csPCa Dice",
            "val_target_cspca_dice_mean",
            "val_target_cspca_dice_std",
            "val_target_cspca_dice_n",
            "dice",
        ),
        ("ROI AUROC", "val_tbx_roi_auc", None, "val_tbx_roi_n", "point"),
        ("ROI AUPRC", "val_tbx_roi_auprc", None, "val_tbx_roi_n", "point"),
        ("Patient AUROC", "val_patient_auc", None, "val_patient_n", "point"),
        ("Patient AUPRC", "val_patient_auprc", None, "val_patient_n", "point"),
        ("Region AUROC", "val_region_auc", None, "val_region_n", "point"),
        ("Region AUPRC", "val_region_auprc", None, "val_region_n", "point"),
    ]
    by_model = {row["model"]: row for _, row in frame.iterrows()}
    md_rows = []
    csv_rows = []
    for label, key, std_key, count_key, kind in specs:
        values = {}
        for model in MODEL_ORDER:
            row = by_model.get(model)
            if row is None or number(row.get(count_key)) <= 0:
                values[model] = math.nan
            else:
                values[model] = number(row.get(key))
        flags = best_flags(values)
        md_cells = [label + " \u2191"]
        csv_record = {"Metric": label + " \u2191"}
        for model in MODEL_ORDER:
            row = by_model.get(model)
            if not math.isfinite(values[model]):
                text = "N/A"
            elif kind == "dice":
                text = (
                    f"{values[model]:.4f} \u00b1 {number(row.get(std_key)):.4f} "
                    f"(n={int(number(row.get(count_key)))})"
                )
            else:
                text = f"{values[model]:.4f}"
            csv_record[model] = text
            if flags[model] and text != "N/A":
                text = f"**{text}**"
            md_cells.append(text)
        md_rows.append(md_cells)
        csv_rows.append(csv_record)
    headers = ["Metric"] + MODEL_ORDER
    return markdown_table(headers, md_rows), pd.DataFrame(csv_rows, columns=headers)


def validation_peaks(experiments: dict[str, dict]) -> pd.DataFrame:
    records = []
    for model, experiment in experiments.items():
        frame = experiment["train"]
        for label, key, mode, count_key in VAL_PEAKS:
            values = pd.to_numeric(frame.get(key), errors="coerce")
            if count_key and count_key in frame:
                values = values.where(pd.to_numeric(frame[count_key], errors="coerce").fillna(0).gt(0))
            if values.dropna().empty:
                continue
            index = values.idxmin() if mode == "min" else values.idxmax()
            records.append(
                {
                    "model": model,
                    "status": experiment["status"],
                    "metric": label,
                    "column": key,
                    "mode": mode,
                    "value": number(values.loc[index]),
                    "epoch": int(frame.loc[index, "epoch"]),
                }
            )
        scores = selection_series(model, frame)
        index = scores.idxmax()
        records.append(
            {
                "model": model,
                "status": experiment["status"],
                "metric": "Selection composite",
                "column": "selection_composite",
                "mode": "max",
                "value": number(scores.loc[index]),
                "epoch": int(frame.loc[index, "epoch"]),
            }
        )
    return pd.DataFrame(records)


def test_checkpoint_table(experiments: dict[str, dict]) -> pd.DataFrame:
    columns = [
        "test_dataset_label",
        "checkpoint_label",
        "checkpoint_epoch",
        "test_lesion_dice_mean",
        "test_lesion_dice_std",
        "test_target_cspca_dice_mean",
        "test_target_cspca_dice_std",
        "test_target_cspca_best_threshold_dice_mean",
        "test_target_cspca_topk_dice_mean",
        "test_tbx_roi_auc",
        "test_tbx_roi_auprc",
        "test_tbx_roi_sens_at_fixed_spec",
        "test_tbx_roi_spec_at_fixed_sens",
        "test_patient_auc",
        "test_patient_auprc",
        "test_patient_sens_at_fixed_spec",
        "test_patient_spec_at_fixed_sens",
        "test_region_auc",
        "test_region_auprc",
        "test_region_sens_at_fixed_spec",
        "test_region_spec_at_fixed_sens",
        "test_lesion_sens_at_fp_per_patient_0p5",
        "test_lesion_sens_at_fp_per_patient_1p0",
        "test_lesion_sens_at_fp_per_patient_2p0",
        "test_target_cspca_sens_at_fp_per_patient_0p5",
        "test_target_cspca_sens_at_fp_per_patient_1p0",
        "test_target_cspca_sens_at_fp_per_patient_2p0",
    ]
    records = []
    for model, experiment in experiments.items():
        if experiment["test"] is None:
            continue
        for _, row in experiment["test"].iterrows():
            record = {"model": model, "split": split_from_row(row)}
            for column in columns:
                record[column] = row.get(column)
            records.append(record)
    return pd.DataFrame(records)


def checkpoint_ranges(checkpoints: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Lesion Dice", "test_lesion_dice_mean", "internal"),
        ("Target Dice", "test_target_cspca_dice_mean", "internal"),
        ("ROI AUROC", "test_tbx_roi_auc", "internal"),
        ("ROI AUPRC", "test_tbx_roi_auprc", "internal"),
        ("Patient AUROC", "test_patient_auc", "external"),
        ("Patient AUPRC", "test_patient_auprc", "external"),
        ("Region AUROC", "test_region_auc", "external"),
        ("Region AUPRC", "test_region_auprc", "external"),
    ]
    records = []
    for model in checkpoints["model"].drop_duplicates():
        for label, key, split in specs:
            rows = checkpoints[(checkpoints["model"].eq(model)) & (checkpoints["split"].eq(split))]
            values = pd.to_numeric(rows[key], errors="coerce").dropna()
            if values.empty:
                continue
            records.append(
                {
                    "model": model,
                    "split": split,
                    "metric": label,
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "range": values.max() - values.min(),
                    "num_checkpoints": len(values),
                }
            )
    return pd.DataFrame(records)


def load_font(size: int, bold: bool = False):
    names = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    )
    for path in names + ["/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def curve_data(frame: pd.DataFrame, key: str, count_key: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(frame["epoch"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if count_key and count_key in frame:
        counts = pd.to_numeric(frame[count_key], errors="coerce").fillna(0).to_numpy(dtype=float)
        valid &= counts > 0
    return x[valid], y[valid]


def dashed_segment(draw, start, end, fill, width=2, dash=8, gap=5) -> None:
    x1, y1 = start
    x2, y2 = end
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance == 0:
        return
    ux, uy = (x2 - x1) / distance, (y2 - y1) / distance
    position = 0.0
    while position < distance:
        segment_end = min(position + dash, distance)
        draw.line(
            (
                x1 + ux * position,
                y1 + uy * position,
                x1 + ux * segment_end,
                y1 + uy * segment_end,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def draw_chart_grid(path: Path, title: str, panels: list[dict], columns: int = 2) -> None:
    rows = math.ceil(len(panels) / columns)
    width = 1600
    panel_width = width // columns
    panel_height = 440
    header_height = 76
    height = header_height + rows * panel_height + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    panel_font = load_font(18, bold=True)
    font = load_font(14)
    small = load_font(12)
    draw.text((42, 22), title, fill="#20242A", font=title_font)

    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x0 = column * panel_width
        y0 = header_height + row * panel_height
        draw.text((x0 + 38, y0 + 12), panel["title"], fill="#20242A", font=panel_font)

        legend_x, legend_y = x0 + 40, y0 + 44
        for curve in panel.get("curves", []):
            color = ImageColor.getrgb(curve["color"])
            sample_end = legend_x + 26
            if curve.get("style") == "dash":
                dashed_segment(draw, (legend_x, legend_y + 7), (sample_end, legend_y + 7), color, width=3)
            else:
                draw.line((legend_x, legend_y + 7, sample_end, legend_y + 7), fill=color, width=3)
            draw.text((sample_end + 8, legend_y), curve["label"], fill="#40464D", font=small)
            item_width = 46 + int(draw.textlength(curve["label"], font=small))
            legend_x += item_width
            if legend_x > x0 + panel_width - 210:
                legend_x = x0 + 40
                legend_y += 23

        left, right = x0 + 92, x0 + panel_width - 34
        top, bottom = y0 + 106, y0 + panel_height - 58
        curves = []
        all_x: list[float] = []
        all_y: list[float] = []
        for curve in panel.get("curves", []):
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(curve["y"], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if x.size == 0:
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            curves.append((curve, x, y))
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())
        if not curves:
            draw.text((left + 180, top + 90), "No available values", fill="#777777", font=font)
            continue

        xlim = panel.get("xlim")
        xmin, xmax = xlim if xlim else (min(all_x), max(all_x))
        ylim = panel.get("ylim")
        if ylim:
            ymin, ymax = ylim
        else:
            ymin, ymax = min(all_y), max(all_y)
            margin = max((ymax - ymin) * 0.08, 0.02)
            ymin, ymax = max(0.0, ymin - margin), ymax + margin
        if math.isclose(xmin, xmax):
            xmax = xmin + 1
        if math.isclose(ymin, ymax):
            ymax = ymin + 1

        def sx(value):
            return left + (value - xmin) / (xmax - xmin) * (right - left)

        def sy(value):
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        y_ticks = panel.get("yticks", np.linspace(ymin, ymax, 5))
        for value in y_ticks:
            yy = sy(value)
            draw.line((left, yy, right, yy), fill="#D9DDE3", width=1)
            label = f"{value:.2f}"
            draw.text((left - 50, yy - 7), label, fill="#656B73", font=small)
        x_ticks = panel.get("xticks", np.linspace(xmin, xmax, 5))
        for value in x_ticks:
            xx = sx(value)
            draw.line((xx, bottom, xx, bottom + 5), fill="#60656B", width=1)
            label = f"{value:g}"
            draw.text((xx - draw.textlength(label, font=small) / 2, bottom + 10), label, fill="#656B73", font=small)
        draw.line((left, top, left, bottom), fill="#555B62", width=2)
        draw.line((left, bottom, right, bottom), fill="#555B62", width=2)

        for value, color in panel.get("vlines", []):
            if xmin <= value <= xmax:
                dashed_segment(draw, (sx(value), top), (sx(value), bottom), ImageColor.getrgb(color), width=2)
        for curve, x, y in curves:
            color = ImageColor.getrgb(curve["color"])
            points = [(sx(a), sy(b)) for a, b in zip(x, y)]
            for start, end in zip(points[:-1], points[1:]):
                if curve.get("style") == "dash":
                    dashed_segment(draw, start, end, color, width=3)
                else:
                    draw.line((*start, *end), fill=color, width=3)
            if curve.get("markers"):
                for px, py in points:
                    draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="white", width=1)
        draw.text(((left + right) / 2 - 18, bottom + 31), panel.get("xlabel", "Epoch"), fill="#555B62", font=small)
    image.save(path)


def plot_loss_curves(experiments: dict[str, dict]) -> None:
    panels = []
    for model in LATEST_MODELS:
        experiment = experiments[model]
        frame = experiment["train"]
        x, train_total = curve_data(frame, "train_loss_total")
        _, val_total = curve_data(frame, "val_loss_total")
        panels.append(
            {
                "title": f"{model}: total loss (selected epoch {experiment['selected_epoch']})",
                "curves": [
                    {"label": "Train total", "x": x, "y": train_total, "color": COLORS[model]},
                    {"label": "Validation total", "x": x, "y": val_total, "color": "#333333"},
                ],
                "vlines": [(experiment["selected_epoch"], "#D4A72C")],
                "ylabel": "Loss",
                "xticks": [1, 50, 100, 150] if model == "RA+TBx" else [1, 25, 50, 75, 100],
            }
        )
    for model in LATEST_MODELS:
        experiment = experiments[model]
        frame = experiment["train"]
        components = [
            ("train_loss_lesion_dense", "Dense RA", COLORS["RA"]),
            (
                "train_loss_lesion_sparse" if model == "RA+TBx" else "train_loss_lesion_sys",
                "TBx ROI" if model == "RA+TBx" else "SBx region",
                COLORS["TBx"] if model == "RA+TBx" else COLORS["SBx"],
            ),
            ("train_loss_lesion_outside_gland", "Outside gland", "#59A14F"),
            ("train_loss_lesion_patient", "Patient risk", "#B07AA1"),
        ]
        curves = []
        for key, label, color in components:
            x, y = curve_data(frame, key)
            curves.append({"label": label, "x": x, "y": y, "color": color})
        panels.append(
            {
                "title": f"{model}: train loss components",
                "curves": curves,
                "vlines": [(experiment["selected_epoch"], "#D4A72C")],
                "ylabel": "Loss",
                "xticks": [1, 50, 100, 150] if model == "RA+TBx" else [1, 25, 50, 75, 100],
            }
        )
    draw_chart_grid(OUT / "loss_curves.png", "Latest N2/N3 loss trajectories", panels)


def plot_metric_grid(
    experiments: dict[str, dict],
    panels: list[tuple[str, str, str | None]],
    filename: str,
    title: str,
    shape: tuple[int, int],
) -> None:
    del shape
    chart_panels = []
    for panel_title, key, count_key in panels:
        curves = []
        vlines = []
        for model in LATEST_MODELS:
            experiment = experiments[model]
            x, y = curve_data(experiment["train"], key, count_key)
            curves.append({"label": model, "x": x, "y": y, "color": COLORS[model]})
            vlines.append((experiment["selected_epoch"], COLORS[model]))
        chart_panels.append(
            {
                "title": panel_title,
                "curves": curves,
                "vlines": vlines,
                "ylim": (0, 1),
                "xticks": [1, 50, 100, 150],
            }
        )
    draw_chart_grid(OUT / filename, title, chart_panels)


def plot_fixed_operating_curves(experiments: dict[str, dict]) -> None:
    specs = [
        ("Lesion voxel", "val_lesion_voxel_sens_at_fixed_spec", "val_lesion_voxel_spec_at_fixed_sens", "val_lesion_voxel_n"),
        ("TBx ROI", "val_tbx_roi_sens_at_fixed_spec", "val_tbx_roi_spec_at_fixed_sens", "val_tbx_roi_n"),
        ("Patient", "val_patient_sens_at_fixed_spec", "val_patient_spec_at_fixed_sens", "val_patient_n"),
        ("Region", "val_region_sens_at_fixed_spec", "val_region_spec_at_fixed_sens", "val_region_n"),
    ]
    panels = []
    for title, sens_key, spec_key, count_key in specs:
        curves = []
        for model in LATEST_MODELS:
            frame = experiments[model]["train"]
            x, sens = curve_data(frame, sens_key, count_key)
            _, spec = curve_data(frame, spec_key, count_key)
            curves.extend(
                [
                    {"label": f"{model} Sens@Spec=.95", "x": x, "y": sens, "color": COLORS[model]},
                    {
                        "label": f"{model} Spec@Sens=.90",
                        "x": x,
                        "y": spec,
                        "color": COLORS[model],
                        "style": "dash",
                    },
                ]
            )
        panels.append(
            {"title": title, "curves": curves, "ylim": (0, 1), "xticks": [1, 50, 100, 150]}
        )
    draw_chart_grid(
        OUT / "fixed_operating_curves.png",
        "Validation fixed-operating-point trajectories",
        panels,
    )


def plot_internal_froc(best_test: pd.DataFrame) -> None:
    internal = best_test[best_test["split"].eq("internal")]
    by_model = {row["model"]: row for _, row in internal.iterrows()}
    specs = [
        (
            "RA dense-lesion FROC (primary)",
            "test_lesion_froc_num_gt",
            [
                "test_lesion_sens_at_fp_per_patient_0p5",
                "test_lesion_sens_at_fp_per_patient_1p0",
                "test_lesion_sens_at_fp_per_patient_2p0",
            ],
            [
                "test_lesion_actual_fp_per_patient_0p5",
                "test_lesion_actual_fp_per_patient_1p0",
                "test_lesion_actual_fp_per_patient_2p0",
            ],
        ),
        (
            "TBx-target FROC (auxiliary)",
            "test_target_cspca_froc_num_gt",
            [
                "test_target_cspca_sens_at_fp_per_patient_0p5",
                "test_target_cspca_sens_at_fp_per_patient_1p0",
                "test_target_cspca_sens_at_fp_per_patient_2p0",
            ],
            [
                "test_target_cspca_actual_fp_per_patient_0p5",
                "test_target_cspca_actual_fp_per_patient_1p0",
                "test_target_cspca_actual_fp_per_patient_2p0",
            ],
        ),
    ]
    panels = []
    for title, count_key, sensitivity_keys, actual_keys in specs:
        curves = []
        for model in MODEL_ORDER:
            row = by_model.get(model)
            if row is None or number(row.get(count_key)) <= 0:
                continue
            points = {}
            for sensitivity_key, actual_key in zip(sensitivity_keys, actual_keys):
                actual = number(row.get(actual_key))
                sensitivity = number(row.get(sensitivity_key))
                if math.isfinite(actual) and math.isfinite(sensitivity):
                    points[actual] = max(points.get(actual, 0.0), sensitivity)
            if not points:
                continue
            x = np.array(sorted(points))
            curves.append(
                {
                    "label": model,
                    "x": x,
                    "y": np.array([points[value] for value in x]),
                    "color": COLORS[model],
                    "markers": True,
                }
            )
        panels.append(
            {
                "title": title,
                "curves": curves,
                "xlim": (0.0, 2.0),
                "ylim": (0, 1.0),
                "xticks": [0.0, 0.5, 1.0, 2.0],
                "yticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                "xlabel": "Actual FP per patient",
                "ylabel": "Sensitivity",
            }
        )
    draw_chart_grid(
        OUT / "internal_froc_comparison.png",
        "Internal-test FROC using actual achieved FP/p",
        panels,
    )


def plot_test_checkpoint_curves(checkpoints: pd.DataFrame) -> None:
    n2 = checkpoints[checkpoints["model"].eq("RA+TBx")].copy()
    if n2.empty:
        return
    best_epoch = int(
        n2[n2["checkpoint_label"].astype(str).str.lower().eq("best")].iloc[0]["checkpoint_epoch"]
    )
    specs = [
        ("Internal Dice", "internal", [("Lesion Dice", "test_lesion_dice_mean"), ("Target Dice", "test_target_cspca_dice_mean")]),
        ("Internal ROI", "internal", [("ROI AUROC", "test_tbx_roi_auc"), ("ROI AUPRC", "test_tbx_roi_auprc")]),
        ("External patient", "external", [("Patient AUROC", "test_patient_auc"), ("Patient AUPRC", "test_patient_auprc")]),
        ("External region", "external", [("Region AUROC", "test_region_auc"), ("Region AUPRC", "test_region_auprc")]),
    ]
    panels = []
    line_colors = [COLORS["RA+TBx"], "#D4A72C"]
    for title, split, metrics in specs:
        rows = n2[n2["split"].eq(split)].copy().sort_values("checkpoint_epoch")
        curves = []
        for (label, key), color in zip(metrics, line_colors):
            curves.append(
                {
                    "label": label,
                    "x": pd.to_numeric(rows["checkpoint_epoch"], errors="coerce").to_numpy(),
                    "y": pd.to_numeric(rows[key], errors="coerce").to_numpy(),
                    "color": color,
                    "markers": True,
                }
            )
        panels.append(
            {
                "title": title,
                "curves": curves,
                "vlines": [(best_epoch, "#333333")],
                "ylim": (0, 1),
                "xticks": [25, 50, 75, 100, 125, 150],
            }
        )
    draw_chart_grid(
        OUT / "n2_test_checkpoint_curves.png",
        "N2 test metrics across logged checkpoints (diagnostic only)",
        panels,
    )


def test_row(best_test: pd.DataFrame, model: str, split: str) -> pd.Series | None:
    rows = best_test[(best_test["model"].eq(model)) & (best_test["split"].eq(split))]
    return None if rows.empty else rows.iloc[0]


def delta_text(current: float, baseline: float) -> str:
    return f"{current - baseline:+.4f}"


def write_report(
    experiments: dict[str, dict],
    best_test: pd.DataFrame,
    validation_table: pd.DataFrame,
    peaks: pd.DataFrame,
    ranges: pd.DataFrame,
) -> None:
    internal_seg_md, _ = comparison_table(best_test, "internal", SEGMENTATION_METRICS)
    internal_loc_md, _ = comparison_table(best_test, "internal", LOCALISATION_METRICS)
    internal_clinical_md, _ = comparison_table(best_test, "internal", CLINICAL_METRICS)
    external_clinical_md, _ = comparison_table(best_test, "external", CLINICAL_METRICS)
    validation_md, _ = validation_comparison_table(validation_table)

    status_rows = []
    for model in LATEST_MODELS:
        experiment = experiments[model]
        status_rows.append(
            [
                model,
                "Complete" if experiment["status"] == "complete" else "Training / interim",
                f"{experiment['last_epoch']}/150",
                "ra_tbx_auprc_composite" if model == "RA+TBx" else "ra_sbx_auprc_composite",
                str(experiment["selected_epoch"]),
                fmt(experiment["selection_score"]),
                "Yes" if experiment["test"] is not None else "No",
            ]
        )

    n2_internal = test_row(best_test, "RA+TBx", "internal")
    n2_external = test_row(best_test, "RA+TBx", "external")
    ra_internal = test_row(best_test, "RA", "internal")
    tbx_internal = test_row(best_test, "TBx", "internal")
    tbx_sbx_internal = test_row(best_test, "TBx+SBx", "internal")
    n3_validation = validation_table[validation_table["model"].eq("RA+SBx")].iloc[0]

    prior_models = ["TBx", "RA", "SBx", "TBx+SBx"]
    prior_external_patient = {
        model: metric_value(test_row(best_test, model, "external"), CLINICAL_METRICS[0])
        for model in prior_models
    }
    prior_external_region = {
        model: metric_value(test_row(best_test, model, "external"), CLINICAL_METRICS[4])
        for model in prior_models
    }
    best_patient_model = max(prior_external_patient, key=prior_external_patient.get)
    best_region_model = max(prior_external_region, key=prior_external_region.get)

    n2_train = experiments["RA+TBx"]["train"]
    n3_train = experiments["RA+SBx"]["train"]
    n2_last = n2_train.iloc[-1]
    n3_last = n3_train.iloc[-1]
    n2_min_val = peaks[(peaks["model"].eq("RA+TBx")) & (peaks["metric"].eq("Validation loss"))].iloc[0]
    n3_min_val = peaks[(peaks["model"].eq("RA+SBx")) & (peaks["metric"].eq("Validation loss"))].iloc[0]

    n2_range_lines = []
    for metric in ["Lesion Dice", "ROI AUROC", "Patient AUROC", "Region AUROC"]:
        row = ranges[(ranges["model"].eq("RA+TBx")) & (ranges["metric"].eq(metric))]
        if row.empty:
            continue
        item = row.iloc[0]
        n2_range_lines.append(
            f"- {metric}: {item['minimum']:.4f} to {item['maximum']:.4f} "
            f"(range {item['range']:.4f}) across {int(item['num_checkpoints'])} logged checkpoints."
        )

    report = [
        "# Latest N2/N3 experiment analysis",
        "",
        f"Snapshot: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## Experiment status",
        "",
        markdown_table(
            ["Model", "Status", "Epochs", "Selection metric", "Selected epoch", "Score", "Test log"],
            status_rows,
        ),
        "",
        "N2 (RA+TBx) is complete. N3 (RA+SBx) was still training at this snapshot, so its selected epoch and validation values are interim and it has no internal/external test results yet.",
        "",
        "Selection definitions: N2 = 0.5 x validation Lesion Dice + 0.5 x validation ROI AUPRC; N3 = 0.5 x validation Lesion Dice + 0.5 x validation Region AUPRC.",
        "",
        "## Training and validation",
        "",
        f"N2 selected epoch {experiments['RA+TBx']['selected_epoch']} with composite {fmt(experiments['RA+TBx']['selection_score'])}. Its validation loss reached a minimum of {n2_min_val['value']:.4f} at epoch {int(n2_min_val['epoch'])}, while the final validation loss was {number(n2_last['val_loss_total']):.4f}. The separation between decreasing train loss and later validation deterioration is consistent with overfitting after the useful early/mid-training window.",
        "",
        f"N3 currently selects epoch {experiments['RA+SBx']['selected_epoch']} with interim composite {fmt(experiments['RA+SBx']['selection_score'])}. Its minimum validation loss so far is {n3_min_val['value']:.4f} at epoch {int(n3_min_val['epoch'])}; at epoch {experiments['RA+SBx']['last_epoch']} it is {number(n3_last['val_loss_total']):.4f}. This trend is provisional until epoch 150 and final testing complete.",
        "",
        "### Validation-selected checkpoint comparison",
        "",
        validation_md,
        "",
        "The validation table is useful for monitoring N3, but B2/B3 used a different validation cohort in the earlier runs. Cross-family claims should therefore be based primarily on matched test cohorts, not validation peaks.",
        "",
        "## Internal test",
        "",
        "### Segmentation and voxel operating points",
        "",
        internal_seg_md,
        "",
        "### ROI localisation and FROC",
        "",
        "Primary internal lesion FROC is computed on RA cases with dense `lesion_mask` ground truth (`has_lesion`). The TBx-target rows instead use biopsy-confirmed `target_mask` regions and are retained only as an auxiliary localisation analysis; they must not replace RA dense-lesion FROC.",
        "",
        "Each FROC cell reports lesion sensitivity together with the actual achieved FP/p. `target unmet` means the logged threshold sweep had no operating point at or below the requested FP/p, so that cell is not a valid estimate at the named target.",
        "",
        internal_loc_md,
        "",
        "### Patient and region classification",
        "",
        internal_clinical_md,
        "",
        "## External PROMIS test",
        "",
        "PROMIS has no compatible dense-lesion or TBx-target masks in these logs, so Dice, ROI and FROC are unavailable. Patient and region metrics remain reportable.",
        "",
        external_clinical_md,
        "",
        "## Main findings",
        "",
        (
            f"1. N2 retains dense segmentation reasonably well but does not improve it over RA: internal Lesion Dice is "
            f"{dice_text(n2_internal, SEGMENTATION_METRICS[0])} versus {dice_text(ra_internal, SEGMENTATION_METRICS[0])} for RA "
            f"(mean delta {delta_text(metric_value(n2_internal, SEGMENTATION_METRICS[0]), metric_value(ra_internal, SEGMENTATION_METRICS[0]))})."
        ),
        (
            f"2. Adding TBx supervision improves ROI ranking over RA: N2 ROI AUROC/AUPRC = "
            f"{fmt(metric_value(n2_internal, LOCALISATION_METRICS[0]))}/{fmt(metric_value(n2_internal, LOCALISATION_METRICS[1]))}, "
            f"versus RA {fmt(metric_value(ra_internal, LOCALISATION_METRICS[0]))}/{fmt(metric_value(ra_internal, LOCALISATION_METRICS[1]))}. "
            f"However, this remains below TBx-only "
            f"{fmt(metric_value(tbx_internal, LOCALISATION_METRICS[0]))}/{fmt(metric_value(tbx_internal, LOCALISATION_METRICS[1]))} "
            f"and TBx+SBx {fmt(metric_value(tbx_sbx_internal, LOCALISATION_METRICS[0]))}/{fmt(metric_value(tbx_sbx_internal, LOCALISATION_METRICS[1]))}."
        ),
        (
            f"3. The primary RA dense-lesion FROC uses 44 RA cases and 61 ground-truth lesions. At the requested 0.5 FP/p, neither RA nor N2 reached the target in the logged sweep: RA's closest point is sensitivity "
            f"{number(ra_internal['test_lesion_sens_at_fp_per_patient_0p5']):.4f} at actual FP/p "
            f"{number(ra_internal['test_lesion_actual_fp_per_patient_0p5']):.4f}, while N2 is "
            f"{number(n2_internal['test_lesion_sens_at_fp_per_patient_0p5']):.4f} at "
            f"{number(n2_internal['test_lesion_actual_fp_per_patient_0p5']):.4f}. These must not be reported as valid FROC@0.5 values."
        ),
        (
            f"4. At the valid RA dense-lesion operating points, N2 is worse than RA under 1 FP/p "
            f"({number(n2_internal['test_lesion_sens_at_fp_per_patient_1p0']):.4f} at actual "
            f"{number(n2_internal['test_lesion_actual_fp_per_patient_1p0']):.4f} versus "
            f"{number(ra_internal['test_lesion_sens_at_fp_per_patient_1p0']):.4f} at "
            f"{number(ra_internal['test_lesion_actual_fp_per_patient_1p0']):.4f}), but better under 2 FP/p "
            f"({number(n2_internal['test_lesion_sens_at_fp_per_patient_2p0']):.4f} at actual "
            f"{number(n2_internal['test_lesion_actual_fp_per_patient_2p0']):.4f} versus "
            f"{number(ra_internal['test_lesion_sens_at_fp_per_patient_2p0']):.4f} at "
            f"{number(ra_internal['test_lesion_actual_fp_per_patient_2p0']):.4f}). N2 therefore improves only the higher-FP part of the RA FROC curve."
        ),
        (
            f"5. The separate TBx-target FROC gives N2 sensitivity "
            f"{number(n2_internal['test_target_cspca_sens_at_fp_per_patient_0p5']):.4f} at actual FP/p "
            f"{number(n2_internal['test_target_cspca_actual_fp_per_patient_0p5']):.4f}. This describes localisation inside biopsy target regions, not dense RA lesion detection, and is auxiliary only."
        ),
        (
            f"6. N2's fixed operating points show a trade-off: ROI Spec@Sens=.90 improves to "
            f"{fmt(metric_value(n2_internal, LOCALISATION_METRICS[3]))}, but ROI Sens@Spec=.95 is only "
            f"{fmt(metric_value(n2_internal, LOCALISATION_METRICS[2]))}. Its internal patient AUROC is "
            f"{fmt(metric_value(n2_internal, CLINICAL_METRICS[0]))}, indicating poor patient-level ranking despite useful localisation metrics."
        ),
        (
            f"7. N2 clinical generalisation is weak. External patient AUROC is "
            f"{fmt(metric_value(n2_external, CLINICAL_METRICS[0]))}, below the prior best {best_patient_model} "
            f"({fmt(prior_external_patient[best_patient_model])}); external region AUROC is "
            f"{fmt(metric_value(n2_external, CLINICAL_METRICS[4]))}, below the prior best {best_region_model} "
            f"({fmt(prior_external_region[best_region_model])})."
        ),
        (
            f"8. N3's interim epoch-{experiments['RA+SBx']['selected_epoch']} validation result combines Lesion Dice "
            f"{number(n3_validation['val_lesion_dice_mean']):.4f} with Patient AUROC/AUPRC "
            f"{number(n3_validation['val_patient_auc']):.4f}/{number(n3_validation['val_patient_auprc']):.4f}, "
            f"but Target Dice is only {number(n3_validation['val_target_cspca_dice_mean']):.4f} and Region AUPRC "
            f"only {number(n3_validation['val_region_auprc']):.4f}. The high patient validation score therefore cannot yet be treated as external generalisation."
        ),
        "9. No test-level conclusion is valid for N3 yet. Its validation behaviour can guide monitoring, but it must not be placed in a final internal/external results table until `test_log.csv` exists.",
        "",
        "## Checkpoint sensitivity",
        "",
        "The following ranges describe N2 test values across the logged best/periodic checkpoints. They are diagnostic ranges, not uncertainty estimates and must not be reported as mean +/- SD:",
        "",
        *n2_range_lines,
        "",
        "## Reporting notes",
        "",
        "- AUC in the logs is ROC-AUC, i.e. AUROC.",
        "- Dice is reported as case-level mean +/- SD; this SD measures variation among test cases, not checkpoints or random seeds.",
        "- Other metrics are single-run point estimates. Paper-level stability requires repeated independent seeds or confidence intervals.",
        "- FROC must be interpreted using actual achieved FP/p. A requested target is valid only when actual FP/p is less than or equal to that target; fallback points are explicitly marked `target unmet`.",
        "- Bold values are the highest available point estimate at four-decimal reporting precision, not a statistical-significance claim.",
        "- N3 figures and validation tables are a live snapshot and should be regenerated after training/testing finishes.",
        "",
        "## Figures",
        "",
        "- `loss_curves.png`",
        "- `dice_curves.png`",
        "- `auc_auprc_curves.png`",
        "- `fixed_operating_curves.png`",
        "- `internal_froc_comparison.png`",
        "- `n2_test_checkpoint_curves.png`",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = load_latest()
    best_test = load_best_test_table(experiments)
    validation_selected = load_validation_selected(experiments)
    peaks = validation_peaks(experiments)
    checkpoints = test_checkpoint_table(experiments)
    ranges = checkpoint_ranges(checkpoints)

    best_test.to_csv(OUT / "best_checkpoint_test_metrics_raw.csv", index=False)
    validation_selected.to_csv(OUT / "validation_selected_checkpoint_raw.csv", index=False)
    peaks.to_csv(OUT / "validation_metric_peaks.csv", index=False)
    checkpoints.to_csv(OUT / "test_checkpoint_metrics.csv", index=False)
    ranges.to_csv(OUT / "test_checkpoint_ranges.csv", index=False)

    for filename, split, specs in [
        ("internal_segmentation_comparison.csv", "internal", SEGMENTATION_METRICS),
        ("internal_localisation_comparison.csv", "internal", LOCALISATION_METRICS),
        ("internal_clinical_comparison.csv", "internal", CLINICAL_METRICS),
        ("external_clinical_comparison.csv", "external", CLINICAL_METRICS),
    ]:
        _, table = comparison_table(best_test, split, specs)
        table.to_csv(OUT / filename, index=False)
    _, validation_csv = validation_comparison_table(validation_selected)
    validation_csv.to_csv(OUT / "validation_selected_comparison.csv", index=False)

    plot_loss_curves(experiments)
    plot_metric_grid(
        experiments,
        [
            ("Lesion Dice", "val_lesion_dice_mean", "val_lesion_dice_n"),
            ("Target csPCa Dice", "val_target_cspca_dice_mean", "val_target_cspca_dice_n"),
            (
                "Best-threshold target Dice",
                "val_target_cspca_best_threshold_dice_mean",
                "val_target_cspca_best_threshold_dice_n",
            ),
            ("Top-k target Dice", "val_target_cspca_topk_dice_mean", "val_target_cspca_topk_dice_n"),
        ],
        "dice_curves.png",
        "Validation Dice trajectories",
        (2, 2),
    )
    plot_metric_grid(
        experiments,
        [
            ("ROI AUROC", "val_tbx_roi_auc", "val_tbx_roi_n"),
            ("ROI AUPRC", "val_tbx_roi_auprc", "val_tbx_roi_n"),
            ("Patient AUROC", "val_patient_auc", "val_patient_n"),
            ("Patient AUPRC", "val_patient_auprc", "val_patient_n"),
            ("Region AUROC", "val_region_auc", "val_region_n"),
            ("Region AUPRC", "val_region_auprc", "val_region_n"),
        ],
        "auc_auprc_curves.png",
        "Validation AUROC and AUPRC trajectories",
        (3, 2),
    )
    plot_fixed_operating_curves(experiments)
    plot_internal_froc(best_test)
    plot_test_checkpoint_curves(checkpoints)
    write_report(experiments, best_test, validation_selected, peaks, ranges)


if __name__ == "__main__":
    main()
