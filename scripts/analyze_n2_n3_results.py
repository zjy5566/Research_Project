from __future__ import annotations

import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE_ROOT / "result"
OUT = ROOT / "n2_n3_analysis"
VIS_OUT = OUT / "n2-n3-experiment-curves.html"

LABEL_RULES = [
    ("20260712_1359_B3_", "N3/B3 TBx+SBx"),
    ("20260712_1112_B2_", "N2/B2 SBx-only"),
    ("20260711_1346_N1_", "N1 dense"),
    ("LR5e-5_PosW1", "B1 LR5e-5+PosW1"),
    ("20260710_1914_B1_", "B1 PosW1"),
    ("20260710_1607_B1_", "B1 LR5e-5"),
    ("20260710_1302_B1_", "B1 Default"),
]

COLORS = {
    "N3/B3 TBx+SBx": (197, 66, 55),
    "N2/B2 SBx-only": (27, 119, 94),
    "N1 dense": (42, 92, 160),
    "B1 LR5e-5+PosW1": (122, 86, 171),
    "B1 PosW1": (221, 132, 52),
    "B1 LR5e-5": (74, 119, 161),
    "B1 Default": (105, 105, 105),
}

LATEST = ["N2/B2 SBx-only", "N3/B3 TBx+SBx"]
PRIOR = [
    "N1 dense",
    "B1 LR5e-5+PosW1",
    "B1 PosW1",
    "B1 LR5e-5",
    "B1 Default",
]
PLOT_MODELS = [
    "N3/B3 TBx+SBx",
    "N2/B2 SBx-only",
    "N1 dense",
    "B1 PosW1",
    "B1 Default",
]

VAL_METRICS = [
    ("Loss", "Validation loss", "val_loss_total", "min"),
    ("Segmentation", "Lesion Dice", "val_lesion_dice", "max"),
    ("Segmentation", "Target csPCa Dice", "val_target_cspca_dice", "max"),
    ("Segmentation", "Best-threshold target Dice", "val_target_cspca_best_threshold_dice", "max"),
    ("Voxel operating point", "Lesion sens @ spec=.95", "val_lesion_voxel_sens_at_fixed_spec", "max"),
    ("Voxel operating point", "Lesion spec @ sens=.90", "val_lesion_voxel_spec_at_fixed_sens", "max"),
    ("Voxel operating point", "Target sens @ spec=.95", "val_target_cspca_voxel_sens_at_fixed_spec", "max"),
    ("Voxel operating point", "Target spec @ sens=.90", "val_target_cspca_voxel_spec_at_fixed_sens", "max"),
    ("ROI", "ROI AUROC", "val_tbx_roi_auc", "max"),
    ("ROI", "ROI AUPRC", "val_tbx_roi_auprc", "max"),
    ("ROI", "ROI sens @ spec=.95", "val_tbx_roi_sens_at_fixed_spec", "max"),
    ("ROI", "ROI spec @ sens=.90", "val_tbx_roi_spec_at_fixed_sens", "max"),
    ("Patient", "Patient AUROC", "val_patient_auc", "max"),
    ("Patient", "Patient AUPRC", "val_patient_auprc", "max"),
    ("Patient", "Patient sens @ spec=.95", "val_patient_sens_at_fixed_spec", "max"),
    ("Patient", "Patient spec @ sens=.90", "val_patient_spec_at_fixed_sens", "max"),
    ("Region", "Region AUROC", "val_region_auc", "max"),
    ("Region", "Region AUPRC", "val_region_auprc", "max"),
    ("Region", "Region sens @ spec=.95", "val_region_sens_at_fixed_spec", "max"),
    ("Region", "Region spec @ sens=.90", "val_region_spec_at_fixed_sens", "max"),
    ("FROC", "Lesion FROC sens @ 0.5 FP/p", "val_lesion_sens_at_fp_per_patient_0p5", "max"),
    ("FROC", "Lesion FROC sens @ 1 FP/p", "val_lesion_sens_at_fp_per_patient_1p0", "max"),
    ("FROC", "Lesion FROC sens @ 2 FP/p", "val_lesion_sens_at_fp_per_patient_2p0", "max"),
    ("FROC", "Target FROC sens @ 0.5 FP/p", "val_target_cspca_sens_at_fp_per_patient_0p5", "max"),
    ("FROC", "Target FROC sens @ 1 FP/p", "val_target_cspca_sens_at_fp_per_patient_1p0", "max"),
    ("FROC", "Target FROC sens @ 2 FP/p", "val_target_cspca_sens_at_fp_per_patient_2p0", "max"),
]

TEST_METRICS = [
    "test_loss_total",
    "test_lesion_dice",
    "test_target_cspca_dice",
    "test_target_cspca_best_threshold_dice",
    "test_lesion_voxel_sens_at_fixed_spec",
    "test_lesion_voxel_spec_at_fixed_sens",
    "test_target_cspca_voxel_sens_at_fixed_spec",
    "test_target_cspca_voxel_spec_at_fixed_sens",
    "test_tbx_roi_bacc",
    "test_tbx_roi_auc",
    "test_tbx_roi_auprc",
    "test_tbx_roi_sens_at_fixed_spec",
    "test_tbx_roi_spec_at_fixed_sens",
    "test_patient_bacc",
    "test_patient_auc",
    "test_patient_auprc",
    "test_patient_sens_at_fixed_spec",
    "test_patient_spec_at_fixed_sens",
    "test_region_bacc",
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
    "test_lesion_dice_n",
    "test_target_cspca_dice_n",
    "test_tbx_roi_n",
    "test_patient_n",
    "test_region_n",
    "test_lesion_froc_n",
    "test_lesion_froc_num_gt",
    "test_target_cspca_froc_n",
    "test_target_cspca_froc_num_gt",
]

VAL_AVAILABILITY = {
    "val_lesion_dice": "val_lesion_dice_n",
    "val_target_cspca_dice": "val_target_cspca_dice_n",
    "val_target_cspca_best_threshold_dice": "val_target_cspca_best_threshold_dice_n",
    "val_lesion_voxel_sens_at_fixed_spec": "val_lesion_voxel_n",
    "val_lesion_voxel_spec_at_fixed_sens": "val_lesion_voxel_n",
    "val_target_cspca_voxel_sens_at_fixed_spec": "val_target_cspca_voxel_n",
    "val_target_cspca_voxel_spec_at_fixed_sens": "val_target_cspca_voxel_n",
    "val_tbx_roi_auc": "val_tbx_roi_n",
    "val_tbx_roi_auprc": "val_tbx_roi_n",
    "val_tbx_roi_sens_at_fixed_spec": "val_tbx_roi_n",
    "val_tbx_roi_spec_at_fixed_sens": "val_tbx_roi_n",
    "val_patient_auc": "val_patient_n",
    "val_patient_auprc": "val_patient_n",
    "val_patient_sens_at_fixed_spec": "val_patient_n",
    "val_patient_spec_at_fixed_sens": "val_patient_n",
    "val_region_auc": "val_region_n",
    "val_region_auprc": "val_region_n",
    "val_region_sens_at_fixed_spec": "val_region_n",
    "val_region_spec_at_fixed_sens": "val_region_n",
    "val_lesion_sens_at_fp_per_patient_0p5": "val_lesion_froc_num_gt",
    "val_lesion_sens_at_fp_per_patient_1p0": "val_lesion_froc_num_gt",
    "val_lesion_sens_at_fp_per_patient_2p0": "val_lesion_froc_num_gt",
    "val_target_cspca_sens_at_fp_per_patient_0p5": "val_target_cspca_froc_num_gt",
    "val_target_cspca_sens_at_fp_per_patient_1p0": "val_target_cspca_froc_num_gt",
    "val_target_cspca_sens_at_fp_per_patient_2p0": "val_target_cspca_froc_num_gt",
}

EXTERNAL_METRICS = [
    "test_patient_auc",
    "test_patient_auprc",
    "test_patient_sens_at_fixed_spec",
    "test_patient_spec_at_fixed_sens",
    "test_region_auc",
    "test_region_auprc",
    "test_region_sens_at_fixed_spec",
    "test_region_spec_at_fixed_sens",
]


def label_for(name: str) -> str:
    for token, label in LABEL_RULES:
        if token in name:
            return label
    return name


def num(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def load_experiments() -> dict[str, dict]:
    experiments = {}
    pattern = re.compile(r"^20\d{6}_\d{4}_")
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or not pattern.match(directory.name):
            continue
        train_path = directory / "train_log.csv"
        test_path = directory / "test_log.csv"
        if not train_path.exists() or not test_path.exists():
            continue
        label = label_for(directory.name)
        test = pd.read_csv(test_path)
        best = test[test["checkpoint_label"].astype(str).str.lower().eq("best")]
        if best.empty:
            best = test[test["is_best_checkpoint"].astype(str).isin(["1", "True", "true"])]
        epoch = int(best.iloc[0]["checkpoint_epoch"])
        experiments[label] = {
            "directory": directory,
            "train": pd.read_csv(train_path),
            "test": test,
            "best_test": best,
            "best_epoch": epoch,
        }
    return experiments


def dataset_type(path: str) -> str:
    lower = path.lower()
    if "internal_test" in lower:
        return "internal"
    if "promis_external" in lower:
        return "external"
    return "other"


def metric_peak(df: pd.DataFrame, column: str, mode: str) -> tuple[float, int | None]:
    if column not in df.columns:
        return math.nan, None
    series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    availability = VAL_AVAILABILITY.get(column)
    if availability in df.columns:
        series = series.where(pd.to_numeric(df[availability], errors="coerce").fillna(0).gt(0))
    if series.dropna().empty:
        return math.nan, None
    idx = series.idxmin() if mode == "min" else series.idxmax()
    return num(series.loc[idx]), int(df.loc[idx, "epoch"])


def build_tables(experiments: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    peak_rows = []
    selected_val_rows = []
    test_rows = []
    for model, exp in experiments.items():
        train = exp["train"]
        for category, metric, column, mode in VAL_METRICS:
            value, epoch = metric_peak(train, column, mode)
            peak_rows.append(
                {
                    "model": model,
                    "category": category,
                    "metric": metric,
                    "column": column,
                    "mode": mode,
                    "best_value": value,
                    "epoch": epoch,
                }
            )
        selected = train[pd.to_numeric(train["epoch"], errors="coerce").eq(exp["best_epoch"])]
        if not selected.empty:
            row = selected.iloc[0]
            record = {"model": model, "checkpoint_epoch": exp["best_epoch"]}
            for _, _, column, _ in VAL_METRICS:
                record[column] = num(row.get(column))
            for column in ["train_loss_total", "val_tbx_roi_bacc", "val_patient_bacc", "val_region_bacc"]:
                record[column] = num(row.get(column))
            record["clinical_bacc"] = np.nanmean([record["val_patient_bacc"], record["val_region_bacc"]])
            selected_val_rows.append(record)
        for _, row in exp["best_test"].iterrows():
            record = {
                "model": model,
                "checkpoint_epoch": int(row["checkpoint_epoch"]),
                "dataset_type": dataset_type(str(row.get("test_csv", ""))),
                "test_csv": Path(str(row.get("test_csv", ""))).name,
                "selection_metric": row.get("best_model_metric_name"),
                "selection_metric_value": num(row.get("checkpoint_best_metric_value")),
            }
            for column in TEST_METRICS:
                record[column] = num(row.get(column))
            record["lesion_seg_available"] = record["test_lesion_dice_n"] > 0
            record["target_seg_available"] = record["test_target_cspca_dice_n"] > 0
            record["roi_available"] = record["test_tbx_roi_n"] > 0
            record["patient_available"] = record["test_patient_n"] > 0
            record["region_available"] = record["test_region_n"] > 0
            record["lesion_froc_available"] = record["test_lesion_froc_num_gt"] > 0
            record["target_froc_available"] = record["test_target_cspca_froc_num_gt"] > 0
            test_rows.append(record)
    return pd.DataFrame(peak_rows), pd.DataFrame(selected_val_rows), pd.DataFrame(test_rows)


def compare_with_history(peaks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, metric, column, mode in VAL_METRICS:
        subset = peaks[peaks["column"].eq(column)]
        prior = subset[subset["model"].isin(PRIOR)].dropna(subset=["best_value"])
        if prior.empty:
            continue
        idx = prior["best_value"].idxmin() if mode == "min" else prior["best_value"].idxmax()
        baseline = prior.loc[idx]
        for model in LATEST:
            current = subset[subset["model"].eq(model)]
            if current.empty:
                continue
            current = current.iloc[0]
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "column": column,
                    "model": model,
                    "value": current["best_value"],
                    "epoch": current["epoch"],
                    "prior_best_model": baseline["model"],
                    "prior_best_value": baseline["best_value"],
                    "delta": current["best_value"] - baseline["best_value"],
                    "improves_prior": (
                        current["best_value"] < baseline["best_value"]
                        if mode == "min"
                        else current["best_value"] > baseline["best_value"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_external_with_history(test_table: pd.DataFrame) -> pd.DataFrame:
    external = test_table[test_table["dataset_type"].eq("external")]
    rows = []
    for metric in EXTERNAL_METRICS:
        prior = external[external["model"].isin(PRIOR)].dropna(subset=[metric])
        if prior.empty:
            continue
        baseline = prior.loc[prior[metric].idxmax()]
        for model in LATEST:
            current = external[external["model"].eq(model)]
            if current.empty:
                continue
            current = current.iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "value": current[metric],
                    "prior_best_model": baseline["model"],
                    "prior_best_value": baseline[metric],
                    "delta": current[metric] - baseline[metric],
                    "improves_prior": current[metric] > baseline[metric],
                    "patient_n": current["test_patient_n"],
                    "region_n": current["test_region_n"],
                }
            )
    return pd.DataFrame(rows)


def load_font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_small_multiples(
    path: Path,
    title: str,
    panels: list[tuple[str, str]],
    series_by_model: dict[str, pd.DataFrame],
    x_column: str = "epoch",
    x_ticks: list[float] | None = None,
) -> None:
    cols = 2
    rows = math.ceil(len(panels) / cols)
    width, panel_h = 1600, 420
    height = 120 + rows * panel_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(25)
    font = load_font(17)
    small = load_font(14)
    draw.text((55, 25), title, fill=(25, 25, 30), font=title_font)
    lx = 55
    for model in series_by_model:
        color = COLORS[model]
        draw.line((lx, 78, lx + 30, 78), fill=color, width=5)
        draw.text((lx + 38, 67), model, fill=(35, 35, 40), font=small)
        lx += 38 + int(draw.textlength(model, font=small)) + 28

    for panel_index, (panel_title, metric_column) in enumerate(panels):
        row, col = divmod(panel_index, cols)
        x0 = 45 + col * 790
        y0 = 112 + row * panel_h
        left, top, right, bottom = x0 + 72, y0 + 46, x0 + 748, y0 + 362
        draw.text((x0 + 12, y0 + 8), panel_title, fill=(25, 25, 30), font=font)
        curves = []
        all_x, all_y = [], []
        for model, frame in series_by_model.items():
            if metric_column not in frame.columns or x_column not in frame.columns:
                continue
            x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[metric_column], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            x, y = x[mask], y[mask]
            curves.append((model, x, y))
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())
        if not curves:
            draw.text((left, top), "No available values", fill=(90, 90, 90), font=small)
            continue
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        if not metric_column.endswith("loss_total"):
            ymin, ymax = 0.0, max(1.0, ymax)
        else:
            margin = max((ymax - ymin) * 0.08, 0.01)
            ymin, ymax = max(0.0, ymin - margin), ymax + margin
        if math.isclose(xmin, xmax):
            xmax = xmin + 1
        if math.isclose(ymin, ymax):
            ymax = ymin + 1

        def sx(value):
            return left + (value - xmin) / (xmax - xmin) * (right - left)

        def sy(value):
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        for i in range(5):
            value = ymin + (ymax - ymin) * i / 4
            yy = sy(value)
            draw.line((left, yy, right, yy), fill=(226, 229, 233), width=1)
            draw.text((x0 + 8, yy - 8), f"{value:.2f}", fill=(80, 80, 85), font=small)
        ticks = x_ticks or list(np.linspace(xmin, xmax, 4))
        for value in ticks:
            xx = sx(value)
            draw.line((xx, bottom, xx, bottom + 5), fill=(70, 70, 75), width=1)
            label = f"{value:g}"
            draw.text((xx - draw.textlength(label, font=small) / 2, bottom + 10), label, fill=(80, 80, 85), font=small)
        draw.line((left, top, left, bottom), fill=(60, 60, 65), width=2)
        draw.line((left, bottom, right, bottom), fill=(60, 60, 65), width=2)
        for model, x, y in curves:
            points = [(sx(a), sy(b)) for a, b in zip(x, y)]
            draw.line(points, fill=COLORS[model], width=3, joint="curve")
    image.save(path)


def mask_unavailable_validation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for metric, availability in VAL_AVAILABILITY.items():
        if metric in frame.columns and availability in frame.columns:
            mask = pd.to_numeric(frame[availability], errors="coerce").fillna(0).le(0)
            frame.loc[mask, metric] = np.nan
    return frame


def build_internal_froc_frames(test_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    for model in ["N3/B3 TBx+SBx", "N2/B2 SBx-only", "N1 dense"]:
        rows = test_table[(test_table["model"].eq(model)) & (test_table["dataset_type"].eq("internal"))]
        if rows.empty:
            continue
        row = rows.iloc[0]
        lesion_available = num(row["test_lesion_froc_num_gt"]) > 0
        target_available = num(row["test_target_cspca_froc_num_gt"]) > 0
        frames[model] = pd.DataFrame(
            {
                "fp_per_patient": [0.5, 1.0, 2.0],
                "lesion_froc": [
                    row["test_lesion_sens_at_fp_per_patient_0p5"] if lesion_available else np.nan,
                    row["test_lesion_sens_at_fp_per_patient_1p0"] if lesion_available else np.nan,
                    row["test_lesion_sens_at_fp_per_patient_2p0"] if lesion_available else np.nan,
                ],
                "target_froc": [
                    row["test_target_cspca_sens_at_fp_per_patient_0p5"] if target_available else np.nan,
                    row["test_target_cspca_sens_at_fp_per_patient_1p0"] if target_available else np.nan,
                    row["test_target_cspca_sens_at_fp_per_patient_2p0"] if target_available else np.nan,
                ],
            }
        )
    return frames


def svg_panel(
    title: str,
    metric: str,
    frames: dict[str, pd.DataFrame],
    series_classes: dict[str, str],
    x_column: str = "epoch",
) -> str:
    width, height = 340, 225
    left, top, right, bottom = 42, 30, 330, 195
    curves = []
    all_x, all_y = [], []
    for model, frame in frames.items():
        if x_column not in frame or metric not in frame:
            continue
        x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        x, y = x[mask], y[mask]
        curves.append((model, x, y))
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    if not curves:
        return ""
    xmin, xmax = min(all_x), max(all_x)
    if metric.endswith("loss_total"):
        ymin, ymax = min(all_y), max(all_y)
        margin = max((ymax - ymin) * 0.08, 0.01)
        ymin, ymax = max(0, ymin - margin), ymax + margin
    else:
        ymin, ymax = 0.0, 1.0

    def sx(value):
        return left + (value - xmin) / max(xmax - xmin, 1e-9) * (right - left)

    def sy(value):
        return bottom - (value - ymin) / max(ymax - ymin, 1e-9) * (bottom - top)

    parts = [
        f'<svg class="metric-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<text class="panel-title" x="6" y="17">{html.escape(title)}</text>',
    ]
    for i in range(5):
        value = ymin + (ymax - ymin) * i / 4
        yy = sy(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="tick" x="4" y="{yy + 3:.1f}">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    for value in [xmin, (xmin + xmax) / 2, xmax]:
        xx = sx(value)
        label = f"{value:g}" if x_column != "epoch" else f"{int(round(value))}"
        parts.append(f'<text class="tick center" x="{xx:.1f}" y="212">{label}</text>')
    for model, x, y in curves:
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
        parts.append(
            f'<polyline class="series {series_classes[model]}" points="{points}" '
            f'aria-label="{html.escape(model)}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_inline_visual(experiments: dict[str, dict], froc_frames: dict[str, pd.DataFrame]) -> None:
    models = [model for model in PLOT_MODELS if model in experiments]
    frames = {model: mask_unavailable_validation_metrics(experiments[model]["train"]) for model in models}
    classes = {model: f"series-{index + 1}" for index, model in enumerate(models)}
    froc_classes = {model: classes.get(model, f"series-{index + 1}") for index, model in enumerate(froc_frames)}
    panels = [
        ("Validation loss", "val_loss_total"),
        ("Lesion Dice", "val_lesion_dice"),
        ("Target csPCa Dice", "val_target_cspca_dice"),
        ("Lesion sens @ spec=.95", "val_lesion_voxel_sens_at_fixed_spec"),
        ("Lesion spec @ sens=.90", "val_lesion_voxel_spec_at_fixed_sens"),
        ("ROI AUROC", "val_tbx_roi_auc"),
        ("Patient AUROC", "val_patient_auc"),
        ("Region AUROC", "val_region_auc"),
    ]
    svgs = [svg_panel(title, metric, frames, classes) for title, metric in panels]
    svgs.append(svg_panel("Internal lesion FROC", "lesion_froc", froc_frames, froc_classes, "fp_per_patient"))
    svgs.append(svg_panel("Internal target FROC", "target_froc", froc_frames, froc_classes, "fp_per_patient"))
    legend = "".join(
        f'<span class="legend-item"><span class="legend-line {classes[model]}"></span>{html.escape(model)}</span>'
        for model in models
    )
    fragment = f'''<div id="n2-n3-experiment-curves">
<style>
#n2-n3-experiment-curves {{ color: var(--foreground); width: 100%; }}
#n2-n3-experiment-curves .legend {{ display: flex; flex-wrap: wrap; gap: .55rem 1rem; margin-bottom: .5rem; }}
#n2-n3-experiment-curves .legend-item {{ display: inline-flex; align-items: center; gap: .35rem; }}
#n2-n3-experiment-curves .legend-line {{ width: 1.6rem; height: 0; border-top: 3px solid; }}
#n2-n3-experiment-curves .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .35rem .65rem; }}
#n2-n3-experiment-curves .metric-chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
#n2-n3-experiment-curves .panel-title {{ fill: var(--foreground); font-weight: 500; font-size: calc(var(--font-size-base) * .82); }}
#n2-n3-experiment-curves .tick {{ fill: var(--muted-foreground); font-size: calc(var(--font-size-base) * .64); }}
#n2-n3-experiment-curves .center {{ text-anchor: middle; }}
#n2-n3-experiment-curves .axis {{ stroke: var(--muted-foreground); stroke-width: 1; }}
#n2-n3-experiment-curves .grid {{ stroke: var(--border); stroke-width: 1; }}
#n2-n3-experiment-curves .series {{ fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }}
#n2-n3-experiment-curves .series-1 {{ stroke: var(--viz-series-1); border-color: var(--viz-series-1); }}
#n2-n3-experiment-curves .series-2 {{ stroke: var(--viz-series-2); border-color: var(--viz-series-2); }}
#n2-n3-experiment-curves .series-3 {{ stroke: var(--viz-series-3); border-color: var(--viz-series-3); }}
#n2-n3-experiment-curves .series-4 {{ stroke: var(--viz-series-4); border-color: var(--viz-series-4); }}
#n2-n3-experiment-curves .series-5 {{ stroke: var(--viz-series-5); border-color: var(--viz-series-5); }}
@media (max-width: 520px) {{ #n2-n3-experiment-curves .charts {{ grid-template-columns: 1fr; }} }}
</style>
<div class="legend" aria-label="Model legend">{legend}</div>
<div class="charts">{''.join(svgs)}</div>
</div>
'''
    VIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    VIS_OUT.write_text(fragment, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments()
    peaks, selected_val, test_table = build_tables(experiments)
    history = compare_with_history(peaks)
    external_history = compare_external_with_history(test_table)

    peaks.to_csv(OUT / "validation_metric_peaks.csv", index=False)
    selected_val.to_csv(OUT / "selected_checkpoint_validation_metrics.csv", index=False)
    test_table.to_csv(OUT / "selected_checkpoint_test_metrics.csv", index=False)
    test_table[test_table["dataset_type"].eq("internal")].to_csv(
        OUT / "internal_test_selected_checkpoint.csv", index=False
    )
    test_table[test_table["dataset_type"].eq("external")].to_csv(
        OUT / "external_test_selected_checkpoint.csv", index=False
    )
    history.to_csv(OUT / "validation_comparison_vs_prior_best.csv", index=False)
    external_history.to_csv(OUT / "external_comparison_vs_prior_best.csv", index=False)

    plot_frames = {
        model: mask_unavailable_validation_metrics(experiments[model]["train"])
        for model in PLOT_MODELS
        if model in experiments
    }
    draw_small_multiples(
        OUT / "loss_curves.png",
        "Train and validation loss",
        [("Train total loss", "train_loss_total"), ("Validation total loss", "val_loss_total")],
        plot_frames,
        x_ticks=[1, 50, 100, 150],
    )
    draw_small_multiples(
        OUT / "segmentation_fixed_curves.png",
        "Validation segmentation and voxel operating points",
        [
            ("Lesion Dice", "val_lesion_dice"),
            ("Target csPCa Dice", "val_target_cspca_dice"),
            ("Best-threshold target Dice", "val_target_cspca_best_threshold_dice"),
            ("Lesion sens @ spec=.95", "val_lesion_voxel_sens_at_fixed_spec"),
            ("Lesion spec @ sens=.90", "val_lesion_voxel_spec_at_fixed_sens"),
            ("Target sens @ spec=.95", "val_target_cspca_voxel_sens_at_fixed_spec"),
            ("Target spec @ sens=.90", "val_target_cspca_voxel_spec_at_fixed_sens"),
        ],
        plot_frames,
        x_ticks=[1, 50, 100, 150],
    )
    draw_small_multiples(
        OUT / "auc_auprc_curves.png",
        "Validation AUROC (AUC) and AUPRC",
        [
            ("ROI AUROC", "val_tbx_roi_auc"),
            ("ROI AUPRC", "val_tbx_roi_auprc"),
            ("Patient AUROC", "val_patient_auc"),
            ("Patient AUPRC", "val_patient_auprc"),
            ("Region AUROC", "val_region_auc"),
            ("Region AUPRC", "val_region_auprc"),
        ],
        plot_frames,
        x_ticks=[1, 50, 100, 150],
    )
    draw_small_multiples(
        OUT / "classification_fixed_curves.png",
        "Validation classification operating points",
        [
            ("ROI sens @ spec=.95", "val_tbx_roi_sens_at_fixed_spec"),
            ("ROI spec @ sens=.90", "val_tbx_roi_spec_at_fixed_sens"),
            ("Patient sens @ spec=.95", "val_patient_sens_at_fixed_spec"),
            ("Patient spec @ sens=.90", "val_patient_spec_at_fixed_sens"),
            ("Region sens @ spec=.95", "val_region_sens_at_fixed_spec"),
            ("Region spec @ sens=.90", "val_region_spec_at_fixed_sens"),
        ],
        plot_frames,
        x_ticks=[1, 50, 100, 150],
    )
    froc_frames = build_internal_froc_frames(test_table)
    draw_small_multiples(
        OUT / "internal_test_froc_curves.png",
        "Selected-checkpoint internal-test FROC (internal sets differ)",
        [
            ("Lesion FROC sensitivity (N2/N3 unavailable)", "lesion_froc"),
            ("Target csPCa FROC sensitivity", "target_froc"),
        ],
        froc_frames,
        x_column="fp_per_patient",
        x_ticks=[0.5, 1.0, 2.0],
    )
    write_inline_visual(experiments, froc_frames)

    inventory = pd.DataFrame(
        [
            {
                "model": model,
                "directory": exp["directory"].name,
                "best_epoch": exp["best_epoch"],
                "train_rows": len(exp["train"]),
                "test_rows": len(exp["test"]),
            }
            for model, exp in experiments.items()
        ]
    )
    inventory.to_csv(OUT / "inventory.csv", index=False)
    with (OUT / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# N2/B2 and N3/B3 experiment analysis\n\n")
        handle.write("The directories requested as N2/N3 are named B2/B3 in the logs. ")
        handle.write("AUC columns are ROC-AUC/AUROC; there is no separate AUROC column.\n\n")
        handle.write("## Outputs\n\n")
        for filename in [
            "validation_metric_peaks.csv",
            "selected_checkpoint_validation_metrics.csv",
            "internal_test_selected_checkpoint.csv",
            "external_test_selected_checkpoint.csv",
            "validation_comparison_vs_prior_best.csv",
            "external_comparison_vs_prior_best.csv",
            "loss_curves.png",
            "segmentation_fixed_curves.png",
            "auc_auprc_curves.png",
            "classification_fixed_curves.png",
            "internal_test_froc_curves.png",
        ]:
            handle.write(f"- {filename}\n")


if __name__ == "__main__":
    main()
