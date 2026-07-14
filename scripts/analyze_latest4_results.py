from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE_ROOT / "result"
OUT = ROOT / "latest_vs_previous_analysis"


EXPERIMENT_LABELS = {
    "LR5e-5": "LR5e-5",
    "PosW1": "PosW1_LR1e-4",
    "LR5e-5_PosW1": "LR5e-5_PosW1",
    "Default": "Default_empty",
}


COLORS = {
    "N1_Dense_latest": (36, 120, 80),
    "LR5e-5": (36, 99, 172),
    "PosW1_LR1e-4": (219, 105, 37),
    "LR5e-5_PosW1": (110, 94, 180),
    "Default_LR1e-4": (130, 80, 155),
    "Default_empty": (130, 130, 130),
}


def label_for_dir(name: str) -> str:
    if name.startswith("20260711_1346_N1_") or "N1_PUBDenseOnly" in name:
        return "N1_Dense_latest"
    if "LR5e-5_PosW1" in name:
        return "LR5e-5_PosW1"
    if "LR5e-5" in name:
        return "LR5e-5"
    if "PosW1" in name:
        return "PosW1_LR1e-4"
    if "Default" in name:
        return "Default_LR1e-4"
    return name[:32]


def safe_float(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def best_row(df: pd.DataFrame) -> pd.Series:
    if "checkpoint_label" in df.columns:
        rows = df[df["checkpoint_label"].astype(str).str.lower() == "best"]
        if len(rows):
            if "test_lesion_dice_n" in rows.columns:
                n = pd.to_numeric(rows["test_lesion_dice_n"], errors="coerce").fillna(0)
                nonzero = rows[n > 0]
                if len(nonzero):
                    return nonzero.iloc[0]
            return rows.iloc[0]
    if "is_best_checkpoint" in df.columns:
        rows = df[df["is_best_checkpoint"].astype(str).isin(["1", "True", "true"])]
        if len(rows):
            return rows.iloc[0]
    return df.iloc[0]


def max_with_epoch(df: pd.DataFrame, col: str) -> tuple[float, int | None]:
    if col not in df.columns:
        return np.nan, None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan, None
    idx = s.idxmax()
    return float(df.loc[idx, col]), int(df.loc[idx, "epoch"]) if "epoch" in df.columns else None


def min_with_epoch(df: pd.DataFrame, col: str) -> tuple[float, int | None]:
    if col not in df.columns:
        return np.nan, None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan, None
    idx = s.idxmin()
    return float(df.loc[idx, col]), int(df.loc[idx, "epoch"]) if "epoch" in df.columns else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(fmt(value, 4))
            elif pd.isna(value):
                vals.append("-")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def load_font(size: int = 18):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_line_chart(
    path: Path,
    title: str,
    series: list[dict],
    ylabel: str,
    width: int = 1500,
    height: int = 860,
) -> None:
    font = load_font(18)
    small = load_font(15)
    title_font = load_font(24)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 95, 82, width - 280, height - 105
    draw.text((left, 24), title, fill=(20, 20, 20), font=title_font)

    clean = []
    xs_all, ys_all = [], []
    for item in series:
        xs = np.asarray(item["x"], dtype=float)
        ys = np.asarray(item["y"], dtype=float)
        mask = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        clean.append({**item, "x": xs, "y": ys})
        xs_all.extend(xs.tolist())
        ys_all.extend(ys.tolist())
    if not clean:
        draw.text((left, top), "No plottable data", fill=(60, 60, 60), font=font)
        img.save(path)
        return

    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    if math.isclose(ymin, ymax):
        ymin -= 0.05
        ymax += 0.05
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    if ymin >= 0 and min(ys_all) >= 0:
        ymin = max(0.0, ymin)

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (right - left) if xmax > xmin else left

    def sy(y):
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    axis = (55, 60, 70)
    grid = (226, 230, 235)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        py = sy(y)
        draw.line((left, py, right, py), fill=grid, width=1)
        draw.text((18, py - 9), fmt(y, 3), fill=(70, 70, 70), font=small)
    for x in np.linspace(xmin, xmax, 6):
        px = sx(x)
        draw.line((px, bottom, px, bottom + 6), fill=axis, width=1)
        draw.text((px - 14, bottom + 12), f"{int(round(x))}", fill=(70, 70, 70), font=small)
    draw.text((left + (right - left) / 2 - 24, height - 54), "epoch", fill=(50, 50, 50), font=font)
    draw.text((18, top - 34), ylabel, fill=(50, 50, 50), font=font)

    dash_patterns = {
        "solid": None,
        "dash": [10, 7],
        "dot": [3, 6],
        "dashdot": [12, 5, 3, 5],
    }

    def draw_segmented(points, color, width_px, pattern):
        if pattern is None:
            draw.line(points, fill=color, width=width_px, joint="curve")
            return
        for p0, p1 in zip(points[:-1], points[1:]):
            x0, y0 = p0
            x1, y1 = p1
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            ux, uy = dx / dist, dy / dist
            pos = 0.0
            on = True
            k = 0
            while pos < dist:
                step = pattern[k % len(pattern)]
                end = min(dist, pos + step)
                if on:
                    draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end), fill=color, width=width_px)
                on = not on
                pos = end
                k += 1

    legend_y = top
    for item in clean:
        color = COLORS.get(item["label"], (90, 90, 90))
        points = [(sx(x), sy(y)) for x, y in zip(item["x"], item["y"])]
        pattern = dash_patterns.get(item.get("style", "solid"))
        draw_segmented(points, color, 3, pattern)
        lx, ly = right + 28, legend_y
        draw.line((lx, ly + 9, lx + 36, ly + 9), fill=color, width=4)
        if pattern:
            draw.line((lx, ly + 9, lx + 36, ly + 9), fill=(255, 255, 255), width=1)
        draw.text((lx + 46, ly), item["legend"], fill=(35, 35, 35), font=small)
        legend_y += 24

    img.save(path)


def pct_delta(value: float, base: float) -> str:
    if any(math.isnan(x) for x in [value, base]) or base == 0:
        return "-"
    return f"{(value - base) * 100:+.2f} pp"


def main() -> None:
    date_prefix = re.compile(r"^20\d{6}_\d{4}_")
    dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and date_prefix.match(p.name)], key=lambda p: p.name, reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = []
    inventory = []
    for d in dirs:
        train_path = d / "train_log.csv"
        test_path = d / "test_log.csv"
        label = label_for_dir(d.name)
        record = {
            "label": label,
            "directory": str(d),
            "has_train_log": train_path.exists(),
            "has_test_log": test_path.exists(),
            "train_rows": 0,
            "test_rows": 0,
            "status": "missing logs",
        }
        if train_path.exists() and test_path.exists():
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            record.update({"train_rows": len(train), "test_rows": len(test), "status": "complete"})
            experiments.append({"label": label, "dir": d, "train": train, "test": test, "best": best_row(test)})
        inventory.append(record)
    write_csv(OUT / "latest4_inventory.csv", inventory)

    test_metrics = [
        ("ROI", "AUC/AUROC", "test_tbx_roi_auc"),
        ("ROI", "AUPRC", "test_tbx_roi_auprc"),
        ("ROI", "sens@spec=0.95", "test_tbx_roi_sens_at_fixed_spec"),
        ("ROI", "spec@sens=0.90", "test_tbx_roi_spec_at_fixed_sens"),
        ("ROI", "BACC", "test_tbx_roi_bacc"),
        ("Patient", "AUC/AUROC", "test_patient_auc"),
        ("Patient", "AUPRC", "test_patient_auprc"),
        ("Patient", "sens@spec=0.95", "test_patient_sens_at_fixed_spec"),
        ("Patient", "spec@sens=0.90", "test_patient_spec_at_fixed_sens"),
        ("Patient", "BACC", "test_patient_bacc"),
        ("Region", "AUC/AUROC", "test_region_auc"),
        ("Region", "AUPRC", "test_region_auprc"),
        ("Region", "sens@spec=0.95", "test_region_sens_at_fixed_spec"),
        ("Region", "spec@sens=0.90", "test_region_spec_at_fixed_sens"),
        ("Region", "BACC", "test_region_bacc"),
        ("Segmentation", "lesion dice", "test_lesion_dice"),
        ("Segmentation", "target csPCa dice", "test_target_cspca_dice"),
        ("FROC", "lesion sens@0.5 FP/p", "test_lesion_sens_at_fp_per_patient_0p5"),
        ("FROC", "lesion sens@1.0 FP/p", "test_lesion_sens_at_fp_per_patient_1p0"),
        ("FROC", "lesion sens@2.0 FP/p", "test_lesion_sens_at_fp_per_patient_2p0"),
        ("FROC", "target sens@0.5 FP/p", "test_target_cspca_sens_at_fp_per_patient_0p5"),
        ("FROC", "target sens@1.0 FP/p", "test_target_cspca_sens_at_fp_per_patient_1p0"),
        ("FROC", "target sens@2.0 FP/p", "test_target_cspca_sens_at_fp_per_patient_2p0"),
    ]
    test_rows = []
    for exp in experiments:
        row = exp["best"]
        base = {
            "model": exp["label"],
            "directory": exp["dir"].name,
            "checkpoint_epoch": int(row.get("checkpoint_epoch", -1)),
            "checkpoint_best_metric_value": safe_float(row.get("checkpoint_best_metric_value")),
            "test_loss_total": safe_float(row.get("test_loss_total")),
        }
        for category, metric, col in test_metrics:
            test_rows.append({**base, "category": category, "metric": metric, "column": col, "value": safe_float(row.get(col))})
    write_csv(OUT / "best_checkpoint_test_metrics_long.csv", test_rows)

    pivot_rows = []
    for category, metric, col in test_metrics:
        pr = {"category": category, "metric": metric, "column": col}
        for exp in experiments:
            pr[exp["label"]] = safe_float(exp["best"].get(col))
        pivot_rows.append(pr)
    write_csv(OUT / "best_checkpoint_test_metrics_wide.csv", pivot_rows)

    val_specs = [
        ("Loss", "min val_loss_total", "val_loss_total", "min"),
        ("Loss", "min train_loss_total", "train_loss_total", "min"),
        ("Segmentation", "max val_lesion_dice", "val_lesion_dice", "max"),
        ("Segmentation", "max val_target_cspca_dice", "val_target_cspca_dice", "max"),
        ("Segmentation", "max val_target_cspca_best_threshold_dice", "val_target_cspca_best_threshold_dice", "max"),
        ("Voxel fixed", "lesion sens@spec=0.95", "val_lesion_voxel_sens_at_fixed_spec", "max"),
        ("Voxel fixed", "lesion spec@sens=0.90", "val_lesion_voxel_spec_at_fixed_sens", "max"),
        ("Voxel fixed", "target sens@spec=0.95", "val_target_cspca_voxel_sens_at_fixed_spec", "max"),
        ("Voxel fixed", "target spec@sens=0.90", "val_target_cspca_voxel_spec_at_fixed_sens", "max"),
        ("FROC", "lesion sens@0.5 FP/p", "val_lesion_sens_at_fp_per_patient_0p5", "max"),
        ("FROC", "lesion sens@1.0 FP/p", "val_lesion_sens_at_fp_per_patient_1p0", "max"),
        ("FROC", "lesion sens@2.0 FP/p", "val_lesion_sens_at_fp_per_patient_2p0", "max"),
        ("FROC", "target sens@0.5 FP/p", "val_target_cspca_sens_at_fp_per_patient_0p5", "max"),
        ("FROC", "target sens@1.0 FP/p", "val_target_cspca_sens_at_fp_per_patient_1p0", "max"),
        ("FROC", "target sens@2.0 FP/p", "val_target_cspca_sens_at_fp_per_patient_2p0", "max"),
        ("ROI", "max val_tbx_roi_auc", "val_tbx_roi_auc", "max"),
        ("ROI", "max val_tbx_roi_auprc", "val_tbx_roi_auprc", "max"),
        ("ROI", "max val_tbx_roi_sens@spec=0.95", "val_tbx_roi_sens_at_fixed_spec", "max"),
        ("ROI", "max val_tbx_roi_spec@sens=0.90", "val_tbx_roi_spec_at_fixed_sens", "max"),
        ("Patient", "max val_patient_auc", "val_patient_auc", "max"),
        ("Patient", "max val_patient_auprc", "val_patient_auprc", "max"),
        ("Patient", "max val_patient_sens@spec=0.95", "val_patient_sens_at_fixed_spec", "max"),
        ("Patient", "max val_patient_spec@sens=0.90", "val_patient_spec_at_fixed_sens", "max"),
        ("Region", "max val_region_auc", "val_region_auc", "max"),
        ("Region", "max val_region_auprc", "val_region_auprc", "max"),
        ("Region", "max val_region_sens@spec=0.95", "val_region_sens_at_fixed_spec", "max"),
        ("Region", "max val_region_spec@sens=0.90", "val_region_spec_at_fixed_sens", "max"),
    ]
    val_rows = []
    for exp in experiments:
        for category, metric, col, mode in val_specs:
            value, epoch = (min_with_epoch if mode == "min" else max_with_epoch)(exp["train"], col)
            val_rows.append({
                "model": exp["label"],
                "category": category,
                "metric": metric,
                "column": col,
                "mode": mode,
                "best_value": value,
                "epoch": epoch,
            })
    write_csv(OUT / "validation_best_metrics_long.csv", val_rows)

    val_wide = []
    for category, metric, col, mode in val_specs:
        pr = {"category": category, "metric": metric, "column": col, "mode": mode}
        for exp in experiments:
            value, epoch = (min_with_epoch if mode == "min" else max_with_epoch)(exp["train"], col)
            pr[exp["label"]] = value
            pr[exp["label"] + "_epoch"] = epoch
        val_wide.append(pr)
    write_csv(OUT / "validation_best_metrics_wide.csv", val_wide)

    # Test checkpoint trend table.
    checkpoint_rows = []
    ckpt_cols = [
        "test_tbx_roi_auc", "test_tbx_roi_auprc", "test_tbx_roi_sens_at_fixed_spec", "test_tbx_roi_spec_at_fixed_sens",
        "test_patient_auc", "test_patient_auprc", "test_patient_sens_at_fixed_spec", "test_patient_spec_at_fixed_sens",
        "test_region_auc", "test_region_auprc", "test_region_sens_at_fixed_spec", "test_region_spec_at_fixed_sens",
    ]
    for exp in experiments:
        for _, row in exp["test"].iterrows():
            out = {"model": exp["label"], "checkpoint_label": row.get("checkpoint_label"), "checkpoint_epoch": row.get("checkpoint_epoch")}
            for col in ckpt_cols:
                out[col] = safe_float(row.get(col))
            checkpoint_rows.append(out)
    write_csv(OUT / "test_checkpoint_trends.csv", checkpoint_rows)

    # Charts.
    loss_series = []
    for exp in experiments:
        x = pd.to_numeric(exp["train"]["epoch"], errors="coerce")
        for col, suffix, style in [
            ("train_loss_total", "train", "solid"),
            ("val_loss_total", "val", "dash"),
        ]:
            if col in exp["train"].columns:
                loss_series.append({
                    "label": exp["label"],
                    "legend": f"{exp['label']} {suffix}",
                    "x": x,
                    "y": pd.to_numeric(exp["train"][col], errors="coerce"),
                    "style": style,
                })
    draw_line_chart(OUT / "loss_curves.png", "Training and validation total loss", loss_series, "loss")

    def metric_chart(filename: str, title: str, specs: list[tuple[str, str, str]]):
        series = []
        for exp in experiments:
            x = pd.to_numeric(exp["train"]["epoch"], errors="coerce")
            for col, short, style in specs:
                if col in exp["train"].columns:
                    series.append({
                        "label": exp["label"],
                        "legend": f"{exp['label']} {short}",
                        "x": x,
                        "y": pd.to_numeric(exp["train"][col], errors="coerce"),
                        "style": style,
                    })
        draw_line_chart(OUT / filename, title, series, "metric")

    metric_chart(
        "roi_metric_curves.png",
        "Validation ROI metrics",
        [
            ("val_tbx_roi_auc", "AUC", "solid"),
            ("val_tbx_roi_auprc", "AUPRC", "dash"),
            ("val_tbx_roi_sens_at_fixed_spec", "sens@spec=.95", "dot"),
            ("val_tbx_roi_spec_at_fixed_sens", "spec@sens=.90", "dashdot"),
        ],
    )
    metric_chart(
        "patient_metric_curves.png",
        "Validation patient metrics",
        [
            ("val_patient_auc", "AUC", "solid"),
            ("val_patient_auprc", "AUPRC", "dash"),
            ("val_patient_sens_at_fixed_spec", "sens@spec=.95", "dot"),
            ("val_patient_spec_at_fixed_sens", "spec@sens=.90", "dashdot"),
        ],
    )
    metric_chart(
        "segmentation_froc_curves.png",
        "Validation segmentation, voxel operating point, and FROC",
        [
            ("val_lesion_dice", "lesion dice", "solid"),
            ("val_target_cspca_dice", "target dice", "dash"),
            ("val_lesion_voxel_sens_at_fixed_spec", "les sens@spec", "dot"),
            ("val_lesion_voxel_spec_at_fixed_sens", "les spec@sens", "dashdot"),
            ("val_lesion_sens_at_fp_per_patient_1p0", "les froc@1", "solid"),
            ("val_target_cspca_sens_at_fp_per_patient_1p0", "target froc@1", "dash"),
        ],
    )
    metric_chart(
        "region_metric_curves.png",
        "Validation region metrics",
        [
            ("val_region_auc", "AUC", "solid"),
            ("val_region_auprc", "AUPRC", "dash"),
            ("val_region_sens_at_fixed_spec", "sens@spec=.95", "dot"),
            ("val_region_spec_at_fixed_sens", "spec@sens=.90", "dashdot"),
        ],
    )

    # Compact markdown report for quick inspection.
    best_test = pd.DataFrame(test_rows).pivot_table(index=["category", "metric", "column"], columns="model", values="value", aggfunc="first").reset_index()
    val_best = pd.DataFrame(val_wide)
    with (OUT / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Latest vs previous result analysis\n\n")
        f.write("All timestamped experiment directories were selected by the timestamp prefix in the directory name. The latest experiment is N1_Dense_latest; previous experiments are the four B1 runs from 20260710.\n\n")
        f.write("## Inventory\n\n")
        f.write(markdown_table(pd.DataFrame(inventory)))
        f.write("\n\n## Best checkpoint test metrics\n\n")
        f.write(markdown_table(best_test))
        f.write("\n\n## Validation best metrics\n\n")
        f.write(markdown_table(val_best))
        f.write("\n\n## Charts\n\n")
        for name in [
            "loss_curves.png",
            "roi_metric_curves.png",
            "patient_metric_curves.png",
            "segmentation_froc_curves.png",
            "region_metric_curves.png",
        ]:
            f.write(f"- {name}\n")


if __name__ == "__main__":
    main()
