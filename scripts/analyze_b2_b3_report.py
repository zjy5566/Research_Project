from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_n2_n3_results import (
    COLORS,
    PRIOR,
    dataset_type,
    draw_small_multiples,
    load_experiments,
    mask_unavailable_validation_metrics,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUT = WORKSPACE_ROOT / "result" / "b2_b3_analysis"
LATEST = ["N2/B2 SBx-only", "N3/B3 TBx+SBx"]
INTERNAL_REFERENCE = "N1 dense"
DISPLAY = {
    "N2/B2 SBx-only": "B2 SBx-only",
    "N3/B3 TBx+SBx": "B3 TBx+SBx",
    "N1 dense": "N1 dense",
    "B1 LR5e-5+PosW1": "B1 LR5e-5+PosW1",
    "B1 PosW1": "B1 PosW1",
    "B1 LR5e-5": "B1 LR5e-5",
    "B1 Default": "B1 Default",
}


METRICS = [
    ("Loss", "Test loss", "test_loss_total", None),
    ("Segmentation", "Lesion Dice", "test_lesion_dice_mean", "test_lesion_dice_n"),
    ("Segmentation", "Target csPCa Dice", "test_target_cspca_dice_mean", "test_target_cspca_dice_n"),
    (
        "Segmentation",
        "Target Dice (best threshold)",
        "test_target_cspca_best_threshold_dice_mean",
        "test_target_cspca_best_threshold_dice_n",
    ),
    (
        "Segmentation",
        "Target Dice (top-k)",
        "test_target_cspca_topk_dice_mean",
        "test_target_cspca_topk_dice_n",
    ),
    (
        "Voxel operating point",
        "Target sens @ spec=.95",
        "test_target_cspca_voxel_sens_at_fixed_spec",
        "test_target_cspca_voxel_n",
    ),
    (
        "Voxel operating point",
        "Target spec @ sens=.90",
        "test_target_cspca_voxel_spec_at_fixed_sens",
        "test_target_cspca_voxel_n",
    ),
    ("ROI", "ROI AUROC", "test_tbx_roi_auc", "test_tbx_roi_n"),
    ("ROI", "ROI AUPRC", "test_tbx_roi_auprc", "test_tbx_roi_n"),
    (
        "ROI",
        "ROI sens @ spec=.95",
        "test_tbx_roi_sens_at_fixed_spec",
        "test_tbx_roi_n",
    ),
    (
        "ROI",
        "ROI spec @ sens=.90",
        "test_tbx_roi_spec_at_fixed_sens",
        "test_tbx_roi_n",
    ),
    ("Patient", "Patient AUROC", "test_patient_auc", "test_patient_n"),
    ("Patient", "Patient AUPRC", "test_patient_auprc", "test_patient_n"),
    (
        "Patient",
        "Patient sens @ spec=.95",
        "test_patient_sens_at_fixed_spec",
        "test_patient_n",
    ),
    (
        "Patient",
        "Patient spec @ sens=.90",
        "test_patient_spec_at_fixed_sens",
        "test_patient_n",
    ),
    ("Region", "Region AUROC", "test_region_auc", "test_region_n"),
    ("Region", "Region AUPRC", "test_region_auprc", "test_region_n"),
    (
        "Region",
        "Region sens @ spec=.95",
        "test_region_sens_at_fixed_spec",
        "test_region_n",
    ),
    (
        "Region",
        "Region spec @ sens=.90",
        "test_region_spec_at_fixed_sens",
        "test_region_n",
    ),
    (
        "FROC",
        "Target FROC sens @ 0.5 FP/p",
        "test_target_cspca_sens_at_fp_per_patient_0p5",
        "test_target_cspca_froc_num_gt",
    ),
    (
        "FROC",
        "Target FROC sens @ 1.0 FP/p",
        "test_target_cspca_sens_at_fp_per_patient_1p0",
        "test_target_cspca_froc_num_gt",
    ),
    (
        "FROC",
        "Target FROC sens @ 2.0 FP/p",
        "test_target_cspca_sens_at_fp_per_patient_2p0",
        "test_target_cspca_froc_num_gt",
    ),
]

METRIC_META = {column: (category, label, availability) for category, label, column, availability in METRICS}

DICE_STD_COLUMNS = {
    "test_lesion_dice_mean": "test_lesion_dice_std",
    "test_target_cspca_dice_mean": "test_target_cspca_dice_std",
    "test_target_cspca_best_threshold_dice_mean": "test_target_cspca_best_threshold_dice_std",
    "test_target_cspca_topk_dice_mean": "test_target_cspca_topk_dice_std",
}

NUMERIC_COLUMNS = sorted(
    {
        column
        for _, _, column, availability in METRICS
        for column in [column, availability]
        if column is not None
    }
    | set(DICE_STD_COLUMNS.values())
    | {
        "checkpoint_epoch",
        "checkpoint_best_metric_value",
        "test_target_cspca_dice_n",
        "test_target_cspca_best_threshold_dice_n",
        "test_target_cspca_topk_dice_n",
        "test_target_cspca_voxel_n",
        "test_tbx_roi_n",
        "test_patient_n",
        "test_region_n",
        "test_lesion_dice_n",
        "test_lesion_froc_num_gt",
        "test_target_cspca_froc_num_gt",
    }
)

EXTERNAL_COLUMNS = [
    "test_patient_auc",
    "test_patient_auprc",
    "test_patient_sens_at_fixed_spec",
    "test_patient_spec_at_fixed_sens",
    "test_region_auc",
    "test_region_auprc",
    "test_region_sens_at_fixed_spec",
    "test_region_spec_at_fixed_sens",
]

INTERNAL_COLUMNS = [
    "test_target_cspca_dice_mean",
    "test_target_cspca_best_threshold_dice_mean",
    "test_target_cspca_topk_dice_mean",
    "test_target_cspca_voxel_sens_at_fixed_spec",
    "test_target_cspca_voxel_spec_at_fixed_sens",
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
    "test_target_cspca_sens_at_fp_per_patient_0p5",
    "test_target_cspca_sens_at_fp_per_patient_1p0",
    "test_target_cspca_sens_at_fp_per_patient_2p0",
]


def finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def build_checkpoint_table(experiments: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, experiment in experiments.items():
        for _, source in experiment["test"].iterrows():
            record = {
                "model": model,
                "dataset_type": dataset_type(str(source.get("test_csv", ""))),
                "test_csv": Path(str(source.get("test_csv", ""))).name,
                "checkpoint_label": str(source.get("checkpoint_label", "")),
                "is_best_checkpoint": str(source.get("is_best_checkpoint", "")),
                "best_model_metric_name": str(source.get("best_model_metric_name", "")),
            }
            for column in NUMERIC_COLUMNS:
                record[column] = finite(source.get(column))
            rows.append(record)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["model", "dataset_type", "checkpoint_epoch", "checkpoint_label"])


def valid_metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    availability = METRIC_META[metric][2]
    valid = pd.to_numeric(frame[metric], errors="coerce").notna()
    if availability is not None:
        valid &= pd.to_numeric(frame[availability], errors="coerce").fillna(0).gt(0)
    return frame.loc[valid]


def build_stability_table(checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), group in checkpoints.groupby(["model", "dataset_type"], sort=True):
        best_rows = group[group["checkpoint_label"].str.lower().eq("best")]
        best = best_rows.iloc[0] if not best_rows.empty else group.iloc[0]
        for category, label, metric, availability in METRICS:
            valid = valid_metric_rows(group, metric)
            if valid.empty:
                continue
            values = pd.to_numeric(valid[metric], errors="coerce").dropna()
            best_available = availability is None or finite(best.get(availability)) > 0
            record = {
                "model": model,
                "dataset_type": split,
                "category": category,
                "metric": label,
                "column": metric,
                "checkpoint_n": int(values.size),
                "checkpoint_mean": float(values.mean()),
                "checkpoint_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "checkpoint_min": float(values.min()),
                "checkpoint_max": float(values.max()),
                "best_checkpoint_epoch": int(best["checkpoint_epoch"]),
                "best_checkpoint_value": finite(best.get(metric)) if best_available else math.nan,
                "best_case_std": finite(best.get(DICE_STD_COLUMNS.get(metric))) if metric in DICE_STD_COLUMNS else math.nan,
                "best_case_n": int(finite(best.get(availability))) if availability and best_available else math.nan,
            }
            rows.append(record)
    return pd.DataFrame(rows)


def learning_dynamics(experiments: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model in LATEST:
        experiment = experiments[model]
        train = experiment["train"].copy()
        selected_epoch = experiment["best_epoch"]
        selected = train.loc[pd.to_numeric(train["epoch"], errors="coerce").eq(selected_epoch)].iloc[0]
        min_idx = pd.to_numeric(train["val_loss_total"], errors="coerce").idxmin()
        sys_enabled = pd.to_numeric(train["lesion_sys_enabled_this_epoch"], errors="coerce").fillna(0)
        activation_rows = train.loc[sys_enabled.gt(0), "epoch"]
        activation_epoch = int(activation_rows.iloc[0]) if not activation_rows.empty else 1
        comparable = train[pd.to_numeric(train["epoch"], errors="coerce").ge(activation_epoch)]
        post_idx = pd.to_numeric(comparable["val_loss_total"], errors="coerce").idxmin()
        final = train.iloc[-1]
        rows.append(
            {
                "model": model,
                "selection_metric": str(experiment["test"].iloc[0]["best_model_metric_name"]),
                "selection_metric_value": finite(experiment["test"].iloc[0]["checkpoint_best_metric_value"]),
                "selected_epoch": selected_epoch,
                "selected_train_loss": finite(selected["train_loss_total"]),
                "selected_val_loss": finite(selected["val_loss_total"]),
                "minimum_val_loss": finite(train.loc[min_idx, "val_loss_total"]),
                "minimum_val_loss_epoch": int(train.loc[min_idx, "epoch"]),
                "all_active_epoch": activation_epoch,
                "post_activation_min_val_loss": finite(train.loc[post_idx, "val_loss_total"]),
                "post_activation_min_epoch": int(train.loc[post_idx, "epoch"]),
                "final_train_loss": finite(final["train_loss_total"]),
                "final_val_loss": finite(final["val_loss_total"]),
                "final_vs_post_min_pct": 100.0
                * (finite(final["val_loss_total"]) / finite(train.loc[post_idx, "val_loss_total"]) - 1.0),
            }
        )
    return pd.DataFrame(rows)


def stability_row(stability: pd.DataFrame, model: str, split: str, metric: str) -> pd.Series | None:
    rows = stability[
        stability["model"].eq(model)
        & stability["dataset_type"].eq(split)
        & stability["column"].eq(metric)
    ]
    return None if rows.empty else rows.iloc[0]


def fmt(value: float, digits: int = 4) -> str:
    return "N/A" if not math.isfinite(finite(value)) else f"{float(value):.{digits}f}"


def fmt_delta(value: float) -> str:
    if not math.isfinite(finite(value)):
        return "N/A"
    return f"{float(value):+.4f}"


def best_and_stability(stability: pd.DataFrame, model: str, split: str, metric: str) -> str:
    row = stability_row(stability, model, split, metric)
    if row is None:
        return "N/A"
    return (
        f"{fmt(row['best_checkpoint_value'])}; "
        f"{fmt(row['checkpoint_mean'])} ± {fmt(row['checkpoint_std'])}"
    )


def dice_cell(stability: pd.DataFrame, model: str, metric: str) -> str:
    row = stability_row(stability, model, "internal", metric)
    if row is None or not math.isfinite(finite(row["best_checkpoint_value"])):
        return "N/A"
    return (
        f"{fmt(row['best_checkpoint_value'])} ± {fmt(row['best_case_std'])} "
        f"(case n={int(row['best_case_n'])}); ckpt "
        f"{fmt(row['checkpoint_mean'])} ± {fmt(row['checkpoint_std'])}"
    )


def external_comparison(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in EXTERNAL_COLUMNS:
        candidates = []
        for model in PRIOR:
            row = stability_row(stability, model, "external", metric)
            if row is not None and math.isfinite(finite(row["best_checkpoint_value"])):
                candidates.append(row)
        prior_best = max(candidates, key=lambda item: item["best_checkpoint_value"])
        record = {
            "category": METRIC_META[metric][0],
            "metric": METRIC_META[metric][1],
            "prior_best_model": prior_best["model"],
            "prior_best_value": prior_best["best_checkpoint_value"],
            "prior_checkpoint_mean": prior_best["checkpoint_mean"],
            "prior_checkpoint_std": prior_best["checkpoint_std"],
        }
        for model in LATEST:
            current = stability_row(stability, model, "external", metric)
            record[f"{model}_best"] = current["best_checkpoint_value"]
            record[f"{model}_checkpoint_mean"] = current["checkpoint_mean"]
            record[f"{model}_checkpoint_std"] = current["checkpoint_std"]
            record[f"{model}_delta_vs_prior_best"] = (
                current["best_checkpoint_value"] - prior_best["best_checkpoint_value"]
            )
        rows.append(record)
    return pd.DataFrame(rows)


def build_plots(experiments: dict[str, dict], checkpoints: pd.DataFrame) -> None:
    latest_validation = {
        model: mask_unavailable_validation_metrics(experiments[model]["train"])
        for model in LATEST
    }
    draw_small_multiples(
        OUT / "loss_curves.png",
        "B2/B3 training and validation losses",
        [
            ("Train total loss", "train_loss_total"),
            ("Validation total loss", "val_loss_total"),
            ("Validation sparse loss", "val_loss_lesion_sparse"),
            ("Validation systematic loss", "val_loss_lesion_sys"),
        ],
        latest_validation,
        x_ticks=[1, 50, 100, 150],
    )
    draw_small_multiples(
        OUT / "validation_key_metric_curves.png",
        "B2/B3 validation metrics",
        [
            ("Target csPCa Dice", "val_target_cspca_dice"),
            ("Best-threshold target Dice", "val_target_cspca_best_threshold_dice"),
            ("ROI AUROC", "val_tbx_roi_auc"),
            ("ROI AUPRC", "val_tbx_roi_auprc"),
            ("Patient AUROC", "val_patient_auc"),
            ("Patient AUPRC", "val_patient_auprc"),
            ("Region AUROC", "val_region_auc"),
            ("Region AUPRC", "val_region_auprc"),
        ],
        latest_validation,
        x_ticks=[1, 50, 100, 150],
    )
    draw_small_multiples(
        OUT / "validation_fixed_operating_curves.png",
        "B2/B3 validation fixed operating points",
        [
            ("ROI sens @ spec=.95", "val_tbx_roi_sens_at_fixed_spec"),
            ("ROI spec @ sens=.90", "val_tbx_roi_spec_at_fixed_sens"),
            ("Patient sens @ spec=.95", "val_patient_sens_at_fixed_spec"),
            ("Patient spec @ sens=.90", "val_patient_spec_at_fixed_sens"),
            ("Region sens @ spec=.95", "val_region_sens_at_fixed_spec"),
            ("Region spec @ sens=.90", "val_region_spec_at_fixed_sens"),
        ],
        latest_validation,
        x_ticks=[1, 50, 100, 150],
    )

    external_models = ["N3/B3 TBx+SBx", "N2/B2 SBx-only", "N1 dense", "B1 Default"]
    external_frames = {}
    for model in external_models:
        frame = checkpoints[
            checkpoints["model"].eq(model) & checkpoints["dataset_type"].eq("external")
        ].copy()
        if not frame.empty:
            external_frames[model] = frame.sort_values("checkpoint_epoch")
    draw_small_multiples(
        OUT / "external_test_checkpoint_curves.png",
        "PROMIS external test metrics across checkpoints",
        [
            ("Patient AUROC", "test_patient_auc"),
            ("Patient AUPRC", "test_patient_auprc"),
            ("Patient sens @ spec=.95", "test_patient_sens_at_fixed_spec"),
            ("Patient spec @ sens=.90", "test_patient_spec_at_fixed_sens"),
            ("Region AUROC", "test_region_auc"),
            ("Region AUPRC", "test_region_auprc"),
            ("Region sens @ spec=.95", "test_region_sens_at_fixed_spec"),
            ("Region spec @ sens=.90", "test_region_spec_at_fixed_sens"),
        ],
        external_frames,
        x_column="checkpoint_epoch",
        x_ticks=[25, 75, 125, 150],
    )

    internal_models = ["N3/B3 TBx+SBx", "N2/B2 SBx-only", "N1 dense"]
    internal_frames = {}
    for model in internal_models:
        frame = checkpoints[
            checkpoints["model"].eq(model) & checkpoints["dataset_type"].eq("internal")
        ].copy()
        if not frame.empty:
            internal_frames[model] = frame.sort_values("checkpoint_epoch")
    draw_small_multiples(
        OUT / "internal_test_checkpoint_curves.png",
        "Internal test metrics across checkpoints (split labels differ)",
        [
            ("Target csPCa Dice", "test_target_cspca_dice_mean"),
            ("Best-threshold target Dice", "test_target_cspca_best_threshold_dice_mean"),
            ("ROI AUROC", "test_tbx_roi_auc"),
            ("ROI AUPRC", "test_tbx_roi_auprc"),
            ("Target FROC @ 0.5 FP/p", "test_target_cspca_sens_at_fp_per_patient_0p5"),
            ("Target FROC @ 1.0 FP/p", "test_target_cspca_sens_at_fp_per_patient_1p0"),
            ("Target FROC @ 2.0 FP/p", "test_target_cspca_sens_at_fp_per_patient_2p0"),
            ("Region AUROC", "test_region_auc"),
        ],
        internal_frames,
        x_column="checkpoint_epoch",
        x_ticks=[25, 75, 125, 150],
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    experiments: dict[str, dict],
    stability: pd.DataFrame,
    dynamics: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    lines = [
        "# B2/B3 experimental result analysis",
        "",
        "## Scope and uncertainty notation",
        "",
        "- B2 is `20260712_1112_B2...`; B3 is `20260712_1359_B3...`. Both contain 150 training epochs and 7 evaluated test checkpoints (`best`, 25, 50, 75, 100, 125, 150).",
        "- In Dice rows, the first `mean ± SD` is the case-level distribution stored by the evaluator. The second `ckpt mean ± SD` is variation across the 7 evaluated checkpoints.",
        "- For AUROC/AUPRC/fixed-operating/FROC metrics, `mean ± SD` is across checkpoints. It measures checkpoint sensitivity, not run-to-run or seed stability; no repeated-seed logs are available.",
        "- `AUC` in these logs is ROC-AUC, i.e. AUROC. External PROMIS has patient/region labels but no dense lesion or ROI ground truth, so external Dice/ROI/FROC zeros are treated as unavailable rather than real zero performance.",
        "",
        "## Main findings",
        "",
        "1. **B3 substantially improves internal ROI detection over B2.** At the selected checkpoint, ROI AUROC/AUPRC rise from 0.4524/0.6240 to 0.6577/0.7840. Target csPCa Dice also rises from 0.0057 ± 0.0146 to 0.0292 ± 0.0511, but remains below the N1 dense reference (0.1825 ± 0.2144).",
        "2. **B3 is strongest for internal FROC at 1–2 FP/p.** Target sensitivity is 0.6923 at 0.5 FP/p and 1.0000 at both 1.0 and 2.0 FP/p; checkpoint stability is 0.7088 ± 0.0732, 0.8984 ± 0.1245, and 0.8984 ± 0.1245, respectively. At 0.5 FP/p, N1's selected checkpoint remains higher (0.7500).",
        "3. **B2 is better for PROMIS external patient discrimination.** Best-checkpoint patient AUROC/AUPRC are 0.5843/0.6034, exceeding the previous best selected-checkpoint results by +0.0194/+0.0241. However, checkpoint means are 0.5655 ± 0.0171 and 0.5833 ± 0.0194, so the AUROC/AUPRC gain is checkpoint-dependent.",
        "4. **B3 is better for PROMIS external region prediction but does not surpass N1 overall.** B3 reaches region AUROC/AUPRC 0.5496/0.1613 versus B2 0.5059/0.1418, while N1 remains higher at 0.5603/0.1702.",
        "5. **Both runs overfit late.** B2 final validation loss is 128.4% above its minimum. B3 activates the systematic branch at epoch 10; relative to the post-activation minimum, final validation loss is 91.1% higher. The selected B3 checkpoint (epoch 35) is close to the post-activation loss minimum, while B2's selected epoch 64 is later than its loss minimum at epoch 36.",
        "",
        "## Training dynamics",
        "",
    ]

    dynamics_rows = []
    for model in LATEST:
        row = dynamics[dynamics["model"].eq(model)].iloc[0]
        dynamics_rows.append(
            [
                DISPLAY[model],
                f"{int(row['selected_epoch'])} ({fmt(row['selection_metric_value'])})",
                f"{fmt(row['selected_train_loss'])} / {fmt(row['selected_val_loss'])}",
                f"{fmt(row['post_activation_min_val_loss'])} @ {int(row['post_activation_min_epoch'])}",
                f"{fmt(row['final_train_loss'])} / {fmt(row['final_val_loss'])}",
                f"{row['final_vs_post_min_pct']:+.1f}%",
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "Model",
                    "Selected epoch (clinical BACC)",
                    "Selected train/val loss",
                    "Min comparable val loss",
                    "Final train/val loss",
                    "Final vs min",
                ],
                dynamics_rows,
            ),
            "",
            "B3 uses curriculum: the systematic loss becomes active at epoch 10. Therefore, its global validation-loss minimum at epoch 8 is not comparable with later two-branch epochs; the table uses the post-activation minimum.",
            "",
            "![B2/B3 loss curves](loss_curves.png)",
            "",
            "## Internal test comparison",
            "",
            "The internal filenames differ (`B2_TCIA_SBx_only_internal_test.csv`, `B3_TCIA_TBx_SBx_internal_test.csv`, and N1's radiologist-only split). Target Dice/ROI/region counts match (48/41,978/1,032), while patient counts differ (N1: 143; B2/B3: 99). The local split CSVs are unavailable, so treat N1/B2/B3 as parallel references rather than a proven paired test.",
            "",
        ]
    )

    internal_rows = []
    for metric in INTERNAL_COLUMNS:
        category, label, _ = METRIC_META[metric]
        cells = []
        for model in [INTERNAL_REFERENCE] + LATEST:
            if metric in DICE_STD_COLUMNS:
                cells.append(dice_cell(stability, model, metric))
            else:
                cells.append(best_and_stability(stability, model, "internal", metric))
        internal_rows.append([category, label] + cells)
    lines.extend(
        [
            "Each non-Dice cell is `best checkpoint; checkpoint mean ± SD`.",
            "",
            markdown_table(
                ["Category", "Metric", "N1 previous", "B2", "B3"],
                internal_rows,
            ),
            "",
            "Lesion Dice and lesion FROC are not reportable for B2/B3 (`lesion_dice_n=0`, `lesion_froc_num_gt=0`). Target csPCa Dice/FROC are reportable (`n=48` cases and 52 target lesions at the selected checkpoint). ROI and target-voxel fixed-operating metrics are numerically identical in these logs because both are finalized from the same 41,978 recorded ROI/voxel samples.",
            "",
            "![Internal checkpoint curves](internal_test_checkpoint_curves.png)",
            "",
            "## PROMIS external comparison with previous experiments",
            "",
            "All models use the PROMIS external split for the metrics below. Each model cell is `best checkpoint; checkpoint mean ± SD`; deltas compare selected best checkpoints with the strongest previous N1/B1 selected checkpoint for that metric.",
            "",
        ]
    )

    external_rows = []
    for _, row in external.iterrows():
        prior = row["prior_best_model"]
        prior_cell = (
            f"{DISPLAY[prior]}: {fmt(row['prior_best_value'])}; "
            f"{fmt(row['prior_checkpoint_mean'])} ± {fmt(row['prior_checkpoint_std'])}"
        )
        b2 = LATEST[0]
        b3 = LATEST[1]
        external_rows.append(
            [
                row["category"],
                row["metric"],
                prior_cell,
                f"{fmt(row[f'{b2}_best'])}; {fmt(row[f'{b2}_checkpoint_mean'])} ± {fmt(row[f'{b2}_checkpoint_std'])}",
                fmt_delta(row[f"{b2}_delta_vs_prior_best"]),
                f"{fmt(row[f'{b3}_best'])}; {fmt(row[f'{b3}_checkpoint_mean'])} ± {fmt(row[f'{b3}_checkpoint_std'])}",
                fmt_delta(row[f"{b3}_delta_vs_prior_best"]),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["Category", "Metric", "Previous best", "B2", "B2 delta", "B3", "B3 delta"],
                external_rows,
            ),
            "",
            "Interpretation: B2's selected checkpoint is best for patient AUROC/AUPRC and patient specificity at 90% sensitivity, but its region metrics regress. B3 restores region performance and slightly improves best-checkpoint region specificity at 90% sensitivity, yet N1 still leads region AUROC/AUPRC. Fixed-operating sensitivity at 95% specificity is low and variable for every model, so these operating points should not be the sole basis for model selection.",
            "",
            "![External checkpoint curves](external_test_checkpoint_curves.png)",
            "",
            "## Validation metric curves",
            "",
            "B3 generally dominates B2 on ROI AUROC/AUPRC and target-Dice-related validation metrics. B2 is stronger on region AUROC and has lower total validation loss, although B2 and B3 total losses aggregate different active branches and should not be compared as identical objectives.",
            "",
            "![Validation key metrics](validation_key_metric_curves.png)",
            "",
            "![Validation fixed operating points](validation_fixed_operating_curves.png)",
            "",
            "## Recommended reporting language",
            "",
            "- Report Dice as case-level `mean ± SD` and add checkpoint `mean ± SD` in parentheses when discussing stability.",
            "- Report AUROC (the log's `*_auc`), AUPRC, sens@spec=.95, spec@sens=.90, and FROC with checkpoint `mean ± SD`, while noting that these 7 checkpoints are correlated snapshots rather than independent runs.",
            "- Do not report external Dice/ROI/FROC as 0.0000; mark them `N/A` because the corresponding ground-truth counts are zero.",
            "- For a paper-level stability claim, rerun at least 3 independent seeds and report seed-level mean ± SD (or confidence intervals).",
            "",
            "## Generated files",
            "",
            "- `checkpoint_metrics.csv`: all raw test checkpoint metrics used here.",
            "- `checkpoint_stability.csv`: long-format checkpoint mean/SD/min/max and best values.",
            "- `external_comparison.csv`: PROMIS comparison with previous N1/B1 best checkpoints.",
            "- `learning_dynamics.csv`: selected/min/final loss diagnostics.",
            "- `loss_curves.png`, `validation_key_metric_curves.png`, `validation_fixed_operating_curves.png`, `external_test_checkpoint_curves.png`, `internal_test_checkpoint_curves.png`.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments()
    missing = [model for model in LATEST + [INTERNAL_REFERENCE] if model not in experiments]
    if missing:
        raise RuntimeError(f"Missing expected experiments: {missing}")

    checkpoints = build_checkpoint_table(experiments)
    stability = build_stability_table(checkpoints)
    dynamics = learning_dynamics(experiments)
    external = external_comparison(stability)

    checkpoints.to_csv(OUT / "checkpoint_metrics.csv", index=False)
    stability.to_csv(OUT / "checkpoint_stability.csv", index=False)
    dynamics.to_csv(OUT / "learning_dynamics.csv", index=False)
    external.to_csv(OUT / "external_comparison.csv", index=False)
    build_plots(experiments, checkpoints)
    write_report(experiments, stability, dynamics, external)


if __name__ == "__main__":
    main()
