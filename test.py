"""
Test/inference script for the revised lesion-segmentation + MIL setting.

This version removes grade/gland evaluation and reports:
  - voxel-level lesion metrics when dense lesion masks are available, e.g. PUB
  - patient-level cancer/csPCa metrics from lesion probability maps
  - region-level MIL metrics for SBx zones, e.g. TCIA/PROMIS
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ProstateUnifiedDataset

try:
    from model import ProstateSegMILNet as ModelClass
except ImportError:  # pragma: no cover - transition compatibility
    from model import ProstateMixedSupervisionNet as ModelClass

import utils


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(Config, name, default)


def build_dataset(csv_path: str):
    task = _cfg("TASK", _cfg("DATASET_TASK", "mixed"))
    try:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
            task=task,
        )
    except TypeError:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
        )


def build_model(device: torch.device):
    try:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            max_zones=_cfg("MAX_ZONES", 20),
            base_channels=_cfg("BASE_CHANNELS", 32),
            dropout_rate=_cfg("DROPOUT_RATE", 0.0),
            mil_pooling=_cfg("MIL_POOLING", "lme"),
            lme_r=_cfg("LME_R", 8.0),
            return_dict=True,
        )
    except TypeError:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            num_grade_classes=_cfg("NUM_CLASSES", 7),
            max_zones=_cfg("MAX_ZONES", 20),
        )
    return model.to(device)


def load_model_weights(model, model_path: str, device: torch.device):
    """Load a checkpoint or a plain state_dict.

    The function first tries exact loading. If the architecture has been renamed
    during migration, it falls back to loading only matching keys.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Remove DataParallel prefix if present.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    try:
        model.load_state_dict(cleaned, strict=True)
        print("✅ Loaded model weights with strict=True")
    except RuntimeError as err:
        print(f"⚠️ Strict loading failed: {err}")
        model_state = model.state_dict()
        matched = {
            k: v for k, v in cleaned.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        print(f"✅ Loaded {len(matched)}/{len(model_state)} matching tensors with strict=False")

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": cleaned}


def unpack_model_output(raw_outputs):
    if hasattr(utils, "unpack_model_output"):
        return utils.unpack_model_output(raw_outputs)
    if isinstance(raw_outputs, dict):
        return raw_outputs
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) >= 5:
        return {
            "lesion_logits": raw_outputs[2],
            "region_logits": raw_outputs[3],
            "region_valid_mask": None,
        }
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 3:
        return {
            "lesion_logits": raw_outputs[0],
            "region_logits": raw_outputs[1],
            "region_valid_mask": raw_outputs[2],
        }
    raise TypeError("Unsupported model output format.")


def infer_dataset_type(batch: Dict, b: int) -> str:
    if hasattr(utils, "infer_dataset_type"):
        return utils.infer_dataset_type(batch, b)
    pid = str(batch.get("pid", [""])[b]) if "pid" in batch else ""
    if pid.startswith("PUB"):
        return "PUB"
    if pid.startswith("TCIA"):
        return "TCIA"
    if pid.startswith("PROMIS"):
        return "PROMIS"
    if batch.get("has_lesion", torch.zeros(1))[b].item() > 0:
        return "PUB"
    if batch.get("has_target", torch.zeros(1))[b].item() > 0:
        return "TCIA"
    if batch.get("has_sys", torch.zeros(1))[b].item() > 0:
        return "PROMIS"
    return "OTHER"


def compute_patient_label(batch: Dict, b: int, positive_threshold: int, invalid_sys_label: int) -> int:
    """Positive if any available supervision indicates cancer/csPCa."""
    label = 0

    if batch.get("has_lesion", torch.zeros(1))[b].item() > 0:
        if batch["lesion_mask"][b].max().item() > 0:
            label = 1

    if batch.get("has_target", torch.zeros(1))[b].item() > 0:
        if batch["target_mask"][b].max().item() >= positive_threshold:
            label = 1

    if batch.get("has_sys", torch.zeros(1))[b].item() > 0:
        labels = batch["sys_labels"][b]
        valid = labels != invalid_sys_label
        if valid.any() and labels[valid].max().item() >= positive_threshold:
            label = 1

    return int(label)


def compute_patient_score(lesion_prob: torch.Tensor, gland_mask: torch.Tensor | None = None) -> float:
    """Patient-level score as max lesion probability, restricted to gland when possible."""
    if gland_mask is not None and gland_mask.max().item() > 0:
        values = lesion_prob[gland_mask > 0]
        if values.numel() > 0:
            return float(values.max().detach().cpu().item())
    return float(lesion_prob.max().detach().cpu().item())


def make_sys_label_volume(zones_mask: np.ndarray, sys_labels: np.ndarray, invalid_sys_label: int) -> np.ndarray:
    out = np.zeros_like(zones_mask, dtype=np.float32)
    for z_idx in range(1, min(20, len(sys_labels)) + 1):
        if int(sys_labels[z_idx - 1]) != invalid_sys_label:
            out[zones_mask == z_idx] = float(sys_labels[z_idx - 1])
    return out


def choose_visual_slice(lesion_prob: np.ndarray, lesion_gt: np.ndarray, target_gt: np.ndarray, sys_gt: np.ndarray) -> int:
    if lesion_gt.sum() > 0:
        return int(np.argmax(lesion_gt.sum(axis=(1, 2))))
    if target_gt.sum() > 0:
        return int(np.argmax(target_gt.sum(axis=(1, 2))))
    if sys_gt.sum() > 0:
        return int(np.argmax(sys_gt.sum(axis=(1, 2))))
    if lesion_prob.sum() > 0:
        return int(np.argmax(lesion_prob.sum(axis=(1, 2))))
    return int(lesion_prob.shape[0] // 2)


def save_seg_mil_vis(
    img_vol: np.ndarray,
    lesion_gt: np.ndarray,
    target_gt: np.ndarray,
    zones_mask: np.ndarray,
    sys_labels: np.ndarray,
    lesion_prob: np.ndarray,
    pid: str,
    save_path: str,
):
    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    sys_gt = make_sys_label_volume(zones_mask, sys_labels, invalid_sys_label)
    sys_pos = (sys_gt >= positive_threshold).astype(np.float32)

    z = choose_visual_slice(lesion_prob, lesion_gt, target_gt, sys_gt)
    s_img = img_vol[z]
    s_prob = lesion_prob[z]
    s_lesion = lesion_gt[z]
    s_target = target_gt[z]
    s_zones = zones_mask[z]
    s_sys_pos = sys_pos[z]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Patient: {pid} | Slice: {z}/{img_vol.shape[0]}", fontsize=16, fontweight="bold")

    axes[0, 0].imshow(s_img, cmap="gray")
    axes[0, 0].set_title("T2 MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(s_img, cmap="gray")
    risk_masked = np.ma.masked_where(s_prob < 0.1, s_prob)
    im_risk = axes[0, 1].imshow(risk_masked, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted lesion probability")
    axes[0, 1].axis("off")
    fig.colorbar(im_risk, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[0, 2].imshow(np.ma.masked_where(s_lesion == 0, s_lesion), cmap="autumn", alpha=0.55)
    axes[0, 2].set_title("GT dense lesion mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(s_img, cmap="gray")
    if s_target.sum() > 0:
        target_pos = (s_target >= positive_threshold).astype(np.float32)
        axes[1, 0].imshow(np.ma.masked_where(target_pos == 0, target_pos), cmap="autumn", alpha=0.55)
    axes[1, 0].set_title("TBx positive needle-track voxels")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(s_img, cmap="gray")
    if s_zones.max() > 0:
        axes[1, 1].imshow(np.ma.masked_where(s_zones == 0, s_zones), cmap="tab20", alpha=0.35)
    if s_sys_pos.sum() > 0:
        axes[1, 1].contour(s_sys_pos, levels=[0.5], linewidths=1.5)
    axes[1, 1].set_title("SBx zones / positive regions")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[1, 2].contour(s_lesion, levels=[0.5], linewidths=2)
    if s_prob.max() > 0:
        axes[1, 2].contour((s_prob >= 0.5).astype(np.float32), levels=[0.5], linewidths=2, linestyles="dashed")
    axes[1, 2].set_title("Contours: GT solid / Pred dashed")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def get_test_dir() -> str:
    return _cfg("TEST_DIR", os.path.join(_cfg("BASE_DIR", "."), "test"))


def get_test_csv() -> str:
    if _cfg("TEST_CSV", None) is not None:
        return _cfg("TEST_CSV")
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    for name in [
        "N4_mixed_PROMIS_external_val.csv",
        "task_cls_external_val.csv",
        "external_val.csv",
        "test.csv",
    ]:
        candidate = os.path.join(split_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(split_dir, "N4_mixed_PROMIS_external_val.csv")


def get_model_path(test_dir: str) -> str:
    if _cfg("TEST_MODEL_PATH", None) is not None:
        return _cfg("TEST_MODEL_PATH")
    for name in ["best_checkpoint.pth", "best_model.pth"]:
        candidate = os.path.join(test_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"❌ 找不到模型文件，请检查 TEST_MODEL_PATH 或 {test_dir}")


def main():
    if hasattr(Config, "set_seed"):
        Config.set_seed()
    device = torch.device(_cfg("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    test_dir = get_test_dir()
    vis_dir = os.path.join(test_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    model_path = get_model_path(test_dir)
    test_csv = get_test_csv()

    print(f"🚀 [Test Start] Model: {model_path}")
    print(f"📄 [Test CSV]   {test_csv}")
    print(f"📷 [Vis Output] {vis_dir}")

    test_dataset = build_dataset(test_csv)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)
    load_model_weights(model, model_path, device)
    model.eval()

    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    prob_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))

    mil_evaluator = utils.LesionMILEvaluator(
        prob_threshold=prob_threshold,
        positive_threshold=positive_threshold,
        invalid_sys_label=invalid_sys_label,
    )

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing & Rendering"):
            batch = utils.move_batch_to_device(batch, device) if hasattr(utils, "move_batch_to_device") else {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            imgs = batch["input"]
            zones_mask = batch.get("zones_mask", None)
            pid = batch.get("pid", ["Unknown_Patient"])[0]
            source = infer_dataset_type(batch, 0)

            raw_outputs = model(imgs, zones_mask)
            outputs = unpack_model_output(raw_outputs)
            lesion_logits = outputs["lesion_logits"]
            lesion_probs = torch.sigmoid(lesion_logits)
            lesion_prob_3d = lesion_probs[0, 0]

            mil_evaluator.update_from_batch(
                lesion_probs=lesion_probs,
                batch=batch,
                region_logits=outputs.get("region_logits"),
                region_valid_mask=outputs.get("region_valid_mask"),
            )

            has_lesion = bool(batch.get("has_lesion", torch.zeros(1, device=device))[0].item() > 0)
            has_target = bool(batch.get("has_target", torch.zeros(1, device=device))[0].item() > 0)
            has_sys = bool(batch.get("has_sys", torch.zeros(1, device=device))[0].item() > 0)
            has_gland = bool(batch.get("has_gland", torch.zeros(1, device=device))[0].item() > 0)

            lesion_dice = lesion_f1 = lesion_sens = lesion_spec = np.nan
            if has_lesion:
                pred_binary = (lesion_prob_3d >= prob_threshold).float()
                gt_lesion_tensor = batch["lesion_mask"][0, 0].float()
                lesion_dice = utils.compute_dice(pred_binary.unsqueeze(0), gt_lesion_tensor.unsqueeze(0))
                lesion_f1 = utils.compute_f1(pred_binary, gt_lesion_tensor)
                lesion_sens = utils.compute_sens(pred_binary, gt_lesion_tensor)
                lesion_spec = utils.compute_spec(pred_binary, gt_lesion_tensor) if hasattr(utils, "compute_spec") else np.nan

            gland_tensor = batch["gland_mask"][0, 0] if has_gland and "gland_mask" in batch else None
            patient_score = compute_patient_score(lesion_prob_3d, gland_tensor)
            patient_label = compute_patient_label(batch, 0, positive_threshold, invalid_sys_label)
            patient_pred = int(patient_score >= prob_threshold)

            # Summarise model-pooled region scores for this patient when available.
            region_positive_gt = np.nan
            region_positive_pred = np.nan
            if has_sys and "sys_labels" in batch:
                labels = batch["sys_labels"][0]
                valid = labels != invalid_sys_label
                if valid.any():
                    region_positive_gt = int((labels[valid] >= positive_threshold).sum().item())

                    region_logits = outputs.get("region_logits")
                    if region_logits is not None:
                        region_scores = torch.sigmoid(region_logits[0])
                        if region_scores.dim() == 2 and region_scores.size(-1) == 1:
                            region_scores = region_scores.squeeze(-1)
                        region_scores = region_scores[: labels.numel()]
                        region_positive_pred = int((region_scores[valid] >= prob_threshold).sum().item())

            results.append(
                {
                    "Patient_ID": pid,
                    "Source": source,
                    "has_lesion": int(has_lesion),
                    "has_target": int(has_target),
                    "has_sys": int(has_sys),
                    "Lesion_Dice": lesion_dice,
                    "Lesion_F1": lesion_f1,
                    "Lesion_Sens": lesion_sens,
                    "Lesion_Spec": lesion_spec,
                    "Patient_Label": patient_label,
                    "Patient_Score": patient_score,
                    "Patient_Pred": patient_pred,
                    "Region_Positive_GT_Count": region_positive_gt,
                    "Region_Positive_Pred_Count": region_positive_pred,
                }
            )

            img_t2 = imgs[0, 0].detach().cpu().numpy()
            lesion_gt = batch.get("lesion_mask", torch.zeros_like(lesion_probs))[0, 0].detach().cpu().numpy()
            target_gt = batch.get("target_mask", torch.zeros_like(lesion_probs))[0, 0].detach().cpu().numpy()
            gt_zones = batch.get("zones_mask", torch.zeros_like(lesion_probs))[0, 0].detach().cpu().numpy()
            gt_sys_labels = batch.get("sys_labels", torch.full((1, 20), invalid_sys_label, device=device))[0].detach().cpu().numpy()
            pred_prob_np = lesion_prob_3d.detach().cpu().numpy()

            save_seg_mil_vis(
                img_vol=img_t2,
                lesion_gt=lesion_gt,
                target_gt=target_gt,
                zones_mask=gt_zones,
                sys_labels=gt_sys_labels,
                lesion_prob=pred_prob_np,
                pid=pid,
                save_path=os.path.join(vis_dir, f"{pid}.png"),
            )

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(test_dir, "test_metrics_per_patient.csv")
    df_results.to_csv(csv_path, index=False)

    mil_metrics = mil_evaluator.compute_metrics()

    print("\n" + "=" * 60)
    print("🎯 TEST METRICS SUMMARY: Lesion Segmentation + MIL")
    print("=" * 60)
    print(f"🖼️ Visualisations saved to: {vis_dir}")
    print(f"📊 Per-patient metrics saved to: {csv_path}\n")

    if df_results["Lesion_Dice"].notna().any():
        dense_df = df_results[df_results["Lesion_Dice"].notna()]
        print("📈 Dense lesion segmentation metrics:")
        print(f"   - N:         {len(dense_df)}")
        print(f"   - Mean Dice: {dense_df['Lesion_Dice'].mean():.4f}")
        print(f"   - Mean F1:   {dense_df['Lesion_F1'].mean():.4f}")
        print(f"   - Mean Sens: {dense_df['Lesion_Sens'].mean():.4f}")
        if dense_df["Lesion_Spec"].notna().any():
            print(f"   - Mean Spec: {dense_df['Lesion_Spec'].mean():.4f}")
        print()

    print("📈 Patient-level MIL metrics:")
    print(f"   - N:     {mil_metrics['patient_n']}")
    print(f"   - Sens:  {mil_metrics['patient_sens']:.4f}")
    print(f"   - Spec:  {mil_metrics['patient_spec']:.4f}")
    print(f"   - BAcc:  {mil_metrics['patient_bacc']:.4f}")
    print(f"   - AUC:   {mil_metrics['patient_auc']:.4f}")
    print(f"   - AUPRC: {mil_metrics['patient_auprc']:.4f}\n")

    print("📈 Region-level SBx MIL metrics:")
    print(f"   - N:     {mil_metrics['region_n']}")
    print(f"   - Sens:  {mil_metrics['region_sens']:.4f}")
    print(f"   - Spec:  {mil_metrics['region_spec']:.4f}")
    print(f"   - BAcc:  {mil_metrics['region_bacc']:.4f}")
    print(f"   - AUC:   {mil_metrics['region_auc']:.4f}")
    print(f"   - AUPRC: {mil_metrics['region_auprc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
