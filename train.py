"""
Training script for the current lesion-risk-map + MIL setting.

Current training contract:
  - The model learns one voxel-level lesion risk map.
  - Only lesion-related losses are logged: PUB dense lesion, TCIA TBx ROI,
    and SBx MIL.
  - TCIA TBx ROI loss follows Config.USE_TBX_POSITIVE_ONLY_LOSS; the current
    B-series default is sampled positive + sampled negative ROI BCE.
  - Grade/gland outputs, losses, metrics, and best-model criteria are removed.
  - The script accepts the new dictionary model/loss outputs, but is tolerant of
    the old 5-output model during transition.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ProstateUnifiedDataset
from Loss_function import MixedSupervisionLoss

# Prefer the new segmentation+MIL model class. Fall back to the old class name so
# that the script can still run while files are being migrated.
try:
    from model import ProstateSegMILNet as ModelClass
except ImportError:  # pragma: no cover - transition compatibility
    from model import ProstateMixedSupervisionNet as ModelClass

import utils


class Logger:
    """Write console output to both terminal and a log file."""

    def __init__(self, filename: str = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(Config, name, default)


def get_dataset_task(is_train: bool, split: str = "val") -> str:
    if is_train:
        return _cfg("TRAIN_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed")))
    if split == "test":
        return _cfg(
            "TEST_DATASET_TASK",
            _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed"))),
        )
    return _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed")))


def build_dataset(csv_path: str, is_train: bool, split: str = "val"):
    """Create dataset with split-specific task filtering when configured."""
    task = get_dataset_task(is_train, split=split)
    try:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=is_train,
            task=task,
        )
    except TypeError:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=is_train,
        )


def build_model(device: torch.device):
    """Instantiate either the new SegMIL model or the old transition model."""
    common_kwargs: Dict[str, Any] = {
        "in_channels": _cfg("IN_CHANNELS", 3),
        "max_zones": _cfg("MAX_ZONES", 20),
    }

    # New model signature.
    try:
        model = ModelClass(
            **common_kwargs,
            base_channels=_cfg("BASE_CHANNELS", 32),
            dropout_rate=_cfg("DROPOUT_RATE", 0.0),
            mil_pooling=_cfg("MIL_POOLING", "lme"),
            lme_r=_cfg("LME_R", 8.0),
            return_dict=True,
        )
    except TypeError:
        # Old model signature. num_grade_classes is ignored by new code paths.
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            num_grade_classes=_cfg("NUM_CLASSES", 7),
            max_zones=_cfg("MAX_ZONES", 20),
        )
    return model.to(device)


def build_criterion(device: torch.device):
    """Instantiate the lesion loss with Config-controlled TBx/SBx behaviour."""
    positive_threshold = _cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1))
    kwargs = {
        "positive_threshold": positive_threshold,
        "invalid_sys_label": _cfg("INVALID_SYS_LABEL", -1),
        "pos_weight_val": _cfg("POS_WEIGHT_VAL", 2.0),
        "sys_pos_weight_val": _cfg("SYS_POS_WEIGHT_VAL", _cfg("POS_WEIGHT_VAL", 2.0)),
        "sys_focal_alpha": _cfg("SYS_FOCAL_ALPHA", 0.75),
        "sys_focal_gamma": _cfg("SYS_FOCAL_GAMMA", 2.0),
        "use_sys_class_balanced_bce": _cfg("USE_SYS_CLASS_BALANCED_BCE", True),
        "use_tbx_positive_only_loss": _cfg("USE_TBX_POSITIVE_ONLY_LOSS", False),
        "return_dict": True,
    }
    try:
        criterion = MixedSupervisionLoss(**kwargs)
    except TypeError:
        # Compatibility with the old loss constructor.
        criterion = MixedSupervisionLoss(
            csPCa_threshold=positive_threshold,
            invalid_sys_label=_cfg("INVALID_SYS_LABEL", -1),
            pos_weight_val=_cfg("POS_WEIGHT_VAL", 2.0),
        )
    return criterion.to(device)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if hasattr(utils, "move_batch_to_device"):
        return utils.move_batch_to_device(batch, device)
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


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


def call_loss(criterion, outputs, batch):
    if hasattr(utils, "call_criterion"):
        loss_output = utils.call_criterion(criterion, outputs, batch)
    else:
        loss_output = criterion(outputs, batch)

    if hasattr(utils, "normalise_loss_output"):
        return utils.normalise_loss_output(loss_output)

    if isinstance(loss_output, dict):
        return loss_output
    raise TypeError("The current utils.py cannot normalise this loss output.")


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _mask_for_visualisation(batch: Dict[str, Any], key: str, b: int, fallback_like: torch.Tensor):
    if key not in batch:
        return fallback_like.detach().cpu().numpy()
    return batch[key][b, 0].detach().cpu().numpy()


def maybe_save_train_visualizations(
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    save_dir: str,
    epoch: int,
    saved_patients: set,
    max_to_save: int,
) -> int:
    """Save a few non-repeated training predictions for visual QA."""
    lesion_logits = outputs.get("lesion_logits")
    if lesion_logits is None or max_to_save <= 0:
        return 0

    vis_dir = os.path.join(save_dir, _cfg("VIS_SUBDIR", "visualizations"), "train", f"epoch_{epoch:03d}")
    os.makedirs(vis_dir, exist_ok=True)

    lesion_probs = torch.sigmoid(lesion_logits.detach())
    saved_count = 0
    batch_size = lesion_probs.size(0)

    for b in range(batch_size):
        if saved_count >= max_to_save:
            break

        pid = batch["pid"][b] if "pid" in batch else f"case_{epoch}_{b}"
        pid = str(pid)
        if pid in saved_patients:
            continue

        empty_like = torch.zeros_like(lesion_probs[b, 0])
        gt_dict = {
            "type": utils.infer_dataset_type(batch, b),
            "lesion_mask": _mask_for_visualisation(batch, "lesion_mask", b, empty_like),
            "target_mask": _mask_for_visualisation(batch, "target_mask", b, empty_like),
            "zones_mask": _mask_for_visualisation(batch, "zones_mask", b, empty_like),
            "sys_labels": batch["sys_labels"][b].detach().cpu().numpy() if "sys_labels" in batch else None,
        }

        filename = f"{saved_count + 1:02d}_{gt_dict['type']}_{_safe_filename(pid)}.png"
        try:
            utils.visualize_predictions(
                input_tensor=batch["input"][b],
                risk_map=lesion_probs[b],
                gt_dict=gt_dict,
                save_path=os.path.join(vis_dir, filename),
                patient_id=pid,
            )
        except Exception as exc:
            print(f"Warning: failed to save training visualization for {pid}: {exc}")
            continue
        saved_patients.add(pid)
        saved_count += 1

    return saved_count


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    save_dir: str = "",
    saved_train_vis_patients=None,
):
    model.train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    tracker = utils.MetricTracker()
    pbar = tqdm(loader, desc="Training")
    should_save_train_vis = (
        bool(_cfg("SAVE_TRAIN_VIS", False))
        and bool(save_dir)
        and int(_cfg("TRAIN_VIS_EVERY_N_EPOCHS", 0)) > 0
        and epoch % int(_cfg("TRAIN_VIS_EVERY_N_EPOCHS", 0)) == 0
    )
    max_train_vis = int(_cfg("TRAIN_VIS_MAX_PER_EPOCH", 0))
    saved_this_epoch = 0
    if saved_train_vis_patients is None:
        saved_train_vis_patients = set()

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        imgs = batch["input"]
        zones_mask = batch.get("zones_mask", None)

        optimizer.zero_grad(set_to_none=True)

        raw_outputs = model(imgs, zones_mask)
        outputs = unpack_model_output(raw_outputs)
        loss_dict = call_loss(criterion, outputs, batch)
        total_loss = loss_dict["total_loss"]

        if torch.is_tensor(total_loss) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_cfg("GRAD_CLIP_NORM", 12.0))
            optimizer.step()

        tracker.update_losses(loss_dict)
        if should_save_train_vis and saved_this_epoch < max_train_vis:
            saved_this_epoch += maybe_save_train_visualizations(
                batch=batch,
                outputs=outputs,
                save_dir=save_dir,
                epoch=epoch,
                saved_patients=saved_train_vis_patients,
                max_to_save=max_train_vis - saved_this_epoch,
            )
        pbar.set_postfix({"Total Loss": f"{float(total_loss.detach().cpu()):.4f}"})

    return tracker


def select_validation_metric(v_track) -> float:
    """Metric for best-model saving. Higher is better."""
    metric_name = str(_cfg("BEST_MODEL_METRIC", "lesion_dice")).lower()

    if metric_name in {"loss", "val_loss", "val_loss_total"}:
        return -float(v_track.loss_total.avg)
    if metric_name in {"lesion_dice", "dice", "val_lesion_dice"}:
        return float(v_track.lesion_dice.avg)
    if metric_name in {"target_cspca_dice", "val_target_cspca_dice", "cspca_dice"}:
        return float(getattr(v_track, "target_cspca_dice").avg)
    if metric_name in {"tbx_roi_sens_at_fixed_spec", "val_tbx_roi_sens_at_fixed_spec", "tbx_roi_sens_at_fpr5"}:
        return float(getattr(v_track, "tbx_roi_sens_at_fixed_spec", 0.0))
    if metric_name in {"tbx_roi_auc", "val_tbx_roi_auc"}:
        return float(getattr(v_track, "tbx_roi_auc", 0.0))
    if metric_name in {"tbx_roi_auprc", "val_tbx_roi_auprc"}:
        return float(getattr(v_track, "tbx_roi_auprc", 0.0))
    if metric_name in {"tbx_roi_bacc", "val_tbx_roi_bacc"}:
        return float(getattr(v_track, "tbx_roi_bacc", 0.0))
    if metric_name == "lesion_f1":
        return float(v_track.lesion_f1.avg)
    if metric_name == "patient_sens_at_fixed_spec":
        return float(getattr(v_track, "patient_sens_at_fixed_spec", 0.0))
    if metric_name == "patient_spec_at_fixed_sens":
        return float(getattr(v_track, "patient_spec_at_fixed_sens", 0.0))
    if metric_name == "region_sens_at_fixed_spec":
        return float(getattr(v_track, "region_sens_at_fixed_spec", 0.0))
    if metric_name == "region_spec_at_fixed_sens":
        return float(getattr(v_track, "region_spec_at_fixed_sens", 0.0))
    if metric_name == "region_bacc":
        return float(v_track.region_bacc)
    if metric_name == "region_auc":
        return float(getattr(v_track, "region_auc", 0.0))
    if metric_name == "patient_bacc":
        return float(getattr(v_track, "patient_bacc", 0.0))
    if metric_name == "patient_auc":
        return float(getattr(v_track, "patient_auc", 0.0))
    if metric_name == "clinical_bacc":
        return 0.5 * float(getattr(v_track, "patient_bacc", 0.0)) + 0.5 * float(v_track.region_bacc)
    if metric_name == "composite":
        return (
            0.50 * float(v_track.lesion_dice.avg)
            + 0.25 * float(getattr(v_track, "patient_bacc", 0.0))
            + 0.25 * float(v_track.region_bacc)
        )

    print(f"⚠️ Unknown BEST_MODEL_METRIC='{metric_name}', using lesion_dice.")
    return float(v_track.lesion_dice.avg)


def save_checkpoint(path, model, criterion, optimizer, scheduler, epoch: int, best_metric: float, config_name: str):
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config_name": config_name,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict() if criterion is not None else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def get_best_model_path(save_path: str) -> str:
    """Prefer the rich checkpoint, but allow old runs with only best_model.pth."""
    for name in ["best_checkpoint.pth", "best_model.pth"]:
        candidate = os.path.join(save_path, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No best model found in {save_path}")


def load_best_weights(model, criterion, model_path: str, device: torch.device) -> Dict[str, Any]:
    """Load a best checkpoint or plain state_dict into the current model."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=True)

    if isinstance(checkpoint, dict) and checkpoint.get("criterion_state_dict") is not None:
        try:
            criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
        except TypeError:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": cleaned}


def run_final_test(model, criterion, device: torch.device, save_path: str, metric_name: str):
    """Evaluate the saved best model on TEST_CSV after training has finished."""
    test_csv = get_csv_path("TEST_CSV", "N4_mixed_PROMIS_external_val.csv")
    best_model_path = get_best_model_path(save_path)

    # Keep test evaluation post-hoc: validation still selects the checkpoint,
    # then the frozen best model is loaded once for external reporting.
    print("\n" + "=" * 60)
    print("Final test using best model")
    print("=" * 60)
    print(f"Best model: {best_model_path}")
    print(f"Test CSV:   {test_csv}")
    print(f"Test dataset task: {get_dataset_task(is_train=False, split='test')}")

    checkpoint = load_best_weights(model, criterion, best_model_path, device)
    best_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    if hasattr(criterion, "set_epoch") and best_epoch > 0:
        criterion.set_epoch(best_epoch)

    test_loader = DataLoader(
        build_dataset(test_csv, is_train=False, split="test"),
        batch_size=_cfg("BATCH_SIZE", 1),
        shuffle=False,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

    test_track = utils.validate(model, test_loader, criterion, device, best_epoch, save_dir="")
    print(f"Test  | {test_track.print_val_summary()}")

    test_log = {
        "best_epoch": best_epoch,
        "best_model_metric_name": metric_name,
        "best_model_metric_value": float(checkpoint.get("best_metric", float("nan")))
        if isinstance(checkpoint, dict)
        else float("nan"),
        "best_model_path": best_model_path,
        "test_csv": test_csv,
    }
    for key, value in test_track.get_val_dict().items():
        test_key = key.replace("val_", "test_", 1) if key.startswith("val_") else f"test_{key}"
        test_log[test_key] = value

    test_log_csv = os.path.join(save_path, "test_log.csv")
    pd.DataFrame([test_log]).to_csv(test_log_csv, index=False)
    print(f"Test log saved to: {test_log_csv}")
    return test_track


def get_experiment_name() -> str:
    if hasattr(Config, "get_experiment_name"):
        return Config.get_experiment_name()
    return str(_cfg("EXP_NAME", "SegMIL_experiment"))


def get_csv_path(name: str, fallback: str) -> str:
    value = _cfg(name, None)
    if value is not None:
        return value
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    return os.path.join(split_dir, fallback)


def main():
    if hasattr(Config, "set_seed"):
        Config.set_seed()

    device = torch.device(_cfg("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    exp_name = get_experiment_name()
    save_path = os.path.join(_cfg("EXP_DIR", "experiments"), exp_name)
    os.makedirs(save_path, exist_ok=True)

    log_file_path = os.path.join(save_path, "console_output.log")
    sys.stdout = Logger(log_file_path)
    print(f"✅ Console outputs will be saved to: {log_file_path}")
    if hasattr(Config, "show"):
        Config.show()

    train_csv = get_csv_path("TRAIN_CSV", "N4_mixed_PUB_TCIA_train.csv")
    val_csv = get_csv_path("VAL_CSV", "N4_mixed_PUB_TCIA_internal_val.csv")
    test_csv = get_csv_path("TEST_CSV", "N4_mixed_PROMIS_external_val.csv")
    print(f"📄 Train CSV: {train_csv}")
    print(f"📄 Val CSV:   {val_csv}")
    print(f"📄 Test CSV:  {test_csv}")
    print(f"📌 Train dataset task: {get_dataset_task(is_train=True)}")
    print(f"📌 Val dataset task:   {get_dataset_task(is_train=False)}")
    print(f"📌 Test dataset task:  {get_dataset_task(is_train=False, split='test')}")

    train_loader = DataLoader(
        build_dataset(train_csv, is_train=True),
        batch_size=_cfg("BATCH_SIZE", 1),
        shuffle=True,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        build_dataset(val_csv, is_train=False),
        batch_size=_cfg("BATCH_SIZE", 1),
        shuffle=False,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)
    criterion = build_criterion(device)

    em_lr_multiplier = float(_cfg("EM_LR_MULTIPLIER", 10.0))
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": _cfg("LR", 1e-4), "weight_decay": _cfg("WEIGHT_DECAY", 0.0)},
            {"params": criterion.parameters(), "lr": _cfg("LR", 1e-4) * em_lr_multiplier, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_cfg("NUM_EPOCHS", 100))

    best_metric = -float("inf")
    early_stop_counter = 0
    history = []
    metric_name = str(_cfg("BEST_MODEL_METRIC", "lesion_dice"))
    saved_train_vis_patients = set()

    for epoch in range(1, int(_cfg("NUM_EPOCHS", 100)) + 1):
        print(f"\nEpoch {epoch}/{_cfg('NUM_EPOCHS', 100)}")
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        if hasattr(criterion, "is_enabled"):
            print(
                "Curriculum/task status | "
                f"Dense: {int(criterion.is_enabled('lesion_dense'))} | "
                f"TBx ROI BCE: {int(criterion.is_enabled('lesion_sparse'))} | "
                f"Sys MIL: {int(criterion.is_enabled('lesion_sys'))} | "
                f"Outside gland: {int(criterion.is_enabled('lesion_outside_gland'))} | "
                f"Patient risk: {int(criterion.is_enabled('lesion_patient'))}"
            )

        train_track = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            save_dir=save_path,
            saved_train_vis_patients=saved_train_vis_patients,
        )
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        val_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)

        print(f"Train | {train_track.print_train_summary()}")
        print(f"Val   | {val_track.print_val_summary()}")

        current_weights = criterion.get_current_weights() if hasattr(criterion, "get_current_weights") else {}
        print("--- Lesion EM / Loss Multipliers ---")
        print(
            f"Dense: {current_weights.get('lesion_dense', 1.0):.3f} | "
            f"TBx ROI BCE: {current_weights.get('lesion_sparse', 1.0):.3f} | "
            f"Sys MIL: {current_weights.get('lesion_sys', 1.0):.3f} | "
            f"Outside gland: {current_weights.get('lesion_outside_gland', 0.0):.3f} | "
            f"Patient risk: {current_weights.get('lesion_patient', 0.0):.3f}"
        )

        epoch_log = {"epoch": epoch}
        epoch_log.update(train_track.get_train_dict())
        epoch_log.update(val_track.get_val_dict())
        epoch_log.update(
            {
                "best_model_metric_name": metric_name,
                "use_em_weighting": int(_cfg("USE_EM_WEIGHTING", True)),
                "use_logvar_clamp": int(_cfg("USE_LOGVAR_CLAMP", False)),
                "use_curriculum": int(_cfg("USE_CURRICULUM", False)),
                "em_lr_multiplier": em_lr_multiplier,
                "lesion_dense_enabled_this_epoch": int(criterion.is_enabled("lesion_dense")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_sparse_enabled_this_epoch": int(criterion.is_enabled("lesion_sparse")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_sys_enabled_this_epoch": int(criterion.is_enabled("lesion_sys")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_outside_gland_enabled_this_epoch": int(criterion.is_enabled("lesion_outside_gland")) if hasattr(criterion, "is_enabled") else 0,
                "lesion_patient_enabled_this_epoch": int(criterion.is_enabled("lesion_patient")) if hasattr(criterion, "is_enabled") else 0,
            }
        )
        history.append(epoch_log)

        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

        cur_metric = select_validation_metric(val_track)
        print(f"Selection metric ({metric_name}): {cur_metric:.4f}")

        if cur_metric > best_metric:
            best_metric = cur_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            save_checkpoint(
                os.path.join(save_path, "best_checkpoint.pth"),
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                exp_name,
            )
            print(f"--> Best Model Saved (Score: {best_metric:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= int(_cfg("EARLY_STOP_PATIENCE", 20)):
                print(f"Early stop triggered at epoch {epoch}")
                break

        scheduler.step()

    try:
        run_final_test(model, criterion, device, save_path, metric_name)
    except FileNotFoundError as exc:
        print(f"Warning: final test skipped because the best model was not found: {exc}")


if __name__ == "__main__":
    main()
