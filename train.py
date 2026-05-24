import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ProstateUnifiedDataset
from Loss_function import MixedSupervisionLoss
from model import ProstateMixedSupervisionNet
import utils


class Logger:
    """Write console output to both terminal and a log file."""

    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def unpack_loss_output(loss_output):
    if len(loss_output) == 11:
        return loss_output
    if len(loss_output) == 10:
        return (*loss_output, None)
    raise ValueError(f"Unexpected criterion return length: {len(loss_output)}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = utils.MetricTracker()
    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        imgs = batch["input"].to(device)
        z_mask = batch["zones_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)

        loss_output = criterion(
            g_p,
            s_g_p,
            l_p,
            s_l_p,
            gl_p,
            batch["target_mask"].to(device),
            batch["sys_labels"].to(device),
            batch["lesion_mask"].to(device),
            batch["gland_mask"].to(device),
            batch["has_target"].to(device),
            batch["has_sys"].to(device),
            batch["has_lesion"].to(device),
            batch["has_gland"].to(device),
        )

        total_loss, l_grad_tot, l_grad_tbx, l_grad_sbx, l_les_tot, l_les_dense, l_les_sparse, l_les_sys, l_gland, em_weights, active_tasks = unpack_loss_output(loss_output)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        optimizer.step()

        tracker.update_losses(
            total_loss.item(),
            l_grad_tot.item(),
            l_grad_tbx.item(),
            l_grad_sbx.item(),
            l_les_tot.item(),
            l_les_dense.item(),
            l_les_sparse.item(),
            l_les_sys.item(),
            l_gland.item(),
            em_weights=em_weights,
            active_tasks=active_tasks,
        )
        pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

    return tracker


def select_validation_metric(v_track):
    """Choose the score used for best-model saving and early stopping."""
    metric_name = getattr(Config, "BEST_MODEL_METRIC", "composite")

    lesion_dice = v_track.lesion_dice.avg
    gland_bacc = v_track.gland_bacc
    region_bacc = v_track.region_bacc
    grade_kappa = v_track.grade_kappa.avg
    clinical_bacc = 0.5 * gland_bacc + 0.5 * region_bacc

    if metric_name == "lesion_dice":
        return lesion_dice
    if metric_name == "clinical_bacc":
        return clinical_bacc
    if metric_name == "region_bacc":
        return region_bacc
    if metric_name == "gland_bacc":
        return gland_bacc
    if metric_name == "grade_kappa":
        return grade_kappa

    # Default composite metric for mixed-supervision training.
    # Keeps dense lesion segmentation important while also rewarding clinical detection and grading.
    return 0.40 * lesion_dice + 0.30 * gland_bacc + 0.20 * region_bacc + 0.10 * grade_kappa


def save_checkpoint(path, model, criterion, optimizer, scheduler, epoch, best_metric, config_name):
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "config_name": config_name,
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def main():
    Config.set_seed()
    device = torch.device(Config.DEVICE)
    exp_name = Config.get_experiment_name()
    save_path = os.path.join(Config.EXP_DIR, exp_name)
    os.makedirs(save_path, exist_ok=True)

    log_file_path = os.path.join(save_path, "console_output.log")
    sys.stdout = Logger(log_file_path)
    print(f"✅ Console outputs will be saved to: {log_file_path}")
    Config.show()

    train_loader = DataLoader(
        ProstateUnifiedDataset(Config.TRAIN_CSV, Config.UNIFIED_DATA_DIR, is_train=True),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ProstateUnifiedDataset(Config.VAL_CSV, Config.UNIFIED_DATA_DIR, is_train=False),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = ProstateMixedSupervisionNet(in_channels=Config.IN_CHANNELS, num_grade_classes=Config.NUM_CLASSES).to(device)
    criterion = MixedSupervisionLoss(
        csPCa_threshold=getattr(Config, "CSPC_THRESHOLD", 3),
        invalid_sys_label=getattr(Config, "INVALID_SYS_LABEL", -1),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": Config.LR, "weight_decay": Config.WEIGHT_DECAY},
            # Do not apply weight decay to learned uncertainty/log-variance terms.
            {"params": criterion.parameters(), "lr": Config.LR * 10, "weight_decay": 0.0},
        ]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_metric = -1.0
    early_stop_counter = 0
    history = []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")

        t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)

        print(f"Train | {t_track.print_train_summary()}")
        print(f"Val   | {v_track.print_val_summary()}")

        current_weights = {k: torch.exp(-v.detach()).item() for k, v in criterion.log_vars.items()}
        print("--- Learned EM Multipliers ---")
        print(f"Grade  [TBx: {current_weights['grade_tbx']:.3f} | SBx: {current_weights['grade_sbx']:.3f}]")
        print(
            f"Lesion [Dense: {current_weights['lesion_dense']:.3f} | "
            f"Sparse: {current_weights['lesion_sparse']:.3f} | Sys: {current_weights['lesion_sys']:.3f}]"
        )
        print(f"Gland  [Dense: {current_weights['gland']:.3f}]")

        epoch_log = {"epoch": epoch}
        epoch_log.update(t_track.get_train_dict())
        epoch_log.update(v_track.get_val_dict())
        history.append(epoch_log)

        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

        cur_metric = select_validation_metric(v_track)
        print(f"Selection metric ({getattr(Config, 'BEST_MODEL_METRIC', 'composite')}): {cur_metric:.4f}")

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
            if early_stop_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"Early stop triggered at epoch {epoch}")
                break

        scheduler.step()


if __name__ == "__main__":
    main()
