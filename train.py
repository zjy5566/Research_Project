import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Config
from dataset import ProstateUnifiedDataset
from model import ProstateMixedSupervisionNet
from Loss_function import MixedSupervisionLoss
import utils

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = utils.MetricTracker()
    
    # 获取带有进度条的枚举迭代器 (放在 utils 里统一风格，或者这里写简单的 tqdm)
    from tqdm import tqdm
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        imgs = batch['input'].to(device)
        z_mask = batch['zones_mask'].to(device)
        
        optimizer.zero_grad()
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)
        
        # [修改] 接收新增的三个 lesion 子 loss
        total_loss, l_grad, l_sys, l_les, l_les_dense, l_les_sparse, l_les_sys, l_gland = criterion(
            g_p, s_g_p, l_p, s_l_p, gl_p,
            batch['target_mask'].to(device), batch['sys_labels'].to(device),
            batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
            batch['has_target'].to(device), batch['has_sys'].to(device),
            batch['has_lesion'].to(device), batch['has_gland'].to(device)
        )
        
        total_loss.backward()
        optimizer.step()
        
        # [修改] 传递三个 lesion 子 loss 给 tracker
        tracker.update_losses(
            total_loss.item(), 
            (l_grad + l_sys).item(), 
            l_les.item(), 
            l_les_dense.item(), 
            l_les_sparse.item(), 
            l_les_sys.item(), 
            l_gland.item()
        )
        pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})
        
    return tracker

def main():
    Config.set_seed()
    device = torch.device(Config.DEVICE)
    exp_name = Config.get_experiment_name()
    save_path = os.path.join(Config.EXP_DIR, exp_name)
    os.makedirs(save_path, exist_ok=True)
    Config.show()

    train_loader = DataLoader(ProstateUnifiedDataset(Config.TRAIN_CSV, Config.UNIFIED_DATA_DIR, True), 
                              batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(ProstateUnifiedDataset(Config.VAL_CSV, Config.UNIFIED_DATA_DIR, False), 
                            batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = ProstateMixedSupervisionNet(in_channels=Config.IN_CHANNELS).to(device)
    criterion = MixedSupervisionLoss(
        Config.LAMBDA_GRADE, Config.LAMBDA_SYS, Config.LAMBDA_LESION, Config.LAMBDA_GLAND,
        Config.LESION_W_DENSE, Config.LESION_W_SPARSE, Config.LESION_W_REGIONAL, Config.CSPC_THRESHOLD
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_metric, early_stop_counter, history = -1, 0, []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        
        t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # 将验证过程和画图彻底封装进 utils
        v_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)
        
        print(f"Train | {t_track.print_train_summary()}")
        print(f"Val   | {v_track.print_val_summary()}")

        # 拼接日志字典（训练只取 Loss，验证取 Loss 和 Metrics）
        epoch_log = {'epoch': epoch}
        epoch_log.update(t_track.get_train_dict())
        epoch_log.update(v_track.get_val_dict())
        history.append(epoch_log)
        
        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

        # 选择一个核心指标作为早停依据 (如 Lesion Dice 或 Grade Kappa)
        cur_metric = v_track.lesion_dice.avg if Config.LAMBDA_LESION > 0 else v_track.grade_kappa.avg
        if cur_metric > best_metric:
            best_metric, early_stop_counter = cur_metric, 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"--> Best Model Saved (Score: {best_metric:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"Early stop triggered at epoch {epoch}"); break
        
        scheduler.step()

if __name__ == "__main__": 
    main()