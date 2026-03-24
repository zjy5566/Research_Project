import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Config
from dataset import ProstateUnifiedDataset
from model import ProstateMixedSupervisionNet
from Loss_function import MixedSupervisionLoss
import utils

# ==========================================
# [新增] 动态权重调度函数 (Curriculum Learning)
# ==========================================
def update_loss_weights(criterion, epoch):
    """
    根据 Epoch 动态调整三个子任务的权重。
    阶段 1 (Epoch 1-10): [1.0, 0.0, 0.0] 专注 PUB 强监督，学习基础形状。
    阶段 2 (Epoch 11-30): [1.0, 0.5, 0.5] 引入 TCIA 和 PROMIS，结合系统/靶向数据学习多中心特征。
    阶段 3 (Epoch 31+): [0.5, 1.0, 1.0] 降低 PUB 权重，重点依靠临床活检的弱监督压制假阳性。
    """
    if epoch <= 10:
        criterion.l_w_dense = 1.0
        criterion.l_w_sparse = 0.0
        criterion.l_w_regional = 0.0
    elif 10 < epoch <= 30:
        criterion.l_w_dense = 1.0
        criterion.l_w_sparse = 0.5
        criterion.l_w_regional = 0.5
    else:
        criterion.l_w_dense = 0.5
        criterion.l_w_sparse = 1.0
        criterion.l_w_regional = 1.0
        
    print(f"🚀 [Epoch {epoch} Weight Update] Dense(PUB): {criterion.l_w_dense:.1f} | "
          f"Sparse(TCIA): {criterion.l_w_sparse:.1f} | Regional(PROMIS): {criterion.l_w_regional:.1f}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = utils.MetricTracker()
    
    # 获取带有进度条的枚举迭代器
    from tqdm import tqdm
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        imgs = batch['input'].to(device)
        z_mask = batch['zones_mask'].to(device)
        
        optimizer.zero_grad()
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)
        
        # 接收细分的 lesion 子 loss
        total_loss, l_grad, l_sys, l_les, l_les_dense, l_les_sparse, l_les_sys, l_gland = criterion(
            g_p, s_g_p, l_p, s_l_p, gl_p,
            batch['target_mask'].to(device), batch['sys_labels'].to(device),
            batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
            batch['has_target'].to(device), batch['has_sys'].to(device),
            batch['has_lesion'].to(device), batch['has_gland'].to(device)
        )
        
        total_loss.backward()
        optimizer.step()
        
        tracker.update_losses(
            total_loss.item(), (l_grad + l_sys).item(), l_les.item(), 
            l_les_dense.item(), l_les_sparse.item(), l_les_sys.item(), l_gland.item()
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
    
    # 初始化 Criterion
    criterion = MixedSupervisionLoss(
        Config.LAMBDA_GRADE, Config.LAMBDA_TB, Config.LAMBDA_SYS, Config.LAMBDA_LESION, Config.LAMBDA_GLAND,
        Config.LESION_W_DENSE, Config.LESION_W_SPARSE, Config.LESION_W_REGIONAL, Config.CSPC_THRESHOLD, Config.LESION_W_SMALL
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_metric, early_stop_counter, history = -1, 0, []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        
        # ========================================================
        # [核心调用] 在每个 Epoch 开始前，调用函数更新 Loss 权重！
        # ========================================================
        update_loss_weights(criterion, epoch)
        
        t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)
        
        print(f"Train | {t_track.print_train_summary()}")
        print(f"Val   | {v_track.print_val_summary()}")

        # 拼接日志字典
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
                print(f"Early stop triggered at epoch {epoch}")
                break
        
        scheduler.step()

if __name__ == "__main__": 
    main()