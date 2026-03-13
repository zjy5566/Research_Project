import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from dataset import ProstateUnifiedDataset
from model import ProstateMixedSupervisionNet
from Loss_function_all import MixedSupervisionLoss
import utils

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = utils.MetricTracker()
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        imgs = batch['input'].to(device)
        z_mask = batch['zones_mask'].to(device)
        
        optimizer.zero_grad()
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)
        
        total_loss, l_grad, l_sys, l_les, l_gland = criterion(
            g_p, s_g_p, l_p, s_l_p, gl_p,
            batch['target_mask'].to(device), batch['sys_labels'].to(device),
            batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
            batch['has_target'].to(device), batch['has_sys'].to(device),
            batch['has_lesion'].to(device), batch['has_gland'].to(device)
        )
        
        total_loss.backward()
        optimizer.step()
        
        # 统计 Loss：将主任务 Grade(局部+区域) 合并统计
        tracker.update_losses(total_loss.item(), (l_grad + l_sys).item(), l_les.item(), l_gland.item())
        pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
        
    return tracker

@torch.no_grad()
@torch.no_grad()
def validate(model, loader, device, epoch, save_dir):
    model.eval()
    tracker = utils.MetricTracker()
    vis_dir = os.path.join(save_dir, Config.VIS_SUBDIR, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)

    # 智能采样记录器：限制每种数据集在验证时最多画几张图 (防止磁盘爆满)
    saved_counts = {'PUB': 0, 'TCIA': 0, 'PROMIS': 0}
    max_saves_per_type = 2 

    for i, batch in enumerate(tqdm(loader, desc="Validation")):
        imgs, z_mask = batch['input'].to(device), batch['zones_mask'].to(device)
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)

        # ---------------------------------------------------------
        # [修改] 智能可视化采样逻辑 (Smart Visual Sampling)
        # ---------------------------------------------------------
        r_probs = torch.sigmoid(l_p)
        g_preds = torch.argmax(g_p, dim=1, keepdim=True)
        
        # 为了让 Grade 背景干净，应用物理截断 (使用预测的 Gland Mask)
        if gl_p is not None:
            gland_bin = (torch.sigmoid(gl_p) > 0.5).float()
            r_probs = r_probs * gland_bin
            g_preds = g_preds * gland_bin.long()

        for b in range(imgs.size(0)):
            # 判别当前样本归属的数据集类型
            if batch['has_lesion'][b] > 0:
                d_type = 'PUB'
            elif batch['has_target'][b] > 0:
                d_type = 'TCIA'
            elif batch['has_sys'][b] > 0:
                d_type = 'PROMIS'
            else:
                continue

            # 如果这类数据集还没画够，就执行画图
            if saved_counts[d_type] < max_saves_per_type:
                # 提取供画图使用的 Ground Truth 字典 (转回 CPU Numpy)
                gt_dict = {
                    'type': d_type,
                    'lesion_mask': batch['lesion_mask'][b][0].cpu().numpy(),
                    'target_mask': batch['target_mask'][b][0].cpu().numpy(),
                    'zones_mask': batch['zones_mask'][b][0].cpu().numpy(),
                    'sys_labels': batch['sys_labels'][b].cpu().numpy()
                }
                
                vis_filename = f"{d_type}_{batch['pid'][b]}.png"
                utils.visualize_predictions(
                    imgs[b], r_probs[b], g_preds[b], gt_dict,
                    os.path.join(vis_dir, vis_filename), batch['pid'][b]
                )
                saved_counts[d_type] += 1

        # ---------------------------------------------------------
        # 评估各分支 (逻辑保持原样)
        # ---------------------------------------------------------
        if batch['has_gland'].sum() > 0:
            idx = batch['has_gland'] > 0
            tracker.gland_dice.update(utils.compute_dice((torch.sigmoid(gl_p[idx]) > 0.5).float(), batch['gland_mask'][idx].to(device)))
        
        if batch['has_lesion'].sum() > 0:
            idx = batch['has_lesion'] > 0
            lp, lt = torch.sigmoid(l_p[idx]), batch['lesion_mask'][idx].to(device)
            lb = (lp > 0.5).float()
            tracker.lesion_dice.update(utils.compute_dice(lb, lt))
            tracker.lesion_auc.update(utils.compute_auc(lp, lt))
            sens, spec = utils.compute_sens_spec(lb, lt)
            tracker.lesion_sens.update(sens)
            tracker.lesion_spec.update(spec)

        if batch['has_sys'].sum() > 0:
            idx = batch['has_sys'] > 0
            v_z = batch['sys_labels'][idx] > 0
            if v_z.sum() > 0:
                tracker.grade_kappa.update(utils.compute_kappa(
                    torch.argmax(s_g_p[idx], dim=-1)[v_z], batch['sys_labels'][idx][v_z]
                ))

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

    best_kappa, early_stop_counter, history = -1, 0, []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}")
        t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_track = validate(model, val_loader, device, epoch, save_path)
        
        print(f"Train {t_track.print_summary()}\nVal   {v_track.print_summary()}")

        # 核心修复点：使用 prefix 区分训练和验证，并确保列名与 utils.plot 对齐
        epoch_log = {'epoch': epoch, **t_track.get_dict(prefix='train_'), **v_track.get_dict(prefix='val_')}
        history.append(epoch_log)
        
        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

        cur_kappa = v_track.grade_kappa.avg
        if cur_kappa > best_kappa:
            best_kappa, early_stop_counter = cur_kappa, 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(f"--> Best Model Saved ({best_kappa:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch}"); break
        scheduler.step()

if __name__ == "__main__": main()