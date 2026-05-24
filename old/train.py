# import os
# import sys
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from config import Config
# from dataset import ProstateUnifiedDataset
# from model import ProstateMixedSupervisionNet
# from Loss_function import MixedSupervisionLoss
# import utils

# # ==========================================
# # 控制台日志双写记录器 (Logger)
# # ==========================================
# class Logger(object):
#     """
#     将 sys.stdout 重定向到终端和文件。
#     任何 print() 输出都会同时显示在屏幕上并追加保存到 log_file 中。
#     """
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a", encoding="utf-8")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush() # 强制每次写入后刷新缓冲区，防止程序意外中断时日志丢失

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# # ==========================================
# # 动态权重调度函数 (Curriculum Learning)
# # ==========================================
# def update_loss_weights(criterion, epoch):
#     """
#     受 Config 控制的动态权重调度器。
#     如果 Config 中关闭了该功能，则保持 Loss 类初始化时的默认权重不变。
#     """
#     # 检查开关状态 (兼容之前没写这个变量的情况，默认设为 False)
#     use_dynamic = getattr(Config, 'USE_DYNAMIC_WEIGHTS', False)
#     if not use_dynamic:
#         return  # 不执行动态更新，维持 config 中初始化的静态权重

#     # 获取分段节点，默认为 [10, 30]
#     epochs_nodes = getattr(Config, 'DYNAMIC_WEIGHT_EPOCHS', [10, 30])
#     node1, node2 = epochs_nodes[0], epochs_nodes[1]

#     if epoch <= node1:
#         criterion.l_w_dense = 1.0
#         criterion.l_w_sparse = 0.0
#         criterion.l_w_regional = 0.0
#     elif node1 < epoch <= node2:
#         criterion.l_w_dense = 1.0
#         criterion.l_w_sparse = 0.5
#         criterion.l_w_regional = 0.5
#     else:
#         criterion.l_w_dense = 0.5
#         criterion.l_w_sparse = 1.0
#         criterion.l_w_regional = 1.0
        
#     print(f"🚀 [Epoch {epoch} Weight Update] Dense(PUB): {criterion.l_w_dense:.1f} | "
#           f"Sparse(TCIA): {criterion.l_w_sparse:.1f} | Regional(PROMIS): {criterion.l_w_regional:.1f}")


# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     tracker = utils.MetricTracker()
    
#     from tqdm import tqdm
#     pbar = tqdm(loader, desc="Training")
    
#     for batch in pbar:
#         imgs = batch['input'].to(device)
#         z_mask = batch['zones_mask'].to(device)
        
#         optimizer.zero_grad()
#         g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)
        
#         # 接收新增的三个 lesion 子 loss
#         # 接收细分的 Grade 和 Lesion 子 loss (9个返回值)
#         total_loss, l_grad_tot, l_grad_tbx, l_grad_sbx, l_les_tot, l_les_dense, l_les_sparse, l_les_sys, l_gland = criterion(
#             g_p, s_g_p, l_p, s_l_p, gl_p,
#             batch['target_mask'].to(device), batch['sys_labels'].to(device),
#             batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
#             batch['has_target'].to(device), batch['has_sys'].to(device),
#             batch['has_lesion'].to(device), batch['has_gland'].to(device)
#         )
        
#         total_loss.backward()
#         optimizer.step()
        
#        # 传递给 tracker
#         tracker.update_losses(
#             total_loss.item(), 
#             l_grad_tot.item(), l_grad_tbx.item(), l_grad_sbx.item(), 
#             l_les_tot.item(), l_les_dense.item(), l_les_sparse.item(), l_les_sys.item(), 
#             l_gland.item()
#         )
#         pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})
        
#     return tracker

# def main():
#     Config.set_seed()
#     device = torch.device(Config.DEVICE)
#     exp_name = Config.get_experiment_name()
#     save_path = os.path.join(Config.EXP_DIR, exp_name)
#     os.makedirs(save_path, exist_ok=True)
    
#     # 在生成保存目录后，立即重定向标准输出
#     log_file_path = os.path.join(save_path, "console_output.log")
#     sys.stdout = Logger(log_file_path)
#     print(f"✅ Console outputs will be saved to: {log_file_path}")

#     Config.show()

#     train_loader = DataLoader(ProstateUnifiedDataset(Config.TRAIN_CSV, Config.UNIFIED_DATA_DIR, True), 
#                               batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
#     val_loader = DataLoader(ProstateUnifiedDataset(Config.VAL_CSV, Config.UNIFIED_DATA_DIR, False), 
#                             batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

#     model = ProstateMixedSupervisionNet(in_channels=Config.IN_CHANNELS).to(device)
    
#     # 初始化 Criterion
#     # 初始化 Criterion
#     criterion = MixedSupervisionLoss(
#         lambda_grade=Config.LAMBDA_GRADE, 
#         lambda_lesion=Config.LAMBDA_LESION, 
#         lambda_gland=Config.LAMBDA_GLAND,
#         grade_w_tbx=Config.GRADE_W_TBX,         # [新增] 传入TBx权重
#         grade_w_sbx=Config.GRADE_W_SBX,         # [新增] 传入SBx权重
#         lesion_w_dense=Config.LESION_W_DENSE, 
#         lesion_w_sparse=Config.LESION_W_SPARSE, 
#         lesion_w_regional=Config.LESION_W_REGIONAL, 
#         csPCa_threshold=Config.CSPC_THRESHOLD
#     ).to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

#     best_metric, early_stop_counter, history = -1, 0, []

#     for epoch in range(1, Config.NUM_EPOCHS + 1):
#         print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        
#         # ========================================================
#         # [调用] 依据 Config 检查并更新权重
#         # ========================================================
#         update_loss_weights(criterion, epoch)
        
#         t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         v_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)
        
#         print(f"Train | {t_track.print_train_summary()}")
#         print(f"Val   | {v_track.print_val_summary()}")

#         # 拼接日志字典
#         epoch_log = {'epoch': epoch}
#         epoch_log.update(t_track.get_train_dict())
#         epoch_log.update(v_track.get_val_dict())
#         history.append(epoch_log)
        
#         log_csv = os.path.join(save_path, "train_log.csv")
#         pd.DataFrame(history).to_csv(log_csv, index=False)
#         utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

#         # 选择一个核心指标作为早停依据 (如 Lesion Dice 或 Grade Kappa)
#         cur_metric = v_track.lesion_dice.avg if Config.LAMBDA_LESION > 0 else v_track.grade_kappa.avg
#         if cur_metric > best_metric:
#             best_metric, early_stop_counter = cur_metric, 0
#             torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
#             print(f"--> Best Model Saved (Score: {best_metric:.4f})")
#         else:
#             early_stop_counter += 1
#             if early_stop_counter >= Config.EARLY_STOP_PATIENCE:
#                 print(f"Early stop triggered at epoch {epoch}")
#                 break
        
#         scheduler.step()

# if __name__ == "__main__": 
#     main()
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Config
from dataset import ProstateUnifiedDataset
from model import ProstateMixedSupervisionNet
from Loss_function import MixedSupervisionLoss
import utils

class Logger(object):
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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tracker = utils.MetricTracker()
    from tqdm import tqdm
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        imgs = batch['input'].to(device)
        z_mask = batch['zones_mask'].to(device)
        
        optimizer.zero_grad()
        g_p, s_g_p, l_p, s_l_p, gl_p = model(imgs, z_mask)
        
        # 接收并解包 10 个返回值 (新增 em_weights)
        total_loss, l_grad_tot, l_grad_tbx, l_grad_sbx, l_les_tot, l_les_dense, l_les_sparse, l_les_sys, l_gland, em_weights = criterion(
            g_p, s_g_p, l_p, s_l_p, gl_p,
            batch['target_mask'].to(device), batch['sys_labels'].to(device),
            batch['lesion_mask'].to(device), batch['gland_mask'].to(device),
            batch['has_target'].to(device), batch['has_sys'].to(device),
            batch['has_lesion'].to(device), batch['has_gland'].to(device)
        )
        
        total_loss.backward()
        optimizer.step()
        
        tracker.update_losses(
            total_loss.item(), 
            l_grad_tot.item(), l_grad_tbx.item(), l_grad_sbx.item(), 
            l_les_tot.item(), l_les_dense.item(), l_les_sparse.item(), l_les_sys.item(), 
            l_gland.item(),
            em_weights=em_weights
        )
        pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})
        
    return tracker

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

    train_loader = DataLoader(ProstateUnifiedDataset(Config.TRAIN_CSV, Config.UNIFIED_DATA_DIR, True), 
                              batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(ProstateUnifiedDataset(Config.VAL_CSV, Config.UNIFIED_DATA_DIR, False), 
                            batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = ProstateMixedSupervisionNet(in_channels=Config.IN_CHANNELS).to(device)
    
    # 初始化 Criterion
    criterion = MixedSupervisionLoss(csPCa_threshold=getattr(Config, 'CSPC_THRESHOLD', 3)).to(device)
    
    # 【核心 EM 优化】：将模型参数 W 和 Loss权重参数 λ 一并交给优化器！
    # 对于自动学习的权重，建议分配比主干网络略大的学习率 (x10) 协助其加速寻找稳态
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': Config.LR},
        {'params': criterion.parameters(), 'lr': Config.LR * 10} 
    ], weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    best_metric, early_stop_counter, history = -1, 0, []

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        
        t_track = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_track = utils.validate(model, val_loader, criterion, device, epoch, save_path)
        
        print(f"Train | {t_track.print_train_summary()}")
        print(f"Val   | {v_track.print_val_summary()}")
        
        # 打印验证这一轮后的网络自适应权重 (让我们直观看到它变聪明了多少)
        current_weights = {k: torch.exp(-v).item() for k, v in criterion.log_vars.items()}
        print(f"--- Learned EM Multipliers ---")
        print(f"Grade  [TBx: {current_weights['grade_tbx']:.3f} | SBx: {current_weights['grade_sbx']:.3f}]")
        print(f"Lesion [Dense: {current_weights['lesion_dense']:.3f} | Sparse: {current_weights['lesion_sparse']:.3f} | Sys: {current_weights['lesion_sys']:.3f}]")

        epoch_log = {'epoch': epoch}
        epoch_log.update(t_track.get_train_dict())
        epoch_log.update(v_track.get_val_dict())
        history.append(epoch_log)
        
        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))

        # 这里默认取 Dice 作为最重要的指标进行早停判断
        cur_metric = v_track.lesion_dice.avg
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