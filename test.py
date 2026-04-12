import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ProstateUnifiedDataset
from model import ProstateMixedSupervisionNet
import utils

# =====================================================================
# 可视化核心函数：生成全景对比图 (包含所有 GT 和 Masked 预测)
# =====================================================================
def save_comprehensive_vis(img_vol, gland_gt, lesion_gt, target_gt, zones_mask, sys_labels, 
                           lesion_prob, grade_class, pid, save_path):
    D, H, W = img_vol.shape
    
    # 1. 重建系统活检 (Systematic Biopsy) 的全空间 GT 映射
    sys_gt_vol = np.zeros_like(zones_mask, dtype=np.float32)
    for z_idx in range(1, 21):
        if sys_labels[z_idx - 1] >= 0: # 过滤掉未穿刺的区域 (-1)
            sys_gt_vol[zones_mask == z_idx] = sys_labels[z_idx - 1]
            
    # 2. 智能切片选择机制 
    best_z = 0
    if target_gt.sum() > 0:
        best_z = np.argmax(target_gt.sum(axis=(1, 2)))
    elif lesion_gt.sum() > 0:
        best_z = np.argmax(lesion_gt.sum(axis=(1, 2)))
    elif sys_gt_vol.sum() > 0:
        best_z = np.argmax(sys_gt_vol.sum(axis=(1, 2)))
    else:
        best_z = np.argmax(lesion_prob.sum(axis=(1, 2)))
        
    if best_z == 0 and lesion_prob.sum() == 0:
        best_z = D // 2

    # 3. 提取 2D 切片
    s_img = img_vol[best_z]
    s_gland = gland_gt[best_z]
    s_lesion = lesion_gt[best_z]
    s_target = target_gt[best_z]
    s_sys = sys_gt_vol[best_z]
    
    s_lesion_prob = lesion_prob[best_z]
    s_grade = grade_class[best_z]

    # 4. 开始绘图 (2行 x 4列)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Patient: {pid} | Selected Slice: {best_z}/{D}', fontsize=18, fontweight='bold')
    
    axes[0, 0].imshow(s_img, cmap='gray')
    axes[0, 0].set_title('T2 MRI')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(s_img, cmap='gray')
    axes[0, 1].contour(s_gland, levels=[0.5], colors='cyan', linewidths=2)
    axes[0, 1].set_title('T2 + Gland Contour')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(s_img, cmap='gray')
    risk_masked = np.ma.masked_where(s_lesion_prob < 0.1, s_lesion_prob) 
    im_risk = axes[0, 2].imshow(risk_masked, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted Lesion Risk (Gland Masked)')
    axes[0, 2].axis('off')
    fig.colorbar(im_risk, ax=axes[0, 2], fraction=0.046, pad=0.04)

    cmap_grade = plt.cm.get_cmap('YlOrRd', 6)
    axes[0, 3].imshow(s_img, cmap='gray')
    grade_masked = np.ma.masked_where((s_grade == 0) | (s_gland == 0), s_grade) 
    im_grade = axes[0, 3].imshow(grade_masked, cmap=cmap_grade, alpha=0.6, vmin=0, vmax=5)
    axes[0, 3].set_title('Predicted ISUP Grade (Gland Masked)')
    axes[0, 3].axis('off')
    fig.colorbar(im_grade, ax=axes[0, 3], fraction=0.046, pad=0.04, ticks=range(6))

    axes[1, 0].imshow(s_img, cmap='gray')
    if s_lesion.sum() > 0:
        axes[1, 0].imshow(np.ma.masked_where(s_lesion == 0, s_lesion), cmap='autumn', alpha=0.6)
    axes[1, 0].set_title('GT: Dense Lesion (PUB)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(s_img, cmap='gray')
    if s_target.sum() > 0:
        axes[1, 1].imshow(np.ma.masked_where(s_target == 0, s_target), cmap=cmap_grade, alpha=0.8, vmin=0, vmax=5)
    axes[1, 1].set_title('GT: Target Biopsy (TCIA)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(s_img, cmap='gray')
    if s_sys.sum() > 0:
        axes[1, 2].imshow(np.ma.masked_where(s_sys == 0, s_sys), cmap=cmap_grade, alpha=0.5, vmin=0, vmax=5)
    axes[1, 2].set_title('GT: Systematic Zones')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(s_img, cmap='gray')
    if s_lesion.sum() > 0:
        axes[1, 3].contour(s_lesion, levels=[0.5], colors='green', linewidths=2, label='GT')
    if s_lesion_prob.sum() > 0:
        axes[1, 3].contour((s_lesion_prob > 0.5).astype(int), levels=[0.5], colors='red', linewidths=2, linestyles='dashed')
    axes[1, 3].set_title('Contours: GT(Green) vs Pred(Red)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    # ==========================================
    # 1. 环境准备与路径配置
    # ==========================================
    Config.set_seed()
    device = torch.device(Config.DEVICE)
    
    TEST_DIR = os.path.join(Config.BASE_DIR, "test")
    VIS_DIR = os.path.join(TEST_DIR, "vis")
    os.makedirs(VIS_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(TEST_DIR, "best_model.pth")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到模型文件！请检查路径: {MODEL_PATH}")
    
    print(f"🚀 [Test Start] 加载模型: {MODEL_PATH}")
    print(f"📷 [Test Output] 可视化结果将保存在: {VIS_DIR}")

    # ==========================================
    # 2. 数据与模型初始化
    # ==========================================
    test_dataset = ProstateUnifiedDataset(Config.TEST_CSV, Config.UNIFIED_DATA_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    model = ProstateMixedSupervisionNet(in_channels=Config.IN_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # ==========================================
    # 3. 评测执行循环 (结合 utils 内的指标)
    # ==========================================
    results = []
    
    # 实例化临床指标评估器 (Gland / Region BAcc)
    balanced_evaluator = utils.BalancedAccuracyEvaluator(prob_threshold=0.5, cs_pca_threshold=3)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing & Rendering"):
            imgs = batch['input'].to(device)
            z_mask = batch['zones_mask'].to(device)
            
            pid = batch.get('pid', ["Unknown_Patient"])[0]
            gt_gland = batch['gland_mask'].numpy()[0, 0]       
            gt_lesion = batch['lesion_mask'].numpy()[0, 0]     
            gt_target = batch['target_mask'].numpy()[0, 0]     
            gt_zones = batch['zones_mask'].numpy()[0, 0]       
            gt_sys_labels = batch['sys_labels'].numpy()[0]     
            
            has_lesion = batch['has_lesion'].numpy()[0]
            has_target = batch['has_target'].numpy()[0]
            has_sys = batch['has_sys'].numpy()[0]

            # --- 前向传播 ---
            grade_logits, sys_grade_preds, lesion_logits, sys_lesion_preds, gland_pred = model(imgs, z_mask)
            
            # --- 更新全局临床分类指标 (Gland / Region Spec, Sens, BAcc) ---
            balanced_evaluator.update(
                pred_prob_3d=torch.sigmoid(lesion_logits[0, 0]),
                gland_mask=batch['gland_mask'][0, 0].to(device),
                zones_mask=batch['zones_mask'][0, 0].to(device) if has_sys > 0 else None,
                sys_labels=batch['sys_labels'][0].to(device) if has_sys > 0 else None,
                lesion_mask=batch['lesion_mask'][0, 0].to(device) if has_lesion > 0 else None,
                target_mask=batch['target_mask'][0, 0].to(device) if has_target > 0 else None,
                has_sys=bool(has_sys > 0), has_lesion=bool(has_lesion > 0), has_target=bool(has_target > 0)
            )
            
            # --- 后处理：应用 Gland Mask 屏蔽外部预测 ---
            pred_lesion_prob = torch.sigmoid(lesion_logits[0, 0]).cpu().numpy()
            pred_lesion_prob = pred_lesion_prob * (gt_gland > 0)  
            
            pred_grade_probs = torch.softmax(grade_logits[0], dim=0) 
            pred_grade_class = torch.argmax(pred_grade_probs, dim=0).cpu().numpy()
            pred_grade_class = pred_grade_class * (gt_gland > 0)  

            # --- 获取所有细分评价指标 ---
            dice_score, f1_score_val, sens_val = np.nan, np.nan, np.nan
            kappa_score = np.nan

            # 1. 如果有病灶掩膜 (PUB): 计算 Dice, F1, Sens
            if has_lesion > 0:
                pred_binary = (pred_lesion_prob > 0.5).astype(np.float32)
                t_pred = torch.tensor(pred_binary)
                t_gt = torch.tensor(gt_lesion)
                
                dice_score = utils.compute_dice(t_pred.unsqueeze(0), t_gt.unsqueeze(0))
                f1_score_val = utils.compute_f1(t_pred, t_gt)
                sens_val = utils.compute_sens(t_pred, t_gt)

            # 2. 如果有系统活检标签 (PROMIS): 计算 Kappa 协同分数
            if has_sys > 0 and sys_grade_preds is not None:
                sys_pred_flat = torch.argmax(sys_grade_preds[0], dim=-1).cpu()
                sys_true_flat = torch.tensor(gt_sys_labels)
                valid_mask = sys_true_flat > 0 # 过滤掉 -1 和未病变区域
                if valid_mask.sum() > 0:
                    kappa_score = utils.compute_kappa(sys_pred_flat[valid_mask], sys_true_flat[valid_mask])

            # 将各项细分指标存入表格
            results.append({
                'Patient_ID': pid,
                'Source': 'PUB' if has_lesion > 0 else ('TCIA' if has_target > 0 else 'PROMIS'),
                'Lesion_Dice': dice_score,
                'Lesion_F1': f1_score_val,
                'Lesion_Sens': sens_val,
                'Grade_Kappa': kappa_score
            })
            
            # --- 渲染并保存全景图 ---
            img_t2 = imgs[0, 0].cpu().numpy()
            save_path = os.path.join(VIS_DIR, f"{pid}.png")
            save_comprehensive_vis(
                img_t2, gt_gland, gt_lesion, gt_target, gt_zones, gt_sys_labels,
                pred_lesion_prob, pred_grade_class, pid, save_path
            )

    # ==========================================
    # 4. 汇总导出打印
    # ==========================================
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(TEST_DIR, "test_metrics_per_patient.csv")
    df_results.to_csv(csv_path, index=False)
    
    # 提取全局评估器计算的 Balanced Accuracy 最终指标
    bacc_metrics = balanced_evaluator.compute_metrics()
    
    print("\n" + "="*50)
    print("🎯 测试指标总结 (TEST METRICS SUMMARY)")
    print("="*50)
    print(f"🖼️ 全景可视化图已保存至: {VIS_DIR}")
    print(f"📊 各病人详细指标表已保存至: {csv_path}\n")
    
    if df_results['Lesion_Dice'].notna().any():
        print(f"📈 PUB 数据集 (Voxel-level Metrics):")
        print(f"   - Mean Dice: {df_results['Lesion_Dice'].mean():.4f}")
        print(f"   - Mean F1:   {df_results['Lesion_F1'].mean():.4f}")
        print(f"   - Mean Sens: {df_results['Lesion_Sens'].mean():.4f}\n")
        
    if df_results['Grade_Kappa'].notna().any():
        print(f"📈 PROMIS 数据集 (Region-level Grade):")
        print(f"   - Mean Kappa: {df_results['Grade_Kappa'].mean():.4f}\n")
        
    print(f"📈 整体临床评价指标 (Clinical-level BAcc):")
    print(f"   - Gland Sens:  {bacc_metrics['gland_sens']:.4f}")
    print(f"   - Gland Spec:  {bacc_metrics['gland_spec']:.4f}")
    print(f"   - Gland BAcc:  {bacc_metrics['gland_bacc']:.4f}")
    print(f"   - Region Sens: {bacc_metrics['region_sens']:.4f}")
    print(f"   - Region Spec: {bacc_metrics['region_spec']:.4f}")
    print(f"   - Region BAcc: {bacc_metrics['region_bacc']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()