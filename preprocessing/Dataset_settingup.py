import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_unified_dataset(base_dir):
    # --- 1. 定义路径 ---
    src_pub = os.path.join(base_dir, 'Dataset_prostate_MRI','Dataset_prostate_MRI_nodwi')
    src_promis = os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS_nodwi')
    src_tcia = os.path.join(base_dir, 'Target biosy', 'Processed_TCIA_nodwi')
    
    dst_root = os.path.join(base_dir, 'Unified_Dataset')
    os.makedirs(dst_root, exist_ok=True)
    
    registry_data = []

    # --- 2. 整合 TCIA 靶向活检集 ---
    print("Processing TCIA Target Biopsy Dataset...")
    if os.path.exists(src_tcia):
        tcia_search_dir = src_tcia
        if os.path.exists(os.path.join(src_tcia, 'Processed_PROMIS')):
            tcia_search_dir = os.path.join(src_tcia, 'Processed_PROMIS')
            
        tcia_patients = [d for d in os.listdir(tcia_search_dir) if d.startswith('Prostate-MRI-US-Biopsy-')]
        
        for pid in tqdm(tcia_patients):
            src_p_dir = os.path.join(tcia_search_dir, pid)
            
            # 【修改点 1：降低 TCIA 的强约束】
            # TCIA 的核心是靶向穿刺，系统穿刺 (zones_mask) 现在是可选项了！
            # 我们只需要保证它至少有 input_tensor 和 (靶向 或 系统两者之一)
            if not os.path.exists(os.path.join(src_p_dir, 'input_tensor.npy')):
                continue 
            
            # 检查各项数据是否存在
            has_target = 1 if os.path.exists(os.path.join(src_p_dir, 'target_bx.nii.gz')) else 0
            has_sys_12 = 1 if (os.path.exists(os.path.join(src_p_dir, 'zones_mask.nii.gz')) and 
                               os.path.exists(os.path.join(src_p_dir, 'systematic_labels.npy'))) else 0
            has_gland = 1 if os.path.exists(os.path.join(src_p_dir, 'gland_mask_crop.nii.gz')) else 0
            
            # 如果既没有靶向也没有系统活检，这个病人就是无用数据，跳过
            if has_target == 0 and has_sys_12 == 0:
                continue
            
            new_pid = f"TCIA_{pid.split('-')[-1]}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 拷贝基础图像
            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))
            
            # 【修改点 2：条件拷贝】只有存在的文件才拷贝
            if has_sys_12:
                shutil.copy2(os.path.join(src_p_dir, 'zones_mask.nii.gz'), os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
                shutil.copy2(os.path.join(src_p_dir, 'systematic_labels.npy'), os.path.join(dst_p_dir, 'systematic_labels_12.npy'))
            
            if has_target:
                shutil.copy2(os.path.join(src_p_dir, 'target_bx.nii.gz'), os.path.join(dst_p_dir, 'target_bx.nii.gz'))
            
            if has_gland:
                # 统一重命名为 gland_mask.nii.gz
                shutil.copy2(os.path.join(src_p_dir, 'gland_mask_crop.nii.gz'), os.path.join(dst_p_dir, 'gland_mask.nii.gz'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'TCIA',
                'has_target': has_target, 'has_sys_12': has_sys_12, 'has_sys_20': 0, 'has_lesion': 0, 'has_gland': has_gland
            })

    # --- 3. 整合 PROMIS 数据集 ---
    print("\nProcessing PROMIS Dataset...")
    if os.path.exists(src_promis):
        promis_patients = [d for d in os.listdir(src_promis) if d.startswith('P-')]
        
        for pid in tqdm(promis_patients):
            src_p_dir = os.path.join(src_promis, pid)
            
            req_files = ['input_tensor.npy', 'zones_mask.nii.gz', 'systematic_labels.npy']
            if not all(os.path.exists(os.path.join(src_p_dir, f)) for f in req_files):
                continue
                
            has_gland = 1 if os.path.exists(os.path.join(src_p_dir, 'gland_mask.nii.gz')) else 0
                
            new_pid = f"PROMIS_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(os.path.join(src_p_dir, 'zones_mask.nii.gz'), os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
            shutil.copy2(os.path.join(src_p_dir, 'systematic_labels.npy'), os.path.join(dst_p_dir, 'systematic_labels_20.npy'))
            
            if has_gland:
                shutil.copy2(os.path.join(src_p_dir, 'gland_mask.nii.gz'), os.path.join(dst_p_dir, 'gland_mask.nii.gz'))
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PROMIS',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 1, 'has_lesion': 0, 'has_gland': has_gland
            })

    # --- 4. 整合 公开 MRI 数据集 (PUB) ---
    print("\nProcessing Dataset_prostate_MRI...")
    if os.path.exists(src_pub):
        pub_files = os.listdir(src_pub)
        pub_ids = sorted(list(set([f.split('_')[0] for f in pub_files if f.endswith('_img.npy')])))
        
        for pid in tqdm(pub_ids):
            src_img = os.path.join(src_pub, f"{pid}_img.npy")
            src_lab = os.path.join(src_pub, f"{pid}_lab.npy")
            src_zone = os.path.join(src_pub, f"{pid}_zone.npy") # 新增: PUB的腺体/区域掩膜
            
            # 【完整性检查】必须同时拥有影像、病灶(lab) 和 腺体(zone)
            if not all(os.path.exists(f) for f in [src_img, src_lab, src_zone]):
                continue
                
            new_pid = f"PUB_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 拷贝并规范化命名
            shutil.copy2(src_img, os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(src_lab, os.path.join(dst_p_dir, 'lesion_mask.npy'))
            shutil.copy2(src_zone, os.path.join(dst_p_dir, 'gland_mask.npy')) # 强行改名为 gland_mask.npy
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PUB',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 0, 'has_lesion': 1, 'has_gland': 1
            })

    # --- 5. 生成索引表与分层划分 ---
    print("\nGenerating Data Splits (70% Train, 10% Val, 20% Test)...")
    if len(registry_data) == 0:
        print("Error: No valid data found! Please check your source directories.")
        return
        
    df = pd.DataFrame(registry_data)
    
    splits_dir = os.path.join(dst_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    df.to_csv(os.path.join(splits_dir, 'dataset_registry.csv'), index=False)

    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['source'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['source'])

    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

    print(f"\nDone! Total Patients: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Unified dataset successfully created at: {dst_root}")

if __name__ == "__main__":
    BASE_DIR = r"F:\RP_dataset"
    create_unified_dataset(BASE_DIR)