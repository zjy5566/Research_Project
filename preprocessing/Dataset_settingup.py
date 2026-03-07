import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_unified_dataset(base_dir):
    # --- 1. 定义路径 ---
    # 根据你的结构修正了路径
    src_pub = os.path.join(base_dir, 'Dataset_prostate_MRI')
    src_promis = os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS')
    
    # 注意：如果你的 Processed_TCIA 下面不小心多嵌套了一层 Processed_PROMIS，请将此处修改为对应的真实路径
    # 这里按照标准逻辑指向 Processed_TCIA
    src_tcia = os.path.join(base_dir, 'Target biosy', 'Processed_TCIA')
    
    dst_root = os.path.join(base_dir, 'Unified_Dataset')
    os.makedirs(dst_root, exist_ok=True)
    
    registry_data = []

    # --- 2. 整合 TCIA 靶向活检集 ---
    print("Processing TCIA Target Biopsy Dataset...")
    if os.path.exists(src_tcia):
        # 兼容处理：防患于未然，检查是不是嵌套在子文件夹里
        tcia_search_dir = src_tcia
        if os.path.exists(os.path.join(src_tcia, 'Processed_PROMIS')):
            tcia_search_dir = os.path.join(src_tcia, 'Processed_PROMIS')
            
        tcia_patients = [d for d in os.listdir(tcia_search_dir) if d.startswith('Prostate-MRI-US-Biopsy-')]
        
        for pid in tqdm(tcia_patients):
            src_p_dir = os.path.join(tcia_search_dir, pid)
            
            # 【完整性检查】必须同时拥有输入张量、12区掩膜、12区标签
            req_files = ['input_tensor.npy', 'zones_mask.nii.gz', 'systematic_labels.npy']
            if not all(os.path.exists(os.path.join(src_p_dir, f)) for f in req_files):
                continue # 文件不全，直接跳过，不创建空文件夹
            
            # 判断是否有靶向掩膜
            has_target = 1 if os.path.exists(os.path.join(src_p_dir, 'target_bx.nii.gz')) else 0
            
            new_pid = f"TCIA_{pid.split('-')[-1]}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 复制基础文件
            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(os.path.join(src_p_dir, 'zones_mask.nii.gz'), os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
            # 复制系统标签并重命名以作区分
            shutil.copy2(os.path.join(src_p_dir, 'systematic_labels.npy'), os.path.join(dst_p_dir, 'systematic_labels_12.npy'))
            
            if has_target:
                shutil.copy2(os.path.join(src_p_dir, 'target_bx.nii.gz'), os.path.join(dst_p_dir, 'target_bx.nii.gz'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'TCIA',
                'has_target': has_target, 'has_sys_12': 1, 'has_sys_20': 0, 'has_lesion': 0
            })

    # --- 3. 整合 PROMIS 数据集 ---
    print("\nProcessing PROMIS Dataset...")
    if os.path.exists(src_promis):
        promis_patients = [d for d in os.listdir(src_promis) if d.startswith('P-')]
        
        for pid in tqdm(promis_patients):
            src_p_dir = os.path.join(src_promis, pid)
            
            # 【完整性检查】PROMIS 现在都在同一个文件夹下，要求这三者必备
            req_files = ['input_tensor.npy', 'zones_mask.nii.gz', 'systematic_labels.npy']
            if not all(os.path.exists(os.path.join(src_p_dir, f)) for f in req_files):
                continue
                
            new_pid = f"PROMIS_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 复制文件
            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(os.path.join(src_p_dir, 'zones_mask.nii.gz'), os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
            # 复制系统标签并重命名以作区分
            shutil.copy2(os.path.join(src_p_dir, 'systematic_labels.npy'), os.path.join(dst_p_dir, 'systematic_labels_20.npy'))
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PROMIS',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 1, 'has_lesion': 0
            })

    # --- 4. 整合 公开 MRI 数据集 (Dataset_prostate_MRI) ---
    print("\nProcessing Dataset_prostate_MRI...")
    if os.path.exists(src_pub):
        pub_files = os.listdir(src_pub)
        pub_ids = sorted(list(set([f.split('_')[0] for f in pub_files if f.endswith('_img.npy')])))
        
        for pid in tqdm(pub_ids):
            src_img = os.path.join(src_pub, f"{pid}_img.npy")
            src_lab = os.path.join(src_pub, f"{pid}_lab.npy")
            
            # 【完整性检查】必须同时拥有影像和标注
            if not os.path.exists(src_img) or not os.path.exists(src_lab):
                continue
                
            new_pid = f"PUB_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 重命名为标准的规范命名，并注意 lab 是病灶 (lesion)
            shutil.copy2(src_img, os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(src_lab, os.path.join(dst_p_dir, 'lesion_mask.npy'))
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PUB',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 0, 'has_lesion': 1
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

    # 按照数据源 (source) 进行分层划分
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