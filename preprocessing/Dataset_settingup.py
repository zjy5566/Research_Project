import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_unified_dataset(base_dir):
    # --- 1. 定义路径 ---
    src_pub = os.path.join(base_dir, 'Dataset_prostate_MRI','Dataset_prostate_MRI')
    src_promis_img = os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS')
    src_promis_lab = os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS_Labels_NPY')
    src_tcia = os.path.join(base_dir, 'Target biosy', 'Processed_Target_Biopsy')
    
    dst_root = os.path.join(base_dir, 'Unified_Dataset')
    os.makedirs(dst_root, exist_ok=True)
    
    registry_data = []

    # --- 2. 整合 TCIA 靶向活检集 ---
    print("Processing TCIA Target Biopsy Dataset...")
    if os.path.exists(src_tcia):
        tcia_patients = [d for d in os.listdir(src_tcia) if d.startswith('Prostate-MRI-US-Biopsy-')]
        for pid in tqdm(tcia_patients):
            new_pid = f"TCIA_{pid.split('-')[-1]}" # 变为 TCIA_0001
            src_p_dir = os.path.join(src_tcia, pid)
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 复制文件
            for f_name in ['input_tensor.npy', 'target_bx.nii.gz', 'zones_mask.nii.gz']:
                src_f = os.path.join(src_p_dir, f_name)
                if os.path.exists(src_f):
                    shutil.copy2(src_f, os.path.join(dst_p_dir, f_name))
            
            # 区分 labels 命名
            src_lab = os.path.join(src_p_dir, 'systematic_labels.npy')
            if os.path.exists(src_lab):
                shutil.copy2(src_lab, os.path.join(dst_p_dir, 'systematic_labels_12.npy'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'TCIA',
                'has_target': 1, 'has_sys_12': 1, 'has_sys_20': 0, 'has_gland': 0
            })

    # --- 3. 整合 PROMIS 数据集 ---
    print("\nProcessing PROMIS Dataset...")
    if os.path.exists(src_promis_img):
        promis_patients = [d for d in os.listdir(src_promis_img) if d.startswith('P-')]
        for pid in tqdm(promis_patients):
            new_pid = f"PROMIS_{pid}"
            src_p_dir = os.path.join(src_promis_img, pid)
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            # 复制输入张量
            src_img = os.path.join(src_p_dir, 'input_tensor.npy')
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(dst_p_dir, 'input_tensor.npy'))
                
            # 从另一个文件夹复制 20-zone 标签
            src_lab = os.path.join(src_promis_lab, f"{pid}.npy")
            if os.path.exists(src_lab):
                shutil.copy2(src_lab, os.path.join(dst_p_dir, 'systematic_labels_20.npy'))
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PROMIS',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 1, 'has_gland': 0
            })

    # --- 4. 整合 公开 MRI 数据集 (Dataset_prostate_MRI) ---
    print("\nProcessing Dataset_prostate_MRI...")
    if os.path.exists(src_pub):
        # 提取 000 到 434 的 ID
        pub_files = os.listdir(src_pub)
        pub_ids = sorted(list(set([f.split('_')[0] for f in pub_files if f.endswith('.npy')])))
        
        for pid in tqdm(pub_ids):
            new_pid = f"PUB_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)
            
            src_img = os.path.join(src_pub, f"{pid}_img.npy")
            src_lab = os.path.join(src_pub, f"{pid}_lab.npy")
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(dst_p_dir, 'input_tensor.npy'))
            if os.path.exists(src_lab):
                shutil.copy2(src_lab, os.path.join(dst_p_dir, 'gland_mask.npy'))
                
            registry_data.append({
                'patient_id': new_pid, 'source': 'PUB',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 0, 'has_gland': 1
            })

    # --- 5. 生成索引表与分层划分 ---
    print("\nGenerating Data Splits (70% Train, 10% Val, 20% Test)...")
    df = pd.DataFrame(registry_data)
    
    splits_dir = os.path.join(dst_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    df.to_csv(os.path.join(splits_dir, 'dataset_registry.csv'), index=False)

    # 按照数据源 (source) 进行分层划分，保证各个子集的比例健康
    # 第一步: 提取 20% 作为 Test
    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['source'])
    
    # 第二步: 从剩下的 80% 中提取 12.5% 作为 Val (0.8 * 0.125 = 0.1 -> 10% 总量)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['source'])

    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

    print(f"\nDone! Total Patients: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    BASE_DIR = r"F:\RP_dataset"
    create_unified_dataset(BASE_DIR)