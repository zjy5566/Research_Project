import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. 12分区文献定义字典 ---
ZONE_DICT = {
    'LEFT LATERAL BASE': 1,
    'LEFT LATERAL MID': 2,
    'LEFT LATERAL APEX': 3,
    'LEFT BASE': 4,
    'LEFT MID': 5,
    'LEFT APEX': 6,
    'RIGHT BASE': 7,
    'RIGHT MID': 8,
    'RIGHT APEX': 9,
    'RIGHT LATERAL BASE': 10,
    'RIGHT LATERAL MID': 11,
    'RIGHT LATERAL APEX': 12
}

def get_isup_label(primary, secondary):
    """
    计算 Gleason 分数并映射为训练用的 ISUP Label
    0: 背景 (Background)
    1: 良性 (Benign / Negative)
    2: ISUP 1 (Gleason 3+3=6)
    3: ISUP 2 (Gleason 3+4=7)
    4: ISUP 3 (Gleason 4+3=7)
    5: ISUP 4 (Gleason 8)
    6: ISUP 5 (Gleason 9-10)
    """
    if pd.isna(primary) or pd.isna(secondary): 
        return 1 
    p, s = int(primary), int(secondary)
    if p + s <= 6: return 2
    if p + s == 7: return 3 if p == 3 else 4
    if p + s == 8: return 5
    if p + s >= 9: return 6
    return 1

def create_needle_mask(image_64, tip_idx, base_idx, radius=2):
    """在 3D 空间中生成一根柱状针道"""
    mask_arr = np.zeros(sitk.GetArrayViewFromImage(image_64).shape, dtype=np.uint8)
    
    num_points = 100
    points_z = np.linspace(tip_idx[2], base_idx[2], num_points)
    points_y = np.linspace(tip_idx[1], base_idx[1], num_points)
    points_x = np.linspace(tip_idx[0], base_idx[0], num_points)
    
    for z, y, x in zip(points_z, points_y, points_x):
        iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
        if 0 <= iz < 32 and 0 <= iy < 64 and 0 <= ix < 64:
            mask_arr[iz, iy, ix] = 1

    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(image_64)
    dilated_mask = sitk.BinaryDilate(mask_sitk > 0, [radius, radius, radius], sitk.sitkBall)
    return sitk.GetArrayFromImage(dilated_mask)

def generate_12_zone_mask(crop_t2, gland_binary):
    """
    基于物理坐标空间，自动将前列腺分割为 12 个解剖区域
    背景: 0, 区域: 1~12
    """
    zones_mask = np.zeros_like(gland_binary, dtype=np.uint8)
    if not np.any(gland_binary):
        return sitk.GetImageFromArray(zones_mask)

    z_idx, y_idx, x_idx = np.where(gland_binary > 0)
    
    # 将体素索引转为真实物理毫米坐标 (利用 SimpleITK 的映射系)
    # 在医学标准 LPS 坐标系下: +X 是左 (Left), +Z 是上 (Superior/Base)
    phys_coords = [crop_t2.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in zip(z_idx, y_idx, x_idx)]
    phys_coords = np.array(phys_coords)
    
    X = phys_coords[:, 0]
    Z = phys_coords[:, 2]
    
    x_min, x_max = X.min(), X.max()
    z_min, z_max = Z.min(), Z.max()

    # Z轴划分: APEX (下/小), MID, BASE (上/大)
    z_thresh1 = z_min + (z_max - z_min) / 3
    z_thresh2 = z_min + 2 * (z_max - z_min) / 3

    # X轴划分: 右外侧, 右内侧, 左内侧, 左外侧 (+X为左侧)
    x_mid = (x_min + x_max) / 2
    x_left_mid = x_mid + (x_max - x_mid) / 2
    x_right_mid = x_min + (x_mid - x_min) / 2

    for i in range(len(z_idx)):
        x_val, z_val = X[i], Z[i]

        # 判断 Z 轴 (Base, Mid, Apex)
        if z_val > z_thresh2: z_str = "BASE"
        elif z_val > z_thresh1: z_str = "MID"
        else: z_str = "APEX"

        # 判断 X 轴 (Left/Right, Medial/Lateral)
        if x_val > x_left_mid: x_str = "LEFT LATERAL"
        elif x_val > x_mid: x_str = "LEFT"
        elif x_val > x_right_mid: x_str = "RIGHT"
        else: x_str = "RIGHT LATERAL"

        zone_name = f"{x_str} {z_str}"
        zones_mask[z_idx[i], y_idx[i], x_idx[i]] = ZONE_DICT.get(zone_name, 0)

    zones_sitk = sitk.GetImageFromArray(zones_mask)
    zones_sitk.CopyInformation(crop_t2)
    return zones_sitk

def process_patient_biopsy(patient_id, biopsy_df, processed_dir):
    # 1. 路径与读取
    crop_t2_path = os.path.join(processed_dir, patient_id, 't2_crop.nii.gz')
    gland_mask_path = os.path.join(processed_dir, patient_id, 'gland_mask_crop.nii.gz')
    
    if not os.path.exists(crop_t2_path) or not os.path.exists(gland_mask_path):
        return 

    try:
        crop_t2 = sitk.ReadImage(crop_t2_path)
        gland_mask_crop = sitk.ReadImage(gland_mask_path)
        gland_binary = (sitk.GetArrayFromImage(gland_mask_crop) > 0).astype(np.uint8)
    except Exception as e:
        print(f"Error reading images for {patient_id}: {e}")
        return

    # 2. 数据初始化
    p_data = biopsy_df[biopsy_df['Patient Number'] == patient_id]
    
    target_mask_arr = np.zeros((32, 64, 64), dtype=np.uint8)
    sys_labels = np.zeros(12, dtype=np.uint8)  # 对应 12 个分区的数组
    found_any_target = False

    # 3. 遍历穿刺点
    for _, row in p_data.iterrows():
        try:
            core_label = str(row['Core Label']).strip().upper()
            isup_label = get_isup_label(row['Primary Gleason'], row['Secondary Gleason'])
            
            # --- 分支 A: 靶向穿刺生成 Mask ---
            if core_label == 'TARGET OR PRIOR POSITIVE':
                tip_phys = [row['Bx Tip X (MRI Coord)'], row['Bx Tip Y (MRI Coord)'], row['Bx Tip Z (MRI Coord)']]
                base_phys = [row['Bx Base X (MRI Coord)'], row['Bx Base Y (MRI Coord)'], row['Bx Base Z (MRI Coord)']]
                
                tip_idx_crop = crop_t2.TransformPhysicalPointToIndex(tip_phys)
                base_idx_crop = crop_t2.TransformPhysicalPointToIndex(base_phys)

                needle_arr = create_needle_mask(crop_t2, tip_idx_crop, base_idx_crop)
                # 针道内填入病理 Label (交叉处取最大恶意程度)
                target_mask_arr = np.maximum(target_mask_arr, needle_arr * isup_label)
                found_any_target = True
                
            # --- 分支 B: 系统穿刺提取 Array ---
            else:
                if core_label in ZONE_DICT:
                    idx = ZONE_DICT[core_label] - 1  # 数组 0 索引对应区域 1
                    # 存入对应分区的最高病理分数
                    sys_labels[idx] = max(sys_labels[idx], isup_label)
        except Exception as e:
            continue

    # 4. 约束保存
    save_folder = os.path.join(processed_dir, patient_id)
    
    # 保存靶向 Mask (与 Gland 相乘约束在体内)
    if found_any_target:
        target_mask_arr = target_mask_arr * gland_binary
        t_mask = sitk.GetImageFromArray(target_mask_arr)
        t_mask.CopyInformation(crop_t2)
        sitk.WriteImage(t_mask, os.path.join(save_folder, 'target_bx.nii.gz'))
        
    # 保存生成的 12 分区前列腺掩膜
    zones_sitk = generate_12_zone_mask(crop_t2, gland_binary)
    sitk.WriteImage(zones_sitk, os.path.join(save_folder, 'zones_mask.nii.gz'))
    
    # 保存系统穿刺标签为 Numpy 数组 (1D Array)
    np.save(os.path.join(save_folder, 'systematic_labels.npy'), sys_labels)

# --- 执行主流程 ---
if __name__ == "__main__":
    BIOPSY_EXCEL = r'F:\RP_dataset\Target biosy\TCIA-Biopsy-Data_2020-07-14.xlsx'
    PROCESSED_ROOT = r'F:\RP_dataset\Target biosy\Processed_Target_Biopsy'

    if not os.path.exists(BIOPSY_EXCEL):
        print(f"Error: Cannot find Excel file at {BIOPSY_EXCEL}")
    else:
        print("Loading Excel...")
        df = pd.read_excel(BIOPSY_EXCEL)
        
        patient_list = df['Patient Number'].unique()
        print(f"Total patients in Excel: {len(patient_list)}")

        for pid in tqdm(patient_list, desc="Processing Biopsy"):
            process_patient_biopsy(pid, df, PROCESSED_ROOT)

        print("\nSuccess: Check your folders in Processed_Target_Biopsy")