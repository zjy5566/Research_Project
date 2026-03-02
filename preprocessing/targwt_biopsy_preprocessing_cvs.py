import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def create_needle_mask(image_64, tip_idx, base_idx, radius=2):
    """
    在 64x64x32 的空间内绘制一根柱状针道
    """
    # 获取数组（z, y, x）
    mask_arr = np.zeros(sitk.GetArrayViewFromImage(image_64).shape, dtype=np.uint8)
    
    # 线性插值生成点
    num_points = 100
    points_z = np.linspace(tip_idx[2], base_idx[2], num_points)
    points_y = np.linspace(tip_idx[1], base_idx[1], num_points)
    points_x = np.linspace(tip_idx[0], base_idx[0], num_points)
    
    for z, y, x in zip(points_z, points_y, points_x):
        iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
        # 边界检查
        if 0 <= iz < 32 and 0 <= iy < 64 and 0 <= ix < 64:
            mask_arr[iz, iy, ix] = 1

    # 转回 SITK 并膨胀
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(image_64)
    # 使用球形结构元素进行膨胀，radius=2 像素约等于 1.5-2mm 的物理厚度
    dilated_mask = sitk.BinaryDilate(mask_sitk > 0, [radius, radius, radius], sitk.sitkBall)
    return dilated_mask

def process_patient_biopsy(patient_id, biopsy_df, original_t2_path, processed_dir):
    """
    处理单个病人的所有穿刺点，区分靶向和系统穿刺并生成 Mask
    """
    # 1. 路径检查
    crop_t2_path = os.path.join(processed_dir, patient_id, 't2_crop.nii.gz')
    
    if not os.path.exists(original_t2_path):
        return # 缺少原始图
    if not os.path.exists(crop_t2_path):
        # 如果你裁剪后的文件名不是 t2_crop.nii.gz，请修改此处
        return 

    try:
        orig_t2 = sitk.ReadImage(original_t2_path)
        crop_t2 = sitk.ReadImage(crop_t2_path)
    except Exception as e:
        print(f"Error reading images for {patient_id}: {e}")
        return

    # 2. 计算裁剪偏移量 (必须与前文中心裁剪逻辑一致)
    orig_size = orig_t2.GetSize()
    crop_size = [64, 64, 32]
    roi_start = [(orig_size[i] - crop_size[i]) // 2 for i in range(3)]

    # 3. 提取该病人数据
    p_data = biopsy_df[biopsy_df['Patient Number'] == patient_id]
    
    target_mask_arr = np.zeros((32, 64, 64), dtype=np.uint8)
    systematic_mask_arr = np.zeros((32, 64, 64), dtype=np.uint8)

    found_any = False
    for _, row in p_data.iterrows():
        try:
            # 物理坐标
            tip_phys = [row['Bx Tip X (MRI Coord)'], row['Bx Tip Y (MRI Coord)'], row['Bx Tip Z (MRI Coord)']]
            base_phys = [row['Bx Base X (MRI Coord)'], row['Bx Base Y (MRI Coord)'], row['Bx Base Z (MRI Coord)']]
            
            # 转换坐标
            tip_idx_orig = orig_t2.TransformPhysicalPointToIndex(tip_phys)
            base_idx_orig = orig_t2.TransformPhysicalPointToIndex(base_phys)
            
            # 计算裁剪后的相对坐标
            tip_idx_crop = [tip_idx_orig[i] - roi_start[i] for i in range(3)]
            base_idx_crop = [base_idx_orig[i] - roi_start[i] for i in range(3)]

            # 生成 Mask
            needle_sitk = create_needle_mask(crop_t2, tip_idx_crop, base_idx_crop)
            needle_arr = sitk.GetArrayFromImage(needle_sitk)

            # 分类合并
            if str(row['Core Label']).strip().upper() == 'TARGET OR PRIOR POSITIVE':
                target_mask_arr = np.maximum(target_mask_arr, needle_arr)
            else:
                systematic_mask_arr = np.maximum(systematic_mask_arr, needle_arr)
            found_any = True
        except:
            continue

    # 4. 保存
    if found_any:
        save_folder = os.path.join(processed_dir, patient_id)
        
        t_mask = sitk.GetImageFromArray(target_mask_arr)
        t_mask.CopyInformation(crop_t2)
        sitk.WriteImage(t_mask, os.path.join(save_folder, 'target_bx.nii.gz'))
        
        s_mask = sitk.GetImageFromArray(systematic_mask_arr)
        s_mask.CopyInformation(crop_t2)
        sitk.WriteImage(s_mask, os.path.join(save_folder, 'systematic_bx.nii.gz'))

# --- 执行主流程 ---
if __name__ == "__main__":
    # 请确保以下路径准确无误
    BIOPSY_EXCEL = r'F:\RP_dataset\Target biosy\TCIA-Biopsy-Data_2020-07-14.xlsx'
    ORIG_MRI_ROOT = r'F:\RP_dataset\Extracted_Target_Biopsy'
    PROCESSED_ROOT = r'F:\RP_dataset\Processed_Target_Biopsy'

    if not os.path.exists(BIOPSY_EXCEL):
        print(f"Error: Cannot find Excel file at {BIOPSY_EXCEL}")
    else:
        print("Loading Excel...")
        # 修正：使用 read_excel
        df = pd.read_excel(BIOPSY_EXCEL)
        
        # 修正：拼写 unique
        patient_list = df['Patient Number'].unique()
        print(f"Total patients in Excel: {len(patient_list)}")

        for pid in tqdm(patient_list, desc="Processing Biopsy"):
            # 路径示例: F:\RP_dataset\Extracted_Target_Biopsy\Prostate-MRI-US-Biopsy-0001\t2.nii.gz
            t2_path = os.path.join(ORIG_MRI_ROOT, pid, 't2.nii.gz')
            
            # 如果你的裁剪后的 T2 文件在每个病人文件夹下叫别的名字，请在 process_patient_biopsy 函数内修改
            process_patient_biopsy(pid, df, t2_path, PROCESSED_ROOT)

        print("\nSuccess: Check your folders in Processed_Target_Biopsy")