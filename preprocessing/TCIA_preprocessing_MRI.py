import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. 核心配准函数 (保持不变) ---
def register_images(fixed_image, moving_image, is_label=False):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.15)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(moving_image)

# --- 2. 重采样函数 (保持不变) ---
def resample_to_spacing(image, target_spacing=[1.0, 1.0, 2.24], is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(3)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)

# --- 3. 以前列腺掩膜为中心的裁剪函数 ---
def mask_centered_crop(img, mask, target_size=[64, 64, 32]):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)
    
    if label_shape_filter.GetNumberOfLabels() == 0:
        center_index = [s // 2 for s in img.GetSize()]
    else:
        centroid_world = label_shape_filter.GetCentroid(1) 
        center_index = img.TransformPhysicalPointToIndex(centroid_world)

    roi_start = [center_index[i] - target_size[i] // 2 for i in range(3)]
    img_size = img.GetSize()
    pad_lower = [max(0, -roi_start[i]) for i in range(3)]
    pad_upper = [max(0, roi_start[i] + target_size[i] - img_size[i]) for i in range(3)]
    
    if sum(pad_lower) + sum(pad_upper) > 0:
        img = sitk.ConstantPad(img, pad_lower, pad_upper, 0)
        roi_start = [roi_start[i] + pad_lower[i] for i in range(3)]
    
    return sitk.RegionOfInterest(img, target_size, roi_start)

# ========================================================
# --- 4. 归一化函数 (前景 Z-score 局部归一化) ---
# ========================================================
def normalize_array(img, mask_arr):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    valid_pixels = arr[mask_arr > 0]
    
    if len(valid_pixels) == 0:
        mean_val = np.mean(arr)
        std_val = np.std(arr)
    else:
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)
        
    return (arr - mean_val) / (std_val + 1e-8)


# --- 5. 单个病例处理流程 ---
def process_single_patient(folder_name, src_path, dst_root):
    t2_file = os.path.join(src_path, 't2.nii.gz')
    adc_file = os.path.join(src_path, 'adc.nii.gz')
    dwi_file = os.path.join(src_path, 'dwi.nii.gz')
    mask_file = os.path.join(src_path, 'gland_mask.nii.gz') 
    
    # 扩展：寻找额外的 Mask 文件和 NPY 文件
    target_mask_file = os.path.join(src_path, 'target_mask.nii.gz') # 由 STL 生成的靶点
    needle_mask_file = os.path.join(src_path, 'target_bx_needle.nii.gz') # 由 Excel 坐标生成的针道
    zones_mask_file = os.path.join(src_path, 'zones_mask.nii.gz') # 由 12 分区算法生成的系统活检分区
    sys_labels_file = os.path.join(src_path, 'systematic_labels.npy') # 系统活检结果
    
    save_dir = os.path.join(dst_root, folder_name)

    # 1. 检查四要素是否齐全 (T2, ADC, DWI, Mask)
    if not all([os.path.exists(f) for f in [t2_file, adc_file, dwi_file, mask_file]]):
        return "MISSING_DATA"

    try:
        os.makedirs(save_dir, exist_ok=True)
        
        t2 = sitk.ReadImage(t2_file)
        adc = sitk.ReadImage(adc_file)
        dwi = sitk.ReadImage(dwi_file)
        mask = sitk.ReadImage(mask_file)

        # 重采样
        t2_res = resample_to_spacing(t2)
        adc_res = resample_to_spacing(adc)
        dwi_res = resample_to_spacing(dwi)
        mask_res = resample_to_spacing(mask, is_label=True)

        # 配准
        adc_reg = register_images(t2_res, adc_res)
        dwi_reg = register_images(t2_res, dwi_res)

        # 中心裁剪 (核心金标准位置)
        t2_crop = mask_centered_crop(t2_res, mask_res)
        adc_crop = mask_centered_crop(adc_reg, mask_res)
        dwi_crop = mask_centered_crop(dwi_reg, mask_res)
        mask_crop = mask_centered_crop(mask_res, mask_res)

        # 保存主要影像
        sitk.WriteImage(t2_crop, os.path.join(save_dir, 't2_crop.nii.gz'))
        sitk.WriteImage(adc_crop, os.path.join(save_dir, 'adc_crop.nii.gz'))
        sitk.WriteImage(dwi_crop, os.path.join(save_dir, 'dwi_crop.nii.gz'))
        sitk.WriteImage(mask_crop, os.path.join(save_dir, 'gland_mask_crop.nii.gz'))

        # ========================================================
        # [新增] 额外掩膜处理流程：完全沿用 T2/Gland 的裁剪逻辑
        # ========================================================
        # 1. 靶点 (Target STL 转化来的)
        if os.path.exists(target_mask_file):
            t_mask = sitk.ReadImage(target_mask_file)
            t_mask_res = resample_to_spacing(t_mask, is_label=True)
            # 使用前面算好的腺体 mask_res 作为参照进行空间切割
            t_mask_crop = mask_centered_crop(t_mask_res, mask_res)
            sitk.WriteImage(t_mask_crop, os.path.join(save_dir, 'target_mask_crop.nii.gz'))

        # 2. 针道 (Excel 坐标生成)
        if os.path.exists(needle_mask_file):
            n_mask = sitk.ReadImage(needle_mask_file)
            n_mask_res = resample_to_spacing(n_mask, is_label=True)
            n_mask_crop = mask_centered_crop(n_mask_res, mask_res)
            sitk.WriteImage(n_mask_crop, os.path.join(save_dir, 'target_bx_needle_crop.nii.gz'))

        # 3. 12 分区系统活检掩膜
        if os.path.exists(zones_mask_file):
            z_mask = sitk.ReadImage(zones_mask_file)
            z_mask_res = resample_to_spacing(z_mask, is_label=True)
            z_mask_crop = mask_centered_crop(z_mask_res, mask_res)
            sitk.WriteImage(z_mask_crop, os.path.join(save_dir, 'zones_mask_crop.nii.gz'))
            
        # 4. 拷贝系统活检结果的 NPY 文件
        if os.path.exists(sys_labels_file):
            shutil.copy2(sys_labels_file, os.path.join(save_dir, 'systematic_labels.npy'))

        # ========================================================
        # 提取 mask 数组并传入归一化函数构建多通道张量
        # ========================================================
        mask_arr = sitk.GetArrayFromImage(mask_crop).astype(np.uint8)

        input_tensor = np.stack([
            normalize_array(t2_crop, mask_arr), 
            normalize_array(dwi_crop, mask_arr), 
            normalize_array(adc_crop, mask_arr)
        ], axis=0)
        
        np.save(os.path.join(save_dir, 'input_tensor.npy'), input_tensor)
        return "SUCCESS"
    except Exception as e:
        print(f"\n[Error] {folder_name}: {e}")
        return "FAILED"

# --- 6. 主程序入口 ---
if __name__ == "__main__":
    SRC_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    DST_ROOT = r'F:\RP_dataset\Target biosy\Processed_TCIA'

    # 这里读取真实的文件夹名称 (无论是 Prostate-MRI-US-Biopsy-0159 还是 Prostate-MRI-US-Biopsy-0159_12345 都会被抓取)
    folders = [d for d in os.listdir(SRC_ROOT) if d.startswith('Prostate-MRI-US-Biopsy-') and os.path.isdir(os.path.join(SRC_ROOT, d))]
    print(f"Total potential cases found: {len(folders)}")

    stats = {
        "complete_modalities": 0, 
        "newly_processed": 0,      
        "already_exists": 0,       
        "failed": 0,               
        "missing": 0               
    }

    pbar = tqdm(folders, desc="Batch Processing")
    for folder_name in pbar:
        src_path = os.path.join(SRC_ROOT, folder_name)
        result = process_single_patient(folder_name, src_path, DST_ROOT)
        
        if result == "SUCCESS":
            stats["complete_modalities"] += 1
            stats["newly_processed"] += 1
        elif result == "ALREADY_PROCESSED":
            stats["complete_modalities"] += 1
            stats["already_exists"] += 1
        elif result == "MISSING_DATA":
            stats["missing"] += 1
        elif result == "FAILED":
            stats["failed"] += 1
        
        pbar.set_postfix({
            "Complete": stats["complete_modalities"], 
            "New": stats["newly_processed"],
            "Exist": stats["already_exists"]
        })
            
    print(f"\n" + "="*40)
    print(f"Preprocessing Summary:")
    print(f"  - Total Complete Cases (T2+ADC+DWI+Mask): {stats['complete_modalities']}")
    print(f"    └─ Newly Processed: {stats['newly_processed']}")
    print(f"    └─ Already Exists:  {stats['already_exists']}")
    print(f"  - Incomplete / Missing Data: {stats['missing']}")
    print(f"  - Processing Failed:         {stats['failed']}")
    print(f"  - Output Directory: {DST_ROOT}")
    print("="*40)