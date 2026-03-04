import os
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

# --- 4. 归一化函数 ---
def normalize_array(img):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    std = np.std(arr)
    return (arr - np.mean(arr)) / (std if std > 1e-7 else 1.0)

# --- 5. 单个病例处理流程 ---
def process_single_patient(p_id, src_path, dst_root):
    # 输入路径 (Extracted 文件夹)
    t2_file = os.path.join(src_path, 't2.nii.gz')
    adc_file = os.path.join(src_path, 'adc.nii.gz')
    dwi_file = os.path.join(src_path, 'dwi.nii.gz')
    
    # 输出路径 (Processed 文件夹)
    save_dir = os.path.join(dst_root, p_id)
    # gland_mask 应该位于已经生成的 Processed 目录下对应病人的文件夹中
    mask_file = os.path.join(save_dir, 'gland_mask.nii.gz') 

    # 1. 检查是否存在 input_tensor.npy (跳过已处理)
    # if os.path.exists(os.path.join(save_dir, 'input_tensor.npy')):
    #     return "ALREADY_PROCESSED"

    # 2. 检查四要素是否齐全 (T2, ADC, DWI, Mask)
    if not all([os.path.exists(f) for f in [t2_file, adc_file, dwi_file, mask_file]]):
        return "MISSING_DATA"

    try:
        t2 = sitk.ReadImage(t2_file)
        adc = sitk.ReadImage(adc_file)
        dwi = sitk.ReadImage(dwi_file)
        mask = sitk.ReadImage(mask_file)

        t2_res = resample_to_spacing(t2)
        adc_res = resample_to_spacing(adc)
        dwi_res = resample_to_spacing(dwi)
        mask_res = resample_to_spacing(mask, is_label=True)

        adc_reg = register_images(t2_res, adc_res)
        dwi_reg = register_images(t2_res, dwi_res)

        t2_crop = mask_centered_crop(t2_res, mask_res)
        adc_crop = mask_centered_crop(adc_reg, mask_res)
        dwi_crop = mask_centered_crop(dwi_reg, mask_res)
        mask_crop = mask_centered_crop(mask_res, mask_res)

        os.makedirs(save_dir, exist_ok=True)
        sitk.WriteImage(t2_crop, os.path.join(save_dir, 't2_crop.nii.gz'))
        sitk.WriteImage(adc_crop, os.path.join(save_dir, 'adc_crop.nii.gz'))
        sitk.WriteImage(dwi_crop, os.path.join(save_dir, 'dwi_crop.nii.gz'))
        sitk.WriteImage(mask_crop, os.path.join(save_dir, 'gland_mask_crop.nii.gz'))

        input_tensor = np.stack([
            normalize_array(t2_crop), 
            normalize_array(dwi_crop), 
            normalize_array(adc_crop)
        ], axis=0)
        
        np.save(os.path.join(save_dir, 'input_tensor.npy'), input_tensor)
        return "SUCCESS"
    except Exception as e:
        print(f"\n[Error] {p_id}: {e}")
        return "FAILED"

# --- 6. 主程序入口 ---
if __name__ == "__main__":
    SRC_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    DST_ROOT = r'F:\RP_dataset\Target biosy\Processed_Target_Biopsy'

    patients = [d for d in os.listdir(SRC_ROOT) if d.startswith('Prostate-MRI-US-Biopsy-')]
    print(f"Total potential patients found: {len(patients)}")

    # 统计计数器
    stats = {
        "complete_modalities": 0,  # 模态齐全的病例
        "newly_processed": 0,      # 本次成功处理的
        "already_exists": 0,       # 之前已经做好的
        "failed": 0,               # 处理过程中报错的
        "missing": 0               # 缺模态或缺 Mask 的
    }

    pbar = tqdm(patients, desc="Batch Processing")
    for p_id in pbar:
        p_path = os.path.join(SRC_ROOT, p_id)
        result = process_single_patient(p_id, p_path, DST_ROOT)
        
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
        
        # 实时显示统计
        pbar.set_postfix({
            "Complete": stats["complete_modalities"], 
            "New": stats["newly_processed"],
            "Exist": stats["already_exists"]
        })
            
    # 打印最终汇总报告
    print(f"\n" + "="*40)
    print(f"Preprocessing Summary:")
    print(f"  - Total Complete Cases (T2+ADC+DWI+Mask): {stats['complete_modalities']}")
    print(f"    └─ Newly Processed: {stats['newly_processed']}")
    print(f"    └─ Already Exists:  {stats['already_exists']}")
    print(f"  - Incomplete / Missing Data: {stats['missing']}")
    print(f"  - Processing Failed:         {stats['failed']}")
    print(f"  - Output Directory: {DST_ROOT}")
    print("="*40)
