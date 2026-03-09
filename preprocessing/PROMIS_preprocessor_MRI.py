import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ==========================================
# 1. 图像配准函数 (保持不变)
# ==========================================
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
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_image)


# ==========================================
# 2. 单病例处理流程 (引入前景归一化)
# ==========================================
def preprocess_promis_case(case_dir, output_dir, target_spacing=[1.0, 1.0, 2.24], crop_size=[64, 64, 32]):
    case_id = os.path.basename(case_dir)
    save_path = os.path.join(output_dir, case_id)
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        t2 = sitk.ReadImage(os.path.join(case_dir, 't2.nii.gz'))
        adc = sitk.ReadImage(os.path.join(case_dir, 'adc.nii.gz'))
        dwi = sitk.ReadImage(os.path.join(case_dir, 'dwi.nii.gz'))
        gland = sitk.ReadImage(os.path.join(case_dir, 'gland.nii.gz'))
        zones = sitk.ReadImage(os.path.join(case_dir, 'gland_zone_20level_set1.nii.gz'))
    except Exception as e:
        print(f"\n[Error] Missing files in {case_id}: {e}")
        return

    def resample(image, is_label=False):
        new_size = [int(round(image.GetSize()[i] * image.GetSpacing()[i] / target_spacing[i])) for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # --- 对所有模态和Mask进行重采样 ---
    t2_res = resample(t2)
    gland_res = resample(gland, is_label=True)
    zones_res = resample(zones, is_label=True)
    adc_res_init = resample(adc)
    dwi_res_init = resample(dwi)

    try:
        adc_reg = register_images(t2_res, adc_res_init)
        dwi_reg = register_images(t2_res, dwi_res_init)
    except Exception as e:
        print(f"\n[Error] Registration failed for {case_id}: {e}")
        return

    def pad_if_needed(img, target_size):
        curr_size = img.GetSize()
        pad_lower = [0, 0, 0]; pad_upper = [0, 0, 0]; need_pad = False
        for i in range(3):
            if curr_size[i] < target_size[i]:
                diff = target_size[i] - curr_size[i]
                pad_lower[i] = diff // 2; pad_upper[i] = diff - pad_lower[i]
                need_pad = True
        return sitk.ConstantPad(img, pad_lower, pad_upper, 0.0) if need_pad else img

    # --- 对所有模态和Mask进行越界补齐 ---
    t2_res = pad_if_needed(t2_res, crop_size)
    adc_reg = pad_if_needed(adc_reg, crop_size)
    dwi_reg = pad_if_needed(dwi_reg, crop_size)
    gland_res = pad_if_needed(gland_res, crop_size)
    zones_res = pad_if_needed(zones_res, crop_size)

    # --- 以 Gland 的质心计算裁剪坐标 ---
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(gland_res)
    if not stats.HasLabel(1):
        centroid_index = [s // 2 for s in t2_res.GetSize()]
    else:
        centroid_world = stats.GetCentroid(1)
        centroid_index = t2_res.TransformPhysicalPointToIndex(centroid_world)

    img_size = t2_res.GetSize()
    roi_start = [max(0, min(int(centroid_index[i] - crop_size[i] // 2), img_size[i] - crop_size[i])) for i in range(3)]

    # --- 执行同步裁剪 ---
    t2_crop = sitk.RegionOfInterest(t2_res, crop_size, roi_start)
    adc_crop = sitk.RegionOfInterest(adc_reg, crop_size, roi_start)
    dwi_crop = sitk.RegionOfInterest(dwi_reg, crop_size, roi_start)
    gland_crop = sitk.RegionOfInterest(gland_res, crop_size, roi_start)
    zones_crop = sitk.RegionOfInterest(zones_res, crop_size, roi_start)

    # ========================================================
    # 【核心修改点】：前景 Z-score 局部归一化逻辑
    # ========================================================
    # 先提取出 Gland 的 Numpy 数组，作为计算统计量的掩膜标尺
    gland_arr = sitk.GetArrayFromImage(gland_crop).astype(np.uint8)

    def finalize_foreground(img, mask_arr):
        """仅利用前列腺腺体内部的像素计算均值和方差"""
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # 提取前列腺内部有效像素
        valid_pixels = arr[mask_arr > 0]
        
        # 防御性回退：如果掩膜为空，则全局归一化
        if len(valid_pixels) == 0:
            mean_val = np.mean(arr)
            std_val = np.std(arr)
        else:
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)
            
        return (arr - mean_val) / (std_val + 1e-8)

    # 保存 1: 多模态输入张量 (传入 gland_arr 作为前景计算标尺)
    input_tensor = np.stack([
        finalize_foreground(t2_crop, gland_arr), 
        finalize_foreground(dwi_crop, gland_arr), 
        finalize_foreground(adc_crop, gland_arr)
    ], axis=0)
    
    np.save(os.path.join(save_path, 'input_tensor.npy'), input_tensor)
    
    # 保存 2: Gland 腺体掩膜 
    sitk.WriteImage(gland_crop, os.path.join(save_path, 'gland_mask.nii.gz'))
    
    # 保存 3: 裁剪并对齐后的 20 Zones 掩膜
    sitk.WriteImage(zones_crop, os.path.join(save_path, 'zones_mask.nii.gz'))


# ==========================================
# 3. 批量处理逻辑
# ==========================================
def batch_preprocess(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    all_cases = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d)) and d.startswith('P-')]
    
    print(f"Total cases found: {len(all_cases)}")

    for case_id in tqdm(all_cases, desc="Batch Processing PROMIS"):
        case_dir = os.path.join(src_root, case_id)
        preprocess_promis_case(case_dir, dst_root)

if __name__ == "__main__":
    MRI_DATA_ROOT = r'F:/RP_dataset/derived PROMIS data set/MRI'
    PROCESSED_ROOT = r'F:/RP_dataset/derived PROMIS data set/Processed_PROMIS'
    
    batch_preprocess(MRI_DATA_ROOT, PROCESSED_ROOT)
    print("\nAll done!")