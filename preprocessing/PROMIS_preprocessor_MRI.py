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
# 2. 单病例处理流程 (修改版)
# ==========================================
def preprocess_promis_case(case_dir, output_dir, target_spacing=[1.0, 1.0, 2.24], crop_size=[64, 64, 32]):
    case_id = os.path.basename(case_dir)
    save_path = os.path.join(output_dir, case_id)
    
    # if os.path.exists(os.path.join(save_path, 'zones_mask.nii.gz')):
    #     # print(f"Skip {case_id}: already processed.") 
    #     return
    
    os.makedirs(save_path, exist_ok=True)
    
    try:
        t2 = sitk.ReadImage(os.path.join(case_dir, 't2.nii.gz'))
        adc = sitk.ReadImage(os.path.join(case_dir, 'adc.nii.gz'))
        dwi = sitk.ReadImage(os.path.join(case_dir, 'dwi.nii.gz'))
        gland = sitk.ReadImage(os.path.join(case_dir, 'gland.nii.gz'))
        # --- 新增: 读取开源代码生成的原尺寸 20 zones mask ---
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
        # 对于 Mask 标签（gland 和 zones），严格使用 NearestNeighbor，避免产生小数
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # --- 对所有模态和Mask进行重采样 ---
    t2_res = resample(t2)
    gland_res = resample(gland, is_label=True)
    zones_res = resample(zones, is_label=True) # 新增: 对 20 zones 进行重采样
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
    zones_res = pad_if_needed(zones_res, crop_size) # 新增: 对 20 zones 进行补齐

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
    zones_crop = sitk.RegionOfInterest(zones_res, crop_size, roi_start) # 新增: 对 20 zones 进行同步裁剪

    def finalize(img):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

    # 保存 1: 多模态输入张量
    input_tensor = np.stack([finalize(t2_crop), finalize(dwi_crop), finalize(adc_crop)], axis=0)
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