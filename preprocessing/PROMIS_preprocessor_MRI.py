import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm  # 如果没有安装，请 pip install tqdm

# --- 保持你之前的 register_images 函数不变 ---
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

# --- 保持你之前的 preprocess_promis_case 函数不变 ---
def preprocess_promis_case(case_dir, output_dir, target_spacing=[1.0, 1.0, 2.24], crop_size=[64, 64, 32]):
    case_id = os.path.basename(case_dir)
    save_path = os.path.join(output_dir, case_id)
    if os.path.exists(os.path.join(save_path, 'input_tensor.npy')):
        # print(f"Skip {case_id}: already processed.") # 如果想断点续传可以开启
        return
    
    reg_check_path = os.path.join(save_path, 'registration_check')
    os.makedirs(reg_check_path, exist_ok=True)
    
    try:
        t2 = sitk.ReadImage(os.path.join(case_dir, 't2.nii.gz'))
        adc = sitk.ReadImage(os.path.join(case_dir, 'adc.nii.gz'))
        dwi = sitk.ReadImage(os.path.join(case_dir, 'dwi.nii.gz'))
        gland = sitk.ReadImage(os.path.join(case_dir, 'gland.nii.gz'))
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

    t2_res = resample(t2)
    gland_res = resample(gland, is_label=True)
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

    t2_res = pad_if_needed(t2_res, crop_size)
    adc_reg = pad_if_needed(adc_reg, crop_size)
    dwi_reg = pad_if_needed(dwi_reg, crop_size)
    gland_res = pad_if_needed(gland_res, crop_size)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(gland_res)
    if not stats.HasLabel(1):
        centroid_index = [s // 2 for s in t2_res.GetSize()]
    else:
        centroid_world = stats.GetCentroid(1)
        centroid_index = t2_res.TransformPhysicalPointToIndex(centroid_world)

    img_size = t2_res.GetSize()
    roi_start = [max(0, min(int(centroid_index[i] - crop_size[i] // 2), img_size[i] - crop_size[i])) for i in range(3)]

    t2_crop = sitk.RegionOfInterest(t2_res, crop_size, roi_start)
    adc_crop = sitk.RegionOfInterest(adc_reg, crop_size, roi_start)
    dwi_crop = sitk.RegionOfInterest(dwi_reg, crop_size, roi_start)

    def finalize(img):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

    input_tensor = np.stack([finalize(t2_crop), finalize(dwi_crop), finalize(adc_crop)], axis=0)
    np.save(os.path.join(save_path, 'input_tensor.npy'), input_tensor)
    
    # 选做：保存预览
    sitk.WriteImage(sitk.GetImageFromArray(input_tensor[0]), os.path.join(save_path, 't2_final_crop.nii.gz'))

# --- 新增：批量处理逻辑 ---
def batch_preprocess(src_root, dst_root):
    # 确保输出根目录存在
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 获取所有子文件夹名（病人ID）
    # 这里假设所有病人文件夹都在 src_root 下，且名字以 'P-' 开头
    all_cases = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d)) and d.startswith('P-')]
    
    print(f"Total cases found: {len(all_cases)}")

    # 使用 tqdm 显示处理进度
    for case_id in tqdm(all_cases, desc="Batch Processing PROMIS"):
        case_dir = os.path.join(src_root, case_id)
        preprocess_promis_case(case_dir, dst_root)

if __name__ == "__main__":
    # 原始 MRI 数据路径
    MRI_DATA_ROOT = r'F:\RP_dataset\derived PROMIS data set\MRI'
    # 预处理后保存的路径
    PROCESSED_ROOT = r'F:\RP_dataset\Processed_PROMIS'
    
    batch_preprocess(MRI_DATA_ROOT, PROCESSED_ROOT)
    print("\nAll done!")