import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# --- 核心配准函数 (保持不变) ---
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

# --- 序列自动识别函数 ---
def find_mri_sequences(patient_path):
    """
    遍历病人目录，识别 T2, ADC, DWI 文件夹路径
    """
    paths = {'t2': None, 'adc': None, 'dwi': None}
    for root, dirs, files in os.walk(patient_path):
        if not files: continue
        # 检查是否包含 .dcm 文件
        if any(f.endswith('.dcm') for f in files):
            folder_name = root.lower()
            # 识别逻辑 (基于文件夹关键词)
            if 't2' in folder_name and 'axial' in folder_name:
                paths['t2'] = root
            elif 'adc' in folder_name:
                paths['adc'] = root
            elif ('dwi' in folder_name or 'diff' in folder_name) and 'adc' not in folder_name:
                paths['dwi'] = root
    return paths

# --- 单例处理函数 ---
def process_target_case(patient_id, patient_path, dst_root, target_spacing=[1.0, 1.0, 2.24], crop_size=[64, 64, 32]):
    seq_paths = find_mri_sequences(patient_path)
    
    # 只要缺失任何一个模态，就跳过 (保证数据完整性)
    if not all(seq_paths.values()):
        return False

    try:
        # 读取 DICOM 序列
        def read_dicom(path):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            return reader.Execute()

        t2 = read_dicom(seq_paths['t2'])
        adc = read_dicom(seq_paths['adc'])
        dwi = read_dicom(seq_paths['dwi'])

        # 1. 重采样 (Resample)
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
        adc_res = resample(adc)
        dwi_res = resample(dwi)

        # 2. 配准 (Registration) - 以 T2 为基准
        adc_reg = register_images(t2_res, adc_res)
        dwi_reg = register_images(t2_res, dwi_res)

        # 3. 裁剪 (Center Crop) 
        # 注意：由于 Target Biopsy 没给出 gland 掩码，我们采用图像中心裁剪
        def center_crop(img, size):
            img_size = img.GetSize()
            # 简单的中心对齐
            roi_start = [max(0, (img_size[i] - size[i]) // 2) for i in range(3)]
            # 确保裁剪大小不超过原图，如果不够则需要补齐(pad)
            # 这里复用你之前的 pad_if_needed 逻辑
            return sitk.RegionOfInterest(img, size, roi_start)

        # 此处省略 pad_if_needed 函数调用，逻辑同 PROMIS 代码
        # 执行裁剪
        t2_crop = center_crop(t2_res, crop_size)
        adc_crop = center_crop(adc_reg, crop_size)
        dwi_crop = center_crop(dwi_reg, crop_size)

        # 4. 归一化与堆叠
        def finalize(img):
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

        # 按照 T2, DWI, ADC 顺序堆叠 (注意：finalize 后的顺序需严格对应)
        input_tensor = np.stack([finalize(t2_crop), finalize(dwi_crop), finalize(adc_crop)], axis=0)
        
        # 保存结果
        save_dir = os.path.join(dst_root, patient_id)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'input_tensor.npy'), input_tensor)
        return True

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return False

# --- 主执行逻辑 ---
def batch_process_target(src_root, dst_root):
    if not os.path.exists(dst_root): os.makedirs(dst_root)
    
    # 获取 Prostate-MRI-US-Biopsy-XXXX 列表
    patients = [d for d in os.listdir(src_root) if d.startswith('Prostate-MRI-US-Biopsy-')]
    
    success_count = 0
    for p_id in tqdm(patients, desc="Processing Target Biopsy Data"):
        p_path = os.path.join(src_root, p_id)
        if process_target_case(p_id, p_path, dst_root):
            success_count += 1
            
    print(f"\nProcessing complete. Successfully saved {success_count} patients.")

if __name__ == "__main__":
    SRC = r'F:\RP_dataset\Target biosy\manifest-1694710246744\Prostate-MRI-US-Biopsy'
    DST = r'F:\RP_dataset\Processed_Target_Biopsy'
    batch_process_target(SRC, DST)