import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

class ProstateDataPreprocessor:
    def __init__(self, root_dir, output_dir, target_spacing=[1.0, 1.0, 2.24], target_size=[64, 64, 32]):
        """
        :param root_dir: 包含 imagesTr, labelsTr, zonesTr 的根目录
        :param output_dir: 处理后数据的保存目录
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.target_spacing = target_spacing
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. 重采样函数 (保持与其它数据集一致) ---
    def resample_to_spacing(self, image, is_label=False):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(original_size[i] * original_spacing[i] / self.target_spacing[i])) for i in range(3)]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # --- 2. 以前列腺掩膜为中心的裁剪函数 (保持与其它数据集一致) ---
    def mask_centered_crop(self, img, mask):
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        # 确保 mask 为二值图像
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        label_shape_filter.Execute(binary_mask)
        
        if label_shape_filter.GetNumberOfLabels() == 0:
            # 如果 mask 为空，回退到图像中心裁剪
            center_index = [s // 2 for s in img.GetSize()]
        else:
            # 获取质心物理坐标并转为索引
            centroid_world = label_shape_filter.GetCentroid(1) 
            center_index = img.TransformPhysicalPointToIndex(centroid_world)

        roi_start = [center_index[i] - self.target_size[i] // 2 for i in range(3)]
        
        # 处理 Padding
        img_size = img.GetSize()
        pad_lower = [max(0, -roi_start[i]) for i in range(3)]
        pad_upper = [max(0, roi_start[i] + self.target_size[i] - img_size[i]) for i in range(3)]
        
        if sum(pad_lower) + sum(pad_upper) > 0:
            img = sitk.ConstantPad(img, pad_lower, pad_upper, 0)
            roi_start = [roi_start[i] + pad_lower[i] for i in range(3)]
        
        return sitk.RegionOfInterest(img, self.target_size, roi_start)

    # --- 3. 归一化函数 ---
    def normalize_array(self, img_arr):
        """
        全局 Z-score 归一化 (与 TCIA/PROMIS 数据集保持一致)
        """
        img_arr = img_arr.astype(np.float32)
        std = np.std(img_arr)
        return (img_arr - np.mean(img_arr)) / (std if std > 1e-7 else 1.0)

    def process_patient(self, patient_id):
        # 1. 构建路径
        t2_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0000.nii.gz")
        adc_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0001.nii.gz")
        dwi_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0002.nii.gz")
        label_path = os.path.join(self.root_dir, 'labelsTr', f"{patient_id}.nii.gz")
        zone_path = os.path.join(self.root_dir, 'zonesTr', f"{patient_id}.nii.gz")

        # 严格要求必须存在 label_path (病灶掩膜)。缺失则直接 return 放弃该病人。
        if not all([os.path.exists(p) for p in [t2_path, adc_path, dwi_path, zone_path, label_path]]):
            return

        # 2. 读取数据
        t2 = sitk.ReadImage(t2_path)
        adc = sitk.ReadImage(adc_path)
        dwi = sitk.ReadImage(dwi_path)
        zone = sitk.ReadImage(zone_path)
        label = sitk.ReadImage(label_path)

        # 3. 统一重采样
        t2_res = self.resample_to_spacing(t2)
        adc_res = self.resample_to_spacing(adc)
        dwi_res = self.resample_to_spacing(dwi)
        zone_res = self.resample_to_spacing(zone, is_label=True)
        label_res = self.resample_to_spacing(label, is_label=True)

        # 4. 以前列腺质心进行中心裁剪
        t2_crop = self.mask_centered_crop(t2_res, zone_res)
        adc_crop = self.mask_centered_crop(adc_res, zone_res)
        dwi_crop = self.mask_centered_crop(dwi_res, zone_res)
        label_crop = self.mask_centered_crop(label_res, zone_res)
        zone_crop = self.mask_centered_crop(zone_res, zone_res) # 确保 zone_mask 也被同步裁剪

        # 5. 转为 Numpy 数组并归一化
        t2_arr = sitk.GetArrayFromImage(t2_crop)
        adc_arr = sitk.GetArrayFromImage(adc_crop)
        dwi_arr = sitk.GetArrayFromImage(dwi_crop)

        # 6. 多模态堆叠 (严格遵守 T2, DWI, ADC 的通道顺序！！！)
        stacked_img = np.stack([
            self.normalize_array(t2_arr), 
            self.normalize_array(dwi_arr), 
            self.normalize_array(adc_arr)
        ], axis=0).astype(np.float32)

        # 7. 标签与区域掩膜处理
        # (label_arr > 0) 自动将原数据中被标为 1,2,3... 等多个病灶转化为统一的二值分割掩膜(1)
        label_arr = sitk.GetArrayFromImage(label_crop)
        final_label = (label_arr > 0).astype(np.uint8)

        # 【核心新增点】：提取裁剪后的 zone_mask
        zone_arr = sitk.GetArrayFromImage(zone_crop)
        final_zone = zone_arr.astype(np.uint8) 

        # 8. 保存处理后的结果
        # 命名格式严格遵守 _img.npy, _lab.npy, _zone.npy
        np.save(os.path.join(self.output_dir, f"{patient_id}_img.npy"), stacked_img)
        np.save(os.path.join(self.output_dir, f"{patient_id}_lab.npy"), final_label)
        np.save(os.path.join(self.output_dir, f"{patient_id}_zone.npy"), final_zone) # 新增保存 zone_mask

    def run_all(self):
        t2_files = glob.glob(os.path.join(self.root_dir, 'imagesTr', '*_0000.nii.gz'))
        patient_ids = sorted([os.path.basename(f).replace('_0000.nii.gz', '') for f in t2_files])
        
        valid_count = 0
        for pid in tqdm(patient_ids, desc="Processing MRI Dataset"):
            try:
                # 在处理前可加一层预判，减少报错日志刷屏
                label_path = os.path.join(self.root_dir, 'labelsTr', f"{pid}.nii.gz")
                if not os.path.exists(label_path):
                    continue
                    
                self.process_patient(pid)
                valid_count += 1
            except Exception as e:
                print(f"Error processing {pid}: {e}")
                
        print(f"Processing complete. Successfully processed {valid_count} patients with valid lesion masks.")

# --- 使用示例 ---
if __name__ == "__main__":
    raw_data_path = r"F:\RP_dataset\Dataset_prostate_MRI" 
    processed_path = r"F:\RP_dataset\Dataset_prostate_MRI\Dataset_prostate_MRI"
    
    preprocessor = ProstateDataPreprocessor(raw_data_path, processed_path)
    print("Script started...")
    preprocessor.run_all()
    print(f"All done! Output saved to: {processed_path}")