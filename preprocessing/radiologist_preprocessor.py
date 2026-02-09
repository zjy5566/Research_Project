import os
import glob
import numpy as np
import SimpleITK as sitk

class ProstateDataPreprocessor:
    def __init__(self, root_dir, output_dir):
        """
        :param root_dir: 包含 imagesTr, labelsTr, zonesTr 的根目录
        :param output_dir: 处理后数据的保存目录
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_nifti(self, path):
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img), img

    def normalize_by_zone(self, data, zone_mask):
        """
        基于前列腺腺体区域(zone_mask == 1)进行 Z-score 归一化
        """
        # 获取腺体内的像素值
        zone_pixels = data[zone_mask > 0]
        if len(zone_pixels) == 0: # 防止异常情况
            mean, std = data.mean(), data.std()
        else:
            mean, std = zone_pixels.mean(), zone_pixels.std()
        
        # 归一化并防止除以零
        normalized_data = (data - mean) / (std + 1e-8)
        return normalized_data

    def process_patient(self, patient_id):
        # 1. 构建路径
        t2_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0000.nii.gz")
        adc_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0001.nii.gz")
        dwi_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0002.nii.gz")
        label_path = os.path.join(self.root_dir, 'labelsTr', f"{patient_id}.nii.gz")
        zone_path = os.path.join(self.root_dir, 'zonesTr', f"{patient_id}.nii.gz")

        # 2. 读取数据
        t2_arr, ref_img = self.load_nifti(t2_path)
        adc_arr, _ = self.load_nifti(adc_path)
        dwi_arr, _ = self.load_nifti(dwi_path)
        label_arr, _ = self.load_nifti(label_path) # 病灶索引 0...L
        zone_arr, _ = self.load_nifti(zone_path)   # 腺体 Mask (0, 1)

        # 3. 执行基于腺体区域的归一化 (核心步骤)
        t2_norm = self.normalize_by_zone(t2_arr, zone_arr)
        adc_norm = self.normalize_by_zone(adc_arr, zone_arr)
        dwi_norm = self.normalize_by_zone(dwi_arr, zone_arr)

        # 4. 多模态堆叠 (Channel-first: [Channels, Z, Y, X])
        # 这样可以直接输入 3D CNN
        stacked_img = np.stack([t2_norm, adc_norm, dwi_norm], axis=0).astype(np.float32)

        # 5. 标签处理
        # 如果是简单的二分类检测（是否有癌），将索引图转为 binary
        # 如果是 Mixed-supervision，这里后续需结合病理 CSV 映射 ISUP 等级
        binary_label = (label_arr > 0).astype(np.uint8)

        # 6. 保存处理后的结果 (以 .npy 格式保存加速训练读取，或存回 .nii.gz)
        np.save(os.path.join(self.output_dir, f"{patient_id}_img.npy"), stacked_img)
        np.save(os.path.join(self.output_dir, f"{patient_id}_lab.npy"), binary_label)
        if patient_id == "001":
            sitk.WriteImage(sitk.GetImageFromArray(t2_norm), os.path.join(self.output_dir, f"{patient_id}_ref.nii.gz"))

        
        print(f"Finished processing patient: {patient_id}")

    def run_all(self):
        # 获取所有患者 ID (基于 T2 影像)
        t2_files = glob.glob(os.path.join(self.root_dir, 'imagesTr', '*_0000.nii.gz'))
        patient_ids = [os.path.basename(f).replace('_0000.nii.gz', '') for f in t2_files]
        
        for pid in patient_ids:
            if pid == "002":
                break
            try:
                self.process_patient(pid)
            except Exception as e:
                print(f"Error processing {pid}: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 填入您的实际文件夹路径
    raw_data_path = "D:\\zjy\\study\\research_project\\Dataset_prostate" 
    processed_path = "D:\\zjy\\study\\research_project\\processed_prostrate"
    
    preprocessor = ProstateDataPreprocessor(raw_data_path, processed_path)
    print("Script started...")
    preprocessor.run_all()