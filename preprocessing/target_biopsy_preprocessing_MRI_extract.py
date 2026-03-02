import os
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. 序列自动识别函数 (针对 TCIA 复杂结构优化) ---
def find_mri_sequences(patient_path):
    """
    递归遍历病人目录，识别 T2, ADC, DWI 文件夹路径
    """
    paths = {'t2': None, 'adc': None, 'dwi': None}
    
    for root, dirs, files in os.walk(patient_path):
        # 只有包含 .dcm 文件的文件夹才会被考虑
        if any(f.lower().endswith('.dcm') for f in files):
            folder_name = os.path.basename(root).lower()
            
            # T2 识别逻辑: 通常含有 't2' 和 'axial' (横断位)
            if 't2' in folder_name and 'axial' in folder_name:
                paths['t2'] = root
            
            # ADC 识别逻辑: 文件夹名含有 'adc'
            elif 'adc' in folder_name:
                paths['adc'] = root
            
            # DWI 识别逻辑: 含有 'diff' 或 'dwi'，且排除 'adc'
            # 针对你提到的 'calcbval' (计算的高B值图像) 也可以归类为 DWI
            elif ('dwi' in folder_name or 'diff' in folder_name or 'calcbval' in folder_name) and 'adc' not in folder_name:
                paths['dwi'] = root
                
    return paths

# --- 2. DICOM 读取函数 ---
def read_dicom_series(directory):
    """
    读取指定目录下的 DICOM 序列并返回 3D 图像对象
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    if not dicom_names:
        return None
    reader.SetFileNames(dicom_names)
    return reader.Execute()

# --- 3. 单个病人处理逻辑 ---
def extract_and_save_case(patient_id, patient_path, dst_root):
    # 寻找序列路径
    seq_paths = find_mri_sequences(patient_path)
    
    # 检查是否找到了必要的模态（你可以根据需要修改，比如允许缺失某些模态）
    if not any(seq_paths.values()):
        return False
    
    # 创建输出文件夹 (以 Patient ID 命名)
    save_path = os.path.join(dst_root, patient_id)
    
    success = False
    for modality, folder in seq_paths.items():
        if folder:
            try:
                # 读取并保存为 .nii.gz
                img = read_dicom_series(folder)
                if img:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    # 规范化命名保存
                    output_filename = f"{modality}.nii.gz"
                    sitk.WriteImage(img, os.path.join(save_path, output_filename))
                    success = True
            except Exception as e:
                print(f"\n[Error] Failed to read {modality} for {patient_id}: {e}")
                
    return success

# --- 4. 批量处理主函数 ---
def batch_extract_mri(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    
    # 获取病人文件夹列表 (Prostate-MRI-US-Biopsy-XXXX)
    patients = [d for d in os.listdir(src_root) if d.startswith('Prostate-MRI-US-Biopsy-')]
    print(f"Found {len(patients)} potential patients.")

    count = 0
    for p_id in tqdm(patients, desc="Extracting MRI"):
        p_path = os.path.join(src_root, p_id)
        if extract_and_save_case(p_id, p_path, dst_root):
            count += 1
            
    print(f"\nExtraction complete. Successfully organized data for {count} patients.")

if __name__ == "__main__":
    # 原始数据集路径
    SRC_DIR = r'F:\RP_dataset\Target biosy\manifest-1694710246744\Prostate-MRI-US-Biopsy'
    # 提取后保存的路径
    DST_DIR = r'F:\RP_dataset\Extracted_Target_Biopsy'
    
    batch_extract_mri(SRC_DIR, DST_DIR)