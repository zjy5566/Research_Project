import os
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. 序列自动识别函数 (保持你最新的优化逻辑) ---
def find_mri_sequences(patient_path):
    paths = {'t2': None, 'adc': None, 'dwi': None}
    for root, dirs, files in os.walk(patient_path):
        if any(f.lower().endswith('.dcm') for f in files):
            folder_name = os.path.basename(root).lower()
            parent_name = os.path.basename(os.path.dirname(root)).lower()
            combined_name = folder_name + " " + parent_name

            if 'adc' in combined_name or 'apparent diffusion' in combined_name:
                if 'report' not in combined_name:
                    paths['adc'] = root
                    continue

            dwi_indicators = ['dwi', 'diff', 'b1600', 'b2000', 'b1000', 'b800', 'calcbval']
            if any(x in combined_name for x in dwi_indicators):
                if 'adc' not in folder_name:
                    paths['dwi'] = root
                    continue

            if 't2' in combined_name:
                if not any(x in combined_name for x in ['sag', 'cor', 'haste', 'loc']):
                    paths['t2'] = root
                    continue
    return paths

# --- 2. DICOM 读取函数 ---
def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    if not dicom_names:
        return None
    reader.SetFileNames(dicom_names)
    try:
        return reader.Execute()
    except Exception:
        return None

# --- 3. 单个病人处理逻辑 (增加预检跳过逻辑) ---
def extract_and_save_case(patient_id, patient_path, dst_root):
    save_path = os.path.join(dst_root, patient_id)
    
    # --- 新增：检查是否已经完整提取 ---
    required_files = ['t2.nii.gz', 'adc.nii.gz', 'dwi.nii.gz']
    if os.path.exists(save_path):
        existing_files = os.listdir(save_path)
        if all(f in existing_files for f in required_files):
            # 返回特定标志表示“已存在”，方便统计
            return "ALREADY_EXISTS"

    seq_paths = find_mri_sequences(patient_path)
    
    # 如果一个模态都没找到，跳过
    if not any(seq_paths.values()):
        return "FAILED"
    
    extracted_count = 0
    for modality, folder in seq_paths.items():
        if folder:
            try:
                # 再次检查单个文件，防止部分缺失的情况补提
                output_path = os.path.join(save_path, f"{modality}.nii.gz")
                if os.path.exists(output_path):
                    extracted_count += 1
                    continue

                img = read_dicom_series(folder)
                if img:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    sitk.WriteImage(img, output_path)
                    extracted_count += 1
            except Exception as e:
                print(f"\n[Error] Failed to read {modality} for {patient_id}: {e}")
                
    return "SUCCESS" if extracted_count > 0 else "FAILED"

# --- 4. 批量处理主函数 (增加详细统计) ---
def batch_extract_mri(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    
    patients = [d for d in os.listdir(src_root) if d.startswith('Prostate-MRI-US-Biopsy-')]
    print(f"Found {len(patients)} potential patients.")

    new_count = 0
    skip_count = 0
    fail_count = 0

    pbar = tqdm(patients, desc="Extracting MRI")
    for p_id in pbar:
        p_path = os.path.join(src_root, p_id)
        result = extract_and_save_case(p_id, p_path, dst_root)
        
        if result == "SUCCESS":
            new_count += 1
        elif result == "ALREADY_EXISTS":
            skip_count += 1
        else:
            fail_count += 1
        
        # 在进度条动态显示统计信息
        pbar.set_postfix({"New": new_count, "Skipped": skip_count, "Failed": fail_count})
            
    print(f"\n" + "="*30)
    print(f"Extraction Summary:")
    print(f"  - Newly Extracted: {new_count}")
    print(f"  - Already Exists:  {skip_count}")
    print(f"  - Failed/Missing:  {fail_count}")
    print(f"  - Total Processed: {new_count + skip_count}")
    print("="*30)

if __name__ == "__main__":
    SRC_DIR = r'F:\RP_dataset\Target biosy\manifest-1694710246744\Prostate-MRI-US-Biopsy'
    DST_DIR = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    
    batch_extract_mri(SRC_DIR, DST_DIR)