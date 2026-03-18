import os
import vtk
from vtk.util import numpy_support
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import glob

# ==========================================
# 核心转换引擎 (已修复 VTK 内存泄漏与挖空逻辑)
# ==========================================
def stl_to_numpy_mask(stl_path, template_img):
    try:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        dims = template_img.GetSize()
        spacing = template_img.GetSpacing()

        # 1. 创建底图
        white_image = vtk.vtkImageData()
        white_image.SetDimensions(dims)
        white_image.SetSpacing(spacing)
        white_image.SetOrigin(0, 0, 0)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # 强制将底图的所有像素初始化为 1 (前景)
        scalars = white_image.GetPointData().GetScalars()
        scalars.Fill(1) 

        # 2. 将 STL 表面转换为 3D 模板 (Stencil)
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(template_img.GetOrigin())
        pol2stenc.SetOutputSpacing(template_img.GetSpacing())
        pol2stenc.SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
        pol2stenc.Update()

        # 3. 执行模板切割
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        
        # 关闭反转，外部强制设为 BackgroundValue(0)
        imgstenc.ReverseStencilOff() 
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        # 4. 提取为 Numpy 数组并重塑维度 (Z, Y, X)
        vtk_data = imgstenc.GetOutput().GetPointData().GetScalars()
        numpy_mask = numpy_support.vtk_to_numpy(vtk_data).reshape(dims[::-1])
        
        return numpy_mask
    except Exception as e:
        print(f"Error converting {os.path.basename(stl_path)}: {e}")
        return None

def save_numpy_to_nifti(numpy_array, template_img, output_path):
    final_mask = sitk.GetImageFromArray(numpy_array.astype(np.uint8))
    final_mask.CopyInformation(template_img)
    sitk.WriteImage(final_mask, output_path)

# ==========================================
# 标签映射字典
# ==========================================
def map_ucla_to_isup(ucla_score):
    """
    UCLA Score 映射规则：0/1/2=1, 3=2, 4=3, 5=4, 6=5, 7=6
    """
    try:
        score = int(ucla_score)
        if score in [0, 1, 2]: return 1
        elif score == 3: return 2
        elif score == 4: return 3
        elif score == 5: return 4
        elif score == 6: return 5
        elif score == 7: return 6
        else: return 1
    except:
        return 1

# ==========================================
# 批量处理主函数
# ==========================================
def batch_convert_stls(processed_dir_root, excel_path):
    print("Loading Target Biopsy Clinical Excel...")
    try:
        if excel_path.endswith('.csv'):
            df = pd.read_csv(excel_path)
        else:
            df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Failed to read clinical file: {e}")
        return
        
    patient_targets_info = {}
    for _, row in df.iterrows():
        pid = str(row['Patient ID']).strip()
        t_num = str(row['Target No.']).strip()
        ucla = row['UCLA Score (Similar to PIRADS v2)']
        
        if pd.isna(ucla): ucla = 0
        
        if pid not in patient_targets_info:
            patient_targets_info[pid] = []
        patient_targets_info[pid].append({
            'target_num': t_num,
            'isup_label': map_ucla_to_isup(ucla)
        })

    folders = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))]
    
    gland_success = 0
    target_success = 0
    
    for folder in tqdm(folders, desc="Converting STLs to Masks"):
        folder_path = os.path.join(processed_dir_root, folder)
        mri_template_path = os.path.join(folder_path, 't2.nii.gz')
        
        if not os.path.exists(mri_template_path):
            continue
            
        template_img = sitk.ReadImage(mri_template_path)
        base_pid = folder.split('_')[0] 
        
        # ---------------------------------------------------------
        # 任务 A: 转换 Gland Mask (移除覆盖保护)
        # ---------------------------------------------------------
        stl_path_gland = os.path.join(folder_path, 'prostate_surface.stl')
        output_gland_path = os.path.join(folder_path, 'gland_mask.nii.gz')
        
        if os.path.exists(stl_path_gland):
            gland_np = stl_to_numpy_mask(stl_path_gland, template_img)
            if gland_np is not None:
                save_numpy_to_nifti(gland_np, template_img, output_gland_path)
                gland_success += 1
                
        # ---------------------------------------------------------
        # 任务 B: 转换 Target Masks 并合并 (移除覆盖保护)
        # ---------------------------------------------------------
        output_target_path = os.path.join(folder_path, 'target_mask.nii.gz')
        
        if base_pid not in patient_targets_info:
            continue
            
        targets_info = patient_targets_info[base_pid]
        target_stl_files = glob.glob(os.path.join(folder_path, 'target_*.stl'))
        
        if len(target_stl_files) > 0:
            dims = template_img.GetSize()
            combined_target_mask = np.zeros(dims[::-1], dtype=np.uint8)
            found_any = False
            
            for t_file in target_stl_files:
                t_num = os.path.basename(t_file).replace('target_', '').replace('.stl', '')
                
                matched_label = 1 # 默认阴性
                for t_info in targets_info:
                    if t_info['target_num'] == t_num:
                        matched_label = t_info['isup_label']
                        break
                
                target_np = stl_to_numpy_mask(t_file, template_img)
                if target_np is not None:
                    labeled_target = target_np * matched_label
                    combined_target_mask = np.maximum(combined_target_mask, labeled_target)
                    found_any = True
            
            if found_any:
                save_numpy_to_nifti(combined_target_mask, template_img, output_target_path)
                target_success += 1

    print(f"\n=============================================")
    print(f"Summary:")
    print(f" - Gland Masks Generated:  {gland_success}")
    print(f" - Target Masks Generated: {target_success}")
    print(f"=============================================")


if __name__ == "__main__":
    EXTRACTED_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    EXCEL_PATH = r'F:\RP_dataset\Target biosy\unprocessed_data\Target-Data_2019-12-05-2.xlsx'
    
    batch_convert_stls(EXTRACTED_ROOT, EXCEL_PATH)