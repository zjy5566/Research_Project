import os
import pandas as pd
import vtk
from vtk.util import numpy_support
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# --- STL 转换核心函数 ---
def stl_to_mask(stl_path, template_mri_path, output_mask_path):
    """
    将 STL 转换为与 T2 图像空间对齐的 NIfTI 掩膜
    """
    try:
        template_img = sitk.ReadImage(template_mri_path)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        dims = template_img.GetSize()
        spacing = template_img.GetSpacing()
        origin = template_img.GetOrigin()

        white_image = vtk.vtkImageData()
        white_image.SetDimensions(dims)
        white_image.SetSpacing(spacing)
        white_image.SetOrigin(0, 0, 0)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(template_img.GetOrigin())
        pol2stenc.SetOutputSpacing(template_img.GetSpacing())
        pol2stenc.SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.SetReverseStencil(1)
        imgstenc.SetBackgroundValue(1)
        imgstenc.Update()

        vtk_data = imgstenc.GetOutput().GetPointData().GetScalars()
        numpy_mask = numpy_support.vtk_to_numpy(vtk_data).reshape(dims[::-1])
        
        final_mask = sitk.GetImageFromArray(numpy_mask)
        final_mask.CopyInformation(template_img)
        
        # 强制同步空间信息
        final_mask.SetOrigin(template_img.GetOrigin())
        final_mask.SetSpacing(template_img.GetSpacing())
        final_mask.SetDirection(template_img.GetDirection())

        sitk.WriteImage(final_mask, output_mask_path)
        return True
    except Exception as e:
        print(f"Error converting {os.path.basename(stl_path)}: {e}")
        return False

# --- 自动化检索与批量处理 ---
def batch_convert_stls(excel_path, stl_root, processed_dir_root, save_root):
    print(f"Loading Excel file: {os.path.basename(excel_path)}")
    
    # 修正点 1：使用 read_excel 读取 .xlsx 文件
    # 如果表格有多个 sheet，请确认 'Prostate-MRI-US-Biopsy Target' 是正确的名字
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    # 修正点 2：列名匹配。Target 文件的列名通常是 'Patient ID' 和 'seriesInstanceUID_MR'
    pid_col = 'Patient ID'
    uid_col = 'seriesInstanceUID_MR'
    
    # 去重处理，一个 UID 对应一个前列腺表面模型
    unique_cases = df[[pid_col, uid_col]].drop_duplicates()
    
    success_count = 0
    for _, row in tqdm(unique_cases.iterrows(), total=len(unique_cases), desc="Processing Cases"):
        pid = str(row[pid_col]).strip()
        uid = str(row[uid_col]).strip()
        
        # 构建 STL 文件名：PatientID-ProstateSurface-seriesUID-UID.STL
        stl_name = f"{pid}-ProstateSurface-seriesUID-{uid}.STL"
        stl_path = os.path.join(stl_root, stl_name)
        
        # 模板图路径：指向你之前提取出的 t2.nii.gz
        mri_template_path = os.path.join(processed_dir_root, pid, 't2.nii.gz')
        
        # 输出路径：保存到处理后的病人文件夹
        output_mask_path = os.path.join(save_root, pid, 'gland_mask.nii.gz')
        
        if os.path.exists(stl_path) and os.path.exists(mri_template_path):
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            if stl_to_mask(stl_path, mri_template_path, output_mask_path):
                success_count += 1
        else:
            # 调试信息：如果没找到，查看是缺 STL 还是缺 MRI
            if not os.path.exists(stl_path):
                pass # 很多 UID 可能是超声的，没有对应的 MRI STL 属于正常现象
            if not os.path.exists(mri_template_path):
                print(f"Warning: MRI template missing for {pid}")

    print(f"\nCompleted! Successfully generated {success_count} gland masks.")

if __name__ == "__main__":
    # --- 路径配置 ---
    # 1. 靶向数据 Excel 路径
    EXCEL_PATH = r'F:\RP_dataset\Target biosy\Target-Data_2019-12-05-2.xlsx'
    
    # 2. STL 原始文件夹路径
    STL_DIR = r'F:\RP_dataset\Target biosy\Prostate-MRI-US-Biopsy\STLs\STLs'
    
    # 3. 原始提取的 MRI 路径 (用于做模板)
    EXTRACTED_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    
    # 4. 最终处理结果存放路径
    SAVE_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    
    batch_convert_stls(EXCEL_PATH, STL_DIR, EXTRACTED_ROOT, SAVE_ROOT)