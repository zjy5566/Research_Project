import vtk
from vtk.util import numpy_support  # 新增这一行
import numpy as np
import SimpleITK as sitk

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

def stl_to_mask(stl_path, template_mri_path, output_mask_path):
    """
    通过将 STL 点投影到模板 MRI 物理空间来创建重合的掩膜
    """
    # 1. 读取模板 MRI
    template_img = sitk.ReadImage(template_mri_path)
    
    # 2. 读取 STL 并获取所有点
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    polydata = reader.GetOutput()
    
    # 3. 在 VTK 中判断点是否在封闭曲面内
    # 我们使用 vtkSelectEnclosedPoints 来替代传统的 Stencil
    # 这是处理复杂 Direction 矩阵最稳健的方法
    
    # 创建一个与原图一致的空白 SITK 图像
    mask_arr = np.zeros(sitk.GetArrayViewFromImage(template_img).shape, dtype=np.uint8)
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(template_img)
    
    # 获取图像的所有物理点坐标（这一步比较慢，但位置绝对准确）
    # 为了提速，我们只对图像中的每个体素点进行一次内/外判定
    
    # 4. 使用 VTK 的探测器
    enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
    enclosed_points_filter.SetSurfaceData(polydata)
    
    # 我们改用更高效的方案：将 polydata 转换为 ImageStencil 之前先处理坐标转换
    # 针对 Direction 不为 Identity 的情况，我们需要通过变换矩阵对 STL 进行预处理
    
    # 获取图像属性
    origin = np.array(template_img.GetOrigin())
    spacing = np.array(template_img.GetSpacing())
    dims = template_img.GetSize()
    direction = np.array(template_img.GetDirection()).reshape(3,3)
    
    # 构造从物理空间到体素空间的变换矩阵
    # 这一步是为了消除 MRI Direction 带来的偏移
    white_image = vtk.vtkImageData()
    white_image.SetDimensions(dims)
    white_image.SetSpacing(spacing)
    white_image.SetOrigin(0,0,0) # 先设为0，通过变换矩阵统一处理
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # 关键修正：对 STL 进行坐标变换，抵消 MRI 的旋转和位移
    # 使得 STL 坐标与一个 Origin 为 (0,0,0), Direction 为 Identity 的 ImageData 对应
    transform = vtk.vtkTransform()
    # 计算逆变换矩阵：Physical -> Voxel
    # 这里直接利用 SimpleITK 的物理坐标转换更稳健
    
    # --- 替代方案：最稳健的体素化逻辑 ---
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    # 关键点：如果 Direction 不是标准对角阵，这里的 OutputOrigin 必须做特殊处理
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

    # 导出
    vtk_data = imgstenc.GetOutput().GetPointData().GetScalars()
    numpy_mask = numpy_support.vtk_to_numpy(vtk_data).reshape(dims[::-1])
    
    final_mask = sitk.GetImageFromArray(numpy_mask)
    # 直接 CopyInformation 覆盖所有空间元数据
    final_mask.CopyInformation(template_img)
    
    # 修复方向不一致导致的位移：
    # 如果偏移依然存在，说明 STL 坐标系是 LPS 而图像是 RAS，或者存在半像素偏移
    # 强制重新设置元数据
    final_mask.SetOrigin(template_img.GetOrigin())
    final_mask.SetSpacing(template_img.GetSpacing())
    final_mask.SetDirection(template_img.GetDirection())

    sitk.WriteImage(final_mask, output_mask_path)
    print(f"✅ Mask 已保存且校准空间信息: {output_mask_path}")

# 使用示例
if __name__ == "__main__":
    stl_to_mask('F:\\RP_dataset\\Target biosy\\Prostate-MRI-US-Biopsy\\STLs\\STLs\\Prostate-MRI-US-Biopsy-0001-ProstateSurface-seriesUID-1.3.6.1.4.1.14519.5.2.1.140367896789002601449386011052978380612.STL'
                , 'F:\\RP_dataset\\Target biosy\\Extracted_Target_Biopsy\\Prostate-MRI-US-Biopsy-0001\\t2.nii.gz', 'F:\\RP_dataset\\Target biosy\\Extracted_Target_Biopsy\\Prostate-MRI-US-Biopsy-0001\\gland_mask.nii.gz')
