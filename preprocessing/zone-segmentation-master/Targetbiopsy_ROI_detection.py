import os
import numpy as np
import SimpleITK as sitk
from UNet_zones import anisotopic_UNET

# 全局变量缓存模型，避免重复加载
_MODEL_CACHE = None

def load_model_cached(model_weight_path):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("--- Loading Model Weights... ---")
        network = anisotopic_UNET()
        model = network.get_net(bn=True, do=False)
        model.load_weights(model_weight_path)
        _MODEL_CACHE = model
    return _MODEL_CACHE

def get_prostate_zones_hybrid(t2_path, model_weight_path):
    """
    三步走策略：中心裁剪(64,64,32) -> 重采样(168,168,32) -> 推理 -> 还原补零
    """
    # 1. 加载模型（单例模式）
    model = load_model_cached(model_weight_path)

    # 2. 读取原图
    img_sitk = sitk.ReadImage(t2_path)
    original_size = img_sitk.GetSize() # [x, y, z]
    
    # --- 步骤 A: 中心裁剪到 (64, 64, 32) ---
    crop_size = [128, 128, 32]    # 如果原图比裁剪尺寸还小，先 Padding
    pad_lower = [max(0, (crop_size[i] - original_size[i]) // 2) for i in range(3)]
    pad_upper = [max(0, crop_size[i] - original_size[i] - pad_lower[i]) for i in range(3)]
    
    if sum(pad_lower) + sum(pad_upper) > 0:
        img_working = sitk.ConstantPad(img_sitk, pad_lower, pad_upper, 0)
    else:
        img_working = img_sitk

    curr_size = img_working.GetSize()
    roi_start = [(curr_size[i] - crop_size[i]) // 2 for i in range(3)]
    roi_64 = sitk.RegionOfInterest(img_working, crop_size, roi_start)

    # --- 步骤 B: Resample 到模型输入 (168, 168, 32) ---
    model_input_size = [168, 168, 32]
    spacing_64 = roi_64.GetSpacing()
    new_spacing = [spacing_64[i] * crop_size[i] / model_input_size[i] for i in range(3)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(model_input_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(roi_64.GetOrigin())
    resampler.SetOutputDirection(roi_64.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    roi_168 = resampler.Execute(roi_64)

    # --- 步骤 C: 模型推理 ---
    img_array = sitk.GetArrayFromImage(roi_168).astype(np.float32)
    img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)
    input_tensor = img_array[np.newaxis, ..., np.newaxis] # [1, 32, 168, 168, 1]
    
    preds = model.predict(input_tensor, verbose=0)
    pred_arr = np.asarray(preds)
    # 合并 5 通道分类结果
    mask_168_arr = np.argmax(pred_arr[:, 0, ...], axis=0).astype(np.uint8)
    mask_168_arr[mask_168_arr == 4] = 0 # 背景索引 4 设为 0

    # --- 步骤 D: 逆向还原 ---
    # 1. 168 -> 64 (重采样还原)
    mask_168_itk = sitk.GetImageFromArray(mask_168_arr)
    mask_168_itk.CopyInformation(roi_168)
    
    resampler_back = sitk.ResampleImageFilter()
    resampler_back.SetReferenceImage(roi_64)
    resampler_back.SetInterpolator(sitk.sitkNearestNeighbor) # 标签还原必须用最近邻
    mask_64_itk = resampler_back.Execute(mask_168_itk)
    mask_64_arr = sitk.GetArrayFromImage(mask_64_itk)

    # 2. 贴回画布 (修正索引顺序)
    full_mask_arr = np.zeros(sitk.GetArrayFromImage(img_working).shape, dtype=np.uint8)
    # Numpy 索引顺序: [z, y, x]
    zs, ys, xs = roi_start[2], roi_start[1], roi_start[0]
    # 注意这里使用的是 crop_size 的反序 [32, 64, 64]
    full_mask_arr[zs : zs + crop_size[2], ys : ys + crop_size[1], xs : xs + crop_size[0]] = mask_64_arr
    
    full_mask_itk = sitk.GetImageFromArray(full_mask_arr)
    full_mask_itk.CopyInformation(img_working)

    # 3. 裁剪回原图尺寸
    if sum(pad_lower) + sum(pad_upper) > 0:
        final_mask = sitk.RegionOfInterest(full_mask_itk, original_size, pad_lower)
    else:
        final_mask = full_mask_itk

    final_mask.CopyInformation(img_sitk)
    return final_mask

def batch_zone_segmentation(data_root, model_path):
    # 确保输出目录存在
    patients = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    for p_id in patients:
        t2_file = os.path.join(data_root, p_id, 't2.nii.gz')
        output_file = os.path.join(data_root, p_id, 'zones_mask.nii.gz')
        
        if os.path.exists(t2_file) and not os.path.exists(output_file):
            print(f">>> Hybrid Processing Case: {p_id}")
            try:
                mask = get_prostate_zones_hybrid(t2_file, model_path)
                sitk.WriteImage(mask, output_file)
            except Exception as e:
                print(f"!!! Error in {p_id}: {e}")

if __name__ == "__main__":
    DATA_ROOT = r'F:\RP_dataset\Target biosy\Extracted_Target_Biopsy'
    MODEL_WEIGHTS = r'D:\zjy\study\research_project\code\Research_Project\preprocessing\zone-segmentation-master\model\model.h5'
    batch_zone_segmentation(DATA_ROOT, MODEL_WEIGHTS)