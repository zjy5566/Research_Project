import pandas as pd
import numpy as np
import os

def get_isup_label(primary, secondary):
    """
    计算 Gleason 分数并映射为训练用的 ISUP Label (与 TCIA 标准完全对齐)
    0: 背景 (Background, 未穿刺/未采样区域)
    1: 良性 (Benign / Negative, 穿刺了但没发现癌细胞)
    2: ISUP 1 (Gleason 3+3=6)
    3: ISUP 2 (Gleason 3+4=7)
    4: ISUP 3 (Gleason 4+3=7)
    5: ISUP 4 (Gleason 8)
    6: ISUP 5 (Gleason 9-10)
    """
    if pd.isna(primary) or pd.isna(secondary): 
        return 1 
        
    try:
        p, s = int(float(primary)), int(float(secondary))
    except ValueError:
        return 1

    if p + s <= 6: return 2
    if p + s == 7: return 3 if p == 3 else 4
    if p + s == 8: return 5
    if p + s >= 9: return 6
    return 1

def process_promis_sbx_csv(file_path):
    """
    处理单个 PROMIS CSV 文件，提取 20 区域的 ISUP 标签向量
    """
    try:
        df = pd.read_csv(file_path)
        # 过滤并确保只处理 zone_id 在 1-20 之间的行
        df = df[df['zone_id'].between(1, 20)].copy()
        
        # 初始化一个长度为 20 的全 0 向量 (默认为 0，代表背景/未穿刺)
        labels_vector = np.zeros(20, dtype=int)
        
        for _, row in df.iterrows():
            zone_idx = int(row['zone_id']) - 1  # 转为 0-19 索引
            
            # 1. 检查是否进行了穿刺采样 (samtaken). 
            # 如果没有采样，保持为 0 (背景，不参与 Loss 计算)
            if 'samtaken' in row and pd.notna(row['samtaken']) and int(float(row['samtaken'])) == 0:
                continue
            
            # 2. 获取是否有癌的标志
            has_cancer = row.get('zprescancer', 0)
            
            if pd.isna(has_cancer) or int(float(has_cancer)) == 0:
                # 采样了，但是没有癌 -> 良性 (Label 1)
                labels_vector[zone_idx] = 1
            else:
                # 3. 有癌 -> 优先通过 Gleason 评分精确计算
                p = row.get('zprimgleason', np.nan)
                s = row.get('zsecondgleason', np.nan)
                
                isup_label = get_isup_label(p, s)
                
                # 4. 容错处理 (Fallback)
                # 如果有癌，但是由于医生漏填 Gleason 评分导致返回了 1 (良性)
                if isup_label == 1:
                    # 尝试读取 maxccisup (原表中 1-5 级)。注意：要加 1 才能与我们的 2-6 级对齐！
                    if 'maxccisup' in row and pd.notna(row['maxccisup']) and int(float(row['maxccisup'])) > 0:
                        isup_label = int(float(row['maxccisup'])) + 1
                    else:
                        isup_label = 2 # 终极保障：有癌但啥都没填，至少保底是 ISUP 1 (Label 2)
                        
                labels_vector[zone_idx] = isup_label
                
        return labels_vector
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def batch_convert_csv_to_npy(input_dir, output_root):
    """
    遍历文件夹，转换所有 CSV 并存入对应的病人 ID 文件夹下
    """
    if not os.path.exists(output_root):
        print(f"警告: 输出根目录 {output_root} 不存在，请确保预处理图像的步骤已创建该目录。")
        # os.makedirs(output_root)

    count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and filename.startswith("P-"):
            file_path = os.path.join(input_dir, filename)
            
            labels = process_promis_sbx_csv(file_path)
            
            if labels is not None:
                patient_id = os.path.splitext(filename)[0]
                patient_folder = os.path.join(output_root, patient_id)
                
                if not os.path.exists(patient_folder):
                    os.makedirs(patient_folder)
                
                save_path = os.path.join(patient_folder, "systematic_labels.npy")
                
                np.save(save_path, labels)
                count += 1
                
                if count % 50 == 0:
                    print(f"已处理 {count} 个病例...")

    print(f"\n全部处理完成！共转换 {count} 个标签文件。")
    print(f"所有 systematic_labels.npy 已存入 {output_root} 下的各病人目录。")

# --- 配置路径 ---
input_folder = r'F:\RP_dataset\derived PROMIS data set\Template_biopsy\Template_biopsy'
output_folder = r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS'

# 执行批量转换
batch_convert_csv_to_npy(input_folder, output_folder)