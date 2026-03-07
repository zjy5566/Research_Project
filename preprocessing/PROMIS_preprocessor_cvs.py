import pandas as pd
import numpy as np
import os

def process_promis_sbx_csv(file_path):
    """
    处理单个 PROMIS CSV 文件，提取 20 区域的 ISUP 标签向量
    """
    try:
        df = pd.read_csv(file_path)
        # 过滤并确保只处理 zone_id 在 1-20 之间的行
        df = df[df['zone_id'].between(1, 20)].copy()
        
        # 初始化一个长度为 20 的全 0 向量 (默认为良性/无癌)
        labels_vector = np.zeros(20, dtype=int)
        
        for _, row in df.iterrows():
            zone_idx = int(row['zone_id']) - 1  # 转为 0-19 索引
            
            # 优先获取 maxccisup (ISUP分级)
            # 如果该列不存在或为空，则根据 zprescancer (是否有癌) 设为 1 或 0
            if 'maxccisup' in row and not pd.isna(row['maxccisup']):
                labels_vector[zone_idx] = int(row['maxccisup'])
            elif 'zprescancer' in row and not pd.isna(row['zprescancer']):
                labels_vector[zone_idx] = int(row['zprescancer'])
                
        return labels_vector
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def batch_convert_csv_to_npy(input_dir, output_root):
    """
    遍历文件夹，转换所有 CSV 并存入对应的病人 ID 文件夹下
    """
    # 检查输出根目录是否存在
    if not os.path.exists(output_root):
        print(f"警告: 输出根目录 {output_root} 不存在，请确保预处理图像的步骤已创建该目录。")
        # 如果你希望自动创建根目录，可以取消下面一行的注释
        # os.makedirs(output_root)

    count = 0
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and filename.startswith("P-"):
            file_path = os.path.join(input_dir, filename)
            
            # 执行处理
            labels = process_promis_sbx_csv(file_path)
            
            if labels is not None:
                # 1. 提取 Patient 编号 (例如 P-12345678)
                patient_id = os.path.splitext(filename)[0]
                
                # 2. 构造病人对应的目标文件夹路径
                patient_folder = os.path.join(output_root, patient_id)
                
                # 3. 如果该病人的文件夹不存在则创建 (通常应与图像预处理目录一致)
                if not os.path.exists(patient_folder):
                    os.makedirs(patient_folder)
                
                # 4. 构造最终的文件保存路径，命名为 systematic_labels.npy
                save_path = os.path.join(patient_folder, "systematic_labels.npy")
                
                # 保存为 npy 文件
                np.save(save_path, labels)
                count += 1
                
                if count % 50 == 0:
                    print(f"已处理 {count} 个病例...")

    print(f"\n全部处理完成！共转换 {count} 个标签文件。")
    print(f"所有 systematic_labels.npy 已存入 {output_root} 下的各病人目录。")

# --- 配置路径 ---
# CSV 源文件目录
input_folder = r'F:\RP_dataset\derived PROMIS data set\Template_biopsy\Template_biopsy'
# 目标根目录 (Processed_PROMIS)
output_folder = r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS'

# 执行批量转换
batch_convert_csv_to_npy(input_folder, output_folder)