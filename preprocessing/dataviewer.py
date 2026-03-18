import numpy as np
import matplotlib.pyplot as plt

class VolumeViewer:
    def __init__(self, npy_path):
        # 1. 加载数据
        self.data = np.load(npy_path)
        print(f"原始加载数据维度: {self.data.shape}")
        
        # 2. 自适应维度处理
        if self.data.ndim == 3:
            # 单通道图像 (D, H, W) -> 扩充为 (1, D, H, W)
            self.data = self.data[np.newaxis, ...]
        elif self.data.ndim != 4:
            raise ValueError(f"期望 3D(D, H, W) 或 4D(C, D, H, W) 数据，但得到的是 {self.data.shape}")

        self.num_channels = self.data.shape[0]
        self.max_slices = self.data.shape[1]
        self.slice_idx = self.max_slices // 2  # 默认显示中间切片

        # 3. 动态设置通道标题
        if self.num_channels == 3:
            self.channels = ["T2", "DWI", "ADC"]
        elif self.num_channels == 1:
            self.channels = ["Label / Mask"]
        else:
            self.channels = [f"Channel {i}" for i in range(self.num_channels)]

        # 4. 创建动态宽度的画布
        self.fig, axes = plt.subplots(1, self.num_channels, figsize=(5 * self.num_channels, 5))
        self.fig.canvas.manager.set_window_title(f'Deep Viewer - Slice {self.slice_idx}')
        
        # 统一将 axes 转为列表，防止单通道时报错
        self.axes = [axes] if self.num_channels == 1 else axes
        
        # 5. 初始化图像显示
        self.images = []
        for i in range(self.num_channels):
            # 获取切片，如果是二值/类别掩膜，可以选择更好的 colormap (如 'viridis' 或 'gray')
            img_slice = self.data[i, self.slice_idx, :, :]
            img = self.axes[i].imshow(img_slice, cmap='gray')
            self.axes[i].set_title(self.channels[i])
            self.axes[i].axis('off')
            self.images.append(img)

        # 6. 绑定鼠标滚轮事件
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.tight_layout()
        print(f"格式化后数据维度: {self.data.shape}。请使用鼠标滚轮滚动切片。")
        plt.show()

    def update_display(self):
        """更新显示的图像内容"""
        for i in range(self.num_channels):
            self.images[i].set_data(self.data[i, self.slice_idx, :, :])
            # 对于标签图像，可能不同层最大最小值变化大，可以用这行动态刷新对比度 (可选)
            # self.images[i].set_clim(vmin=self.data[i, self.slice_idx].min(), vmax=self.data[i, self.slice_idx].max())
            
        self.fig.canvas.manager.set_window_title(f'Slice {self.slice_idx} / {self.max_slices - 1}')
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """处理滚动事件"""
        if event.button == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.max_slices - 1)
        elif event.button == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        
        self.update_display()

def npy_viewer(npy_path):
    """用于查看 1D 数组（如分类标签）的小工具"""
    data = np.load(npy_path)
    print("\n--- 数组概览 ---")
    print("数据维度 (Shape):", data.shape)
    print("数据类型 (Dtype):", data.dtype)

    # 如果是 1D 数组（如 PROMIS 的 12/20 区域标签向量），直接打印内容
    if data.ndim == 1:
        print("\n--- 标签详情 ---")
        for i, isup in enumerate(data):
            print(f"区域 {i+1} 分级/值: {isup}")
    else:
        print("数据为高维图像矩阵，建议使用 VolumeViewer 查看。")

# 使用示例
if __name__ == "__main__":
    
    # 【测试例 1】：查看 4D 多模态图像 (3, 32, 64, 64)
    viewer = VolumeViewer(r'F:\RP_dataset\Target biosy\Processed_TCIA\Prostate-MRI-US-Biopsy-0001_90221\input_tensor.npy')
    viewer = VolumeViewer(r'F:\RP_dataset\Dataset_prostate_MRI\Dataset_prostate_MRI_dwi\000_img.npy')

    viewer=VolumeViewer(r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS_dwi\P-10104751\input_tensor.npy')
    
    # 【测试例 2】：查看 3D 单通道掩膜 (32, 64, 64) (如 lab.npy 或 zones_mask.nii.gz 转换的 npy)
    # viewer = VolumeViewer(r'F:\RP_dataset\Dataset_prostate_MRI\Dataset_prostate_MRI\000_lab.npy')

    
    # 【测试例 3】：查看 1D 向量标签
    # npy_viewer(r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS_Labels_NPY\P-12743658.npy')
    # npy_viewer(r'F:\RP_dataset\Target biosy\Processed_TCIA\Prostate-MRI-US-Biopsy-0425\systematic_labels.npy')
    # npy_viewer(r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS\P-11691939\systematic_labels.npy')