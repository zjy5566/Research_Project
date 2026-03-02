import numpy as np
import matplotlib.pyplot as plt

class VolumeViewer:
    def __init__(self, npy_path):
        # 加载数据
        # 预期形状: (C, D, H, W)，即 (3, Depth, Height, Width)
        self.data = np.load(npy_path)
        
        if self.data.ndim != 4:
            raise ValueError(f"期望 4D 数据 (C, D, H, W)，但得到的是 {self.data.shape}")

        self.channels = ["T2", "DWI", "ADC"]
        self.slice_idx = self.data.shape[1] // 2  # 默认显示中间切片
        self.max_slices = self.data.shape[1]

        # 创建画布
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.canvas.manager.set_window_title(f'Deep Viewer - Slice {self.slice_idx}')
        
        self.images = []
        for i in range(3):
            img = self.axes[i].imshow(self.data[i, self.slice_idx, :, :], cmap='gray')
            self.axes[i].set_title(self.channels[i])
            self.axes[i].axis('off')
            self.images.append(img)

        # 绑定鼠标滚轮事件
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.tight_layout()
        print(f"已加载数据: {self.data.shape}。请使用鼠标滚轮滚动切片。")
        plt.show()

    def update_display(self):
        """更新显示的图像内容"""
        for i in range(3):
            self.images[i].set_data(self.data[i, self.slice_idx, :, :])
        
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
    data = np.load(npy_path)
    # 2. 查看基本信息
    print("数据内容:", data)
    print("数据维度 (Shape):", data.shape)
    print("数据类型 (Dtype):", data.dtype)

    # 3. 如果是 20 区域标签向量，你可以直接打印出来看每一个区域的分级
    for i, isup in enumerate(data):
        print(f"区域 {i+1} 的 ISUP 分级: {isup}")

# 使用示例
if __name__ == "__main__":

    viewer = VolumeViewer(r"F:\RP_dataset\derived PROMIS data set\Processed_PROMIS\P-12743658\input_tensor.npy")
    # viewer= VolumeViewer(r'F:\RP_dataset\derived PROMIS data set\Processed_PROMIS_Labels_NPY\P-12743658.npy')
    # viewer = VolumeViewer(r'F:\RP_dataset\Processed_Target_Biopsy\Prostate-MRI-US-Biopsy-0001\input_tensor.npy')
    import numpy as np

    # 1. 加载 npy 文件
    # 假设文件名是 P-10104751.npy
    # viewer = npy_viewer('F:\RP_dataset\derived PROMIS data set\Processed_PROMIS_Labels_NPY\P-12743658.npy')

    