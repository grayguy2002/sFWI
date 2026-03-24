"""
可视化工具函数

合并自 fwi_utils.py 和 fwi_common.py 中的可视化函数。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_velocity_fields(samples, save_path=None):
    """
    可视化生成的速度场

    参数:
        samples: torch.Tensor, 形状为 (N, C, H, W)
        save_path: str, 可选，保存图像的路径
    """
    samples = samples.cpu().numpy()

    n_samples = len(samples)
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    plt.figure(figsize=(15, 15))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        field = samples[i][0]
        plt.imshow(field, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_data(data, mode='velocity', n_samples=None, figsize=(15, 15), save_path=None):
    """
    统一的可视化函数，可以绘制速度模型或地震记录

    参数:
        data: torch.Tensor, 输入数据
            - 速度模型模式: shape为(batch, channel, height, width)
            - 地震记录模式: shape为(batch, time, receivers)
        mode: str, 可选 'velocity' 或 'seismic'
        n_samples: int, 可选，显示多少个样本
        figsize: tuple, 图像大小
        save_path: str, 可选，保存图像的路径
    """
    plt.figure(figsize=figsize)
    data = data.detach().cpu()

    if n_samples is None:
        n_samples = len(data)
    n_samples = min(n_samples, len(data))

    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)

        if mode.lower() == 'velocity':
            field = data[i][0].numpy()
            im = plt.imshow(field, cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.set_label('Velocity (m/s)', fontsize=8)
            cbar.ax.tick_params(labelsize=6)
            plt.title(f'Velocity Model {i+1}', fontsize=8)

        elif mode.lower() == 'seismic':
            field = data[i]
            vmin, vmax = torch.quantile(field, torch.tensor([0.05, 0.95]))
            im = plt.imshow(field.T,
                          cmap='gray',
                          vmin=vmin,
                          vmax=vmax)
            cbar = plt.colorbar(im)
            cbar.set_label('Amplitude', fontsize=8)
            cbar.ax.tick_params(labelsize=6)
            plt.title(f'Seismic Record {i+1}', fontsize=8)

        else:
            raise ValueError("mode must be either 'velocity' or 'seismic'")

        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
