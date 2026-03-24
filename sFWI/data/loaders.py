"""
惰性数据加载函数

来源: fwi_common.py:224-328
将模块级执行的 pickle 加载和 SEAM SEG-Y 读取封装为纯函数，
由调用者显式调用，消除导入时副作用。
"""

import os
import pickle
import torch
import numpy as np


def load_pickle_dataset(dataset_path):
    """
    加载 pickle 格式的地震数据集，返回速度和地震数据张量。

    参数:
        dataset_path: str, pickle 数据集文件路径

    返回:
        v_torch: torch.Tensor, 速度模型 (N, 200, 200)
        s_torch: torch.Tensor, 地震数据 (N, 1, 100, 300)
    """
    with open(dataset_path, 'rb') as f:
        dataset_origin = pickle.load(f)

    velocity = []
    seismic = []
    for i in range(len(dataset_origin)):
        velocity.append(dataset_origin[i]['velocity'])
        seismic.append(dataset_origin[i]['seismic'])

    v_torch = torch.stack(velocity)
    s_torch = torch.stack(seismic)

    return v_torch, s_torch


def load_seam_model(seam_vp_path, patch_size_h=200, patch_size_w=200,
                    stride_h=100, stride_w=100):
    """
    读取 SEAM SEG-Y 速度模型并切割成图块。

    参数:
        seam_vp_path: str, SEAM Vp 速度模型 SEG-Y 文件路径
        patch_size_h: int, 垂直方向图块尺寸
        patch_size_w: int, 水平方向图块尺寸
        stride_h: int, 垂直方向步长
        stride_w: int, 水平方向步长

    返回:
        v_torch_seam: torch.Tensor, 切割后的速度图块 (N, H, W)
    """
    import segyio

    print(f"正在从 '{os.path.basename(seam_vp_path)}' 加载SEAM速度模型...")

    with segyio.open(seam_vp_path, "r", ignore_geometry=True) as sgyfile:
        seam_full_model = sgyfile.trace.raw[:]

    # 转置为 (深度, 距离) 格式
    seam_full_model = seam_full_model.T

    print(f"成功加载模型！完整模型尺寸 (高度, 宽度): {seam_full_model.shape}")
    print(f"正在将完整模型切割成 {patch_size_h}x{patch_size_w} 大小的图块...")

    patches = []
    full_h, full_w = seam_full_model.shape

    for i in range(0, full_h - patch_size_h + 1, stride_h):
        for j in range(0, full_w - patch_size_w + 1, stride_w):
            patch = seam_full_model[i:i+patch_size_h, j:j+patch_size_w]
            patches.append(patch)

    print(f"切割完成！共生成 {len(patches)} 个图块。")

    v_torch_seam = torch.stack([torch.from_numpy(p) for p in patches])

    print(f"最终张量形状: {v_torch_seam.shape}")
    print(f"数据类型: {v_torch_seam.dtype}")

    return v_torch_seam
