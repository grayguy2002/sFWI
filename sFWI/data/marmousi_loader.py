"""
Marmousi数据集加载器

用于加载Marmousi速度模型（支持SEGY和pkl格式），用于OOD泛化实验。
数据来源: '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'
"""
import os
import sys
import pickle
import numpy as np
import torch
import segyio


def _load_pickle_with_compat(pkl_path):
    """
    兼容加载 pickle。

    处理在 notebook/脚本中以 __main__.SeismicPatchDataset 保存的数据，
    避免在其他入口脚本中反序列化时报 AttributeError。
    """
    with open(pkl_path, 'rb') as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
            # 仅对已知历史数据格式启用兼容分支
            if "SeismicPatchDataset" not in str(exc):
                raise
            f.seek(0)

            class _CompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == '__main__' and name == 'SeismicPatchDataset':
                        from sFWI.data.datasets import SeismicPatchDataset
                        return SeismicPatchDataset
                    return super().find_class(module, name)

            return _CompatUnpickler(f).load()


def load_marmousi_from_pkl(pkl_path, image_size=200):
    """
    从pickle文件加载Marmousi速度模型数据集。

    参数:
        pkl_path: str, pickle文件路径
        image_size: int, 目标图像尺寸（默认200）

    返回:
        v_torch: torch.Tensor, 速度模型 (N, H, W)
        s_torch: torch.Tensor, 地震数据 (N, 1, 100, 300) 或 None
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Marmousi pickle文件不存在: {pkl_path}")

    print(f"正在从 '{os.path.basename(pkl_path)}' 加载Marmousi数据集...")
    dataset_origin = _load_pickle_with_compat(pkl_path)

    print(f"数据集包含 {len(dataset_origin)} 个样本")

    # 提取velocity和seismic数据
    velocity = []
    seismic = []
    for i in range(len(dataset_origin)):
        velocity.append(dataset_origin[i]['velocity'])
        if 'seismic' in dataset_origin[i]:
            seismic.append(dataset_origin[i]['seismic'])

    v_torch = torch.stack(velocity)  # (N, 200, 200)
    print(f"速度模型张量形状: {v_torch.shape}")
    print(f"速度范围: {v_torch.min().item():.2f} - {v_torch.max().item():.2f} m/s")

    if seismic:
        s_torch = torch.stack(seismic)  # (N, 1, 100, 300)
        print(f"地震数据张量形状: {s_torch.shape}")
    else:
        s_torch = None

    return v_torch, s_torch


def load_marmousi_model(marmousi_path, patch_size=200, stride=100):
    """
    读取Marmousi速度模型（SEGY格式）并切割成图块。

    参数:
        marmousi_path: str, Marmousi模型文件路径（SEG-Y格式）
        patch_size: int, 图块尺寸（默认200x200）
        stride: int, 滑动步长（默认100，有50%重叠）

    返回:
        v_torch: torch.Tensor, 形状为 (N, H, W) 的速度模型图块
        patch_info: dict, 图块切割信息
    """
    if not os.path.exists(marmousi_path):
        raise FileNotFoundError(f"Marmousi模型文件不存在: {marmousi_path}")

    print(f"正在从 '{os.path.basename(marmousi_path)}' 加载Marmousi速度模型...")

    with segyio.open(marmousi_path, "r", ignore_geometry=True) as sgyfile:
        marmousi_full_model = sgyfile.trace.raw[:]

    marmousi_full_model = marmousi_full_model.T
    print(f"成功加载模型！完整模型尺寸 (高度, 宽度): {marmousi_full_model.shape}")

    # 切割成图块
    patches = []
    full_h, full_w = marmousi_full_model.shape

    for i in range(0, full_h - patch_size + 1, stride):
        for j in range(0, full_w - patch_size + 1, stride):
            patch = marmousi_full_model[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    print(f"切割完成！共生成 {len(patches)} 个 {patch_size}x{patch_size} 图块。")

    v_torch = torch.stack([torch.from_numpy(p) for p in patches])

    patch_info = {
        'full_shape': marmousi_full_model.shape,
        'patch_size': patch_size,
        'stride': stride,
        'n_patches': len(patches),
    }

    print(f"最终张量形状: {v_torch.shape}")
    print(f"速度范围: {v_torch.min().item():.2f} - {v_torch.max().item():.2f} m/s")

    return v_torch, patch_info


def get_marmousi_patch(v_torch_marmousi, index):
    """
    获取单个Marmousi图块，用于DAPS采样。

    参数:
        v_torch_marmousi: torch.Tensor, Marmousi图块张量
        index: int, 图块索引

    返回:
        patch: torch.Tensor, 形状为 (1, H, W) 的单个图块
    """
    patch = v_torch_marmousi[index].unsqueeze(0)
    return patch


def create_marmousi_dataset(v_torch_marmousi, image_size=32, device='cpu'):
    """
    创建Marmousi数据集的VelocityDataset实例。

    参数:
        v_torch_marmousi: torch.Tensor, Marmousi速度图块 (N, H, W)
        image_size: int, 目标图像大小
        device: str, 设备

    返回:
        dataset: VelocityDataset 实例
    """
    from sFWI.data.daps_adapter import create_velocity_dataset

    # 重采样到目标尺寸
    if image_size != 200:
        v_resized = torch.nn.functional.interpolate(
            v_torch_marmousi.unsqueeze(1),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        print(f"Marmousi数据集重采样到 {image_size}x{image_size}")
    else:
        v_resized = v_torch_marmousi

    dataset = create_velocity_dataset(v_resized, image_size=image_size)
    return dataset


def normalize_marmousi_for_training(v_torch):
    """
    归一化Marmousi数据用于训练。

    参数:
        v_torch: torch.Tensor, 原始速度模型 (N, H, W)

    返回:
        v_normalized: torch.Tensor, 归一化后的速度模型
        stats: dict, 归一化统计信息
    """
    v_mean = v_torch.mean()
    v_std = v_torch.std()

    v_normalized = (v_torch - v_mean) / v_std

    stats = {
        'mean': v_mean.item(),
        'std': v_std.item(),
        'min': v_torch.min().item(),
        'max': v_torch.max().item(),
    }

    print(f"Marmousi归一化: mean={v_mean.item():.2f}, std={v_std.item():.2f}")
    print(f"  速度范围: {stats['min']:.2f} - {stats['max']:.2f} m/s")

    return v_normalized, stats


if __name__ == '__main__':
    # 测试加载Marmousi数据
    if len(sys.argv) < 2:
        print("用法: python marmousi_loader.py <path_to_marmousi.pkl_or.sgy>")
        print("  支持pkl格式和SEGY格式")
        sys.exit(1)

    marmousi_path = sys.argv[1]

    if marmousi_path.endswith('.pkl'):
        v_torch, s_torch = load_marmousi_from_pkl(marmousi_path)
        print(f"\n加载完成！共 {v_torch.shape[0]} 个速度模型。")
    else:
        v_torch, info = load_marmousi_model(marmousi_path)
        print(f"\n加载完成！共 {info['n_patches']} 个图块。")
