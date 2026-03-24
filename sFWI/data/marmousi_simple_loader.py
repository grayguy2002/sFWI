"""
Marmousi数据集加载器 - 简化版本

用于快速加载Marmousi数据进行OOD实验。
"""
import os
import sys
import pickle
import torch
import numpy as np


def _load_pickle_with_compat(pkl_path):
    """兼容加载 __main__.SeismicPatchDataset 历史 pickle。"""
    with open(pkl_path, 'rb') as f:
        try:
            return pickle.load(f)
        except AttributeError as exc:
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


def load_marmousi_dataset(pkl_path=None):
    """
    加载Marmousi数据集。

    参数:
        pkl_path: str, pickle文件路径（默认使用Colab路径）

    返回:
        v_torch: torch.Tensor, 速度模型 (N, 200, 200)
        s_torch: torch.Tensor, 地震数据 (N, 1, 100, 300)
    """
    if pkl_path is None:
        pkl_path = '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Marmousi dataset not found: {pkl_path}")

    print(f"正在加载Marmousi数据集: {pkl_path}")

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
    s_torch = torch.stack(seismic) if seismic else None

    print(f"速度模型: {v_torch.shape}, 范围: [{v_torch.min():.2f}, {v_torch.max():.2f}]")

    if s_torch is not None:
        print(f"地震数据: {s_torch.shape}")

    return v_torch, s_torch


if __name__ == '__main__':
    v, s = load_marmousi_dataset()
    print(f"\n加载完成！共 {len(v)} 个样本。")
