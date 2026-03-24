"""
数据处理工具函数

来源: fwi_utils.py:122-132
"""

import torch


def normalize_data(data, vmin=None, vmax=None):
    """归一化数据到[0, 1]"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return (data - vmin) / (vmax - vmin + 1e-8)


def denormalize_data(data, vmin, vmax):
    """反归一化数据"""
    return data * (vmax - vmin) + vmin
