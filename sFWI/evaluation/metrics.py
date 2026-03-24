"""
评估指标函数

来源: fwi_utils.py:106-116 + exp_rss_dms.py:311-325
合并 calculate_mse 的两个定义，添加 MAE 和相对误差。
"""

import torch
import torch.nn.functional as F


def calculate_mse(pred, target):
    """
    计算均方误差 (MSE)

    Args:
        pred: torch.Tensor, 预测值
        target: torch.Tensor, 目标值

    Returns:
        float or torch.Tensor, MSE 值
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        return F.mse_loss(pred, target).item()
    else:
        return torch.mean((pred - target) ** 2).item()


def calculate_mae(pred, target):
    """计算平均绝对误差 (MAE)"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_relative_error(pred, target):
    """计算相对误差"""
    return (torch.norm(pred - target) / torch.norm(target)).item()
