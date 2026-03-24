"""
Wasserstein 距离评估函数

来源: fwi_common.py:1155-1292
包含有监督和无监督的 Wasserstein 距离评估。
注意：导入此模块会触发 DAPS 全局注册表的副作用（@register_eval_fn）。
"""

import torch
import numpy as np
import ot
from DAPS.eval import EvalFn, register_eval_fn


def check(samples):
    """检查并修复 NaN 值"""
    idx = torch.isnan(samples)
    samples[idx] = torch.zeros_like(samples[idx])
    return samples


def get_gt_posterior(gt, operator, y, oversample=1, sigma=0.05):
    """
    获取 ground truth 后验样本

    Args:
        gt: 实际地震记录
        operator: 正演算子
        y: 观测数据
        oversample: 过采样倍数
        sigma: 噪声水平
    """
    idx = []
    for _ in range(oversample):
        likelihood = operator.likelihood(gt, y)
        resampling = torch.multinomial(
            likelihood / likelihood.sum(), len(gt), replacement=True
        ).to(gt.device)
        idx.append(resampling)
    idx = torch.cat(idx)
    return gt[idx]


def wasserstein(sample1, sample2):
    """计算两个样本集之间的 Wasserstein 距离"""
    sample1, sample2 = check(sample1), check(sample2)
    a = np.ones((sample1.shape[0],)) / sample1.shape[0]
    b = np.ones((sample2.shape[0],)) / sample2.shape[0]
    C = ot.dist(sample1.numpy(), sample2.numpy(), metric='euclidean')
    w = ot.emd2(a, b, C)
    return w


@register_eval_fn(name='w2dist')
class Wasserstein(EvalFn):
    """有监督 Wasserstein 距离评估函数"""
    cmp = 'min'

    def __init__(self, operator):
        self.operator = operator
        self.requires_gt = True

    def __call__(self, gt, measurement, sample, reduction='mean'):
        """
        Args:
            gt: reference ground truth, torch.Tensor([B, *data.shape])
            measurement: noisy measurement, torch.Tensor([B, *measurement.shape])
            sample: posterior samples, torch.Tensor([B, *data.shape])
        """
        gt_sample = get_gt_posterior(gt, self.operator, measurement)
        gt_prob = np.ones((gt_sample.shape[0],)) / gt_sample.shape[0]
        pred_prob = np.ones((sample.shape[0],)) / sample.shape[0]

        gt_sample_2d = gt_sample.detach().cpu().numpy().reshape(gt_sample.shape[0], -1)
        sample_2d = sample.detach().cpu().numpy().reshape(sample.shape[0], -1)

        dist = ot.dist(gt_sample_2d, sample_2d, metric='euclidean')
        w = ot.emd2(gt_prob, pred_prob, dist)

        if reduction == 'none':
            w = torch.tensor([w] * gt.shape[0], device=gt.device)
        else:
            w = torch.tensor([[w]], device=gt.device)
        return w


@register_eval_fn(name='w2dist_unsupervised')
class Wasserstein_us(EvalFn):
    """无监督 Wasserstein 距离评估函数"""
    cmp = 'min'

    def __init__(self, operator):
        self.operator = operator
        self.requires_gt = False

    def __call__(self, gt, measurement, sample, reduction='mean'):
        """
        以批处理方式,计算每个后验样本与单个真实测量数据之间的Wasserstein-2距离

        Args:
            gt: None, 在此函数中未使用
            measurement: 真实的地震数据, shape: [1, *measurement_shape]
            sample: 批处理的后验样本, shape: [B, *data_shape]
        Returns:
            torch.Tensor: 包含B个Wasserstein距离值的张量, shape: [B,]
        """
        pred = self.operator(sample)

        if measurement.shape[0] != 1:
            raise ValueError(f"此函数期望 measurement 的批次大小为1, 但收到了 {measurement.shape[0]}")

        pred_2d = pred.detach().cpu().numpy().reshape(pred.shape[0], -1)
        measurement_2d = measurement.detach().cpu().numpy().reshape(measurement.shape[0], -1)

        dist = ot.dist(pred_2d, measurement_2d, metric='euclidean')
        w = torch.from_numpy(dist).squeeze(dim=1).to(sample.device, dtype=torch.float32)

        return w
