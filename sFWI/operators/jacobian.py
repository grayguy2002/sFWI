"""
雅可比矩阵计算器

来源: fwi_common.py:963-1153
用于计算观测矩阵 H 和噪声协方差矩阵 R。
"""

import torch
import numpy as np


class RHcalculator_0112:
    """雅可比矩阵计算器"""

    def __init__(self, seismic_operator):
        """
        初始化雅可比矩阵计算器

        Parameters:
        -----------
        seismic_operator : SeismicForwardOperator
            地震正演算子实例
        """
        self.operator = seismic_operator
        self.device = seismic_operator.device

    def calculate_H_matrix(self, velocity_model):
        """
        使用批量计算优化 H 矩阵计算
        """
        velocity_model.requires_grad_(True)
        seismic_data = self.operator(velocity_model)

        n_receivers, nt = seismic_data.shape[2:]
        n_model_params = velocity_model.numel()
        H = torch.zeros((n_receivers * nt, n_model_params), device=velocity_model.device)

        seismic_data_flat = seismic_data.view(-1)
        indices = torch.arange(seismic_data_flat.size(0), device=velocity_model.device)

        gradients = []
        for idx in indices:
            if velocity_model.grad is not None:
                velocity_model.grad.zero_()
            seismic_data_flat[idx].backward(retain_graph=True)
            gradients.append(velocity_model.grad.clone().flatten())

        H = torch.stack(gradients, dim=0)
        return H

    def calculate_H_efficient(self, velocity_model, observed_data):
        """
        使用伴随状态方法更高效地计算H矩阵

        Parameters:
        -----------
        velocity_model : torch.Tensor
            速度模型 [1, 1, H, W]
        observed_data : torch.Tensor
            观测数据 [1, 1, n_receivers, nt]

        Returns:
        --------
        H : torch.Tensor
            观测矩阵（以隐式形式）
        """
        velocity_model = velocity_model.clone().detach().requires_grad_(True)
        predicted_data = self.operator(velocity_model)
        residual = predicted_data - observed_data
        loss = 0.5 * torch.sum(residual ** 2)
        loss.backward()
        gradient = velocity_model.grad.clone()
        return gradient

    def setup_noise_covariance(self, velocity_model, snr_db=20):
        """
        使用稀疏矩阵优化噪声协方差
        """
        from scipy.sparse import diags
        seismic_data = self.operator(velocity_model)
        n_receivers, nt = seismic_data.shape[2:]
        n_data = n_receivers * nt

        noise_var = 10 ** (-snr_db / 10)
        R_diag = np.ones(n_data) * noise_var

        space_corr = diags([1, 0.5], [0, 1], shape=(n_data, n_data)).toarray()
        time_corr = diags([1, 0.7], [0, 1], shape=(n_data, n_data)).toarray()
        R = diags(R_diag).toarray() * space_corr * time_corr

        return R
