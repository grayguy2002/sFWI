"""
DAPS 地震正演算子

来源: fwi_common.py:786-961
原名 SeismicForwardOperator，重命名为 DAPSSeismicOperator 以消歧。
继承 DAPS.forward_operator.Operator，单炮、输出归一化到图像尺寸。
"""

import torch
import torch.nn.functional as F
import deepwave
from DAPS.forward_operator import Operator, register_operator


@register_operator(name='seismic_fo')
class DAPSSeismicOperator(Operator):
    """DAPS 地震正演算子（单炮、归一化输出）"""

    def __init__(self, config, image_size=128, sigma=0.05):
        super().__init__(sigma=sigma)
        self.config = config
        self.image_size = image_size
        self.dx = 2.

        # 采集系统参数
        self.n_shots = 1
        self.n_sources_per_shot = 1
        self.n_receivers_per_shot = 100
        self.d_receiver = 2
        self.first_receiver = 0
        self.source_depth = 2

        # 震源参数
        self.freq = 25
        self.dt = 0.002
        self.nt = 300
        self.peak_time = 1.5 / self.freq

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_acquisition_geometry(self, patch_size):
        """设置采集系统几何布置"""
        source_locations = torch.zeros(self.n_shots, self.n_sources_per_shot, 2,
                                     dtype=torch.long, device=self.device)
        source_locations[..., 1] = self.source_depth
        source_locations[:, 0, 0] = torch.clamp(
            torch.arange(1, 2) * (patch_size / 2),
            0, patch_size - 1
        )

        receiver_locations = torch.zeros(self.n_shots, self.n_receivers_per_shot, 2,
                                       dtype=torch.long, device=self.device)
        receiver_locations[..., 1] = 2
        receiver_locations[:, :, 0] = torch.clamp(
            (torch.arange(self.n_receivers_per_shot) * self.d_receiver + self.first_receiver)
            .repeat(self.n_shots, 1),
            0, patch_size - 1
        )

        source_amplitudes = (
            deepwave.wavelets.ricker(self.freq, self.nt, self.dt, self.peak_time)
            .repeat(self.n_shots, self.n_sources_per_shot, 1)
            .to(self.device)
        )

        return source_locations, receiver_locations, source_amplitudes

    def _toVelocityShape(self, seismic_data):
        """
        将地震数据重塑为速度模型形状

        Args:
            seismic_data: shape [B, n_receivers, nt]
        Returns:
            reshaped_data: shape [B, 1, image_size, image_size]
        """
        batch_size = seismic_data.shape[0]
        data = seismic_data.unsqueeze(1)
        reshaped_data = F.interpolate(
            data,
            size=(128, 128),
            mode='bilinear',
            align_corners=True
        )
        return reshaped_data

    def __call__(self, x):
        """
        计算速度模型对应的地震记录并重塑为相同尺寸

        Args:
            x: 速度模型, torch.Tensor([B, 1, H, W])
        Returns:
            y: 重塑后的地震记录, torch.Tensor([B, 1, H, W])
        """
        x = x.to(self.device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.shape[2] < self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=True
            )

        # 逐样本归一化到速度范围，避免 batch 内样本相互耦合
        x_min = torch.amin(x, dim=(1, 2, 3), keepdim=True)
        x_max = torch.amax(x, dim=(1, 2, 3), keepdim=True)
        x_range = (x_max - x_min).clamp_min(1e-8)
        x_normalized = (x - x_min) / x_range
        v_min, v_max = 1500.0, 5500.0
        x = v_min + (v_max - v_min) * x_normalized

        batch_size = x.shape[0]
        x = x.squeeze(1)

        all_receiver_amplitudes = []
        source_locations, receiver_locations, source_amplitudes = self.setup_acquisition_geometry(x.shape[-1])

        for i in range(batch_size):
            velocity = x[i]
            out = deepwave.scalar(
                velocity,
                self.dx,
                self.dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_freq=self.freq
            )
            receiver_amplitudes = out[-1]
            receiver_amplitudes = receiver_amplitudes.reshape(-1, self.nt)
            all_receiver_amplitudes.append(receiver_amplitudes)

        seismic_data = torch.stack(all_receiver_amplitudes, dim=0)
        y = self._toVelocityShape(seismic_data)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # 逐样本归一化，保证候选并行评估时 misfit 可比
        y_min = torch.amin(y, dim=(1, 2, 3), keepdim=True)
        y_max = torch.amax(y, dim=(1, 2, 3), keepdim=True)
        y_range = (y_max - y_min).clamp_min(1e-8)
        y_normalized = (y - y_min) / y_range

        return y_normalized
