"""
经典 FWI 地震正演算子

来源: exp_classic_fwi.py:285-326
独立类，多炮、返回原始接收器数据，用于多频 FWI。
"""

import torch
import deepwave


class ClassicSeismicOperator:
    """经典 FWI 地震正演算子（多炮、原始输出）"""

    def __init__(self, config, f_peak):
        self.config = config
        self.device = config['device']
        self.model_shape = config['model_shape']
        self.dx = config['dx']
        self.dt = config['dt']
        self.nt = config['nt']
        self.f_peak = f_peak
        self._setup_acquisition()

    def _setup_acquisition(self):
        n_shots = self.config['n_shots']
        n_receivers_per_shot = self.config['n_receivers_per_shot']
        model_shape = self.config['model_shape']
        src_depth = self.config['src_depth']
        rec_depth = self.config['rec_depth']

        self.src_locs = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=self.device)
        self.src_locs[:, 0, 0] = torch.linspace(0, model_shape[1] - 1, n_shots).long()
        self.src_locs[:, 0, 1] = src_depth

        self.rec_locs = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=self.device)
        self.rec_locs[:, :, 0] = torch.linspace(0, model_shape[1] - 1, n_receivers_per_shot).long().repeat(n_shots, 1)
        self.rec_locs[:, :, 1] = rec_depth

        self.source_amplitudes = deepwave.wavelets.ricker(
            self.f_peak, self.nt, self.dt, 1.0 / self.f_peak
        ).to(self.device).repeat(n_shots, 1, 1)

    def forward(self, model):
        out = deepwave.scalar(
            model, self.dx, self.dt,
            source_amplitudes=self.source_amplitudes,
            source_locations=self.src_locs,
            receiver_locations=self.rec_locs,
            accuracy=4, pml_width=[10, 10, 10, 10]
        )
        return out[-1]
