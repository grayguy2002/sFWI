"""
sFWI 配置中心

用 dataclass 替代原来散落在 fwi_config.py 和 fwi_common.py 中的模块级全局变量。
所有路径从 code_dir 动态派生，消除硬编码路径。
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PathConfig:
    """路径配置：所有路径从 code_dir 动态派生"""

    code_dir: str = ""

    def __post_init__(self):
        if not self.code_dir:
            self.code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def project_root(self) -> str:
        return os.path.dirname(self.code_dir)

    @property
    def f_path(self) -> str:
        return self.project_root + os.sep

    @property
    def external_data_root(self) -> str:
        return os.path.join(os.path.dirname(self.project_root), 'solving_inverse_in_SGM')

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.external_data_root, 'dataset', 'seismic_dataset.pkl')

    @property
    def seam_model_path(self) -> str:
        return os.path.join(
            self.project_root, 'SEAM_I_2D_Model', 'SEAM_I_2D_Model',
            'SEAM_Vp_Elastic_N23900.sgy'
        )

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.project_root, 'checkpoints', 'checkpoint_5.pth')

    @property
    def seam_finetune_checkpoint_path(self) -> str:
        """SEAM fine-tune checkpoint路径"""
        return os.path.join(self.project_root, 'checkpoints', 'seam_finetune_checkpoint_5.pth')

    @property
    def marmousi_dataset_path(self) -> str:
        """Marmousi数据集路径（用于预训练）"""
        # 优先使用solving_inverse_in_SGM目录下的seismic_dataset.pkl
        external_root = self.external_data_root
        pkl_path = os.path.join(external_root, 'dataset', 'seismic_dataset.pkl')
        if os.path.exists(pkl_path):
            return pkl_path
        # 回退到project_root
        return os.path.join(self.project_root, 'dataset', 'seismic_dataset.pkl')

    @property
    def marmousi_checkpoint_path(self) -> str:
        """Marmousi预训练checkpoint路径"""
        return os.path.join(self.project_root, 'checkpoints', 'marmousi_checkpoint_5.pth')

    @property
    def output_path(self) -> str:
        return os.path.join(self.project_root, 'outputs')

    @property
    def score_sde_path(self) -> str:
        return os.path.join(self.code_dir, 'score_sde_pytorch')

    @property
    def daps_config_dir(self) -> str:
        return os.path.join(self.code_dir, 'DAPS', 'config', 'sampler')


@dataclass
class DAPSHyperparams:
    """DAPS 采样超参数"""

    # Annealing scheduler
    annealing_steps: int = 50
    annealing_sigma_max: float = 0.1
    annealing_sigma_min: float = 0.01
    annealing_sigma_final: float = 0.0

    # Diffusion scheduler
    diffusion_steps: int = 20

    # LGVD config
    langevin_steps: int = 20
    lr: float = 1e-4
    tau: float = 0.07
    lr_min_ratio: float = 1.0
    lambda_prior: float = 1.0
    lambda_prior_min_ratio: float = 1.0

    # Sampling
    batch_size: int = 1
    sigma: float = 0.3
    sampling_eps: float = 1e-5


@dataclass
class FWIConfig:
    """sFWI 主配置，聚合路径、超参数和设备信息"""

    paths: PathConfig = field(default_factory=PathConfig)
    daps: DAPSHyperparams = field(default_factory=DAPSHyperparams)

    # 设备
    device: str = ""

    # 模型参数
    image_size: int = 32
    seed_index: int = 1

    # 经典FWI配置
    fwi_frequencies: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])
    fwi_max_iterations: int = 100

    # 可视化配置
    figsize: Tuple[int, int] = (15, 15)
    dpi: int = 300
    cmap: str = 'viridis'

    def __post_init__(self):
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_daps_configs(cfg: FWIConfig):
    """
    根据 FWIConfig 按需构建 DAPS 所需的 OmegaConf 配置对象。

    返回:
        base_config: OmegaConf, DAPS sampler 基础配置
        lgvd_config: OmegaConf, Langevin dynamics 配置
    """
    from omegaconf import OmegaConf

    daps_params = cfg.daps
    config_file = os.path.join(
        cfg.paths.daps_config_dir, "edm_daps.yaml"
    )
    base_config = OmegaConf.load(config_file)

    # Annealing scheduler
    base_config.annealing_scheduler_config.num_steps = daps_params.annealing_steps
    base_config.annealing_scheduler_config.sigma_max = daps_params.annealing_sigma_max
    base_config.annealing_scheduler_config.sigma_min = daps_params.annealing_sigma_min
    base_config.annealing_scheduler_config.sigma_final = daps_params.annealing_sigma_final

    # Diffusion scheduler
    base_config.diffusion_scheduler_config.num_steps = daps_params.diffusion_steps

    # LGVD config
    lgvd_config = OmegaConf.create()
    lgvd_config.num_steps = daps_params.langevin_steps
    lgvd_config.lr = daps_params.lr
    lgvd_config.tau = daps_params.tau
    lgvd_config.lr_min_ratio = daps_params.lr_min_ratio
    lgvd_config.lambda_prior = daps_params.lambda_prior
    lgvd_config.lambda_prior_min_ratio = daps_params.lambda_prior_min_ratio

    return base_config, lgvd_config
