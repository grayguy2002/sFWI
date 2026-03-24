"""
NCSNpp_DAPS Score 模型

来源: fwi_common.py:643-758
改造：将原来依赖 exec 命名空间的隐式全局变量
(base_config, lgvd_config, f_path) 改为显式构造函数参数。
"""

import torch
from DAPS.model import DiffusionModel, register_model
from DAPS.sampler import get_sampler
from sde_lib import VESDE
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from models import utils as mutils
import models.ncsnpp  # 触发 @register_model('ncsnpp') 注册
from utils import restore_checkpoint
import datasets


@register_model(name='my_model')
class NCSNpp_DAPS(DiffusionModel):
    """
    NCSNpp 模型的 DAPS 适配器。

    改造要点：原代码中 __init__ 直接引用 exec 命名空间中的
    base_config, lgvd_config, f_path。现在全部改为显式构造函数参数。
    """

    def __init__(self, model_config, base_config, lgvd_config, checkpoint_path):
        """
        参数:
            model_config: score_sde_pytorch 的配置对象
            base_config: OmegaConf, DAPS sampler 基础配置
            lgvd_config: OmegaConf, Langevin dynamics 配置
            checkpoint_path: str, 模型检查点文件路径
        """
        super().__init__()

        self.config = model_config

        # 初始化 SDE
        self.sde = VESDE(
            sigma_min=self.config.model.sigma_min,
            sigma_max=self.config.model.sigma_max,
            N=self.config.model.num_scales
        )
        self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

        # 初始化数据处理工具
        self.sigmas = mutils.get_sigmas(self.config)
        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        # 创建和加载模型
        self.score_model = mutils.create_model(self.config)
        self._initialize_model(checkpoint_path)

        # 创建 sampler（显式传入配置）
        sampler_kwargs = {
            'annealing_scheduler_config': base_config.annealing_scheduler_config,
            'diffusion_scheduler_config': base_config.diffusion_scheduler_config,
            'lgvd_config': lgvd_config,
            'sde': self.sde,
            'inverse_scaler': self.inverse_scaler,
            'sampling_eps': 1e-5,
            'latent': False
        }
        self.daps = get_sampler(**sampler_kwargs)

    def forward(self, x, t):
        x = x.to(self.config.device)
        t = t.to(self.config.device)
        return self.score_model(x, t)

    def _initialize_model(self, ckpt_filename):
        """初始化模型，加载检查点"""
        optimizer = get_optimizer(self.config, self.score_model.parameters())
        ema = ExponentialMovingAverage(
            self.score_model.parameters(),
            decay=self.config.model.ema_rate
        )
        state = dict(
            step=0,
            optimizer=optimizer,
            model=self.score_model,
            ema=ema
        )
        state = restore_checkpoint(ckpt_filename, state, self.config.device)
        ema.copy_to(self.score_model.parameters())

    def score(self, x, sigma):
        """计算 score function"""
        if isinstance(sigma, float):
            sigma = torch.ones(x.shape[0], device=x.device) * sigma
        x = x.to(self.config.device)
        sigma = sigma.to(self.config.device)
        score = self.score_model(x, sigma)
        return score

    def set_device(self, device):
        """设置模型设备"""
        self.config.device = torch.device(device)
        self.score_model = self.score_model.to(self.config.device)
        self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

    def load_checkpoint(self, ckpt_filename):
        """加载新的检查点"""
        self._initialize_model(ckpt_filename)
