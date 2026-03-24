"""
Score SDE 路径设置与配置初始化

来源: fwi_common.py:337-411
将 sys.path 操作和 SDE/config 初始化封装为显式函数。
"""

import sys
import os


def setup_score_sde_path(code_dir):
    """
    将 score_sde_pytorch 目录添加到 sys.path。

    参数:
        code_dir: str, code 目录的绝对路径
    """
    score_sde_path = os.path.join(code_dir, 'score_sde_pytorch')
    if score_sde_path not in sys.path:
        sys.path.insert(0, score_sde_path)
    return score_sde_path


def create_sde_config(code_dir, batch_size=64):
    """
    创建 Score SDE 配置和 VESDE 实例。

    必须在 setup_score_sde_path() 之后调用。

    参数:
        code_dir: str, code 目录路径（用于定位 checkpoint）
        batch_size: int, 批次大小

    返回:
        config: 模型配置对象
        sde: VESDE 实例
    """
    from configs.ve import cifar10_ncsnpp_continuous as configs
    from sde_lib import VESDE

    project_root = os.path.dirname(code_dir)

    config = configs.get_config()
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales
    )
    sde.discrete_sigmas = sde.discrete_sigmas.to(config.device)

    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    return config, sde
