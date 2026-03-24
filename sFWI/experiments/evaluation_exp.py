"""
系统性评估实验脚本

功能模块:
  - quantitative:      定量指标计算 (NRMSE, SSIM, PSNR)
  - uncertainty:       TopG 分支后验不确定性量化 (total/between/within)
  - failure_analysis:   失败模式分析 (Mode I-IV 分类与统计)
  - comparison_table:   方法对比汇总表 (LaTeX table 生成)

用法:
  %run sFWI/experiments/evaluation_exp.py -- --mode quantitative
  %run sFWI/experiments/evaluation_exp.py -- --mode uncertainty --sampling_method gss_topg --sm_path /path/to/sm.pt
  %run sFWI/experiments/evaluation_exp.py -- --mode failure_analysis --n_gt 100
  %run sFWI/experiments/evaluation_exp.py -- --mode comparison_table
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from tqdm import tqdm

# ---------- 路径设置 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # code/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

# ---------- sFWI 模块导入 ----------
from sFWI.config import FWIConfig, build_daps_configs
from sFWI.models.sde_setup import create_sde_config
from sFWI.models.score_model import NCSNpp_DAPS
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.evaluation.wasserstein import Wasserstein, Wasserstein_us
from sFWI.utils.file_utils import generate_timestamped_filename

from DAPS.sampler import get_sampler, DAPS
from DAPS.eval import Evaluator

# ---------- 聚类 & 相似度矩阵 ----------
from sFWI.utils.clustering import load_clustering_results

# ---------- 第三方指标库 ----------
try:
    from piq import psnr as piq_psnr, ssim as piq_ssim
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("[WARNING] piq 未安装，SSIM/PSNR 将使用简易实现。pip install piq")

# 与 daps_langevin.py 保持一致的速度模型配色
VELOCITY_CMAP = 'viridis'


# ================================================================
#  Section 1: 定量指标计算
# ================================================================

def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Normalized Root Mean Square Error.

    NRMSE = RMSE / (gt_max - gt_min)
    值域 [0, +inf), 越小越好。
    """
    pred, gt = _align_pred_to_gt(pred, gt)
    mse = F.mse_loss(pred, gt).item()
    rmse = np.sqrt(mse)
    gt_range = (gt.max() - gt.min()).item()
    if gt_range < 1e-8:
        return float('inf')
    return rmse / gt_range


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """确保 tensor 为 4D [B, C, H, W]。"""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor):
    """将 pred 与 gt 对齐到可比较形状。"""
    pred4 = _to_4d(pred).float()
    gt4 = _to_4d(gt).float()

    if pred4.shape[0] != gt4.shape[0]:
        if pred4.shape[0] == 1:
            pred4 = pred4.expand(gt4.shape[0], -1, -1, -1)
        elif gt4.shape[0] == 1:
            gt4 = gt4.expand(pred4.shape[0], -1, -1, -1)
        else:
            raise ValueError(
                f"batch 不可对齐: pred={tuple(pred4.shape)}, gt={tuple(gt4.shape)}"
            )

    if pred4.shape[1] != gt4.shape[1]:
        if pred4.shape[1] == 1:
            pred4 = pred4.expand(-1, gt4.shape[1], -1, -1)
        elif gt4.shape[1] == 1:
            gt4 = gt4.expand(-1, pred4.shape[1], -1, -1)
        else:
            raise ValueError(
                f"channel 不可对齐: pred={tuple(pred4.shape)}, gt={tuple(gt4.shape)}"
            )

    if pred4.shape[-2:] != gt4.shape[-2:]:
        pred4 = F.interpolate(
            pred4, size=gt4.shape[-2:], mode='bilinear', align_corners=False
        )

    return pred4, gt4


def _shift_to_nonneg(pred: torch.Tensor, gt: torch.Tensor):
    """将 pred 和 gt 平移到非负区间, 返回 (pred_shifted, gt_shifted, data_range)。

    piq 的 SSIM/PSNR 要求输入值在 [0, data_range] 内。
    score model 的输出可能包含负值, 需要统一平移。
    """
    global_min = min(pred.min().item(), gt.min().item())
    if global_min < 0:
        pred = pred - global_min
        gt = gt - global_min
    data_range = max(pred.max().item(), gt.max().item())
    if data_range < 1e-8:
        data_range = 1.0
    return pred, gt, data_range


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Structural Similarity Index. 值域 [-1, 1], 越大越好。"""
    pred, gt = _align_pred_to_gt(pred, gt)

    if HAS_PIQ:
        pred_s, gt_s, dr = _shift_to_nonneg(pred.float(), gt.float())
        return piq_ssim(pred_s, gt_s, data_range=dr).item()

    # 简易 SSIM fallback
    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (num / den).item()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB). 越大越好。"""
    pred, gt = _align_pred_to_gt(pred, gt)

    if HAS_PIQ:
        pred_s, gt_s, dr = _shift_to_nonneg(pred.float(), gt.float())
        return piq_psnr(pred_s, gt_s, data_range=dr).item()

    mse = F.mse_loss(pred, gt).item()
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_all_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """计算全部三项指标，返回 dict。"""
    return {
        'NRMSE': compute_nrmse(pred, gt),
        'SSIM':  compute_ssim(pred, gt),
        'PSNR':  compute_psnr(pred, gt),
    }


# ================================================================
#  Section 2: Baseline 模型通用接口
# ================================================================

# ---------- 已知 baseline 模型类注册表 ----------
# 从 conditional_autoencoder_0_1b.py 中提取的四个模型类。
# 为避免循环导入, 延迟导入: 在 load 时动态 import。
MODEL_CLASS_REGISTRY = {}


def register_model_class(name):
    """装饰器: 注册模型类到 MODEL_CLASS_REGISTRY。"""
    def decorator(cls):
        MODEL_CLASS_REGISTRY[name] = cls
        return cls
    return decorator


def _import_baseline_classes():
    """从 conditional_autoencoder_0_1b.py 所在目录导入模型类。

    由于源文件是 Colab 脚本 (包含 drive.mount 等副作用代码),
    这里直接在 evaluation_exp.py 中注册轻量级工厂函数,
    实际模型类定义通过 JSON config 的 'module' 字段动态导入。
    """
    pass  # 模型类通过下方的具体 Baseline 子类内联构建


# ---------- Baseline 抽象接口 ----------
BASELINE_REGISTRY = {}


def register_baseline(type_name):
    """装饰器: 将 BaselineModel 子类注册到全局表。"""
    def decorator(cls):
        BASELINE_REGISTRY[type_name] = cls
        return cls
    return decorator


class BaselineModel:
    """Baseline 模型的抽象接口。

    所有 baseline 必须实现:
      - load(weights_path, device, **kwargs)  加载权重
      - predict(seismic_input)  输入地震数据, 返回速度模型预测

    注意: baseline 模型的输入是地震数据 [B, 1, 100, 300],
    而非 sFWI 的 DAPS operator 输出。evaluation 流程中需要
    用原始地震数据 (而非 DAPS measurement) 调用 baseline。
    """

    def load(self, weights_path: str, device: str, **kwargs):
        raise NotImplementedError

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        """输入: seismic [B, 1, 100, 300], 输出: velocity [B, 1, 200, 200] (归一化)。"""
        raise NotImplementedError

    @staticmethod
    def _normalize_output(output: torch.Tensor) -> torch.Tensor:
        """统一输出为 [B, 1, H, W] 格式。"""
        if output.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            output = output.unsqueeze(0).unsqueeze(0)
        elif output.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            output = output.unsqueeze(1)
        return output


@register_baseline('autoencoder')
class AutoencoderBaseline(BaselineModel):
    """Autoencoder baseline (conditional_autoencoder_0_1b.py: class Autoencoder)。

    forward(x) -> [B, 200, 200]
    权重文件: ae_seismic.pth (state_dict)
    """

    def load(self, weights_path: str, device: str, **kwargs):
        from sFWI.experiments._baseline_models import Autoencoder
        self.model = Autoencoder()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(seismic_input.to(self.device))
        return self._normalize_output(output)


@register_baseline('unet')
class UNetBaseline(BaselineModel):
    """ModernUNet baseline (conditional_autoencoder_0_1b.py: class ModernUNet)。

    forward(x) -> [B, 200, 200]  (内部 squeeze)
    权重文件: ModernUNET_seismic.pth (state_dict)
    """

    def load(self, weights_path: str, device: str, **kwargs):
        from sFWI.experiments._baseline_models import ModernUNet
        self.model = ModernUNet()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(seismic_input.to(self.device))
        return self._normalize_output(output)


@register_baseline('vae')
class VAEBaseline(BaselineModel):
    """VAE baseline (conditional_autoencoder_0_1b.py: class VAE)。

    forward(x) -> (recon [B, 200, 200], mu, logvar)
    权重文件: VAE_seismic.pth (state_dict)
    """

    def load(self, weights_path: str, device: str, **kwargs):
        from sFWI.experiments._baseline_models import VAE
        self.model = VAE()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output, _mu, _logvar = self.model(seismic_input.to(self.device))
        return self._normalize_output(output)


@register_baseline('diffae')
class DiffAEBaseline(BaselineModel):
    """CustomDiffAE baseline (conditional_autoencoder_0_1b.py: class CustomDiffAE)。

    forward(x, time=None, noise_level=0.0) -> (recon [B, 1, 200, 200], z, mean, log_var)
    权重文件: custom_diffae_model.pth (state_dict)
    """

    def load(self, weights_path: str, device: str, **kwargs):
        from sFWI.experiments._baseline_models import CustomDiffAE
        self.model = CustomDiffAE(
            input_shape=(1, 100, 300),
            output_shape=(200, 200),
            latent_dim=256,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            time_emb_dim=256,
            use_attention=True,
        )
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon, _z, _mean, _log_var = self.model(
                seismic_input.to(self.device), time=None, noise_level=0.0
            )
        return self._normalize_output(recon)


@register_baseline('inversionnet')
class InversionNetBaseline(BaselineModel):
    """OpenFWI InversionNet baseline（sFWI 输入输出尺寸适配版）。"""

    def load(self, weights_path: str, device: str, **kwargs):
        from sFWI.models.inversionnet import (
            InversionNetSFWI,
            load_inversionnet_state_dict,
        )

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        init_kwargs = dict(kwargs.get('init_kwargs', {}))
        strict = bool(kwargs.get('strict', False))

        # 若配置中未显式给出输入输出尺寸，则优先从 checkpoint 中恢复。
        if isinstance(checkpoint, dict):
            ckpt_input = checkpoint.get('input_shape')
            ckpt_output = checkpoint.get('output_shape')
            if ('input_shape' not in init_kwargs) and isinstance(ckpt_input, (list, tuple)):
                if len(ckpt_input) == 2:
                    init_kwargs['input_shape'] = (1, int(ckpt_input[0]), int(ckpt_input[1]))
                elif len(ckpt_input) == 3:
                    init_kwargs['input_shape'] = tuple(int(v) for v in ckpt_input)
            if ('output_shape' not in init_kwargs) and isinstance(ckpt_output, (list, tuple)):
                if len(ckpt_output) == 2:
                    init_kwargs['output_shape'] = tuple(int(v) for v in ckpt_output)
                elif len(ckpt_output) == 3:
                    init_kwargs['output_shape'] = tuple(int(v) for v in ckpt_output[-2:])

        self.model = InversionNetSFWI(**init_kwargs)
        incompatible = load_inversionnet_state_dict(
            self.model, checkpoint, strict=strict
        )

        # strict=False 时保留兼容加载信息，便于排查旧版权重字段。
        if not strict and (
            incompatible.missing_keys or incompatible.unexpected_keys
        ):
            print(
                "[WARNING] InversionNet 权重非严格匹配: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        self.model.to(device)
        self.model.eval()
        self.device = device
        self.input_shape = tuple(init_kwargs.get('input_shape', (1, 100, 300)))

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = seismic_input.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if len(self.input_shape) == 3:
                target_hw = tuple(self.input_shape[-2:])
                if tuple(x.shape[-2:]) != target_hw:
                    x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
            output = self.model(x)
        return self._normalize_output(output)


@register_baseline('generic_encoder')
class GenericEncoderBaseline(BaselineModel):
    """通用 encoder-decoder baseline。

    适用于未来新增的 baseline 模型。
    JSON config 中需指定 'module' 和 'class' 字段:
      {"name": "NewModel", "type": "generic_encoder",
       "weights": "/path/to/weights.pth",
       "module": "path.to.module", "class": "ModelClassName"}
    """

    def load(self, weights_path: str, device: str, **kwargs):
        import importlib
        module_path = kwargs.get('module')
        class_name = kwargs.get('class_name')
        if not module_path or not class_name:
            raise ValueError(
                "generic_encoder 需要 'module' 和 'class' 参数。"
                "请在 JSON config 中指定。"
            )
        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, class_name)
        init_kwargs = kwargs.get('init_kwargs', {})
        self.model = model_cls(**init_kwargs)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        # forward 返回值处理: 默认取第一个元素 (如果是 tuple)
        self._returns_tuple = kwargs.get('returns_tuple', False)

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(seismic_input.to(self.device))
        if self._returns_tuple and isinstance(output, tuple):
            output = output[0]
        return self._normalize_output(output)


@register_baseline('precomputed')
class PrecomputedBaseline(BaselineModel):
    """预计算结果 baseline: 直接从 .pt 文件加载已有的推理结果。

    适用于: 已经跑完推理、只需加载结果做指标对比的场景。
    文件格式: torch.save(results_tensor, path)  # [N, 1, H, W]
    """

    def load(self, weights_path: str, device: str, **kwargs):
        self.results = torch.load(weights_path, map_location=device)
        if self.results.dim() == 3:
            self.results = self.results.unsqueeze(1)
        self.device = device
        self._idx = 0

    def predict(self, seismic_input: torch.Tensor) -> torch.Tensor:
        batch_size = seismic_input.shape[0]
        out = self.results[self._idx: self._idx + batch_size]
        self._idx += batch_size
        return out.to(self.device)


def load_baseline(type_name: str, weights_path: str, device: str,
                  **kwargs) -> BaselineModel:
    """工厂函数: 根据 type_name 创建并加载 baseline 模型。"""
    if type_name not in BASELINE_REGISTRY:
        raise ValueError(
            f"未知 baseline 类型 '{type_name}'。"
            f"已注册: {list(BASELINE_REGISTRY.keys())}"
        )
    model = BASELINE_REGISTRY[type_name]()
    model.load(weights_path, device, **kwargs)
    return model


def load_baselines_from_config(config_path: str, device: str) -> OrderedDict:
    """从 JSON 配置文件批量加载 baselines。

    JSON 格式:
    {
      "baselines": [
        {"name": "AE",       "type": "autoencoder",  "weights": "/path/to/ae_seismic.pth"},
        {"name": "U-Net",    "type": "unet",         "weights": "/path/to/ModernUNET_seismic.pth"},
        {"name": "VAE",      "type": "vae",          "weights": "/path/to/VAE_seismic.pth"},
        {"name": "DiffAE",   "type": "diffae",       "weights": "/path/to/custom_diffae_model.pth"},
        {"name": "InversionNet", "type": "inversionnet", "weights": "/path/to/inversionnet.pth",
         "init_kwargs": {"input_shape": [1, 100, 300], "output_shape": [200, 200]},
         "strict": false},
        {"name": "ClassicFWI", "type": "precomputed", "weights": "/path/to/fwi_results.pt"},
        {"name": "NewModel", "type": "generic_encoder",
         "weights": "/path/to/new.pth",
         "module": "my_module", "class": "NewModelClass",
         "returns_tuple": true, "init_kwargs": {}}
      ]
    }
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    baselines = OrderedDict()
    for entry in cfg['baselines']:
        name = entry['name']
        type_name = entry['type']
        weights = entry['weights']
        # 提取额外参数 (module, class, init_kwargs 等)
        extra = {k: v for k, v in entry.items()
                 if k not in ('name', 'type', 'weights')}
        # JSON 中 'class' 是保留字, 映射为 'class_name'
        if 'class' in extra:
            extra['class_name'] = extra.pop('class')
        print(f"  加载 baseline: {name} ({type_name}) <- {weights}")
        baselines[name] = load_baseline(type_name, weights, device, **extra)
    return baselines


# ================================================================
#  Section 3: 公共初始化
# ================================================================

def _load_eval_dataset(cfg, image_size=200, eval_patches_path=None):
    """加载评估域数据。

    默认:
      - 读取 SEAM 原始切片（224 个）
    可选:
      - 通过 --eval_patches_path 覆盖评估集，支持:
        1) .pkl (Marmousi 风格): 自动提取 velocity
        2) 直接 Tensor: [N,H,W] 或 [N,1,H,W]
        3) dict: 包含 test_v_patches / velocity_patches / patches / v_torch
           若这些键不存在，会自动尝试第一个形状合法的 Tensor 字段
    """
    if eval_patches_path:
        if not os.path.isfile(eval_patches_path):
            raise FileNotFoundError(f"评估集文件不存在: {eval_patches_path}")

        ext = os.path.splitext(eval_patches_path)[1].lower()
        if ext == '.pkl':
            v_torch, _ = load_marmousi_from_pkl(eval_patches_path)
        else:
            blob = torch.load(eval_patches_path, weights_only=False)
            if isinstance(blob, torch.Tensor):
                v_torch = blob
            elif isinstance(blob, dict):
                candidate_keys = [
                    'test_v_patches',
                    'velocity_patches',
                    'patches',
                    'v_torch',
                ]
                v_torch = None
                for key in candidate_keys:
                    if key in blob and isinstance(blob[key], torch.Tensor):
                        v_torch = blob[key]
                        break
                if v_torch is None:
                    for _k, _v in blob.items():
                        if isinstance(_v, torch.Tensor) and _v.dim() in (3, 4):
                            v_torch = _v
                            break
                if v_torch is None:
                    raise KeyError(
                        f"{eval_patches_path} 未找到可用速度张量字段。"
                        f"keys={list(blob.keys())}"
                    )
            else:
                raise TypeError(
                    f"不支持的评估集文件类型: {type(blob)}, file={eval_patches_path}"
                )

        if v_torch.dim() == 4 and v_torch.shape[1] == 1:
            v_torch = v_torch[:, 0]
        if v_torch.dim() != 3:
            raise ValueError(
                f"评估集张量形状需为 [N,H,W] 或 [N,1,H,W]，当前: {tuple(v_torch.shape)}"
            )
        v_torch = v_torch.detach().cpu().float()
        data = create_velocity_dataset(v_torch, image_size=image_size)
        source_desc = f"external:{eval_patches_path}"
        return data, v_torch, source_desc

    v_torch = load_seam_model(cfg.paths.seam_model_path)
    if v_torch.dim() == 4 and v_torch.shape[1] == 1:
        v_torch = v_torch[:, 0]
    v_torch = v_torch.detach().cpu().float()
    data = create_velocity_dataset(v_torch, image_size=image_size)
    source_desc = f"seam_default:{cfg.paths.seam_model_path}"
    return data, v_torch, source_desc


def setup_environment(args):
    """初始化配置、模型、数据、算子，返回统一的上下文 dict。"""
    cfg = FWIConfig()
    cfg.daps.batch_size = 1
    cfg.daps.sigma = args.sigma

    code_dir = cfg.paths.code_dir
    config, sde = create_sde_config(code_dir, batch_size=cfg.daps.batch_size)
    base_config, lgvd_config = build_daps_configs(cfg)

    # 数据（默认 SEAM；可由 --eval_patches_path 覆盖）
    data, v_torch_eval, dataset_source = _load_eval_dataset(
        cfg,
        image_size=cfg.image_size,
        eval_patches_path=args.eval_patches_path,
    )
    print(f"[setup] 数据集: {len(data)} 个样本")
    print(f"[setup] 数据来源: {dataset_source}")
    print(f"[setup] 速度张量形状: {tuple(v_torch_eval.shape)}")

    # Score 模型
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=cfg.paths.checkpoint_path,
    )
    model.set_device(cfg.device)
    print(f"[setup] Score 模型已加载 -> {cfg.device}")

    # 算子 & 评估器
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
    eval_fn = Wasserstein(operator)
    eval_us_fn = Wasserstein_us(operator)
    evaluator = Evaluator((eval_fn,))
    evaluator_us = Evaluator((eval_us_fn,))

    # 输出目录
    output_dir = os.path.join(cfg.paths.f_path, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    return {
        'cfg': cfg,
        'config': config,
        'data': data,
        'model': model,
        'operator': operator,
        'evaluator': evaluator,
        'evaluator_us': evaluator_us,
        'output_dir': output_dir,
        'dataset_source': dataset_source,
    }


# ================================================================
#  Section 4: Mode 1 — quantitative (定量指标计算)
# ================================================================

def _run_sfwi_sample(env, gt_seed, master_seed=0, n_candidates=50):
    """对单个 GT 执行 sFWI RSS (暴力随机搜索) 采样。

    流程:
      1. 并行生成 n_candidates 个候选后验样本
      2. 用 Wasserstein 距离 (无监督) 评估每个候选
      3. 选取 W-distance 最小的候选作为最终反演结果

    Args:
        env: 环境上下文 dict
        gt_seed: GT 样本索引
        master_seed: 主随机种子 (控制可复现性)
        n_candidates: 候选样本数量 (默认 50)

    Returns:
        (best_sample, gt): 最优候选 tensor [1, C, H, W] 和 GT tensor
    """
    data = env['data']
    model = env['model']
    operator = env['operator']
    cfg = env['cfg']
    device = cfg.device

    gt = data.get_data(1, 0, seed=gt_seed).to(device)
    measurement = operator(gt)

    # 设置可复现的随机状态
    iteration_seed = master_seed + gt_seed
    torch.manual_seed(iteration_seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(iteration_seed)

    # 并行生成 n_candidates 个样本, 同时获取 W-distances 和样本 batch
    w_distances, x0hat_batch = model.daps.sample(
        model=model,
        x_start=model.daps.get_start(gt),
        operator=operator,
        measurement=measurement,
        evaluator_us=env['evaluator_us'],
        seed=range(n_candidates),
        return_batch=True,
        record=False,
        verbose=False,
        gt=gt,
    )

    best_idx = torch.argmin(w_distances).item()
    best_sample = x0hat_batch[best_idx].unsqueeze(0)  # [1, C, H, W]

    print(f"  GT #{gt_seed}: RSS 从 {n_candidates} 个候选中选取 idx={best_idx}, "
          f"W-dist={w_distances[best_idx]:.4f} (min={w_distances.min():.4f}, "
          f"max={w_distances.max():.4f})")

    return best_sample, gt


def _compute_centroid_distances_from_sm(sm_info, measurement, operator, data, device):
    """计算 measurement 到各个 group centroid 的距离。

    优先使用 SM 资产中的 `d_centroids_2d`（快速）；
    若缺失则回退到在线正演 centroid patch（兼容旧资产）。
    """
    meas_flat = measurement.reshape(1, -1)

    if 'd_centroids_2d' in sm_info and isinstance(sm_info['d_centroids_2d'], torch.Tensor):
        d_centroids_2d = sm_info['d_centroids_2d'].to(device)
        return torch.norm(d_centroids_2d - meas_flat, dim=1)

    centroid_indices = sm_info.get('centroid_indices', None)
    if centroid_indices is None:
        raise KeyError("SM 缺少 centroid_indices，无法计算 centroid 距离。")
    if isinstance(centroid_indices, torch.Tensor):
        centroid_indices = centroid_indices.detach().cpu().tolist()
    elif isinstance(centroid_indices, np.ndarray):
        centroid_indices = centroid_indices.tolist()

    k = int(sm_info.get('k', len(centroid_indices)))
    centroid_distances = torch.zeros(k, device=device)
    for i in range(k):
        centroid_idx = int(centroid_indices[i])
        centroid_patch = data[centroid_idx].unsqueeze(0).to(device)
        with torch.no_grad():
            d_centroid = operator(centroid_patch)
            centroid_distances[i] = torch.norm(measurement - d_centroid)
    return centroid_distances


def _resolve_group_representative_indices(sm_info, sm, n_seeds, device):
    """解析每个 group 的代表 seed（与 group_topg 逻辑保持一致）。"""
    centroid_seed_indices = sm_info.get('centroid_seed_indices', None)
    group_members = sm_info.get('group_members', None)
    if isinstance(centroid_seed_indices, torch.Tensor):
        centroid_seed_indices = centroid_seed_indices.detach().cpu().tolist()
    elif isinstance(centroid_seed_indices, np.ndarray):
        centroid_seed_indices = centroid_seed_indices.tolist()
    if isinstance(group_members, torch.Tensor):
        group_members = group_members.detach().cpu().tolist()
    elif isinstance(group_members, np.ndarray):
        group_members = group_members.tolist()

    rep_seeds = []
    rep_groups = []
    seen = set()
    n_groups = int(sm.shape[0])

    def _rep_from_group_members(g_idx: int):
        if not isinstance(group_members, (list, tuple)):
            return None
        if g_idx >= len(group_members):
            return None
        members_raw = group_members[g_idx]
        if not isinstance(members_raw, (list, tuple)):
            return None
        members = [int(v) for v in members_raw if 0 <= int(v) < n_seeds]
        if len(members) == 0:
            return None
        mem_t = torch.tensor(members, device=device, dtype=torch.long)
        local = sm[g_idx, mem_t]
        best_pos = int(torch.argmin(local).item())
        return int(members[best_pos])

    for g in range(n_groups):
        seed = None
        if isinstance(centroid_seed_indices, (list, tuple)) and g < len(centroid_seed_indices):
            cand = int(centroid_seed_indices[g])
            if 0 <= cand < n_seeds:
                seed = cand
        if seed is None:
            seed = _rep_from_group_members(g)
        if seed is None:
            # 兼容旧资产：退化为该 group 行最优 seed
            seed = int(torch.argmin(sm[g]).item())
        if seed not in seen:
            seen.add(seed)
            rep_seeds.append(seed)
            rep_groups.append(int(g))

    if len(rep_seeds) == 0:
        raise RuntimeError("group_topg 未找到有效 group 代表 seed。")

    rep_indices = torch.tensor(rep_seeds, device=device, dtype=torch.long)
    rep_group_indices = torch.tensor(rep_groups, device=device, dtype=torch.long)
    return rep_indices, rep_group_indices


def _precompute_rep_d2d_by_operator(x0hat_batch, rep_indices, operator, eval_bs=16):
    """一次性预计算代表 seed 的正演数据并拉平。"""
    device = x0hat_batch.device
    rep_indices = rep_indices.to(device=device, dtype=torch.long)
    n_rep = int(rep_indices.numel())
    out = []
    eval_bs = max(1, int(eval_bs))
    with torch.no_grad():
        for s in range(0, n_rep, eval_bs):
            e = min(s + eval_bs, n_rep)
            cand_b = x0hat_batch[rep_indices[s:e]]
            d_b = operator(cand_b).reshape(e - s, -1).detach()
            out.append(d_b)
    return torch.cat(out, dim=0)


def _build_gss_topg_light_cache(sm_info, operator, device, eval_bs=16):
    """构建 gss_topg_light 缓存（代表 seed + 对应数据域向量）。"""
    sm = sm_info['similarity_matrix'].to(device)     # [k, n_seeds]
    x0hat_batch = sm_info['x0hat_batch'].to(device)  # [n_seeds, C, H, W]
    n_seeds = int(sm.shape[1])
    if n_seeds < 1:
        raise RuntimeError("SM 中 x0hat/seed 数为 0。")

    rep_indices, rep_group_indices = _resolve_group_representative_indices(
        sm_info=sm_info, sm=sm, n_seeds=n_seeds, device=device
    )

    # 强制与在线选择口径一致：始终用当前 operator 计算代表 seed 的数据域向量。
    source = 'operator'
    rep_d_2d = _precompute_rep_d2d_by_operator(
        x0hat_batch=x0hat_batch,
        rep_indices=rep_indices,
        operator=operator,
        eval_bs=eval_bs,
    ).detach()

    return {
        # 关键: 缓存 device 侧 SM 资产，避免每个 GT 重复 .to(device) 传输。
        'sm': sm.detach(),
        'x0hat_batch': x0hat_batch.detach(),
        'rep_indices': rep_indices.detach(),
        'rep_group_indices': rep_group_indices.detach(),
        'rep_d_2d': rep_d_2d,
        'source': source,
    }


def _select_x0_candidate_group_topg_light(
    sm_info,
    measurement,
    operator,
    data,
    group_top_g=20,
    light_cache=None,
    eval_bs=16,
    need_centroid_distance=False,
):
    """gss_topg_light：使用缓存并行评分，直接返回 x0hat。"""
    device = measurement.device
    x0hat_batch = None
    sm = None
    if isinstance(light_cache, dict):
        x0hat_batch = light_cache.get('x0hat_batch', None)
        sm = light_cache.get('sm', None)
    if not isinstance(x0hat_batch, torch.Tensor):
        x0hat_batch = sm_info['x0hat_batch'].to(device)  # [n_seeds, C, H, W]
    else:
        x0hat_batch = x0hat_batch.to(device)
    if not isinstance(sm, torch.Tensor):
        sm = sm_info['similarity_matrix'].to(device)
    else:
        sm = sm.to(device)

    if light_cache is None:
        light_cache = _build_gss_topg_light_cache(
            sm_info=sm_info,
            operator=operator,
            device=device,
            eval_bs=eval_bs,
        )

    rep_indices = light_cache['rep_indices'].to(device=device, dtype=torch.long)
    rep_group_indices = light_cache['rep_group_indices'].to(device=device, dtype=torch.long)
    rep_d_2d = light_cache['rep_d_2d'].to(device=device)

    meas_flat = measurement.reshape(1, -1)
    if rep_d_2d.shape[1] != meas_flat.shape[1]:
        # 兜底: 若缓存维度与当前 measurement 不一致，退回在线预计算一次。
        rep_d_2d = _precompute_rep_d2d_by_operator(
            x0hat_batch=x0hat_batch,
            rep_indices=rep_indices,
            operator=operator,
            eval_bs=eval_bs,
        ).detach()
        light_cache['rep_d_2d'] = rep_d_2d
        light_cache['source'] = 'operator_fallback'

    rep_candidate_distances = torch.norm(rep_d_2d - meas_flat, dim=1)
    n_groups = int(sm.shape[0])
    n_pick_groups = max(1, min(int(group_top_g), n_groups))
    order = torch.argsort(rep_candidate_distances)
    pick = order[:n_pick_groups]
    top_indices = rep_indices[pick]
    selected_group_indices = rep_group_indices[pick]
    candidate_distances = rep_candidate_distances[pick]

    best_local_idx = int(torch.argmin(candidate_distances).item())
    best_global_idx = int(top_indices[best_local_idx].item())
    best_group = int(selected_group_indices[best_local_idx].item())
    centroid_distance = float('nan')
    if need_centroid_distance:
        centroid_distances = _compute_centroid_distances_from_sm(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            data=data,
            device=device,
        )
        centroid_distance = float(centroid_distances[best_group].item())
    best_candidate = x0hat_batch[best_global_idx].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_global_idx,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'centroid_distance': centroid_distance,
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
        'selected_group_indices': selected_group_indices,
    }
    return best_candidate, meta


def _select_x0_candidate_from_sm(
    sm_info,
    measurement,
    operator,
    data,
    top_k=50,
    group_top_g=20,
    candidate_mode='gss_topk',
):
    """从 SM 中选取 x0 候选。

    candidate_mode:
      - gss_topk: 旧版 GSS，先选 best_group，再取组内 top-k seed。
      - group_topg: 新版 gss_topg，每组选代表 seed，按真实 m0 选 top-g group。
    """
    device = measurement.device
    sm = sm_info['similarity_matrix'].to(device)    # [k, n_seeds]
    x0hat_batch = sm_info['x0hat_batch'].to(device) # [n_seeds, C, H, W]
    n_seeds = int(sm.shape[1])
    if n_seeds < 1:
        raise RuntimeError("SM 中 x0hat/seed 数为 0。")

    centroid_distances = _compute_centroid_distances_from_sm(
        sm_info=sm_info,
        measurement=measurement,
        operator=operator,
        data=data,
        device=device,
    )
    best_group = int(torch.argmin(centroid_distances).item())
    centroid_distance = float(centroid_distances[best_group].item())

    meas_flat = measurement.reshape(1, -1)
    selected_group_indices = None
    candidate_distances = None

    if candidate_mode == 'group_topg':
        n_groups = int(sm.shape[0])
        n_pick_groups = max(1, min(int(group_top_g), n_groups))
        rep_indices, rep_group_indices = _resolve_group_representative_indices(
            sm_info=sm_info,
            sm=sm,
            n_seeds=n_seeds,
            device=device,
        )

        # 关键: 按真实物理误差 m0 排序（而非 centroid 近似距离）
        n_rep = int(rep_indices.numel())
        rep_candidate_distances = torch.empty(n_rep, device=device, dtype=measurement.dtype)
        eval_bs = 16
        with torch.no_grad():
            for s in range(0, n_rep, eval_bs):
                e = min(s + eval_bs, n_rep)
                cand_b = x0hat_batch[rep_indices[s:e]]
                d_b = operator(cand_b)
                meas_b = measurement.expand(e - s, -1, -1, -1)
                rep_candidate_distances[s:e] = torch.norm(
                    (d_b - meas_b).reshape(e - s, -1), dim=1
                )

        order = torch.argsort(rep_candidate_distances)
        pick = order[:n_pick_groups]
        top_indices = rep_indices[pick]
        selected_group_indices = rep_group_indices[pick]
        candidate_distances = rep_candidate_distances[pick]
    else:
        n_select = max(1, min(int(top_k), n_seeds))
        _, top_indices = torch.topk(sm[best_group], n_select, largest=False)

    if candidate_distances is None:
        if 'd_samples_2d' in sm_info and isinstance(sm_info['d_samples_2d'], torch.Tensor):
            d_samples_2d = sm_info['d_samples_2d'].to(device)
            cand_flat = d_samples_2d[top_indices]
        else:
            candidates = x0hat_batch[top_indices]
            n_curr = int(top_indices.shape[0])
            candidates_high = F.interpolate(
                candidates, size=(128, 128), mode='bilinear', align_corners=True
            )
            with torch.no_grad():
                d_candidates = operator(candidates_high)
            cand_flat = d_candidates.reshape(n_curr, -1)
        candidate_distances = torch.norm(cand_flat - meas_flat, dim=1)

    best_local_idx = int(torch.argmin(candidate_distances).item())
    best_global_idx = int(top_indices[best_local_idx].item())
    if candidate_mode == 'group_topg' and selected_group_indices is not None:
        best_group = int(selected_group_indices[best_local_idx].item())
        centroid_distance = float(centroid_distances[best_group].item())
    best_candidate = x0hat_batch[best_global_idx].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_global_idx,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'centroid_distance': centroid_distance,
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
        'selected_group_indices': selected_group_indices,
    }
    return best_candidate, meta


def _set_manual_seed(seed, device):
    """设置随机种子，确保退火采样可复现。"""
    if seed is None:
        return
    seed_i = int(seed)
    torch.manual_seed(seed_i)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed_i)


def _compute_data_misfit_l2(sample, measurement, operator):
    """计算单样本数据域 L2 失配: ||F(x)-y||_2。"""
    with torch.no_grad():
        d_pred = operator(sample)
    return float(torch.norm((d_pred - measurement).reshape(1, -1), dim=1).item())


def _refine_candidate_with_daps(env, x0_candidate, measurement, noise_seed=None):
    """对给定 x0 候选执行 DAPS-FWI 精细化，返回最终 x0hat。"""
    model = env['model']
    operator = env['operator']
    daps = model.daps
    annealing = daps.annealing_scheduler
    device = env['cfg'].device
    from DAPS.sampler import Scheduler, DiffusionSampler

    _set_manual_seed(noise_seed, device)

    xt = x0_candidate
    x0hat = x0_candidate
    for step in range(annealing.num_steps):
        sigma = annealing.sigma_steps[step]
        # 1) reverse diffusion
        diff_scheduler = Scheduler(**daps.diffusion_scheduler_config, sigma_max=sigma)
        diff_sampler = DiffusionSampler(diff_scheduler)
        x0hat = diff_sampler.sample(model, xt, SDE=False, verbose=False)
        # 2) physics-informed Langevin correction
        x0y = daps.lgvd.sample(
            x0hat, operator, measurement, sigma, step / annealing.num_steps
        )
        # 3) renoise to next annealing level
        xt = x0y + torch.randn_like(x0y) * annealing.sigma_steps[step + 1]
    return x0hat


def _collect_topg_branch_ensemble(
    env,
    gt_seed,
    sm_info,
    sampling_method='gss_topg',
    group_top_g=20,
    refine_top_b=None,
    branch_repeats=1,
    master_seed=0,
    n_candidates=50,
):
    """基于 gss_topg 选出的 top-g 分支，构建后验样本集合。

    返回:
      dict with keys:
        - gt: [1,C,H,W]
        - measurement: [1,1,Hd,Wd]
        - samples: [K,C,H,W]
        - misfits: [K]
        - weights_init_m0: [K]  # 初始分支 m0（重复给每个 repeat）
        - branch_ids: [K]       # 0...(B-1)
        - source_seed_ids: [K]  # x0hat_batch 中 seed 索引
        - source_group_ids: [K] # group 索引，若未知为 -1
        - branch_count: B
        - repeats: R
    """
    data = env['data']
    operator = env['operator']
    device = env['cfg'].device

    gt = data.get_data(1, 0, seed=gt_seed).to(device)
    measurement = operator(gt)

    # 1) 选取 top-g 分支候选
    is_light = sampling_method == 'gss_topg_light'
    if is_light:
        _best_candidate, meta = _select_x0_candidate_group_topg_light(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            data=data,
            group_top_g=group_top_g,
            light_cache=env.get('gss_topg_light_cache'),
            eval_bs=int(env.get('gss_topg_light_eval_bs', 16)),
            need_centroid_distance=False,
        )
        candidate_mode = 'group_topg'
    else:
        candidate_mode = 'group_topg' if sampling_method == 'gss_topg' else 'gss_topk'
        _best_candidate, meta = _select_x0_candidate_from_sm(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            data=data,
            top_k=n_candidates,
            group_top_g=group_top_g,
            candidate_mode=candidate_mode,
        )

    top_indices = meta.get('top_indices', None)
    top_candidate_distances = meta.get('top_candidate_distances', None)
    selected_group_indices = meta.get('selected_group_indices', None)
    if top_indices is None or top_candidate_distances is None:
        raise RuntimeError("TopG 分支信息缺失: top_indices/top_candidate_distances。")

    top_indices = top_indices.detach().to(device=device, dtype=torch.long)
    top_candidate_distances = top_candidate_distances.detach().to(
        device=device, dtype=torch.float32
    )
    if selected_group_indices is None:
        selected_group_indices = torch.full_like(top_indices, fill_value=-1)
    else:
        selected_group_indices = selected_group_indices.detach().to(
            device=device, dtype=torch.long
        )

    if is_light and isinstance(env.get('gss_topg_light_cache'), dict):
        x0hat_batch = env['gss_topg_light_cache'].get('x0hat_batch', None)
    else:
        x0hat_batch = None
    if not isinstance(x0hat_batch, torch.Tensor):
        x0hat_batch = sm_info['x0hat_batch'].to(device)  # [n_seeds,C,H,W]
    else:
        x0hat_batch = x0hat_batch.to(device)

    b_default = int(group_top_g)
    if refine_top_b is not None and int(refine_top_b) > 0:
        b_default = int(refine_top_b)
    branch_count = max(1, min(b_default, int(top_indices.numel())))
    repeats = max(1, int(branch_repeats))

    # 2) 每个分支做 repeats 次精细化（或 light 直出 x0hat）
    samples = []
    misfits = []
    weights_init_m0 = []
    branch_ids = []
    source_seed_ids = []
    source_group_ids = []

    for b in range(branch_count):
        seed_idx = int(top_indices[b].item())
        group_idx = int(selected_group_indices[b].item())
        m0 = float(top_candidate_distances[b].item())
        x0_candidate = x0hat_batch[seed_idx].unsqueeze(0)

        for r in range(repeats):
            noise_seed = int(master_seed) + int(gt_seed) * 100003 + b * 997 + r * 31
            if is_light:
                sample = x0_candidate
            else:
                sample = _refine_candidate_with_daps(
                    env=env,
                    x0_candidate=x0_candidate,
                    measurement=measurement,
                    noise_seed=noise_seed,
                )

            m = _compute_data_misfit_l2(sample, measurement, operator)
            samples.append(sample.squeeze(0).detach().cpu())
            misfits.append(m)
            weights_init_m0.append(m0)
            branch_ids.append(b)
            source_seed_ids.append(seed_idx)
            source_group_ids.append(group_idx)

    if len(samples) == 0:
        raise RuntimeError("TopG 分支采样失败：未生成任何样本。")

    return {
        'gt': gt.detach().cpu(),
        'measurement': measurement.detach().cpu(),
        'samples': torch.stack(samples, dim=0),  # [K,C,H,W]
        'misfits': np.asarray(misfits, dtype=np.float64),
        'weights_init_m0': np.asarray(weights_init_m0, dtype=np.float64),
        'branch_ids': np.asarray(branch_ids, dtype=np.int64),
        'source_seed_ids': np.asarray(source_seed_ids, dtype=np.int64),
        'source_group_ids': np.asarray(source_group_ids, dtype=np.int64),
        'branch_count': int(branch_count),
        'repeats': int(repeats),
        'candidate_mode': candidate_mode,
    }


def _run_sfwi_sample_gss(
    env,
    gt_seed,
    master_seed=0,
    n_candidates=50,
    sm_info=None,
    sampling_method='gss',
    group_top_g=20,
):
    """对单个 GT 执行 sFWI GSS 采样（支持 gss/gss_topg/gss_topg_light）。

    流程:
      1. 获取 GT 的观测数据 d_obs
      2. 在数据域直接匹配最优 group: i* = argmin_i ||d_obs - F(centroid_i)||
      3. 从相似度矩阵第 i* 行选取 top-k 最优 sample 索引
      4. 从预生成的 x0hat_batch 中取出候选，正演后与 d_obs 比较，选最优
      5. 对最优候选执行 DAPS-FWI 精细化 (退火 + 反向扩散 + Langevin dynamics)

    Args:
        env: 环境上下文 dict
        gt_seed: GT 样本索引
        master_seed: 主随机种子
        n_candidates: gss 模式下组内候选数量
        sm_info: dict, 包含:
            - similarity_matrix: Tensor [k, n_seeds]
            - centroid_indices: list[int]
            - k: int
            - x0hat_batch: Tensor [n_seeds, C, H, W] 预生成的无条件样本
        sampling_method: 'gss' / 'gss_topg' / 'gss_topg_light'
        group_top_g: gss_topg 模式下保留的 group 数

    Returns:
        (best_sample, gt): 最优候选 tensor [1, C, H, W] 和 GT tensor
    """
    data = env['data']
    operator = env['operator']
    cfg = env['cfg']
    device = cfg.device

    gt = data.get_data(1, 0, seed=gt_seed).to(device)
    measurement = operator(gt)  # d_obs

    is_light = sampling_method == 'gss_topg_light'
    if is_light:
        best_candidate, meta = _select_x0_candidate_group_topg_light(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            data=data,
            group_top_g=group_top_g,
            light_cache=env.get('gss_topg_light_cache'),
            eval_bs=int(env.get('gss_topg_light_eval_bs', 16)),
            need_centroid_distance=False,
        )
        candidate_mode = 'group_topg'
    else:
        candidate_mode = (
            'group_topg' if sampling_method == 'gss_topg' else 'gss_topk'
        )
        best_candidate, meta = _select_x0_candidate_from_sm(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            data=data,
            top_k=n_candidates,
            group_top_g=group_top_g,
            candidate_mode=candidate_mode,
        )

    best_group = int(meta['best_group'])
    best_seed = int(meta['best_seed'])
    best_dist = float(meta['candidate_distance'])

    if not is_light:
        centroid_distance = float(meta.get('centroid_distance', float('nan')))
        centroid_indices = sm_info.get('centroid_indices', [])
        centroid_idx = None
        if isinstance(centroid_indices, torch.Tensor):
            centroid_indices = centroid_indices.detach().cpu().tolist()
        elif isinstance(centroid_indices, np.ndarray):
            centroid_indices = centroid_indices.tolist()
        if isinstance(centroid_indices, (list, tuple)) and 0 <= best_group < len(centroid_indices):
            centroid_idx = int(centroid_indices[best_group])

        if centroid_idx is not None:
            print(
                f"  GT #{gt_seed}: Direct Match -> Group {best_group} "
                f"(centroid=patch[{centroid_idx}], dist={centroid_distance:.4f})"
            )
        else:
            print(
                f"  GT #{gt_seed}: Direct Match -> Group {best_group} "
                f"(dist={centroid_distance:.4f})"
            )

        if candidate_mode == 'group_topg' and meta.get('selected_group_indices') is not None:
            g_list = [int(v) for v in meta['selected_group_indices'].detach().cpu().tolist()]
            show = g_list[:8]
            suffix = "..." if len(g_list) > 8 else ""
            print(
                f"  GT #{gt_seed}: GSS-TopG 选组={show}{suffix}, "
                f"best_seed={best_seed}, m0={best_dist:.4f}"
            )
        else:
            n_select = int(meta['top_indices'].numel())
            print(
                f"  GT #{gt_seed}: GSS 从 Group {best_group} 的 {n_select} 个候选中"
                f"选取 sample={best_seed}, dist={best_dist:.4f}"
            )

    # 轻量模式: 直接返回选中的 x0hat，不做 DAPS-FWI 精细化。
    if is_light:
        return best_candidate, gt

    # --- Step 4: DAPS-FWI 精细化 ---
    best_sample = _refine_candidate_with_daps(
        env=env,
        x0_candidate=best_candidate,
        measurement=measurement,
        noise_seed=(master_seed + gt_seed),
    )
    n_steps = int(env['model'].daps.annealing_scheduler.num_steps)
    print(f"  GT #{gt_seed}: DAPS-FWI 精细化完成 ({n_steps} steps)")

    return best_sample, gt


def _run_agnostic_sample(env, gt_seed, sample_seed=0):
    """对单个 GT 执行一次 physics-agnostic 采样 (纯 score 先验, 无 Langevin dynamics)。

    与 sFWI 的区别: 仅通过 PC sampler 做一次完整的 reverse diffusion,
    不进入退火循环, 不执行 Langevin dynamics 物理约束校正。
    对应源码 DAPS.sample() 的 batch 采样逻辑。
    """
    data = env['data']
    model = env['model']
    cfg = env['cfg']
    device = cfg.device

    gt = data.get_data(1, 0, seed=gt_seed).to(device)
    measurement = env['operator'](gt)

    torch.manual_seed(sample_seed)
    # DAPS.sample (batch版) 仅执行 PC sampler, 不含 Langevin dynamics
    w_distances, x0hat_batch = model.daps.sample(
        model,
        model.daps.get_start(gt),
        env['operator'],
        measurement,
        env['evaluator_us'],
        record=False,
        verbose=False,
        seed=[sample_seed],
        return_batch=True,
    )
    # x0hat_batch: [1, C, H, W] at score model native resolution
    # 上采样到与 GT 相同的分辨率
    agn_sample = F.interpolate(
        x0hat_batch,
        size=(gt.shape[-2], gt.shape[-1]),
        mode='bilinear',
        align_corners=True,
    )
    return agn_sample, gt


def run_quantitative(env, args):
    """Mode 1: 对指定 GT 列表计算 sFWI + baselines 的定量指标。

    输出:
      - 每个方法在每个 GT 上的 NRMSE / SSIM / PSNR
      - 每个方法的推理耗时
      - 汇总 CSV 文件
      - 带指标标注的对比可视化图
    """
    print("\n" + "=" * 60)
    print("Mode: quantitative — 定量指标计算")
    print("=" * 60)

    gt_indices = _parse_gt_indices(args)
    output_dir = env['output_dir']
    data = env['data']
    device = env['cfg'].device

    # --- 加载 baselines ---
    baselines = OrderedDict()
    if args.baseline_config and os.path.isfile(args.baseline_config):
        baselines = load_baselines_from_config(args.baseline_config, device)
    print(f"Baselines 数量: {len(baselines)}")

    # --- 收集结果 ---
    # results[method_name][gt_idx] = {'NRMSE': ..., 'SSIM': ..., 'PSNR': ...}
    results = OrderedDict()
    results['sFWI'] = {}

    for bl_name in baselines:
        results[bl_name] = {}

    # 缓存 prediction tensors 用于可视化
    # predictions[method_name][gt_idx] = tensor (cpu, squeezed to 2D)
    predictions = OrderedDict()
    predictions['sFWI'] = {}
    for bl_name in baselines:
        predictions[bl_name] = {}

    # 缓存 GT tensors
    gt_cache = {}

    # 计时统计: timing[method_name] = [elapsed_seconds_per_gt, ...]
    timing = OrderedDict()
    timing['sFWI'] = []
    for bl_name in baselines:
        timing[bl_name] = []
    zero_range_gt_count = 0
    zero_range_gt_examples = []

    for gt_idx in tqdm(gt_indices, desc="quantitative"):
        gt = data.get_data(1, 0, seed=gt_idx).to(device)
        measurement = env['operator'](gt)
        gt_cache[gt_idx] = gt.squeeze().detach().cpu()
        gt_range = float((gt.max() - gt.min()).item())
        if gt_range < 1e-8:
            zero_range_gt_count += 1
            if len(zero_range_gt_examples) < 10:
                zero_range_gt_examples.append(int(gt_idx))

        # sFWI 采样 (带计时)
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        if args.sampling_method in ('gss', 'gss_topg', 'gss_topg_light') and 'sm_info' in env:
            sfwi_pred, _ = _run_sfwi_sample_gss(
                env, gt_idx, master_seed=args.master_seed,
                n_candidates=args.n_candidates,
                sm_info=env['sm_info'],
                sampling_method=args.sampling_method,
                group_top_g=args.group_top_g,
            )
        else:
            sfwi_pred, _ = _run_sfwi_sample(
                env, gt_idx, master_seed=args.master_seed,
                n_candidates=args.n_candidates)
        torch.cuda.synchronize() if device == 'cuda' else None
        timing['sFWI'].append(time.time() - t0)

        if isinstance(sfwi_pred, torch.Tensor):
            results['sFWI'][gt_idx] = compute_all_metrics(sfwi_pred, gt)
            predictions['sFWI'][gt_idx] = sfwi_pred.squeeze().detach().cpu()

        # Baselines (带计时)
        for bl_name, bl_model in baselines.items():
            torch.cuda.synchronize() if device == 'cuda' else None
            t0 = time.time()
            bl_pred = bl_model.predict(measurement)
            torch.cuda.synchronize() if device == 'cuda' else None
            timing[bl_name].append(time.time() - t0)

            gt_for_metrics = gt if gt.dim() == 4 else gt.unsqueeze(1)
            bl_pred_norm = bl_pred if bl_pred.dim() == 4 else bl_pred.unsqueeze(1)
            results[bl_name][gt_idx] = compute_all_metrics(bl_pred_norm, gt_for_metrics)
            predictions[bl_name][gt_idx] = bl_pred.squeeze().detach().cpu()

    # --- 输出汇总 ---
    _print_metrics_summary(results, gt_indices, timing)
    if zero_range_gt_count > 0:
        show = zero_range_gt_examples
        suffix = "..." if zero_range_gt_count > len(show) else ""
        print(
            "[quantitative][warn] 检测到 GT 动态范围接近 0，"
            f"NRMSE 会记为 inf。count={zero_range_gt_count}, "
            f"examples={show}{suffix}"
        )
    csv_path = os.path.join(output_dir, generate_timestamped_filename('quantitative', '.csv'))
    _save_metrics_csv(results, gt_indices, csv_path, timing)

    # --- 可视化 ---
    fig_path = os.path.join(output_dir, generate_timestamped_filename('quantitative_comparison', '.pdf'))
    _plot_quantitative_comparison(results, gt_indices, gt_cache, predictions, fig_path)

    print(f"\n[quantitative] 完成。结果保存至 {output_dir}")


def _parse_gt_indices(args):
    """从 args.gt_indices 解析 GT 索引列表。"""
    if args.gt_indices:
        return args.gt_indices
    # 默认: 使用 5 个均匀分布的索引
    return list(range(0, 50, 10))


def _print_metrics_summary(results, gt_indices, timing=None):
    """终端打印指标汇总表。"""
    methods = list(results.keys())
    metrics = ['NRMSE', 'SSIM', 'PSNR']

    # 表头
    header = f"{'Method':<20}"
    for m in metrics:
        header += f"  {m:>10}"
    if timing:
        header += f"  {'Time(s)':>10}"
    print("\n" + header)
    print("-" * len(header))

    dropped = {}
    for method in methods:
        vals = results[method]
        if not vals:
            continue
        row = f"{method:<20}"
        for m in metrics:
            arr_raw = [vals[gi][m] for gi in gt_indices if gi in vals]
            arr = [float(v) for v in arr_raw if np.isfinite(float(v))]
            drop_n = len(arr_raw) - len(arr)
            if drop_n > 0:
                dropped[(method, m)] = drop_n
            if arr:
                mean_v = np.mean(arr)
                row += f"  {mean_v:>10.4f}"
            else:
                row += f"  {'N/A':>10}"
        if timing and method in timing and timing[method]:
            mean_t = np.mean(timing[method])
            row += f"  {mean_t:>10.2f}"
        print(row)
    if dropped:
        print("[summary][warn] 均值已忽略非有限值（inf/nan）:")
        for method in methods:
            for m in metrics:
                n = dropped.get((method, m), 0)
                if n > 0:
                    print(f"  {method} {m}: dropped={n}")
    print()


def _save_metrics_csv(results, gt_indices, csv_path, timing=None):
    """将指标保存为 CSV 文件。"""
    methods = list(results.keys())
    metrics = ['NRMSE', 'SSIM', 'PSNR']

    lines = []
    # 表头
    header_parts = ['method', 'gt_index'] + metrics + ['time_s']
    lines.append(','.join(header_parts))

    for method in methods:
        for i, gi in enumerate(gt_indices):
            if gi not in results[method]:
                continue
            vals = results[method][gi]
            row = [method, str(gi)]
            for m in metrics:
                row.append(f"{vals[m]:.6f}")
            # 每个 GT 对应的推理耗时
            if timing and method in timing and i < len(timing[method]):
                row.append(f"{timing[method][i]:.4f}")
            else:
                row.append('N/A')
            lines.append(','.join(row))

    # 追加均值行
    for method in methods:
        vals = results[method]
        if not vals:
            continue
        row = [method, 'mean']
        for m in metrics:
            arr_raw = [vals[gi][m] for gi in gt_indices if gi in vals]
            arr = [float(v) for v in arr_raw if np.isfinite(float(v))]
            row.append(f"{np.mean(arr):.6f}" if arr else 'N/A')
        if timing and method in timing and timing[method]:
            row.append(f"{np.mean(timing[method]):.4f}")
        else:
            row.append('N/A')
        lines.append(','.join(row))

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[quantitative] CSV 已保存: {csv_path}")


def _plot_quantitative_comparison(results, gt_indices, gt_cache, predictions, save_path):
    """生成论文级对比可视化: 每行一个 GT, 列为各方法, 子图下方标注 NRMSE/SSIM。

    Args:
        results: OrderedDict, results[method][gt_idx] = {'NRMSE':..., 'SSIM':..., 'PSNR':...}
        gt_indices: list of GT indices
        gt_cache: dict, gt_cache[gt_idx] = 2D cpu tensor
        predictions: OrderedDict, predictions[method][gt_idx] = 2D cpu tensor
        save_path: output PDF path
    """
    methods = list(results.keys())
    n_gt = len(gt_indices)
    n_methods = len(methods) + 1  # +1 for GT column

    fig, axes = plt.subplots(n_gt, n_methods, figsize=(3 * n_methods, 3 * n_gt))
    if n_gt == 1:
        axes = axes[np.newaxis, :]

    # 确定全局 colorbar 范围 (基于所有 GT)
    all_gt_vals = [gt_cache[gi].numpy() for gi in gt_indices if gi in gt_cache]
    if all_gt_vals:
        vmin = min(v.min() for v in all_gt_vals)
        vmax = max(v.max() for v in all_gt_vals)
    else:
        vmin, vmax = 0, 1

    for row, gi in enumerate(gt_indices):
        gt_np = gt_cache[gi].numpy() if gi in gt_cache else np.zeros((200, 200))

        # GT 列
        ax = axes[row, 0]
        im = ax.imshow(gt_np, cmap=VELOCITY_CMAP, aspect='auto', vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'GT #{gi}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # 各方法列
        for col, method in enumerate(methods):
            ax = axes[row, col + 1]

            # 渲染真实 prediction
            if gi in predictions.get(method, {}):
                pred_np = predictions[method][gi].numpy()
                ax.imshow(pred_np, cmap=VELOCITY_CMAP, aspect='auto', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.full_like(gt_np, np.nan), cmap='gray', aspect='auto')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')

            # 标注指标
            if gi in results.get(method, {}):
                nrmse_val = results[method][gi]['NRMSE']
                ssim_val = results[method][gi]['SSIM']
                psnr_val = results[method][gi]['PSNR']
                ax.set_xlabel(
                    f'NRMSE={nrmse_val:.3f}  SSIM={ssim_val:.3f}\nPSNR={psnr_val:.1f}dB',
                    fontsize=8,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method, fontsize=11, fontweight='bold')

    # 添加共享 colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Velocity (m/s)', fontsize=10)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[quantitative] 对比图已保存: {save_path}")


# ================================================================
#  Section 5: Mode 2 — uncertainty (不确定性量化)
# ================================================================

def _rankdata_simple(x):
    """简易 rank（不做 ties 平均），用于 Spearman 近似。"""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    rx = _rankdata_simple(x)
    ry = _rankdata_simple(y)
    return _safe_pearson(rx, ry)


def _risk_coverage_curve(uncertainty_flat, abs_error_flat, n_points=20):
    """像素级 risk-coverage 曲线（按不确定性从低到高扩展覆盖率）。"""
    u = np.asarray(uncertainty_flat, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error_flat, dtype=np.float64).reshape(-1)
    if u.size == 0 or e.size == 0 or u.size != e.size:
        return np.array([]), np.array([]), np.nan
    order = np.argsort(u)
    coverages = np.linspace(0.05, 1.0, int(max(2, n_points)))
    risks = []
    n = u.size
    for c in coverages:
        k = max(1, min(n, int(round(c * n))))
        idx = order[:k]
        risks.append(float(e[idx].mean()))
    risks = np.asarray(risks, dtype=np.float64)
    aurc = float(np.trapz(risks, coverages))
    return coverages, risks, aurc


def _misfit_to_weights(misfits, tau=-1.0):
    """根据 misfit 生成样本权重 w ∝ exp(-m/tau)。"""
    m = np.asarray(misfits, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return np.asarray([], dtype=np.float64), 1.0
    if tau is None or float(tau) <= 0:
        tau_eff = float(np.std(m))
        if tau_eff < 1e-12:
            tau_eff = float(np.mean(np.abs(m)))
        if tau_eff < 1e-12:
            tau_eff = 1.0
    else:
        tau_eff = float(tau)
    logits = -(m - m.min()) / max(tau_eff, 1e-12)
    logits = logits - logits.max()
    w = np.exp(logits)
    w_sum = float(w.sum())
    if w_sum < 1e-12:
        w = np.full_like(w, 1.0 / max(1, w.size))
    else:
        w = w / w_sum
    return w.astype(np.float64), tau_eff


def _weighted_posterior_stats(samples, misfits, branch_ids, branch_count, tau):
    """计算加权后验统计与方差分解。

    Args:
      samples: [K,C,H,W] (CPU tensor)
      misfits: [K] ndarray
      branch_ids: [K] ndarray，取值 0...(B-1)
      branch_count: B
      tau: 温度，<=0 时自动估计
    """
    if samples.dim() != 4:
        raise ValueError(f"samples 形状应为 [K,C,H,W]，当前: {tuple(samples.shape)}")

    K = int(samples.shape[0])
    if K < 1:
        raise RuntimeError("空样本集合，无法计算不确定性。")

    weights_np, tau_eff = _misfit_to_weights(misfits, tau=tau)
    if weights_np.size != K:
        raise RuntimeError(
            f"权重与样本数不一致: weights={weights_np.size}, K={K}"
        )

    device = samples.device
    dtype = samples.dtype
    weights = torch.tensor(weights_np, device=device, dtype=dtype)
    w4 = weights.view(K, 1, 1, 1)

    mean_map = (w4 * samples).sum(dim=0)
    var_total = (w4 * (samples - mean_map) ** 2).sum(dim=0)
    var_total = torch.clamp(var_total, min=0.0)

    var_between = torch.zeros_like(var_total)
    var_within = torch.zeros_like(var_total)

    branch_ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    if branch_ids.size != K:
        raise RuntimeError(
            f"branch_ids 与样本数不一致: branch_ids={branch_ids.size}, K={K}"
        )

    branch_weights = []
    for b in range(int(branch_count)):
        idx = np.where(branch_ids == b)[0]
        if idx.size == 0:
            branch_weights.append(0.0)
            continue
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        wb = weights[idx_t].sum()
        wb_f = float(wb.item())
        branch_weights.append(wb_f)
        if wb_f < 1e-12:
            continue
        w_local = weights[idx_t] / wb
        w_local4 = w_local.view(-1, 1, 1, 1)
        sb = samples[idx_t]
        mean_b = (w_local4 * sb).sum(dim=0)
        var_b = (w_local4 * (sb - mean_b) ** 2).sum(dim=0)
        var_within += wb * torch.clamp(var_b, min=0.0)
        var_between += wb * (mean_b - mean_map) ** 2

    var_between = torch.clamp(var_between, min=0.0)
    var_within = torch.clamp(var_within, min=0.0)

    eps = 1e-12
    entropy = float(-(weights_np * np.log(weights_np + eps)).sum())
    entropy_norm = float(entropy / np.log(max(2, K)))

    return {
        'weights': weights_np,
        'tau_eff': float(tau_eff),
        'mean_map': mean_map,
        'var_total': var_total,
        'var_between': var_between,
        'var_within': var_within,
        'std_total': torch.sqrt(var_total),
        'std_between': torch.sqrt(var_between),
        'std_within': torch.sqrt(var_within),
        'entropy': entropy,
        'entropy_norm': entropy_norm,
        'branch_weights': np.asarray(branch_weights, dtype=np.float64),
    }


def _save_uncertainty_casewise_csv(records, csv_path):
    fields = [
        'gt_idx',
        'branch_count',
        'branch_repeats',
        'sample_count',
        'tau_eff',
        'misfit_min',
        'misfit_mean',
        'misfit_std',
        'nrmse',
        'ssim',
        'psnr',
        'u_mean',
        'u_p90',
        'between_ratio',
        'entropy_norm',
        'pearson_u_err',
        'spearman_u_err',
        'aurc',
        'hcw_pixel_ratio',
        'hcw_case',
    ]
    lines = [','.join(fields)]
    for r in records:
        row = []
        for k in fields:
            v = r.get(k, 'N/A')
            if isinstance(v, (float, np.floating)):
                row.append(f"{float(v):.8f}")
            else:
                row.append(str(v))
        lines.append(','.join(row))
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[uncertainty] casewise CSV 已保存: {csv_path}")


def _plot_uncertainty_gallery(gallery_items, output_dir):
    """绘制章节配图: 每行一个案例, 列为 GT / Posterior Mean / Pixel Std。"""
    if not gallery_items:
        return None

    n = len(gallery_items)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 2.9 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, item in enumerate(gallery_items):
        gt_np = item['gt'].squeeze().numpy()
        mean_np = item['mean_map'].squeeze().numpy()
        std_np = item['std_total'].squeeze().numpy()

        vmin = float(min(gt_np.min(), mean_np.min()))
        vmax = float(max(gt_np.max(), mean_np.max()))

        panels = [
            ('Ground Truth', gt_np, VELOCITY_CMAP, vmin, vmax),
            ('Posterior Mean', mean_np, VELOCITY_CMAP, vmin, vmax),
            ('Pixel-wise Std', std_np, 'hot', None, None),
        ]
        for j, (title, arr, cmap, pmin, pmax) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(arr, cmap=cmap, aspect='auto', vmin=pmin, vmax=pmax)
            if i == 0:
                ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"GT #{item['gt_idx']}", fontsize=9)
            ax.set_xlabel(
                (
                    f"NRMSE={item['nrmse']:.3f}\n"
                    f"Umean={item['u_mean']:.4f}, HCW={item['hcw_pixel_ratio']:.3f}"
                ),
                fontsize=8,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Uncertainty Gallery (GT / Prediction / Pixel-wise Uncertainty)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, generate_timestamped_filename('uncertainty_gallery', '.pdf'))
    plt.savefig(path, dpi=240, bbox_inches='tight')
    plt.close()
    print(f"[uncertainty] gallery 图已保存: {path}")
    return path


def _save_uncertainty_latex_table(records, summary, latex_path):
    """生成 uncertainty 子章节可直接引用的 LaTeX 汇总表。"""
    if not records:
        return

    def _mean_std(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan'), float('nan')
        return float(arr.mean()), float(arr.std())

    nrmse_m, nrmse_s = _mean_std([r['nrmse'] for r in records])
    ssim_m, ssim_s = _mean_std([r['ssim'] for r in records])
    psnr_m, psnr_s = _mean_std([r['psnr'] for r in records])
    u_m, u_s = _mean_std([r['u_mean'] for r in records])
    br_m, br_s = _mean_std([r['between_ratio'] for r in records])
    hcw_pix_m, hcw_pix_s = _mean_std([r['hcw_pixel_ratio'] for r in records])

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Uncertainty quantification summary under TopG posterior branches.}')
    lines.append(r'\label{tab:uq_summary}')
    lines.append(r'\begin{tabular}{lc}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Item} & \textbf{Value} \\')
    lines.append(r'\hline')
    lines.append(f"Sampling method & {summary.get('sampling_method', 'N/A')} \\\\")
    lines.append(f"Cases (N) & {summary.get('n_cases', 'N/A')} \\\\")
    lines.append(
        f"TopG / B / R & "
        f"{summary.get('group_top_g', 'N/A')} / "
        f"{summary.get('refine_top_b', 'N/A')} / "
        f"{summary.get('branch_repeats', 'N/A')} \\\\"
    )
    lines.append(f"NRMSE & ${nrmse_m:.4f} \\pm {nrmse_s:.4f}$ \\\\")
    lines.append(f"SSIM & ${ssim_m:.4f} \\pm {ssim_s:.4f}$ \\\\")
    lines.append(f"PSNR (dB) & ${psnr_m:.4f} \\pm {psnr_s:.4f}$ \\\\")
    lines.append(f"Mean uncertainty $\\bar{{U}}$ & ${u_m:.4f} \\pm {u_s:.4f}$ \\\\")
    lines.append(f"Between/Total ratio & ${br_m:.4f} \\pm {br_s:.4f}$ \\\\")
    lines.append(f"HCW pixel ratio & ${hcw_pix_m:.4f} \\pm {hcw_pix_s:.4f}$ \\\\")
    lines.append(
        f"HCW case rate & {float(summary.get('hcw_case_rate', float('nan'))):.4f} \\\\"
    )
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[uncertainty] LaTeX 表已保存: {latex_path}")


def _parse_uncertainty_gt_indices(args, data_len):
    """解析 uncertainty 模式使用的 GT 索引集合。"""
    if args.gt_indices:
        valid = [int(v) for v in args.gt_indices if 0 <= int(v) < int(data_len)]
        return valid
    if args.uq_n_gt > 0:
        req = int(args.uq_n_gt)
        if int(data_len) < req:
            raise ValueError(
                f"uq_n_gt={req} 但当前评估集仅有 {int(data_len)} 个样本。"
                "请通过 --eval_patches_path 指向足够大的测试集（例如 n=1000）。"
            )
        return list(range(req))
    return _parse_gt_indices(args)


def _plot_uncertainty_panel(
    gt,
    mean_map,
    std_total,
    std_between,
    std_within,
    abs_error,
    gt_idx,
    output_dir,
    title_info,
):
    """绘制单 GT 六联图（含 total/between/within 像素级不确定性）。"""
    gt_np = gt.squeeze().numpy()
    mean_np = mean_map.squeeze().numpy()
    std_total_np = std_total.squeeze().numpy()
    std_between_np = std_between.squeeze().numpy()
    std_within_np = std_within.squeeze().numpy()
    err_np = abs_error.squeeze().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    vmin = float(min(gt_np.min(), mean_np.min()))
    vmax = float(max(gt_np.max(), mean_np.max()))

    panels = [
        ('Ground Truth', gt_np, VELOCITY_CMAP, vmin, vmax),
        ('Posterior Mean', mean_np, VELOCITY_CMAP, vmin, vmax),
        ('Std Total', std_total_np, 'hot', None, None),
        ('Std Between', std_between_np, 'hot', None, None),
        ('Std Within', std_within_np, 'hot', None, None),
        ('|Mean - GT|', err_np, 'hot', None, None),
    ]

    for ax, (title, arr, cmap, pmin, pmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, aspect='auto', vmin=pmin, vmax=pmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"GT #{gt_idx} | "
        f"NRMSE={title_info['nrmse']:.3f}, SSIM={title_info['ssim']:.3f}, "
        f"Umean={title_info['u_mean']:.4f}, HCWpix={title_info['hcw_pixel_ratio']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    fname = generate_timestamped_filename(f'uncertainty_gt{gt_idx}', '.pdf')
    plt.savefig(os.path.join(output_dir, fname), dpi=220, bbox_inches='tight')
    plt.close()


def run_uncertainty(env, args):
    """Mode 2: 基于 gss_topg_light 的分支后验不确定性量化（统一口径）。

    核心流程:
      1) 从 gss_topg_light 取 top-g 分支（默认 refine_top_b=group_top_g）
      2) 每个分支生成 1 个样本（light 模式下 repeats 自动限制为 1）
      3) 按样本 data-misfit 计算权重并融合后验
      4) 输出像素级 std_total/std_between/std_within + HCW 统计
    """
    print("\n" + "=" * 60)
    print("Mode: uncertainty — TopG 后验不确定性量化")
    print("=" * 60)

    if 'sm_info' not in env:
        print(
            "[ERROR] uncertainty 模式需要 gss 资产（--sm_path）。"
            "请使用 --sampling_method gss_topg 或 gss_topg_light。"
        )
        return

    dataset_source = str(env.get('dataset_source', 'unknown'))
    if args.uq_n_gt >= 1000 and not dataset_source.startswith('external:'):
        print(
            "[ERROR] 大样本 uncertainty (uq_n_gt>=1000) 必须使用外部测试集。"
            f"当前数据来源: {dataset_source}。请提供 --eval_patches_path。"
        )
        return

    try:
        gt_indices = _parse_uncertainty_gt_indices(args, len(env['data']))
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    output_dir = env['output_dir']
    sampling_for_uq = 'gss_topg_light'
    if args.sampling_method != 'gss_topg_light':
        print(
            f"[uncertainty][info] 为统一口径，uncertainty 模式固定使用 gss_topg_light "
            f"(忽略传入 sampling_method={args.sampling_method})。"
        )

    refine_top_b = int(args.refine_top_b) if int(args.refine_top_b) > 0 else int(args.group_top_g)
    branch_repeats = max(1, int(args.branch_repeats))
    if branch_repeats > 1:
        print(
            "[uncertainty][warn] gss_topg_light 不含 DAPS 随机精细化，"
            "branch_repeats>1 会产生重复样本；已自动设为 1。"
        )
        branch_repeats = 1
    print(
        f"GT 数量: {len(gt_indices)}, group_top_g={args.group_top_g}, "
        f"refine_top_b={refine_top_b}, branch_repeats={branch_repeats}"
    )

    records = []
    gallery_items = []
    gallery_limit = max(0, int(args.uq_gallery_cases))
    for gt_idx in tqdm(gt_indices, desc="uncertainty_topg"):
        bundle = _collect_topg_branch_ensemble(
            env=env,
            gt_seed=gt_idx,
            sm_info=env['sm_info'],
            sampling_method=sampling_for_uq,
            group_top_g=args.group_top_g,
            refine_top_b=refine_top_b,
            branch_repeats=branch_repeats,
            master_seed=args.master_seed,
            n_candidates=args.n_candidates,
        )

        samples = bundle['samples'].float()  # [K,C,H,W]
        gt_cpu = bundle['gt'].squeeze(0).float()  # [C,H,W]
        misfits = bundle['misfits']
        branch_ids = bundle['branch_ids']
        K = int(samples.shape[0])

        stats = _weighted_posterior_stats(
            samples=samples,
            misfits=misfits,
            branch_ids=branch_ids,
            branch_count=bundle['branch_count'],
            tau=args.uq_weight_tau,
        )
        mean_map = stats['mean_map']
        std_total = stats['std_total']
        std_between = stats['std_between']
        std_within = stats['std_within']

        abs_error = (mean_map - gt_cpu).abs()
        u_flat = std_total.flatten().numpy()
        err_flat = abs_error.flatten().numpy()

        pearson_u_err = _safe_pearson(u_flat, err_flat)
        spearman_u_err = _safe_spearman(u_flat, err_flat)
        _, _, aurc = _risk_coverage_curve(u_flat, err_flat, n_points=20)

        # 像素级 HCW: 低不确定性 + 高误差
        uq_q = float(args.hcw_uncertainty_q)
        err_q = float(args.hcw_error_q)
        u_thr = float(np.quantile(u_flat, uq_q))
        e_thr = float(np.quantile(err_flat, err_q))
        hcw_pixel_ratio = float(np.mean((u_flat <= u_thr) & (err_flat >= e_thr)))

        pred4 = mean_map.unsqueeze(0)
        gt4 = gt_cpu.unsqueeze(0)
        nrmse = compute_nrmse(pred4, gt4)
        ssim = compute_ssim(pred4, gt4)
        psnr = compute_psnr(pred4, gt4)

        var_total_mean = float(stats['var_total'].mean().item())
        var_between_mean = float(stats['var_between'].mean().item())
        between_ratio = float(var_between_mean / (var_total_mean + 1e-12))

        rec = {
            'gt_idx': int(gt_idx),
            'branch_count': int(bundle['branch_count']),
            'branch_repeats': int(bundle['repeats']),
            'sample_count': int(K),
            'tau_eff': float(stats['tau_eff']),
            'misfit_min': float(np.min(misfits)),
            'misfit_mean': float(np.mean(misfits)),
            'misfit_std': float(np.std(misfits)),
            'nrmse': float(nrmse),
            'ssim': float(ssim),
            'psnr': float(psnr),
            'u_mean': float(std_total.mean().item()),
            'u_p90': float(np.quantile(u_flat, 0.90)),
            'between_ratio': float(between_ratio),
            'entropy_norm': float(stats['entropy_norm']),
            'pearson_u_err': float(pearson_u_err),
            'spearman_u_err': float(spearman_u_err),
            'aurc': float(aurc),
            'hcw_pixel_ratio': float(hcw_pixel_ratio),
            'hcw_case': 0,  # 先占位，后续按全体分位线打标
        }
        records.append(rec)

        if args.save_uq_ensemble:
            pt_name = f"uncertainty_case_gt{gt_idx}_B{bundle['branch_count']}_R{bundle['repeats']}.pt"
            torch.save(
                {
                    'gt_idx': int(gt_idx),
                    'samples': samples,
                    'weights': stats['weights'],
                    'misfits': misfits,
                    'branch_ids': branch_ids,
                    'source_seed_ids': bundle['source_seed_ids'],
                    'source_group_ids': bundle['source_group_ids'],
                    'mean_map': mean_map,
                    'std_total': std_total,
                    'std_between': std_between,
                    'std_within': std_within,
                    'abs_error': abs_error,
                },
                os.path.join(output_dir, pt_name),
            )

        need_plot = (not args.uq_disable_plots)
        if need_plot and (args.uq_plot_max_cases <= 0 or len(records) <= args.uq_plot_max_cases):
            _plot_uncertainty_panel(
                gt=gt_cpu,
                mean_map=mean_map,
                std_total=std_total,
                std_between=std_between,
                std_within=std_within,
                abs_error=abs_error,
                gt_idx=gt_idx,
                output_dir=output_dir,
                title_info=rec,
            )

        if len(gallery_items) < gallery_limit:
            gallery_items.append(
                {
                    'gt_idx': int(gt_idx),
                    'gt': gt_cpu.detach().cpu(),
                    'mean_map': mean_map.detach().cpu(),
                    'std_total': std_total.detach().cpu(),
                    'nrmse': float(nrmse),
                    'u_mean': float(rec['u_mean']),
                    'hcw_pixel_ratio': float(hcw_pixel_ratio),
                }
            )

        print(
            f"  GT #{gt_idx}: K={K}, NRMSE={nrmse:.4f}, "
            f"Umean={rec['u_mean']:.4f}, HCWpix={hcw_pixel_ratio:.4f}"
        )

    if len(records) == 0:
        print("[uncertainty] 无有效记录，结束。")
        return

    # case-level HCW: 低不确定性 + 高误差（按样本分位定义）
    u_arr = np.asarray([r['u_mean'] for r in records], dtype=np.float64)
    e_arr_raw = np.asarray([r['nrmse'] for r in records], dtype=np.float64)
    e_mask = np.isfinite(e_arr_raw)
    if e_mask.any():
        u_case_thr = float(np.quantile(u_arr, float(args.hcw_uncertainty_q)))
        e_case_thr = float(np.quantile(e_arr_raw[e_mask], float(args.hcw_error_q)))
        for r in records:
            r['hcw_case'] = int((r['u_mean'] <= u_case_thr) and (r['nrmse'] >= e_case_thr))
    else:
        u_case_thr = float('nan')
        e_case_thr = float('nan')

    # 保存
    csv_path = os.path.join(output_dir, generate_timestamped_filename('uncertainty_casewise', '.csv'))
    _save_uncertainty_casewise_csv(records, csv_path)

    def _mean_finite(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan')
        return float(arr.mean())

    summary = {
        'n_cases': int(len(records)),
        'sampling_method': sampling_for_uq,
        'group_top_g': int(args.group_top_g),
        'refine_top_b': int(refine_top_b),
        'branch_repeats': int(branch_repeats),
        'hcw_uncertainty_q': float(args.hcw_uncertainty_q),
        'hcw_error_q': float(args.hcw_error_q),
        'hcw_case_threshold_u': u_case_thr,
        'hcw_case_threshold_nrmse': e_case_thr,
        'mean_nrmse': _mean_finite([r['nrmse'] for r in records]),
        'mean_ssim': _mean_finite([r['ssim'] for r in records]),
        'mean_psnr': _mean_finite([r['psnr'] for r in records]),
        'mean_u': _mean_finite([r['u_mean'] for r in records]),
        'mean_between_ratio': _mean_finite([r['between_ratio'] for r in records]),
        'mean_entropy_norm': _mean_finite([r['entropy_norm'] for r in records]),
        'mean_pearson_u_err': _mean_finite([r['pearson_u_err'] for r in records]),
        'mean_spearman_u_err': _mean_finite([r['spearman_u_err'] for r in records]),
        'mean_hcw_pixel_ratio': _mean_finite([r['hcw_pixel_ratio'] for r in records]),
        'hcw_case_rate': _mean_finite([r['hcw_case'] for r in records]),
    }
    summary_path = os.path.join(output_dir, generate_timestamped_filename('uncertainty_summary', '.json'))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if gallery_limit > 0:
        _plot_uncertainty_gallery(gallery_items, output_dir)

    if not args.uq_skip_latex_table:
        latex_path = os.path.join(
            output_dir, generate_timestamped_filename('uncertainty_table', '.tex')
        )
        _save_uncertainty_latex_table(records, summary, latex_path)

    print(f"[uncertainty] summary JSON 已保存: {summary_path}")
    print(
        "[uncertainty] 汇总: "
        f"mean_u={summary['mean_u']:.4f}, "
        f"mean_between_ratio={summary['mean_between_ratio']:.4f}, "
        f"hcw_case_rate={summary['hcw_case_rate']:.4f}"
    )
    print(f"[uncertainty] 完成。结果保存至 {output_dir}")


# ================================================================
#  Section 6: Mode 3 — failure_analysis (失败模式分析)
# ================================================================

# Mode 分类阈值 (基于 NRMSE)
MODE_THRESHOLDS = {
    'III': 0.05,   # GT reconstruction:      NRMSE < 0.05
    'II':  0.15,   # Near-optimal:      0.05 <= NRMSE < 0.15
    'I':   0.35,   # Plausible sub-optimal: 0.15 <= NRMSE < 0.35
                   # Mode IV (failure):      NRMSE >= 0.35
}


def _classify_mode(nrmse: float) -> str:
    """根据 NRMSE 将结果分类为 Mode I-IV。"""
    if nrmse < MODE_THRESHOLDS['III']:
        return 'III'
    elif nrmse < MODE_THRESHOLDS['II']:
        return 'II'
    elif nrmse < MODE_THRESHOLDS['I']:
        return 'I'
    else:
        return 'IV'


def _parse_failure_gt_indices(args, data_len):
    """解析 failure_analysis 使用的 GT 索引集合。"""
    if args.gt_indices:
        valid = [int(v) for v in args.gt_indices if 0 <= int(v) < int(data_len)]
        if not valid:
            raise ValueError(
                f"--gt_indices 全部越界（data_len={int(data_len)}）。"
            )
        return valid

    req = int(args.n_gt)
    if req < 1:
        raise ValueError("--n_gt 必须 >= 1")
    if int(data_len) < req:
        raise ValueError(
            f"n_gt={req} 但当前评估集仅有 {int(data_len)} 个样本。"
            "请通过 --eval_patches_path 指向足够大的测试集（例如 n=1000）。"
        )
    return list(range(req))


def _save_failure_distribution_csv(records, skip_agnostic, csv_path):
    """保存 Mode 分布统计（用于论文表格）。"""
    modes = ['III', 'II', 'I', 'IV']
    lines = ['method,mode,count,pct,n_total']

    sfwi_modes = [r.get('sfwi_mode') for r in records if 'sfwi_mode' in r]
    total_s = len(sfwi_modes)
    for m in modes:
        c = sfwi_modes.count(m)
        pct = 100.0 * c / total_s if total_s > 0 else 0.0
        lines.append(f"sFWI,{m},{c},{pct:.6f},{total_s}")

    if not skip_agnostic:
        agn_modes = [r.get('agnostic_mode') for r in records if 'agnostic_mode' in r]
        total_a = len(agn_modes)
        for m in modes:
            c = agn_modes.count(m)
            pct = 100.0 * c / total_a if total_a > 0 else 0.0
            lines.append(f"agnostic,{m},{c},{pct:.6f},{total_a}")

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[failure_analysis] Mode 分布 CSV 已保存: {csv_path}")


def run_failure_analysis(env, args):
    """Mode 3: 大样本 failure mode 分布统计（统一 gss_topg_light 口径）。"""
    print("\n" + "=" * 60)
    print("Mode: failure_analysis — 失败模式分析")
    print("=" * 60)

    output_dir = env['output_dir']
    data = env['data']
    device = env['cfg'].device
    dataset_source = str(env.get('dataset_source', 'unknown'))

    try:
        gt_indices = _parse_failure_gt_indices(args, len(data))
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    if len(gt_indices) >= 1000 and not dataset_source.startswith('external:'):
        print(
            "[ERROR] 大样本 failure_analysis (N>=1000) 必须使用外部测试集。"
            f"当前数据来源: {dataset_source}。请提供 --eval_patches_path。"
        )
        return

    sampling_for_failure = 'gss_topg_light'
    if args.sampling_method != 'gss_topg_light':
        print(
            f"[failure_analysis][info] 为统一口径，failure_analysis 固定使用 gss_topg_light "
            f"(忽略传入 sampling_method={args.sampling_method})。"
        )

    print(
        f"GT 样本数: {len(gt_indices)}, 阈值: {MODE_THRESHOLDS}, "
        f"group_top_g={args.group_top_g}"
    )
    print(f"数据来源: {dataset_source}")

    run_agnostic = (not args.skip_agnostic)
    if run_agnostic and len(gt_indices) >= 200 and (not args.force_agnostic_large_scale):
        print(
            "[failure_analysis][info] 大样本模式默认跳过 physics-agnostic（该分支开销显著）。"
            "如需强制运行，请添加 --force_agnostic_large_scale。"
        )
        run_agnostic = False
    print(f"[failure_analysis] run_agnostic={run_agnostic}")

    # 收集: {gt_idx: {'sfwi': nrmse, 'agnostic': nrmse}}
    records = []

    for gt_idx in tqdm(gt_indices, desc="failure_analysis"):
        gt = data.get_data(1, 0, seed=gt_idx).to(device)

        entry = {'gt_idx': gt_idx}

        # --- sFWI GSS-TopG-Light 采样 ---
        sfwi_sample, _ = _run_sfwi_sample_gss(
            env, gt_idx, master_seed=args.master_seed,
            n_candidates=args.n_candidates,
            sm_info=env['sm_info'],
            sampling_method=sampling_for_failure,
            group_top_g=args.group_top_g,
        )
        if isinstance(sfwi_sample, torch.Tensor):
            sfwi_nrmse = float(compute_nrmse(sfwi_sample, gt))
            if not np.isfinite(sfwi_nrmse):
                sfwi_nrmse = float('inf')
            sfwi_ssim = float(compute_ssim(sfwi_sample, gt))
            if not np.isfinite(sfwi_ssim):
                sfwi_ssim = float('nan')
            entry['sfwi_nrmse'] = sfwi_nrmse
            entry['sfwi_ssim'] = sfwi_ssim
            entry['sfwi_mode'] = _classify_mode(entry['sfwi_nrmse'])

        # --- Physics-agnostic 采样 (纯 score 先验, 无 Langevin dynamics) ---
        # 仅通过 PC sampler 做 reverse diffusion, 不进入退火循环,
        # 不执行物理约束校正。对应 DAPS.sample() 的 batch 采样逻辑。
        if run_agnostic:
            sample_seed = int(args.master_seed + gt_idx)
            torch.manual_seed(sample_seed)
            agn_sample, _ = _run_agnostic_sample(env, gt_idx, sample_seed=sample_seed)
            if isinstance(agn_sample, torch.Tensor):
                agn_nrmse = float(compute_nrmse(agn_sample, gt))
                if not np.isfinite(agn_nrmse):
                    agn_nrmse = float('inf')
                agn_ssim = float(compute_ssim(agn_sample, gt))
                if not np.isfinite(agn_ssim):
                    agn_ssim = float('nan')
                entry['agnostic_nrmse'] = agn_nrmse
                entry['agnostic_ssim'] = agn_ssim
                entry['agnostic_mode'] = _classify_mode(entry['agnostic_nrmse'])

        records.append(entry)

    # --- 统计 ---
    _print_failure_statistics(records, (not run_agnostic))

    # --- 可视化 ---
    _plot_failure_distribution(records, (not run_agnostic), output_dir)
    _plot_failure_representatives(records, env, output_dir)

    # --- 保存原始数据 ---
    json_path = os.path.join(output_dir, generate_timestamped_filename('failure_analysis', '.json'))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[failure_analysis] 原始数据已保存: {json_path}")

    csv_path = os.path.join(output_dir, generate_timestamped_filename('failure_distribution', '.csv'))
    _save_failure_distribution_csv(records, (not run_agnostic), csv_path)

    print(f"[failure_analysis] 完成。结果保存至 {output_dir}")


def _print_failure_statistics(records, skip_agnostic):
    """终端打印 Mode 分布统计。"""
    modes = ['III', 'II', 'I', 'IV']
    mode_labels = {
        'III': 'Mode III (GT reconstruction)',
        'II':  'Mode II  (Near-optimal)',
        'I':   'Mode I   (Plausible sub-optimal)',
        'IV':  'Mode IV  (Convergence failure)',
    }

    def _print_nrmse_stats(tag, values):
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        n_total = int(arr.size)
        n_valid = int(np.sum(finite))
        n_invalid = n_total - n_valid
        if n_valid == 0:
            print(f"  {tag}: no finite values (invalid={n_invalid}/{n_total})")
            return
        vals = arr[finite]
        print(
            f"  {tag}: mean={np.mean(vals):.4f}, "
            f"median={np.median(vals):.4f}, std={np.std(vals):.4f}, "
            f"valid={n_valid}/{n_total}, invalid={n_invalid}"
        )

    # sFWI 统计
    sfwi_modes = [r.get('sfwi_mode') for r in records if 'sfwi_mode' in r]
    total = len(sfwi_modes)

    print(f"\n{'='*50}")
    print(f"sFWI Mode 分布 (N={total})")
    print(f"{'='*50}")
    for m in modes:
        count = sfwi_modes.count(m)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {mode_labels[m]}: {count:>4d} ({pct:5.1f}%)")

    sfwi_nrmses = [r['sfwi_nrmse'] for r in records if 'sfwi_nrmse' in r]
    if sfwi_nrmses:
        _print_nrmse_stats("NRMSE", sfwi_nrmses)

    # Physics-agnostic 统计
    if not skip_agnostic:
        agn_modes = [r.get('agnostic_mode') for r in records if 'agnostic_mode' in r]
        total_a = len(agn_modes)
        print(f"\n{'='*50}")
        print(f"Physics-agnostic Mode 分布 (N={total_a})")
        print(f"{'='*50}")
        for m in modes:
            count = agn_modes.count(m)
            pct = 100.0 * count / total_a if total_a > 0 else 0
            print(f"  {mode_labels[m]}: {count:>4d} ({pct:5.1f}%)")

        agn_nrmses = [r['agnostic_nrmse'] for r in records if 'agnostic_nrmse' in r]
        if agn_nrmses:
            _print_nrmse_stats("NRMSE", agn_nrmses)
    print()


def _plot_failure_distribution(records, skip_agnostic, output_dir):
    """绘制 Mode 分布柱状图 (sFWI vs physics-agnostic 并列对比)。"""
    modes = ['III', 'II', 'I', 'IV']
    mode_names = ['Mode III\n(GT recon.)', 'Mode II\n(Near-opt.)',
                  'Mode I\n(Plausible)', 'Mode IV\n(Failure)']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    sfwi_modes = [r.get('sfwi_mode') for r in records if 'sfwi_mode' in r]
    sfwi_counts = [sfwi_modes.count(m) for m in modes]
    total_s = len(sfwi_modes)

    n_groups = 2 if (not skip_agnostic) else 1
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    # sFWI 柱状图
    ax = axes[0]
    bars = ax.bar(mode_names, sfwi_counts, color=colors,
                  edgecolor='black', linewidth=0.5)
    for bar, cnt in zip(bars, sfwi_counts):
        pct = 100.0 * cnt / total_s if total_s > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_title(f'sFWI (N={total_s})', fontsize=12)
    ax.set_ylabel('Count', fontsize=10)

    # Physics-agnostic 柱状图
    if not skip_agnostic:
        agn_modes = [r.get('agnostic_mode') for r in records
                     if 'agnostic_mode' in r]
        agn_counts = [agn_modes.count(m) for m in modes]
        total_a = len(agn_modes)

        ax2 = axes[1]
        bars2 = ax2.bar(mode_names, agn_counts, color=colors,
                        edgecolor='black', linewidth=0.5)
        for bar, cnt in zip(bars2, agn_counts):
            pct = 100.0 * cnt / total_a if total_a > 0 else 0
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{cnt}\n({pct:.1f}%)',
                     ha='center', va='bottom', fontsize=9)
        ax2.set_title(f'Physics-agnostic (N={total_a})', fontsize=12)
        ax2.set_ylabel('Count', fontsize=10)

    plt.tight_layout()
    fname = generate_timestamped_filename('failure_distribution', '.pdf')
    plt.savefig(os.path.join(output_dir, fname),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[failure_analysis] 分布图已保存: {fname}")


def _plot_failure_representatives(records, env, output_dir):
    """为每个 Mode 选取一个代表性案例并可视化。"""
    modes = ['III', 'II', 'I', 'IV']
    data = env['data']
    device = env['cfg'].device

    # 按 Mode 分组, 取每组中 NRMSE 最接近中位数的样本
    representatives = {}
    for m in modes:
        group = [r for r in records
                 if r.get('sfwi_mode') == m and 'sfwi_nrmse' in r]
        if not group:
            continue
        nrmses = [r['sfwi_nrmse'] for r in group]
        median_val = np.median(nrmses)
        best = min(group, key=lambda r: abs(r['sfwi_nrmse'] - median_val))
        representatives[m] = best

    if not representatives:
        print("[failure_analysis] 无代表性案例可展示")
        return

    n_modes = len(representatives)
    fig, axes = plt.subplots(n_modes, 2, figsize=(8, 3.5 * n_modes))
    if n_modes == 1:
        axes = axes[np.newaxis, :]

    for row, m in enumerate(modes):
        if m not in representatives:
            continue
        rec = representatives[m]
        gt_idx = rec['gt_idx']
        gt = data.get_data(1, 0, seed=gt_idx).to(device)
        gt_np = gt.squeeze().detach().cpu().numpy()

        # GT
        ax_gt = axes[row, 0]
        ax_gt.imshow(gt_np, cmap=VELOCITY_CMAP, aspect='auto')
        ax_gt.set_title(f'GT #{gt_idx}', fontsize=9)
        ax_gt.set_ylabel(f'Mode {m}', fontsize=10, fontweight='bold')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

        # sFWI (占位 — 实际 pred 需缓存或重新采样)
        ax_pred = axes[row, 1]
        ax_pred.imshow(np.zeros_like(gt_np), cmap='gray',
                       aspect='auto', vmin=0, vmax=1)
        nrmse = rec['sfwi_nrmse']
        ssim_v = rec.get('sfwi_ssim', 0)
        ax_pred.set_title(
            f'sFWI  NRMSE={nrmse:.3f}  SSIM={ssim_v:.3f}',
            fontsize=9,
        )
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

    fig.suptitle('Representative Cases per Mode', fontsize=12)
    plt.tight_layout()
    fname = generate_timestamped_filename('failure_representatives', '.pdf')
    plt.savefig(os.path.join(output_dir, fname),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[failure_analysis] 代表性案例图已保存: {fname}")


# ================================================================
#  Section 7: Mode 4 — comparison_table (方法对比汇总表)
# ================================================================

def run_comparison_table(env, args):
    """Mode 4: 汇总所有方法在所有 GT 上的指标, 生成 LaTeX 表格。

    可复用 quantitative 模式的 CSV 结果, 也可独立运行。
    """
    print("\n" + "=" * 60)
    print("Mode: comparison_table — 方法对比汇总表")
    print("=" * 60)

    output_dir = env['output_dir']
    timing_means = OrderedDict()

    # 优先从已有 CSV 加载
    if args.metrics_csv and os.path.isfile(args.metrics_csv):
        results, gt_indices, methods, timing_means = _load_metrics_from_csv(args.metrics_csv)
        print(f"从 CSV 加载: {args.metrics_csv}")
    else:
        # 重新运行 quantitative 收集
        print("未指定 --metrics_csv, 将重新运行 quantitative 模式收集数据...")
        run_quantitative(env, args)
        # 尝试找到最新生成的 CSV
        import glob
        csvs = sorted(glob.glob(os.path.join(output_dir, 'quantitative_*.csv')))
        if csvs:
            results, gt_indices, methods, timing_means = _load_metrics_from_csv(csvs[-1])
        else:
            print("[comparison_table] 未找到指标 CSV, 退出")
            return

    # 生成 LaTeX 表格
    latex_path = os.path.join(
        output_dir,
        generate_timestamped_filename('comparison_table', '.tex'),
    )
    _generate_latex_table(results, gt_indices, methods, latex_path, timing_means)

    print(f"[comparison_table] 完成。结果保存至 {output_dir}")


def _load_metrics_from_csv(csv_path):
    """从 quantitative 模式生成的 CSV 文件加载指标数据。"""
    results = OrderedDict()
    gt_indices = []
    timing_means = OrderedDict()  # method -> mean time

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    header = lines[0].split(',')
    has_time = 'time_s' in header

    for line in lines[1:]:
        parts = line.split(',')
        method = parts[0]
        gi_str = parts[1]

        if gi_str == 'mean':
            # 提取均值行的 timing
            if has_time and len(parts) > 5 and parts[5] != 'N/A':
                timing_means[method] = float(parts[5])
            continue

        gi = int(gi_str)
        if gi not in gt_indices:
            gt_indices.append(gi)
        if method not in results:
            results[method] = {}

        results[method][gi] = {
            'NRMSE': float(parts[2]),
            'SSIM':  float(parts[3]),
            'PSNR':  float(parts[4]),
        }

    methods = list(results.keys())
    return results, gt_indices, methods, timing_means


def _generate_latex_table(results, gt_indices, methods, save_path, timing_means=None):
    """生成论文级 LaTeX 表格。

    格式:
      Method | Mean NRMSE/SSIM/PSNR | Time(s)
    当 GT 数量较多时采用紧凑格式 (仅显示 Mean + std)，避免表格过宽。
    """
    metrics = ['NRMSE', 'SSIM', 'PSNR']
    n_gt = len(gt_indices)
    # 当 GT 数量 > 3 时使用紧凑格式
    compact = n_gt > 3
    has_timing = timing_means and any(timing_means.values())

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Quantitative comparison of all methods.}')
    lines.append(r'\label{tab:quantitative}')

    if compact:
        # 紧凑格式: Method | NRMSE | SSIM | PSNR | Time(s)
        col_fmt = 'l' + 'c' * len(metrics)
        if has_timing:
            col_fmt += 'c'
        lines.append(r'\begin{tabular}{' + col_fmt + '}')
        lines.append(r'\hline')

        header_parts = [r'\textbf{Method}']
        for m in metrics:
            header_parts.append(r'\textbf{' + m + '}')
        if has_timing:
            header_parts.append(r'\textbf{Time (s)}')
        lines.append(' & '.join(header_parts) + r' \\')
        lines.append(r'\hline')

        for method in methods:
            row_parts = [method.replace('_', r'\_')]
            for m in metrics:
                arr = [results[method][gi][m]
                       for gi in gt_indices if gi in results.get(method, {})]
                if arr:
                    mean_v = np.mean(arr)
                    std_v = np.std(arr)
                    row_parts.append(f'${mean_v:.4f} \\pm {std_v:.4f}$')
                else:
                    row_parts.append('--')
            if has_timing:
                if method in timing_means:
                    row_parts.append(f'{timing_means[method]:.2f}')
                else:
                    row_parts.append('--')
            lines.append(' & '.join(row_parts) + r' \\')

    else:
        # 展开格式: Method | 每个GT三列 | Mean | Time(s)
        col_fmt = 'l' + '|ccc' * n_gt + '|ccc'
        if has_timing:
            col_fmt += '|c'
        lines.append(r'\begin{tabular}{' + col_fmt + '}')
        lines.append(r'\hline')

        # 表头第一行
        header1_parts = [r'\textbf{Method}']
        for gi in gt_indices:
            header1_parts.append(
                r'\multicolumn{3}{c|}{\textbf{GT \#' + str(gi) + '}}'
            )
        header1_parts.append(r'\multicolumn{3}{c}{\textbf{Mean}}')
        if has_timing:
            header1_parts.append(r'\textbf{Time}')
        lines.append(' & '.join(header1_parts) + r' \\')

        # 表头第二行
        header2_parts = ['']
        for _ in range(n_gt + 1):
            for m in metrics:
                header2_parts.append(m)
        if has_timing:
            header2_parts.append('(s)')
        lines.append(' & '.join(header2_parts) + r' \\')
        lines.append(r'\hline')

        # 数据行
        for method in methods:
            row_parts = [method.replace('_', r'\_')]
            all_vals = {m: [] for m in metrics}

            for gi in gt_indices:
                if gi in results.get(method, {}):
                    for m in metrics:
                        v = results[method][gi][m]
                        row_parts.append(f'{v:.4f}')
                        all_vals[m].append(v)
                else:
                    row_parts.extend(['--'] * 3)

            # Mean 列
            for m in metrics:
                if all_vals[m]:
                    row_parts.append(f'{np.mean(all_vals[m]):.4f}')
                else:
                    row_parts.append('--')

            # Time 列
            if has_timing:
                if method in timing_means:
                    row_parts.append(f'{timing_means[method]:.2f}')
                else:
                    row_parts.append('--')

            lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    latex_str = '\n'.join(lines)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"[comparison_table] LaTeX 表格已保存: {save_path}")
    print(f"\n--- LaTeX 预览 ---\n{latex_str}\n--- 预览结束 ---")


# ================================================================
#  Section 8: argparse & main
# ================================================================

def build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='sFWI 系统性评估实验脚本',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ---------- 全局参数 ----------
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['quantitative', 'uncertainty',
                 'failure_analysis', 'comparison_table'],
        help='评估模式:\n'
             '  quantitative     — 定量指标计算 (NRMSE/SSIM/PSNR)\n'
             '  uncertainty      — TopG 分支后验不确定性量化\n'
             '  failure_analysis — 失败模式分析 (Mode I-IV)\n'
             '  comparison_table — 方法对比汇总表 (LaTeX)',
    )
    parser.add_argument(
        '--master_seed', type=int, default=8,
        help='主随机种子 (默认: 8)',
    )
    parser.add_argument(
        '--sigma', type=float, default=0.3,
        help='DAPS 噪声水平 (默认: 0.3)',
    )
    parser.add_argument(
        '--gt_indices', type=int, nargs='+', default=None,
        help='要评估的 GT 样本索引列表 (默认: 0 10 20 30 40)',
    )
    parser.add_argument(
        '--n_candidates', type=int, default=50,
        help='sFWI GSS 每个 GT 的候选样本数 (默认: 50)',
    )
    parser.add_argument(
        '--sampling_method', type=str, default='rss',
        choices=['rss', 'gss', 'gss_topg', 'gss_topg_light'],
        help='sFWI 采样方法:\n'
             '  rss — 暴力随机搜索 (并行 n_candidates 个候选)\n'
             '  gss — 旧版 Group Score Search: best_group + 组内 top-k\n'
             '  gss_topg — 新版 Group Score Search: 每组选代表并按真实 m0 选 top-g 组 + DAPS 精细化\n'
             '  gss_topg_light — 与 gss_topg 相同选种子，但跳过 DAPS 精细化，直接输出 x0hat',
    )
    parser.add_argument(
        '--group_top_g', type=int, default=20,
        help='[gss_topg/gss_topg_light] 每个 GT 选取 top-g 个 group（每组 1 个代表 seed）',
    )
    parser.add_argument(
        '--gss_light_eval_bs', type=int, default=32,
        help='[gss_topg_light] 预计算代表 seed 正演时的批大小',
    )
    parser.add_argument(
        '--sm_path', type=str, default=None,
        help='[gss/gss_topg/gss_topg_light] 相似度矩阵 .pt 文件路径',
    )
    parser.add_argument(
        '--eval_patches_path', type=str, default=None,
        help='可选评估集路径（.pt/.pkl）；提供后覆盖默认 SEAM 224 切片',
    )

    # ---------- Baseline 参数 ----------
    parser.add_argument(
        '--baseline_config', type=str, default=None,
        help='Baseline 配置 JSON 文件路径 (格式见 load_baselines_from_config)',
    )

    # ---------- uncertainty 专用参数 ----------
    parser.add_argument(
        '--n_samples', type=int, default=10,
        help='[uncertainty][兼容保留] 旧版参数，当前不再使用',
    )
    parser.add_argument(
        '--refine_top_b', type=int, default=0,
        help='[uncertainty] 参与后验融合的分支数 B（<=0 时默认等于 group_top_g）',
    )
    parser.add_argument(
        '--branch_repeats', type=int, default=1,
        help='[uncertainty] 每个分支精细化重复次数 R（默认: 1，提速）',
    )
    parser.add_argument(
        '--uq_weight_tau', type=float, default=-1.0,
        help='[uncertainty] 权重温度 tau（<=0 自动估计）',
    )
    parser.add_argument(
        '--hcw_uncertainty_q', type=float, default=0.25,
        help='[uncertainty] HCW 的低不确定性分位阈值 q_u（默认: 0.25）',
    )
    parser.add_argument(
        '--hcw_error_q', type=float, default=0.75,
        help='[uncertainty] HCW 的高误差分位阈值 q_e（默认: 0.75）',
    )
    parser.add_argument(
        '--uq_plot_max_cases', type=int, default=-1,
        help='[uncertainty] 最多绘图 case 数（<=0 表示不限制）',
    )
    parser.add_argument(
        '--uq_disable_plots', action='store_true',
        help='[uncertainty] 关闭可视化，仅保存表格与统计',
    )
    parser.add_argument(
        '--save_uq_ensemble', action='store_true',
        help='[uncertainty] 保存每个 GT 的分支样本集合 (.pt)',
    )
    parser.add_argument(
        '--uq_n_gt', type=int, default=0,
        help='[uncertainty] 评估样本数（仅在未指定 --gt_indices 时生效；若>=1000需配合 --eval_patches_path 测试集）',
    )
    parser.add_argument(
        '--uq_large_scale', action='store_true',
        help='[uncertainty][兼容保留] 统一口径后 uncertainty 固定使用 gss_topg_light',
    )
    parser.add_argument(
        '--uq_gallery_cases', type=int, default=6,
        help='[uncertainty] 章节配图包含的案例数量（0 表示不生成 gallery）',
    )
    parser.add_argument(
        '--uq_skip_latex_table', action='store_true',
        help='[uncertainty] 跳过 LaTeX 汇总表输出',
    )

    # ---------- failure_analysis 专用参数 ----------
    parser.add_argument(
        '--n_gt', type=int, default=100,
        help='[failure_analysis] 覆盖的 GT 样本数 M（若>=1000需提供 --eval_patches_path 测试集）',
    )
    parser.add_argument(
        '--skip_agnostic', action='store_true',
        help='[failure_analysis] 跳过 physics-agnostic 对比（大样本默认也会自动跳过）',
    )
    parser.add_argument(
        '--force_agnostic_large_scale', action='store_true',
        help='[failure_analysis] 大样本时仍强制运行 physics-agnostic（会明显变慢）',
    )
    parser.add_argument(
        '--threshold_mode3', type=float, default=0.05,
        help='[failure_analysis] Mode III 阈值 (默认: 0.05)',
    )
    parser.add_argument(
        '--threshold_mode2', type=float, default=0.15,
        help='[failure_analysis] Mode II 阈值 (默认: 0.15)',
    )
    parser.add_argument(
        '--threshold_mode1', type=float, default=0.35,
        help='[failure_analysis] Mode I 阈值 (默认: 0.35)',
    )

    # ---------- comparison_table 专用参数 ----------
    parser.add_argument(
        '--metrics_csv', type=str, default=None,
        help='[comparison_table] 已有的指标 CSV 路径 (跳过重新计算)',
    )

    return parser


def main():
    """主入口: 解析参数 → 初始化环境 → 分发到对应 mode。"""
    parser = build_parser()

    # 兼容 Colab %run 和命令行两种调用方式。
    # 约定:
    #   - 纯脚本: python xxx.py --mode ...
    #   - Colab/IPython: %run xxx.py -- --mode ...
    raw_argv = list(sys.argv[1:])
    if '--' in raw_argv:
        raw_argv = raw_argv[raw_argv.index('--') + 1:]

    if len(raw_argv) == 0:
        # 无显式参数时回退到定量模式，便于快速 smoke test。
        raw_argv = ['--mode', 'quantitative']
    elif '--mode' not in raw_argv:
        # 兼容只传局部参数的场景。
        raw_argv = ['--mode', 'quantitative'] + raw_argv

    try:
        args, unknown = parser.parse_known_args(raw_argv)
    except SystemExit:
        print(f"[ERROR] 参数解析失败，请检查命令。argv={raw_argv}")
        return

    if unknown:
        print(f"[args][warn] 忽略未知参数: {unknown}")

    # 应用自定义阈值
    global MODE_THRESHOLDS
    MODE_THRESHOLDS['III'] = args.threshold_mode3
    MODE_THRESHOLDS['II'] = args.threshold_mode2
    MODE_THRESHOLDS['I'] = args.threshold_mode1

    print("=" * 60)
    print("sFWI 系统性评估实验")
    print("=" * 60)
    print(f"  模式:       {args.mode}")
    print(f"  主种子:     {args.master_seed}")
    print(f"  sigma:      {args.sigma}")
    print(f"  GT 索引:    {args.gt_indices or '默认'}")
    print(f"  GSS 候选数: {args.n_candidates}")
    print(f"  采样方法:   {args.sampling_method}")
    if args.eval_patches_path:
        print(f"  eval_patches_path: {args.eval_patches_path}")
    if args.sampling_method in ('gss_topg', 'gss_topg_light'):
        print(f"  group_top_g: {args.group_top_g}")
    if args.mode == 'uncertainty':
        refine_top_b = args.refine_top_b if args.refine_top_b > 0 else args.group_top_g
        print(
            f"  [UQ] refine_top_b={refine_top_b}, "
            f"branch_repeats={args.branch_repeats}, tau={args.uq_weight_tau}"
        )
        print(
            f"  [UQ] HCW quantiles: uq_q={args.hcw_uncertainty_q}, "
            f"err_q={args.hcw_error_q}"
        )
        print(
            f"  [UQ] uq_n_gt={args.uq_n_gt}, large_scale={args.uq_large_scale}, "
            f"gallery_cases={args.uq_gallery_cases}"
        )
    if args.mode == 'failure_analysis':
        print(
            f"  [FAIL] n_gt={args.n_gt}, group_top_g={args.group_top_g}, "
            f"skip_agnostic={args.skip_agnostic}, "
            f"force_agnostic_large_scale={args.force_agnostic_large_scale}"
        )

    if args.mode == 'uncertainty':
        if args.refine_top_b < 0:
            print("[ERROR] --refine_top_b 必须 >= 0")
            return
        if args.branch_repeats < 1:
            print("[ERROR] --branch_repeats 必须 >= 1")
            return
        if args.group_top_g < 1:
            print("[ERROR] --group_top_g 必须 >= 1")
            return
        if not (0.0 <= float(args.hcw_uncertainty_q) <= 1.0):
            print("[ERROR] --hcw_uncertainty_q 需在 [0,1] 内")
            return
        if not (0.0 <= float(args.hcw_error_q) <= 1.0):
            print("[ERROR] --hcw_error_q 需在 [0,1] 内")
            return
        if args.uq_n_gt < 0:
            print("[ERROR] --uq_n_gt 必须 >= 0")
            return
        if args.uq_gallery_cases < 0:
            print("[ERROR] --uq_gallery_cases 必须 >= 0")
            return
        if args.uq_n_gt >= 1000 and not args.eval_patches_path:
            print(
                "[ERROR] uq_n_gt>=1000 时必须提供 --eval_patches_path（1000样本测试集），"
                "以避免误用默认训练域切片。"
            )
            return
        if not args.sm_path or not os.path.isfile(args.sm_path):
            print(
                "[ERROR] uncertainty 模式需要 --sm_path 指向有效相似度矩阵文件。"
            )
            return
    if args.mode == 'failure_analysis':
        if args.n_gt < 1:
            print("[ERROR] --n_gt 必须 >= 1")
            return
        if args.group_top_g < 1:
            print("[ERROR] --group_top_g 必须 >= 1")
            return
        if args.n_gt >= 1000 and not args.eval_patches_path:
            print(
                "[ERROR] n_gt>=1000 时必须提供 --eval_patches_path（1000样本测试集），"
                "以避免误用默认训练域切片。"
            )
            return
        if args.gt_indices and len(args.gt_indices) >= 1000 and not args.eval_patches_path:
            print(
                "[ERROR] gt_indices 数量>=1000 时必须提供 --eval_patches_path（1000样本测试集），"
                "以避免误用默认训练域切片。"
            )
            return
        if not args.sm_path or not os.path.isfile(args.sm_path):
            print(
                "[ERROR] failure_analysis 模式需要 --sm_path 指向有效相似度矩阵文件。"
            )
            return

    # 初始化环境
    env = setup_environment(args)

    # 加载相似度矩阵 (GSS 模式)
    need_sm = (
        args.sampling_method in ('gss', 'gss_topg', 'gss_topg_light')
        or args.mode in ('uncertainty', 'failure_analysis')
    )
    if need_sm:
        if not args.sm_path or not os.path.isfile(args.sm_path):
            print(
                f"[ERROR] 模式 {args.mode} 需要有效的 --sm_path，当前: {args.sm_path}"
            )
            return
        if args.group_top_g < 1:
            print("[ERROR] --group_top_g 必须 >= 1")
            return
        sm_data = torch.load(args.sm_path, weights_only=False)
        required_keys = ['similarity_matrix', 'k', 'centroid_indices', 'x0hat_batch']
        miss = [k for k in required_keys if k not in sm_data]
        if miss:
            print(f"[ERROR] SM 文件缺少字段: {miss}")
            return
        env['sm_info'] = sm_data
        print(f"[setup] 已加载相似度矩阵: {args.sm_path}")
        print(f"  矩阵形状: {sm_data['similarity_matrix'].shape}, "
              f"k={sm_data['k']}, centroids={sm_data['centroid_indices']}")
        if args.sampling_method in ('gss_topg', 'gss_topg_light') or args.mode in ('uncertainty', 'failure_analysis'):
            has_rep = ('centroid_seed_indices' in sm_data) or ('group_members' in sm_data)
            has_dcent = ('d_centroids_2d' in sm_data)
            if not has_rep:
                print("[setup][warn] SM 无 centroid_seed_indices/group_members，"
                      "group_topg 将回退到每组 argmin(similarity_row) 代表。")
            if not has_dcent:
                print("[setup][warn] SM 无 d_centroids_2d，group matching 将回退为在线正演 centroid。")
        need_light_cache = (
            args.sampling_method == 'gss_topg_light'
            or args.mode in ('uncertainty', 'failure_analysis')
        )
        if need_light_cache:
            if args.gss_light_eval_bs < 1:
                print("[ERROR] --gss_light_eval_bs 必须 >= 1")
                return
            print("[setup] 预计算 gss_topg_light 缓存（一次性）...")
            t0_cache = time.time()
            light_cache = _build_gss_topg_light_cache(
                sm_info=env['sm_info'],
                operator=env['operator'],
                device=env['cfg'].device,
                eval_bs=args.gss_light_eval_bs,
            )
            env['gss_topg_light_cache'] = light_cache
            env['gss_topg_light_eval_bs'] = int(args.gss_light_eval_bs)
            print(
                "[setup] gss_topg_light 缓存完成: "
                f"rep_n={int(light_cache['rep_indices'].numel())}, "
                f"source={light_cache.get('source', 'unknown')}, "
                f"time={time.time() - t0_cache:.2f}s"
            )

    # 分发
    dispatch = {
        'quantitative':     run_quantitative,
        'uncertainty':      run_uncertainty,
        'failure_analysis': run_failure_analysis,
        'comparison_table': run_comparison_table,
    }

    dispatch[args.mode](env, args)

    print("\n" + "=" * 60)
    print("评估实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
