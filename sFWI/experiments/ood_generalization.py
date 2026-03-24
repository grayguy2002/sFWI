"""
OOD泛化实验脚本

用于评估sFWI模型在out-of-distribution数据上的泛化能力。

实验设计:
  - Marmousi-pretrained模型 -> 测试在SEAM数据上 (OOD)
  - SEAM侧模型(默认SEAM-finetuned) -> 测试在SEAM数据上 (in-distribution)
  - 对比两者的NRMSE/SSIM/PSNR和失败模式分布

用法:
  %run sFWI/experiments/ood_generalization.py --mode ood_comparison
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
from collections import OrderedDict
from tqdm import tqdm

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 设置score_sde路径
from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

# 导入sFWI模块
from sFWI.config import FWIConfig, build_daps_configs
from sFWI.models.sde_setup import create_sde_config
from sFWI.models.score_model import NCSNpp_DAPS
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.evaluation.wasserstein import Wasserstein, Wasserstein_us
from sFWI.utils.file_utils import generate_timestamped_filename

# 导入DAPS
from DAPS.sampler import get_sampler, DAPS
from DAPS.eval import Evaluator

VELOCITY_CMAP = 'viridis'

try:
    from piq import psnr as piq_psnr, ssim as piq_ssim
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("[WARNING] piq 未安装，SSIM/PSNR 将使用简易实现。pip install piq")


# ================================================================
#  Section 1: 辅助函数
# ================================================================

def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """确保 tensor 为 4D [B, C, H, W]。"""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor):
    """对齐 pred 与 gt 的形状/分辨率。"""
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
            pred4,
            size=gt4.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    return pred4, gt4


def _shift_to_nonneg(pred: torch.Tensor, gt: torch.Tensor):
    """将 pred 和 gt 平移到非负区间, 返回 (pred_shifted, gt_shifted, data_range)。"""
    global_min = min(pred.min().item(), gt.min().item())
    if global_min < 0:
        pred = pred - global_min
        gt = gt - global_min
    data_range = max(pred.max().item(), gt.max().item())
    if data_range < 1e-8:
        data_range = 1.0
    return pred, gt, data_range


def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Normalized Root Mean Square Error."""
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    mse = F.mse_loss(pred4, gt4).item()
    rmse = np.sqrt(mse)
    gt_range = (gt4.max() - gt4.min()).item()
    if gt_range < 1e-8:
        return float('inf')
    return rmse / gt_range


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Structural Similarity Index."""
    pred4, gt4 = _align_pred_to_gt(pred, gt)

    if HAS_PIQ:
        pred_s, gt_s, dr = _shift_to_nonneg(pred4.float(), gt4.float())
        return piq_ssim(pred_s, gt_s, data_range=dr).item()

    mu_x = pred4.mean()
    mu_y = gt4.mean()
    sigma_x = pred4.var()
    sigma_y = gt4.var()
    sigma_xy = ((pred4 - mu_x) * (gt4 - mu_y)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return (num / den).item()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    pred4, gt4 = _align_pred_to_gt(pred, gt)

    if HAS_PIQ:
        pred_s, gt_s, dr = _shift_to_nonneg(pred4.float(), gt4.float())
        return piq_psnr(pred_s, gt_s, data_range=dr).item()

    mse = F.mse_loss(pred4, gt4).item()
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_all_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """计算全部定量指标。"""
    nrmse = compute_nrmse(pred, gt)
    ssim = compute_ssim(pred, gt)
    psnr = compute_psnr(pred, gt)
    return {
        'NRMSE': nrmse,
        'SSIM': ssim,
        'PSNR': psnr,
    }


def _classify_mode(nrmse: float) -> str:
    """根据NRMSE将结果分类为Mode I-IV。"""
    if nrmse < 0.05:
        return 'III'
    elif nrmse < 0.15:
        return 'II'
    elif nrmse < 0.35:
        return 'I'
    else:
        return 'IV'


def _resolve_checkpoint_path(default_path, ckpt_dir=None, ckpt_file=None):
    """解析checkpoint路径（支持目录/文件名覆盖）。"""
    if ckpt_file and os.path.isabs(ckpt_file):
        return ckpt_file

    base_dir = ckpt_dir if ckpt_dir else os.path.dirname(default_path)
    filename = ckpt_file if ckpt_file else os.path.basename(default_path)
    return os.path.join(base_dir, filename)


def _load_sm_info(sm_path, expected_model_tag=None):
    """加载并校验缓存的相似度矩阵资产。"""
    if not sm_path or not os.path.isfile(sm_path):
        raise FileNotFoundError(f"GSS相似度矩阵文件不存在: {sm_path}")

    sm_data = torch.load(sm_path, weights_only=False)
    required_keys = ['similarity_matrix', 'k', 'centroid_indices', 'x0hat_batch']
    for key in required_keys:
        if key not in sm_data:
            raise KeyError(f"{sm_path} 缺少必需字段: {key}")

    model_tag = sm_data.get('model_tag')
    if expected_model_tag and model_tag and model_tag != expected_model_tag:
        raise ValueError(
            f"相似度矩阵模型标签不匹配: 期望={expected_model_tag}, 实际={model_tag}, 文件={sm_path}"
        )
    return sm_data


def _set_manual_seed(seed, device):
    """设置随机种子，确保采样可复现。"""
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


def _compute_centroid_distances_from_sm(sm_info, measurement, operator, device):
    """计算 measurement 到各 group centroid 的距离。"""
    meas_flat = measurement.reshape(1, -1)

    d_centroids_2d = sm_info.get('d_centroids_2d', None)
    if isinstance(d_centroids_2d, torch.Tensor):
        d_centroids_2d = d_centroids_2d.to(device)
        if d_centroids_2d.dim() == 2 and d_centroids_2d.shape[1] == meas_flat.shape[1]:
            return torch.norm(d_centroids_2d - meas_flat, dim=1)

    centroid_indices = sm_info.get('centroid_indices', None)
    if centroid_indices is None:
        raise KeyError("SM 缺少 centroid_indices，无法计算 centroid 距离。")
    if isinstance(centroid_indices, torch.Tensor):
        centroid_indices = centroid_indices.detach().cpu().tolist()
    elif isinstance(centroid_indices, np.ndarray):
        centroid_indices = centroid_indices.tolist()

    x0hat_batch = sm_info['x0hat_batch'].to(device)
    n_seeds = int(x0hat_batch.shape[0])
    valid = [int(v) for v in centroid_indices if 0 <= int(v) < n_seeds]
    if len(valid) == 0:
        raise ValueError("centroid_indices 全部越界，无法回退计算 centroid 距离。")

    idx_t = torch.tensor(valid, device=device, dtype=torch.long)
    with torch.no_grad():
        d_centroids = operator(x0hat_batch[idx_t]).reshape(len(valid), -1)
    return torch.norm(d_centroids - meas_flat, dim=1)


def _resolve_group_representative_indices(sm_info, sm, n_seeds, device):
    """解析每个 group 的代表 seed。"""
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

    def _rep_from_group_members(g_idx):
        if not isinstance(group_members, (list, tuple)) or g_idx >= len(group_members):
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
    sm = sm_info['similarity_matrix'].to(device)
    x0hat_batch = sm_info['x0hat_batch'].to(device)
    n_seeds = int(sm.shape[1])
    if n_seeds < 1:
        raise RuntimeError("SM 中 x0hat/seed 数为 0。")

    rep_indices, rep_group_indices = _resolve_group_representative_indices(
        sm_info=sm_info, sm=sm, n_seeds=n_seeds, device=device
    )
    rep_d_2d = _precompute_rep_d2d_by_operator(
        x0hat_batch=x0hat_batch,
        rep_indices=rep_indices,
        operator=operator,
        eval_bs=eval_bs,
    ).detach()

    return {
        'sm': sm.detach(),
        'x0hat_batch': x0hat_batch.detach(),
        'rep_indices': rep_indices.detach(),
        'rep_group_indices': rep_group_indices.detach(),
        'rep_d_2d': rep_d_2d,
        'source': 'operator',
    }


def _select_x0_candidate_group_topg_light(
    sm_info,
    measurement,
    operator,
    group_top_g=20,
    light_cache=None,
    eval_bs=16,
):
    """gss_topg_light: 使用缓存并行评分，直接返回 x0hat。"""
    device = measurement.device
    x0hat_batch = None
    sm = None
    if isinstance(light_cache, dict):
        x0hat_batch = light_cache.get('x0hat_batch', None)
        sm = light_cache.get('sm', None)
    if not isinstance(x0hat_batch, torch.Tensor):
        x0hat_batch = sm_info['x0hat_batch'].to(device)
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
    best_candidate = x0hat_batch[best_global_idx].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_global_idx,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
        'selected_group_indices': selected_group_indices,
    }
    return best_candidate, meta


def _select_x0_candidate_from_sm(
    sm_info,
    measurement,
    operator,
    top_k=50,
    group_top_g=20,
    candidate_mode='gss_topk',
):
    """从 SM 中选取 x0 候选。"""
    device = measurement.device
    sm = sm_info['similarity_matrix'].to(device)
    x0hat_batch = sm_info['x0hat_batch'].to(device)
    n_seeds = int(sm.shape[1])
    if n_seeds < 1:
        raise RuntimeError("SM 中 x0hat/seed 数为 0。")

    centroid_distances = _compute_centroid_distances_from_sm(
        sm_info=sm_info,
        measurement=measurement,
        operator=operator,
        device=device,
    )
    best_group = int(torch.argmin(centroid_distances).item())

    meas_flat = measurement.reshape(1, -1)
    selected_group_indices = None
    candidate_distances = None

    if candidate_mode == 'group_topg':
        n_groups = int(sm.shape[0])
        n_pick_groups = max(1, min(int(group_top_g), n_groups))
        rep_indices, rep_group_indices = _resolve_group_representative_indices(
            sm_info=sm_info, sm=sm, n_seeds=n_seeds, device=device
        )
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
    best_candidate = x0hat_batch[best_global_idx].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_global_idx,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
        'selected_group_indices': selected_group_indices,
    }
    return best_candidate, meta


def _run_rss_sampling(model, gt, measurement, operator, evaluator_us, n_samples):
    """原始RSS路径：并行采样后按W-distance选最优。"""
    w_distances, x0hat_batch = model.daps.sample(
        model=model,
        x_start=model.daps.get_start(gt),
        operator=operator,
        measurement=measurement,
        evaluator_us=evaluator_us,
        seed=range(n_samples),
        return_batch=True,
        record=False,
        verbose=False,
        gt=gt,
    )
    best_idx = torch.argmin(w_distances).item()
    best_sample = x0hat_batch[best_idx].unsqueeze(0)
    meta = {
        'best_idx': int(best_idx),
        'best_distance': float(w_distances[best_idx].item()),
    }
    return best_sample, meta


def _refine_with_daps(model, operator, measurement, init_x0, noise_seed=None):
    """给定候选x0，执行DAPS-FWI精细化（退火+反向扩散+Langevin dynamics）。"""
    from DAPS.sampler import Scheduler, DiffusionSampler

    daps = model.daps
    annealing = daps.annealing_scheduler
    device = str(init_x0.device)
    _set_manual_seed(noise_seed, device)

    xt = init_x0
    n_steps = int(annealing.num_steps)
    if n_steps <= 0:
        return init_x0

    for step in range(n_steps):
        sigma = float(annealing.sigma_steps[step])

        # 1) reverse diffusion
        diff_scheduler = Scheduler(**daps.diffusion_scheduler_config, sigma_max=sigma)
        diff_sampler = DiffusionSampler(diff_scheduler)
        x0hat = diff_sampler.sample(model, xt, SDE=False, verbose=False)

        # 2) physics-informed Langevin correction
        ratio = float(step) / float(max(1, n_steps))
        x0y = daps.lgvd.sample(x0hat, operator, measurement, sigma, ratio)

        # 3) forward diffusion
        next_sigma = float(annealing.sigma_steps[step + 1])
        xt = x0y + torch.randn_like(x0y) * next_sigma

    return x0hat


def _run_gss_sampling(
    model,
    operator,
    measurement,
    sm_info,
    sampling_method='gss_cached',
    top_k=50,
    group_top_g=20,
    light_cache=None,
    light_eval_bs=16,
    master_seed=0,
    gt_seed=0,
):
    """统一 GSS 采样入口：gss_cached / gss_topg / gss_topg_light。"""
    if sampling_method == 'gss_topg_light':
        best_candidate, meta = _select_x0_candidate_group_topg_light(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            group_top_g=group_top_g,
            light_cache=light_cache,
            eval_bs=light_eval_bs,
        )
        return best_candidate, meta

    candidate_mode = 'group_topg' if sampling_method == 'gss_topg' else 'gss_topk'
    best_candidate, meta = _select_x0_candidate_from_sm(
        sm_info=sm_info,
        measurement=measurement,
        operator=operator,
        top_k=top_k,
        group_top_g=group_top_g,
        candidate_mode=candidate_mode,
    )
    best_sample = _refine_with_daps(
        model=model,
        operator=operator,
        measurement=measurement,
        init_x0=best_candidate,
        noise_seed=(int(master_seed) + int(gt_seed)),
    )
    return best_sample, meta


# ================================================================
#  Section 2: 模型加载器
# ================================================================

def load_seam_trained_model(config, base_config, lgvd_config, checkpoint_path, device):
    """加载SEAM-trained的sFWI模型。"""
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=checkpoint_path,
    )
    model.set_device(device)
    return model


def load_marmousi_pretrained_model(config, base_config, lgvd_config, checkpoint_path, device):
    """
    加载Marmousi-pretrained的score model。
    """
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=checkpoint_path,
    )
    model.set_device(device)
    return model


def create_seam_dataset(cfg, image_size=200, eval_patches_path=None):
    """创建评估数据集（默认SEAM，可被外部测试集覆盖）。"""
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
        return data, v_torch, f"external:{eval_patches_path}"

    v_torch = load_seam_model(cfg.paths.seam_model_path)
    if v_torch.dim() == 4 and v_torch.shape[1] == 1:
        v_torch = v_torch[:, 0]
    v_torch = v_torch.detach().cpu().float()
    data = create_velocity_dataset(v_torch, image_size=image_size)
    return data, v_torch, f"seam_default:{cfg.paths.seam_model_path}"


def create_marmousi_dataset(cfg, image_size=200):
    """
    创建Marmousi数据集。
    从seismic_dataset.pkl加载Marmousi速度模型。
    """
    pkl_path = cfg.paths.marmousi_dataset_path
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Marmousi dataset not found: {pkl_path}")

    print(f"正在加载Marmousi数据集: {pkl_path}")
    v_torch, _ = load_marmousi_from_pkl(pkl_path, image_size=image_size)

    # 重采样到目标尺寸
    if image_size != 200:
        v_torch_resized = F.interpolate(
            v_torch.unsqueeze(1),
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    else:
        v_torch_resized = v_torch

    data = create_velocity_dataset(v_torch_resized, image_size=image_size)
    return data, v_torch_resized


# ================================================================
#  Section 3: OOD实验核心逻辑
# ================================================================

def run_ood_comparison(cfg, args):
    """
    OOD泛化实验主流程：

    对比组:
      1. SEAM侧模型(可选SEAM-trained/SEAM-finetuned) + SEAM (in-distribution)
      2. Marmousi-pretrained + SEAM (OOD)

    对每个GT执行DAPS采样，计算定量指标。
    """
    seam_label = 'SEAM-finetuned' if args.seam_model_tag == 'seam_finetune' else 'SEAM-trained'

    print("\n" + "=" * 60)
    print(f"OOD泛化实验 —— {seam_label} vs Marmousi-pretrained")
    print("=" * 60)

    device = cfg.device
    image_size = cfg.image_size

    # ---------- 加载SEAM评估数据 ----------
    print("\n加载SEAM评估数据...")
    seam_data, seam_v_torch, dataset_source = create_seam_dataset(
        cfg, image_size, eval_patches_path=args.eval_patches_path
    )
    print(f"SEAM数据集: {len(seam_data)} 个样本")
    print(f"数据来源: {dataset_source}")
    print(f"速度张量形状: {tuple(seam_v_torch.shape)}")

    # ---------- 创建算子 ----------
    config, sde = create_sde_config(parent_dir, batch_size=1)
    base_config, lgvd_config = build_daps_configs(cfg)
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
    eval_fn = Wasserstein(operator)
    eval_us_fn = Wasserstein_us(operator)
    evaluator = Evaluator((eval_fn,))
    evaluator_us = Evaluator((eval_us_fn,))

    # ---------- 解析并加载 checkpoint ----------
    seam_default_ckpt = (
        cfg.paths.seam_finetune_checkpoint_path
        if args.seam_model_tag == 'seam_finetune'
        else cfg.paths.checkpoint_path
    )
    seam_ckpt_path = _resolve_checkpoint_path(
        seam_default_ckpt,
        ckpt_dir=args.seam_ckpt_dir,
        ckpt_file=args.seam_ckpt_file,
    )
    marmousi_ckpt_path = _resolve_checkpoint_path(
        cfg.paths.marmousi_checkpoint_path,
        ckpt_dir=args.marmousi_ckpt_dir,
        ckpt_file=args.marmousi_ckpt_file,
    )

    if not os.path.isfile(seam_ckpt_path):
        raise FileNotFoundError(f"SEAM checkpoint不存在: {seam_ckpt_path}")

    print(f"\n加载{seam_label}模型: {seam_ckpt_path}")
    seam_model = load_seam_trained_model(
        config, base_config, lgvd_config,
        seam_ckpt_path, device
    )
    seam_model.set_device(device)
    print(f"✓ {seam_label}模型已加载")

    print(f"\n加载Marmousi-pretrained模型: {marmousi_ckpt_path}")
    if os.path.exists(marmousi_ckpt_path):
        marmousi_model = load_marmousi_pretrained_model(
            config, base_config, lgvd_config,
            marmousi_ckpt_path, device
        )
        print("✓ Marmousi-pretrained模型已加载")
    else:
        print(f"⚠ 警告: Marmousi checkpoint不存在，跳过OOD对比")
        marmousi_model = None

    # ---------- GSS资产 ----------
    seam_sm_info = None
    marmousi_sm_info = None
    seam_light_cache = None
    marmousi_light_cache = None
    use_gss = args.sampling_method in ('gss_cached', 'gss_topg', 'gss_topg_light')
    if use_gss:
        seam_sm_info = _load_sm_info(
            args.sm_path_seam,
            expected_model_tag=args.seam_model_tag
        )
        print(f"✓ 已加载SEAM相似度矩阵: {args.sm_path_seam}")
        if marmousi_model is not None:
            if not args.sm_path_marmousi:
                raise ValueError(
                    f"sampling_method={args.sampling_method} 时必须提供 --sm_path_marmousi"
                )
            marmousi_sm_info = _load_sm_info(
                args.sm_path_marmousi, expected_model_tag='marmousi'
            )
            print(f"✓ 已加载Marmousi相似度矩阵: {args.sm_path_marmousi}")
        if args.sampling_method == 'gss_topg_light':
            print("[setup] 预计算 gss_topg_light 缓存（SEAM）...")
            seam_light_cache = _build_gss_topg_light_cache(
                sm_info=seam_sm_info,
                operator=operator,
                device=device,
                eval_bs=args.gss_light_eval_bs,
            )
            print(
                f"[setup] SEAM light cache: rep_n={int(seam_light_cache['rep_indices'].numel())}, "
                f"source={seam_light_cache.get('source', 'unknown')}"
            )
            if marmousi_model is not None:
                print("[setup] 预计算 gss_topg_light 缓存（Marmousi）...")
                marmousi_light_cache = _build_gss_topg_light_cache(
                    sm_info=marmousi_sm_info,
                    operator=operator,
                    device=device,
                    eval_bs=args.gss_light_eval_bs,
                )
                print(
                    f"[setup] Marmousi light cache: rep_n={int(marmousi_light_cache['rep_indices'].numel())}, "
                    f"source={marmousi_light_cache.get('source', 'unknown')}"
                )

    # ---------- 采样参数 ----------
    gt_indices = [0, 10, 20, 30, 40] if args.gt_indices is None else args.gt_indices
    n_samples = int(args.n_samples)
    top_k = int(args.gss_top_k) if args.gss_top_k is not None else n_samples
    group_top_g = int(args.group_top_g) if int(args.group_top_g) > 0 else top_k

    print(f"\n实验配置:")
    print(f"  采样方法: {args.sampling_method}")
    print(f"  GT索引: {gt_indices}")
    print(f"  每个GT采样次数: {n_samples}")
    print(f"  GSS top-k: {top_k}")
    print(f"  Group top-g: {group_top_g}")
    print(f"  DAPS sigma: {cfg.daps.sigma}")
    print(f"  gss_light_eval_bs: {args.gss_light_eval_bs}")

    # ---------- 收集结果 ----------
    results = {
        seam_label: {gi: {} for gi in gt_indices},
        'Marmousi-pretrained': {gi: {} for gi in gt_indices} if marmousi_model else {},
    }
    predictions = {
        seam_label: {gi: None for gi in gt_indices},
        'Marmousi-pretrained': {gi: None for gi in gt_indices} if marmousi_model else {},
    }
    timing = {seam_label: [], 'Marmousi-pretrained': []}

    for gt_idx in tqdm(gt_indices, desc="OOD实验"):
        # 获取GT（严格按索引取样，避免与可视化行标错位）
        if gt_idx < 0 or gt_idx >= len(seam_data):
            raise IndexError(f"gt_index超出范围: {gt_idx}, 数据集大小={len(seam_data)}")
        gt = seam_data[gt_idx].unsqueeze(0).to(device)
        measurement = operator(gt)

        # --- SEAM模型 (in-distribution) ---
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        if use_gss:
            seam_pred, seam_meta = _run_gss_sampling(
                model=seam_model,
                operator=operator,
                measurement=measurement,
                sm_info=seam_sm_info,
                sampling_method=args.sampling_method,
                top_k=top_k,
                group_top_g=group_top_g,
                light_cache=seam_light_cache,
                light_eval_bs=args.gss_light_eval_bs,
                master_seed=args.master_seed,
                gt_seed=gt_idx,
            )
        else:
            seam_pred, seam_meta = _run_rss_sampling(
                seam_model, gt, measurement, operator, evaluator_us, n_samples
            )
        torch.cuda.synchronize() if device == 'cuda' else None
        timing[seam_label].append(time.time() - t0)

        # 计算指标
        results[seam_label][gt_idx] = compute_all_metrics(seam_pred, gt)
        predictions[seam_label][gt_idx] = seam_pred.squeeze().detach().cpu()

        # --- Marmousi-pretrained模型 (OOD) ---
        if marmousi_model:
            torch.cuda.synchronize() if device == 'cuda' else None
            t0 = time.time()

            if use_gss:
                marmousi_pred, marmousi_meta = _run_gss_sampling(
                    model=marmousi_model,
                    operator=operator,
                    measurement=measurement,
                    sm_info=marmousi_sm_info,
                    sampling_method=args.sampling_method,
                    top_k=top_k,
                    group_top_g=group_top_g,
                    light_cache=marmousi_light_cache,
                    light_eval_bs=args.gss_light_eval_bs,
                    master_seed=args.master_seed + 100000,
                    gt_seed=gt_idx,
                )
            else:
                marmousi_pred, marmousi_meta = _run_rss_sampling(
                    marmousi_model, gt, measurement, operator, evaluator_us, n_samples
                )
            torch.cuda.synchronize() if device == 'cuda' else None
            timing['Marmousi-pretrained'].append(time.time() - t0)

            # 计算指标
            results['Marmousi-pretrained'][gt_idx] = compute_all_metrics(marmousi_pred, gt)
            predictions['Marmousi-pretrained'][gt_idx] = marmousi_pred.squeeze().detach().cpu()

    # ---------- 输出汇总 ----------
    _print_ood_summary(results, gt_indices, timing)
    _save_ood_results(results, gt_indices, timing, cfg.paths.output_path)

    # ---------- 可视化 ----------
    _plot_ood_comparison(results, gt_indices, seam_data, seam_v_torch, predictions,
                         cfg.paths.output_path)

    print(f"\n[OOD实验] 完成。结果保存至 {cfg.paths.output_path}")


def _print_ood_summary(results, gt_indices, timing):
    """打印OOD实验汇总表。"""
    print("\n" + "=" * 60)
    print("OOD实验结果汇总")
    print("=" * 60)

    methods = list(results.keys())
    methods = [m for m in methods if isinstance(results[m], dict) and results[m].get(gt_indices[0])]

    header = f"{'Method':<25}"
    for m in ['NRMSE', 'SSIM', 'PSNR']:
        header += f"  {m:>10}"
    header += f"  {'Time(s)':>10}"
    print(header)
    print("-" * len(header))

    for method in methods:
        vals = [results[method][gi] for gi in gt_indices if results[method][gi]]
        if not vals:
            continue

        row = f"{method:<25}"
        finite_info = {}
        for m in ['NRMSE', 'SSIM', 'PSNR']:
            arr = np.asarray([v[m] for v in vals], dtype=np.float64)
            finite_mask = np.isfinite(arr)
            n_valid = int(np.sum(finite_mask))
            n_total = int(arr.size)
            finite_info[m] = (n_valid, n_total)
            if n_valid > 0:
                mean_v = float(np.mean(arr[finite_mask]))
            else:
                mean_v = float('nan')
            row += f"  {mean_v:>10.4f}"
        mean_t = float(np.mean(timing[method])) if len(timing.get(method, [])) > 0 else float('nan')
        row += f"  {mean_t:>10.2f}"
        print(row)
        nrmse_valid, nrmse_total = finite_info['NRMSE']
        ssim_valid, ssim_total = finite_info['SSIM']
        psnr_valid, psnr_total = finite_info['PSNR']
        print(
            f"  [stat] invalid counts -> "
            f"NRMSE={nrmse_total - nrmse_valid}, "
            f"SSIM={ssim_total - ssim_valid}, "
            f"PSNR={psnr_total - psnr_valid}"
        )
    print()


def _save_ood_results(results, gt_indices, timing, output_path):
    """保存OOD实验结果到CSV。"""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, generate_timestamped_filename('ood_results', '.csv'))

    methods = list(results.keys())
    lines = ['method,gt_index,NRMSE,SSIM,PSNR,time_s,is_finite_nrmse']

    for method in methods:
        for gi in gt_indices:
            if not results[method].get(gi):
                continue
            v = results[method][gi]
            t_idx = gt_indices.index(gi)
            t = timing[method][t_idx] if t_idx < len(timing[method]) else 0
            is_finite_nrmse = int(np.isfinite(float(v['NRMSE'])))
            lines.append(
                f"{method},{gi},{float(v['NRMSE']):.6f},{float(v['SSIM']):.6f},{float(v['PSNR']):.6f},{t:.4f},{is_finite_nrmse}"
            )

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[OOD] 结果CSV已保存: {csv_path}")


def _plot_ood_comparison(results, gt_indices, seam_data, seam_v_torch, predictions, output_path):
    """绘制OOD对比可视化图。"""
    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, generate_timestamped_filename('ood_comparison', '.pdf'))

    methods = list(results.keys())
    methods = [m for m in methods if isinstance(results[m], dict) and results[m].get(gt_indices[0])]
    n_gt = len(gt_indices)
    n_methods = len(methods)

    # 全局colorbar范围
    all_gts = [seam_v_torch[gi].numpy() for gi in gt_indices]
    vmin = min(g.min() for g in all_gts)
    vmax = max(g.max() for g in all_gts)

    fig, axes = plt.subplots(n_gt, n_methods + 1, figsize=(4 * (n_methods + 1), 3.5 * n_gt))
    if n_gt == 1:
        axes = axes[np.newaxis, :]

    for row, gi in enumerate(gt_indices):
        gt_np = seam_v_torch[gi].numpy()

        # GT列
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
            pred_tensor = predictions[method].get(gi)
            pred_np = None
            if pred_tensor is not None:
                pred_tensor = pred_tensor.detach().cpu()

                if pred_tensor.dim() == 2:
                    pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
                elif pred_tensor.dim() == 3:
                    pred_tensor = pred_tensor.unsqueeze(0)
                elif pred_tensor.dim() != 4:
                    pred_tensor = None

                if pred_tensor is not None:
                    # 预测在归一化空间，先反归一化到速度值再做尺寸对齐
                    pred_tensor = seam_data.denormalize(pred_tensor)
                    if pred_tensor.shape[-2:] != gt_np.shape:
                        pred_tensor = F.interpolate(
                            pred_tensor,
                            size=gt_np.shape,
                            mode='bilinear',
                            align_corners=False
                        )
                    pred_np = pred_tensor.squeeze().numpy()

            if pred_np is not None:
                ax.imshow(pred_np, cmap=VELOCITY_CMAP, aspect='auto', vmin=vmin, vmax=vmax)
                v = results[method][gi]
                ax.set_xlabel(f'NRMSE={v["NRMSE"]:.3f}\nSSIM={v["SSIM"]:.3f}', fontsize=8)
            else:
                ax.imshow(np.full_like(gt_np, np.nan), cmap='gray', aspect='auto')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(method, fontsize=11, fontweight='bold')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Velocity (m/s)', fontsize=10)

    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OOD] 对比图已保存: {fig_path}")


# ================================================================
#  Section 4: CLI入口
# ================================================================

def build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='OOD泛化实验脚本',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--mode', type=str, default='ood_comparison',
        choices=['ood_comparison'],
        help='实验模式 (默认: ood_comparison)',
    )
    parser.add_argument(
        '--master_seed', type=int, default=8,
        help='主随机种子 (默认: 8)',
    )
    parser.add_argument(
        '--sigma', type=float, default=0.3,
        help='DAPS噪声水平 (默认: 0.3)',
    )
    parser.add_argument(
        '--gt_indices', type=int, nargs='+', default=None,
        help='要评估的GT样本索引列表 (默认: 0 10 20 30 40)',
    )
    parser.add_argument(
        '--n_samples', type=int, default=50,
        help='每个GT的采样次数 (默认: 50)',
    )
    parser.add_argument(
        '--sampling_method', type=str, default='rss',
        choices=['rss', 'gss_cached', 'gss_topg', 'gss_topg_light'],
        help='采样方法: rss / gss_cached(旧版组内top-k) / gss_topg(TopG+精细化) / gss_topg_light(TopG直出x0hat)',
    )
    parser.add_argument(
        '--seam_model_tag', type=str, default='seam_finetune',
        choices=['seam', 'seam_finetune'],
        help='SEAM侧模型标签（影响默认checkpoint与SM模型标签校验）',
    )
    parser.add_argument(
        '--gss_top_k', type=int, default=None,
        help='[gss_cached/gss_topg/gss_topg_light] 候选数；未设置时默认等于 --n_samples',
    )
    parser.add_argument(
        '--group_top_g', type=int, default=20,
        help='[gss_topg/gss_topg_light] 每个GT保留的Top-G分组数（默认: 20；<=0时回退为 gss_top_k）',
    )
    parser.add_argument(
        '--gss_light_eval_bs', type=int, default=32,
        help='[gss_topg_light] 预计算代表seed正演时的batch大小',
    )
    parser.add_argument(
        '--sm_path_seam', type=str, default=None,
        help='[gss_cached/gss_topg/gss_topg_light] SEAM模型对应的相似度矩阵文件路径',
    )
    parser.add_argument(
        '--sm_path_marmousi', type=str, default=None,
        help='[gss_cached/gss_topg/gss_topg_light] Marmousi模型对应的相似度矩阵文件路径',
    )
    parser.add_argument(
        '--seam_ckpt_dir', type=str, default=None,
        help='SEAM checkpoint目录（默认使用配置路径目录）',
    )
    parser.add_argument(
        '--seam_ckpt_file', type=str, default=None,
        help='SEAM checkpoint文件名或绝对路径（默认随 --seam_model_tag 变化）',
    )
    parser.add_argument(
        '--marmousi_ckpt_dir', type=str, default=None,
        help='Marmousi checkpoint目录（默认使用配置路径目录）',
    )
    parser.add_argument(
        '--marmousi_ckpt_file', type=str, default=None,
        help='Marmousi checkpoint文件名或绝对路径（默认marmousi_checkpoint_5.pth）',
    )
    parser.add_argument(
        '--eval_patches_path', type=str, default=None,
        help='可选评估集路径（.pt/.pkl）；提供后覆盖默认SEAM 224切片',
    )
    parser.add_argument(
        '--image_size', type=int, default=200,
        help='图像尺寸 (默认: 200)',
    )

    return parser


def main():
    """主入口。"""
    parser = build_parser()
    raw_argv = list(sys.argv[1:])
    if '--' in raw_argv:
        raw_argv = raw_argv[raw_argv.index('--') + 1:]
    if len(raw_argv) == 0:
        raw_argv = ['--mode', 'ood_comparison']
    elif '--mode' not in raw_argv:
        raw_argv = ['--mode', 'ood_comparison'] + raw_argv
    args, unknown = parser.parse_known_args(raw_argv)
    if unknown:
        print(f"[args][warn] 忽略未知参数: {unknown}")

    # 初始化配置
    cfg = FWIConfig()
    cfg.daps.sigma = args.sigma
    cfg.image_size = args.image_size

    print("=" * 60)
    print("sFWI OOD泛化实验")
    print("=" * 60)
    print(f"  模式:        {args.mode}")
    print(f"  主种子:      {args.master_seed}")
    print(f"  sigma:       {args.sigma}")
    print(f"  采样方法:    {args.sampling_method}")
    print(f"  SEAM标签:    {args.seam_model_tag}")
    print(f"  GT索引:      {args.gt_indices or '默认'}")
    print(f"  采样次数:    {args.n_samples}")
    if args.eval_patches_path:
        print(f"  eval_patches: {args.eval_patches_path}")
    if args.sampling_method in ('gss_cached', 'gss_topg', 'gss_topg_light'):
        print(f"  GSS top-k:   {args.gss_top_k if args.gss_top_k is not None else args.n_samples}")
        eff_group_top_g = args.group_top_g if int(args.group_top_g) > 0 else (args.gss_top_k if args.gss_top_k is not None else args.n_samples)
        print(f"  Group top-g: {eff_group_top_g}")
        print(f"  GSS light bs:{args.gss_light_eval_bs}")
        print(f"  SM(SEAM):    {args.sm_path_seam}")
        print(f"  SM(Marmousi):{args.sm_path_marmousi}")
        if not args.sm_path_seam:
            raise ValueError(f"sampling_method={args.sampling_method} 时必须提供 --sm_path_seam")
        if args.gss_light_eval_bs < 1:
            raise ValueError("--gss_light_eval_bs 必须 >= 1")
        if args.n_samples < 1:
            raise ValueError("--n_samples 必须 >= 1")
    elif args.n_samples < 1:
        raise ValueError("--n_samples 必须 >= 1")

    # 运行实验
    run_ood_comparison(cfg, args)

    print("\n" + "=" * 60)
    print("OOD实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
