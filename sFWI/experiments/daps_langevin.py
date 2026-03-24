"""
DAPS Langevin 机理诊断脚本（固定 GSS 候选 x0）。

目的:
  在固定模型、固定 GT、固定 x0 候选的前提下，展开 DAPS 的
  "reverse diffusion + Langevin + forward noise" 循环，逐步记录:
    - data/prior/total loss
    - data/prior/total 梯度范数
    - Langevin 漂移项与随机噪声项范数
    - 噪声/漂移、噪声/梯度比值

适用场景:
  用于判断当前 DAPS 设置是否处于 "噪声主导" 或 "梯度主导"，
  解释为什么输出更像 x0 的轻微扰动，还是会偏离到新结构。
"""

from __future__ import annotations

import sys
import os
import csv
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 设置 score_sde 路径
from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

from sFWI.config import FWIConfig, build_daps_configs
from sFWI.models.sde_setup import create_sde_config
from sFWI.models.score_model import NCSNpp_DAPS
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.utils.file_utils import generate_timestamped_filename

VELOCITY_CMAP = 'viridis'


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred4 = _to_4d(pred).float()
    gt4 = _to_4d(gt).float()
    if pred4.shape[-2:] != gt4.shape[-2:]:
        pred4 = F.interpolate(
            pred4,
            size=gt4.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    return pred4, gt4


def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    mse = F.mse_loss(pred4, gt4).item()
    rmse = np.sqrt(mse)
    gt_range = (gt4.max() - gt4.min()).item()
    if gt_range < 1e-8:
        return float('inf')
    return rmse / gt_range


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    mu_x = pred4.mean()
    mu_y = gt4.mean()
    sigma_x = pred4.var()
    sigma_y = gt4.var()
    sigma_xy = ((pred4 - mu_x) * (gt4 - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).item()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    nrmse = compute_nrmse(pred, gt)
    return 10.0 * np.log10(1.0 / (nrmse ** 2 + 1e-10))


def _find_all_seed_min_nrmse_x0hat(
    x0hat_batch: torch.Tensor,
    gt: torch.Tensor,
    eval_batch_size: int = 128,
):
    """
    在全 seed 的 x0hat 中，检索与 GT 的 NRMSE 最小样本。
    返回:
      {
        'seed': int,
        'nrmse': float,
        'x0': Tensor[1,C,H,W] (cpu)
      }
    """
    x0b = _to_4d(x0hat_batch).float()
    gt4 = _to_4d(gt).float().to(x0b.device)
    n = int(x0b.shape[0])
    if n < 1:
        raise RuntimeError("x0hat_batch 为空，无法检索全 seed 最小 NRMSE。")

    gt_range = float((gt4.max() - gt4.min()).item())
    if gt_range < 1e-8:
        return {
            'seed': 0,
            'nrmse': float('inf'),
            'x0': x0b[0:1].detach().cpu(),
        }

    bs = max(1, int(eval_batch_size))
    best_seed = 0
    best_nrmse = float('inf')

    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(n, start + bs)
            xb = x0b[start:end]
            if xb.shape[-2:] != gt4.shape[-2:]:
                xb = F.interpolate(
                    xb,
                    size=gt4.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            mse = torch.mean((xb - gt4) ** 2, dim=(1, 2, 3))
            nrmse = torch.sqrt(mse) / gt_range
            local_pos = int(torch.argmin(nrmse).item())
            local_val = float(nrmse[local_pos].item())
            if local_val < best_nrmse:
                best_nrmse = local_val
                best_seed = int(start + local_pos)

    return {
        'seed': int(best_seed),
        'nrmse': float(best_nrmse),
        'x0': x0b[best_seed:best_seed + 1].detach().cpu(),
    }


def compute_normalized_tv(x: torch.Tensor) -> float:
    x4 = _to_4d(x).float()
    dx = torch.abs(x4[:, :, 1:, :] - x4[:, :, :-1, :]).mean()
    dy = torch.abs(x4[:, :, :, 1:] - x4[:, :, :, :-1]).mean()
    denom = torch.abs(x4).mean().clamp_min(1e-8)
    return float(((dx + dy) / denom).item())


def compute_delta_frequency_ratios(delta: torch.Tensor, low_radius_ratio: float = 0.15):
    d4 = _to_4d(delta).float()
    if d4.shape[-2] < 4 or d4.shape[-1] < 4:
        return 0.0, 1.0, 0.0

    field = d4.mean(dim=(0, 1))
    spec = torch.fft.fftshift(torch.fft.fft2(field))
    power = (spec.real ** 2 + spec.imag ** 2)

    h, w = power.shape
    yy = torch.arange(h, device=power.device).view(-1, 1).expand(h, w)
    xx = torch.arange(w, device=power.device).view(1, -1).expand(h, w)
    cy, cx = h // 2, w // 2
    dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    r = max(1.0, float(low_radius_ratio) * float(min(h, w)))
    low_mask = dist <= r

    low_energy = power[low_mask].sum().item()
    high_energy = power[~low_mask].sum().item()
    total = max(1e-12, low_energy + high_energy)
    low_ratio = low_energy / total
    high_ratio = high_energy / total
    low_high_ratio = low_energy / max(1e-12, high_energy)
    return float(low_ratio), float(high_ratio), float(low_high_ratio)


def _save_rows_csv(path: str, rows, headers):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values):
    if not values:
        return float('nan')
    return float(sum(values) / len(values))


def _weighted_mean_std(values, weights):
    """返回加权均值与加权标准差（总体标准差）。"""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0 or w.size == 0:
        return float('nan'), float('nan')
    wsum = float(w.sum())
    if wsum <= 0:
        return float('nan'), float('nan')
    w = w / wsum
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    return mean, float(np.sqrt(max(0.0, var)))


def _resolve_checkpoint_path(cfg, args):
    default_by_tag = {
        'seam': 'checkpoint_5.pth',
        'seam_finetune': 'seam_finetune_checkpoint_5.pth',
        'marmousi': 'marmousi_checkpoint_5.pth',
    }
    ckpt_dir = args.ckpt_dir or os.path.join(cfg.paths.project_root, 'checkpoints')
    if args.ckpt_file:
        if os.path.isabs(args.ckpt_file):
            ckpt_path = args.ckpt_file
        else:
            ckpt_path = os.path.join(ckpt_dir, args.ckpt_file)
    else:
        ckpt_path = os.path.join(ckpt_dir, default_by_tag[args.model_tag])

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint不存在: {ckpt_path}")
    return ckpt_path


def _load_sm_info(sm_path, expected_model_tag=None):
    if not sm_path or not os.path.isfile(sm_path):
        raise FileNotFoundError(f"SM文件不存在: {sm_path}")
    sm_data = torch.load(sm_path, weights_only=False)
    required_keys = ['similarity_matrix', 'k', 'centroid_indices', 'x0hat_batch', 'd_centroids_2d']
    for key in required_keys:
        if key not in sm_data:
            raise KeyError(f"{sm_path} 缺少字段: {key}")

    model_tag = sm_data.get('model_tag')
    if expected_model_tag and model_tag and model_tag != expected_model_tag:
        raise ValueError(
            f"SM模型标签不匹配: expected={expected_model_tag}, actual={model_tag}, file={sm_path}"
        )
    return sm_data


def _load_eval_dataset(cfg, image_size=200, eval_patches_path=None):
    """
    加载评估域数据。

    - 默认: 读取 SEAM 原始切片（224 个）
    - 可选: 从 --eval_patches_path 读取外部评估集，支持:
        A) Marmousi pkl: seismic_dataset.pkl（自动提取 velocity）
        B) 直接保存的 Tensor: [N,H,W] 或 [N,1,H,W]
        C) dict，包含以下任一 key:
           test_v_patches / velocity_patches / patches / v_torch
      其中 B/C 推荐使用 .pt 文件。
    - 旧行为兼容: 仍支持从 torch.save 导出的 .pt 评估集读取。

    注:
      该函数只返回 velocity patches（用于构造 GT/measurement），
      不依赖 seismic 字段。

    详细格式（B/C）:
        1) 直接保存的 Tensor: [N,H,W] 或 [N,1,H,W]
        2) dict，包含以下任一 key:
           test_v_patches / velocity_patches / patches / v_torch
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
                    raise KeyError(
                        f"{eval_patches_path} 未找到可用张量字段，期望 keys: {candidate_keys}"
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
        return data, v_torch

    v_torch_seam = load_seam_model(cfg.paths.seam_model_path)
    data = create_velocity_dataset(v_torch_seam, image_size=image_size)
    return data, v_torch_seam


def _select_x0_candidate_from_sm(
    sm_info,
    measurement,
    operator,
    top_k=50,
    group_top_g=20,
    candidate_mode='gss_topk',
    generator=None,
):
    device = measurement.device
    sm = sm_info['similarity_matrix'].to(device)          # [k, n_seeds]
    x0hat_batch = sm_info['x0hat_batch'].to(device)       # [n_seeds, C, H, W]
    d_centroids_2d = sm_info['d_centroids_2d'].to(device) # [k, D]
    measurement_flat = measurement.reshape(1, -1)

    centroid_distances = torch.norm(d_centroids_2d - measurement_flat, dim=1)
    best_group = int(torch.argmin(centroid_distances).item())
    centroid_distance = float(centroid_distances[best_group].item())

    n_seeds = int(sm.shape[1])
    n_select = max(1, min(int(top_k), n_seeds))
    selected_group_indices = None
    candidate_distances = None
    if candidate_mode == 'random':
        if generator is None:
            generator = torch.Generator(device='cpu')
        perm = torch.randperm(n_seeds, generator=generator, device='cpu')[:n_select]
        top_indices = perm.to(device)
    elif candidate_mode == 'all_seed':
        top_indices = torch.arange(n_seeds, device=device, dtype=torch.long)
    elif candidate_mode == 'group_topg':
        n_groups = int(sm.shape[0])
        n_pick_groups = max(1, min(int(group_top_g), n_groups))

        centroid_seed_indices = sm_info.get('centroid_seed_indices', None)
        group_members = sm_info.get('group_members', None)
        rep_seeds = []
        rep_groups = []
        seen = set()

        def _get_rep_from_group_members(g_idx: int):
            if group_members is None:
                return None
            if isinstance(group_members, torch.Tensor):
                gm = group_members.detach().cpu().tolist()
            else:
                gm = group_members
            if not isinstance(gm, (list, tuple)):
                return None
            if g_idx >= len(gm):
                return None
            members = [int(v) for v in gm[g_idx] if 0 <= int(v) < n_seeds]
            if len(members) == 0:
                return None
            mem_t = torch.tensor(members, device=device, dtype=torch.long)
            local = sm[g_idx, mem_t]
            best_pos = int(torch.argmin(local).item())
            return int(members[best_pos])

        for g in range(n_groups):
            seed = None
            if centroid_seed_indices is not None:
                if isinstance(centroid_seed_indices, torch.Tensor):
                    csi = centroid_seed_indices.detach().cpu().tolist()
                else:
                    csi = centroid_seed_indices
                if isinstance(csi, (list, tuple)) and g < len(csi):
                    cand = int(csi[g])
                    if 0 <= cand < n_seeds:
                        seed = cand

            if seed is None:
                seed = _get_rep_from_group_members(g)

            if seed is None:
                # 兼容旧资产：退化为该 group 行全局最优 seed
                seed = int(torch.argmin(sm[g]).item())

            if seed not in seen:
                seen.add(seed)
                rep_seeds.append(seed)
                rep_groups.append(int(g))

        if len(rep_seeds) == 0:
            raise RuntimeError("group_topg 未找到有效 group 代表 seed。")

        rep_indices = torch.tensor(rep_seeds, device=device, dtype=torch.long)
        rep_group_indices = torch.tensor(rep_groups, device=device, dtype=torch.long)

        # 不再使用组级近似距离（centroid_dist），改为候选级真实物理误差 m0
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
        _, top_indices = torch.topk(sm[best_group], n_select, largest=False)
    if candidate_distances is None:
        if 'd_samples_2d' in sm_info:
            d_samples_2d = sm_info['d_samples_2d'].to(device)
            cand_flat = d_samples_2d[top_indices]
        else:
            candidates = x0hat_batch[top_indices]
            n_curr = int(top_indices.shape[0])
            candidates_high = F.interpolate(candidates, size=(128, 128), mode='bilinear', align_corners=True)
            with torch.no_grad():
                d_candidates = operator(candidates_high)
            cand_flat = d_candidates.reshape(n_curr, -1)
        candidate_distances = torch.norm(cand_flat - measurement_flat, dim=1)

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


def _build_all_seed_short_param(param: LangevinParam, args):
    """构建 all-seed 预筛选用的短程 Langevin 参数（其余超参沿用主参数）。"""
    return LangevinParam(
        langevin_lr=float(param.langevin_lr),
        langevin_steps=int(args.all_seed_short_langevin_steps),
        tau=float(param.tau),
        lr_min_ratio=float(param.lr_min_ratio),
        lambda_prior=float(param.lambda_prior),
        lambda_prior_min_ratio=float(param.lambda_prior_min_ratio),
        annealing_steps=int(args.all_seed_short_annealing_steps),
        sigma_max=float(param.sigma_max),
        sigma_min=float(param.sigma_min),
        sigma_final=float(param.sigma_final),
        beta_langevin_noise=float(param.beta_langevin_noise),
        beta_forward_noise=float(param.beta_forward_noise),
    )


def _all_seed_multistart_short_pick(
    model,
    operator,
    measurement: torch.Tensor,
    x0hat_batch: torch.Tensor,
    top_indices: torch.Tensor,
    top_candidate_distances: torch.Tensor,
    short_param: LangevinParam,
    top_n: int,
    seed_base: int,
):
    """
    all-seed 候选下的多起点短程筛选：
      - 按 d_obs 距离选 top-N
      - 每个候选跑短程 DAPS
      - 以短程后的数据域 misfit 选最优
    """
    device = measurement.device
    if top_indices.numel() == 0:
        raise RuntimeError("all_seed 候选为空，无法执行多起点筛选。")

    n_pool = max(1, min(int(top_n), int(top_indices.numel())))
    order = torch.argsort(top_candidate_distances, dim=0)
    pool_pos = order[:n_pool]
    pool_indices = top_indices[pool_pos]

    best_seed = int(pool_indices[0].item())
    best_x = x0hat_batch[best_seed].unsqueeze(0).detach()
    with torch.no_grad():
        best_misfit = float(torch.norm(operator(best_x) - measurement).item())

    for rank, seed_idx in enumerate(pool_indices.tolist()):
        run_seed = int(seed_base + rank * 10007 + seed_idx)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        init_x0 = x0hat_batch[int(seed_idx)].unsqueeze(0)
        diag = _run_langevin_diagnostics(
            model=model,
            operator=operator,
            measurement=measurement,
            x0=init_x0,
            param=short_param,
            return_trajectory=False,
        )
        x_short = diag['final_x0hat'].detach()
        with torch.no_grad():
            misfit = float(torch.norm(operator(x_short) - measurement).item())

        if misfit < best_misfit:
            best_misfit = misfit
            best_seed = int(seed_idx)
            best_x = x_short

    return best_x.to(device), {
        'short_top_n': int(n_pool),
        'short_best_seed': int(best_seed),
        'short_best_misfit': float(best_misfit),
    }


def _build_run_seed(master_seed: int, gt_idx: int, repeat_id: int, seed_mode: str) -> int:
    if seed_mode == 'shared_across_repeats':
        return int(master_seed + gt_idx * 1009)
    return int(master_seed + gt_idx * 1009 + repeat_id)


def _fmt_float_name_token(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace('-', 'm').replace('.', 'p')


def _uniform_indices(total: int, n_select: int):
    if total <= 0:
        return []
    n = max(1, min(int(n_select), int(total)))
    idx = np.linspace(0, total - 1, num=n, dtype=int).tolist()
    uniq = []
    for i in idx:
        if not uniq or i != uniq[-1]:
            uniq.append(i)
    return uniq


@dataclass
class LangevinParam:
    langevin_lr: float
    langevin_steps: int
    tau: float
    lr_min_ratio: float
    lambda_prior: float
    lambda_prior_min_ratio: float
    annealing_steps: int
    sigma_max: float
    sigma_min: float
    sigma_final: float
    beta_langevin_noise: float
    beta_forward_noise: float


def _run_langevin_diagnostics(
    model,
    operator,
    measurement: torch.Tensor,
    x0: torch.Tensor,
    param: LangevinParam,
    sample_seeds=None,
    return_trajectory: bool = True,
):
    """
    展开 DAPS 迭代并返回逐步诊断日志。

    返回:
      dict 包含:
        - final_x0hat, final_x0y, final_xt
        - inner_rows: 每个 outer/inner step 的日志
        - outer_rows: 每个 outer step 聚合日志
        - trajectory: 快照（可选）
    """
    from DAPS.sampler import Scheduler, DiffusionSampler, LangevinDynamics

    base_annealing = model.daps.annealing_scheduler
    annealing = Scheduler(
        num_steps=param.annealing_steps,
        sigma_max=param.sigma_max,
        sigma_min=param.sigma_min,
        sigma_final=param.sigma_final,
        schedule=base_annealing.schedule,
        timestep=base_annealing.timestep,
    )
    lgvd = LangevinDynamics(
        num_steps=param.langevin_steps,
        lr=param.langevin_lr,
        tau=param.tau,
        lr_min_ratio=param.lr_min_ratio,
        lambda_prior=param.lambda_prior,
        lambda_prior_min_ratio=param.lambda_prior_min_ratio,
    )

    inner_rows = []
    outer_rows = []
    trajectory = []

    xt = x0.detach().clone()
    batch_size = int(xt.shape[0])
    if sample_seeds is not None:
        if len(sample_seeds) != batch_size:
            raise ValueError("sample_seeds 长度必须与 batch 大小一致")
        noise_generators = []
        gen_device = 'cuda' if xt.device.type == 'cuda' else 'cpu'
        for seed in sample_seeds:
            g = torch.Generator(device=gen_device)
            g.manual_seed(int(seed))
            noise_generators.append(g)
    else:
        noise_generators = None
    final_x0hat = xt.detach().clone()
    final_x0y = xt.detach().clone()

    for outer_step in range(annealing.num_steps):
        sigma = float(annealing.sigma_steps[outer_step])
        ratio = float(outer_step) / float(max(1, annealing.num_steps))
        lr = float(lgvd.get_lr(ratio))
        lambda_prior = float(lgvd.get_lambda_prior(ratio))
        sigma_eff = max(float(sigma), 1e-8)

        # 1) reverse diffusion
        diff_scheduler = Scheduler(**model.daps.diffusion_scheduler_config, sigma_max=sigma)
        diff_sampler = DiffusionSampler(diff_scheduler)
        with torch.no_grad():
            x0hat = diff_sampler.sample(model, xt, SDE=False, verbose=False)
        x0hat_detach = x0hat.detach()

        # 2) Langevin dynamics（手动展开，便于记录各项量级）
        x = x0hat_detach
        per_outer = []

        for inner_step in range(param.langevin_steps):
            x = x.detach().requires_grad_(True)
            pred = operator(x)
            err = pred - measurement
            err_sq_vec = (err.reshape(err.shape[0], -1) ** 2).sum(dim=1)
            data_loss_vec = err_sq_vec / (2.0 * (param.tau ** 2))
            data_loss = data_loss_vec.sum()

            prior_diff = x - x0hat_detach
            prior_sq_vec = (prior_diff.reshape(prior_diff.shape[0], -1) ** 2).sum(dim=1)
            prior_loss_vec = lambda_prior * prior_sq_vec / (2.0 * (sigma_eff ** 2))
            prior_loss = prior_loss_vec.sum()
            total_loss = data_loss + prior_loss

            grad_data = torch.autograd.grad(data_loss, x, retain_graph=True)[0]
            grad_prior = torch.autograd.grad(prior_loss, x, retain_graph=False)[0]
            grad_total = grad_data + grad_prior

            grad_data_norm = float(torch.norm(grad_data).item())
            grad_prior_norm = float(torch.norm(grad_prior).item())
            grad_total_norm = float(torch.norm(grad_total).item())

            grad_dot = float((grad_data * grad_prior).sum().item())
            grad_cos = grad_dot / max(1e-12, grad_data_norm * grad_prior_norm)

            with torch.no_grad():
                x_before = x.detach()
                drift = -lr * grad_total
                drift_norm = float(torch.norm(drift).item())

                if noise_generators is None:
                    epsilon = torch.randn_like(x_before)
                else:
                    eps_list = [
                        torch.randn(
                            (1,) + tuple(x_before.shape[1:]),
                            generator=noise_generators[i],
                            device=x_before.device,
                            dtype=x_before.dtype,
                        )
                        for i in range(batch_size)
                    ]
                    epsilon = torch.cat(eps_list, dim=0)
                noise = param.beta_langevin_noise * np.sqrt(2.0 * lr) * epsilon
                noise_norm = float(torch.norm(noise).item())

                x_after = x_before + drift + noise

            data_loss_mean = float(data_loss_vec.mean().item())
            prior_loss_mean = float(prior_loss_vec.mean().item())
            total_loss_mean = float((data_loss_vec + prior_loss_vec).mean().item())
            meas_misfit_mean = float(torch.sqrt(torch.clamp(err_sq_vec, min=0.0)).mean().item())
            row = {
                'outer_step': int(outer_step),
                'inner_step': int(inner_step),
                'sigma': float(sigma),
                'ratio': float(ratio),
                'langevin_lr_effective': float(lr),
                'lambda_prior_effective': float(lambda_prior),
                'tau': float(param.tau),
                'data_loss': data_loss_mean,
                'prior_loss': prior_loss_mean,
                'total_loss': total_loss_mean,
                'prior_over_data_loss': float(prior_loss_mean / max(1e-12, data_loss_mean)),
                'measurement_misfit': meas_misfit_mean,
                'grad_data_norm': grad_data_norm,
                'grad_prior_norm': grad_prior_norm,
                'grad_total_norm': grad_total_norm,
                'grad_data_prior_cos': float(grad_cos),
                'drift_norm': float(drift_norm),
                'noise_norm': float(noise_norm),
                'noise_to_drift_ratio': float(noise_norm / max(1e-12, drift_norm)),
                'noise_to_grad_ratio': float(noise_norm / max(1e-12, grad_total_norm)),
            }
            inner_rows.append(row)
            per_outer.append(row)

            x = x_after

            if torch.isnan(x).any():
                # 与原实现一致：出现 NaN 则退化为全0，提前退出
                x = torch.zeros_like(x)
                break

        x0y = x.detach()
        next_sigma = float(annealing.sigma_steps[outer_step + 1])
        with torch.no_grad():
            if noise_generators is None:
                fw_eps = torch.randn_like(x0y)
            else:
                fw_list = [
                    torch.randn(
                        (1,) + tuple(x0y.shape[1:]),
                        generator=noise_generators[i],
                        device=x0y.device,
                        dtype=x0y.dtype,
                    )
                    for i in range(batch_size)
                ]
                fw_eps = torch.cat(fw_list, dim=0)
            forward_noise = param.beta_forward_noise * next_sigma * fw_eps
            xt = x0y + forward_noise

            misfit_x0hat = torch.norm((operator(x0hat_detach) - measurement).reshape(batch_size, -1), dim=1).mean().item()
            misfit_x0y = torch.norm((operator(x0y) - measurement).reshape(batch_size, -1), dim=1).mean().item()

        outer_rows.append({
            'outer_step': int(outer_step),
            'sigma': float(sigma),
            'next_sigma': float(next_sigma),
            'langevin_lr_effective': float(lr),
            'lambda_prior_effective': float(lambda_prior),
            'mean_data_loss': _mean([r['data_loss'] for r in per_outer]),
            'mean_prior_loss': _mean([r['prior_loss'] for r in per_outer]),
            'mean_total_loss': _mean([r['total_loss'] for r in per_outer]),
            'mean_prior_over_data_loss': _mean([r['prior_over_data_loss'] for r in per_outer]),
            'mean_measurement_misfit': _mean([r['measurement_misfit'] for r in per_outer]),
            'mean_grad_data_norm': _mean([r['grad_data_norm'] for r in per_outer]),
            'mean_grad_prior_norm': _mean([r['grad_prior_norm'] for r in per_outer]),
            'mean_grad_total_norm': _mean([r['grad_total_norm'] for r in per_outer]),
            'mean_drift_norm': _mean([r['drift_norm'] for r in per_outer]),
            'mean_noise_norm': _mean([r['noise_norm'] for r in per_outer]),
            'mean_noise_to_drift_ratio': _mean([r['noise_to_drift_ratio'] for r in per_outer]),
            'mean_noise_to_grad_ratio': _mean([r['noise_to_grad_ratio'] for r in per_outer]),
            'misfit_x0hat': float(misfit_x0hat),
            'misfit_x0y': float(misfit_x0y),
        })

        if return_trajectory:
            trajectory.append({
                'outer_step': int(outer_step),
                'sigma': float(sigma),
                'next_sigma': float(next_sigma),
                'x0hat': x0hat_detach.cpu(),
                'x0y': x0y.detach().cpu(),
                'xt': xt.detach().cpu(),
            })

        final_x0hat = x0hat_detach
        final_x0y = x0y

    return {
        'final_x0hat': final_x0hat.detach(),
        'final_x0y': final_x0y.detach(),
        'final_xt': xt.detach(),
        'inner_rows': inner_rows,
        'outer_rows': outer_rows,
        'trajectory': trajectory,
    }


def _compute_final_metrics(
    pred: torch.Tensor,
    x0: torch.Tensor,
    gt: torch.Tensor,
    measurement: torch.Tensor,
    operator,
    low_freq_radius_ratio: float,
    min_delta_rel_l2: float,
    max_misfit_ratio: float,
    max_high_freq_ratio: float,
    misfit_before: float,
):
    delta = pred - x0
    delta_l2 = torch.norm(delta).item()
    delta_rel_l2 = delta_l2 / max(1e-8, torch.norm(x0).item())
    delta_nrmse = compute_nrmse(pred, x0)
    delta_ssim = compute_ssim(pred, x0)
    delta_tv = compute_normalized_tv(delta)
    low_ratio, high_ratio, low_high_ratio = compute_delta_frequency_ratios(
        delta, low_radius_ratio=low_freq_radius_ratio
    )

    gt_nrmse = compute_nrmse(pred, gt)
    gt_ssim = compute_ssim(pred, gt)
    gt_psnr = compute_psnr(pred, gt)

    with torch.no_grad():
        misfit_after = torch.norm(operator(pred) - measurement).item()
    misfit_ratio = misfit_after / max(1e-8, misfit_before)

    is_structural_variant = int(
        delta_rel_l2 >= min_delta_rel_l2
        and misfit_ratio <= max_misfit_ratio
        and high_ratio <= max_high_freq_ratio
    )

    return {
        'delta_l2': float(delta_l2),
        'delta_rel_l2': float(delta_rel_l2),
        'delta_nrmse': float(delta_nrmse),
        'delta_ssim': float(delta_ssim),
        'delta_tv': float(delta_tv),
        'delta_low_freq_ratio': float(low_ratio),
        'delta_high_freq_ratio': float(high_ratio),
        'delta_low_high_ratio': float(low_high_ratio),
        'gt_nrmse': float(gt_nrmse),
        'gt_ssim': float(gt_ssim),
        'gt_psnr': float(gt_psnr),
        'meas_misfit_after': float(misfit_after),
        'misfit_ratio': float(misfit_ratio),
        'is_structural_variant': int(is_structural_variant),
    }


def _save_inner_curve_plot(
    output_dir: str,
    model_tag: str,
    param: LangevinParam,
    gt_idx: int,
    repeat_id: int,
    inner_rows,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[diag_plot] matplotlib 不可用，跳过曲线图: {e}")
        return None

    if not inner_rows:
        return None

    steps = np.arange(len(inner_rows))
    data_loss = np.array([r['data_loss'] for r in inner_rows], dtype=np.float64)
    prior_loss = np.array([r['prior_loss'] for r in inner_rows], dtype=np.float64)
    total_loss = np.array([r['total_loss'] for r in inner_rows], dtype=np.float64)
    grad_data = np.array([r['grad_data_norm'] for r in inner_rows], dtype=np.float64)
    grad_prior = np.array([r['grad_prior_norm'] for r in inner_rows], dtype=np.float64)
    grad_total = np.array([r['grad_total_norm'] for r in inner_rows], dtype=np.float64)
    drift = np.array([r['drift_norm'] for r in inner_rows], dtype=np.float64)
    noise = np.array([r['noise_norm'] for r in inner_rows], dtype=np.float64)
    r_nd = np.array([r['noise_to_drift_ratio'] for r in inner_rows], dtype=np.float64)
    r_ng = np.array([r['noise_to_grad_ratio'] for r in inner_rows], dtype=np.float64)
    lrs = np.array([r['langevin_lr_effective'] for r in inner_rows], dtype=np.float64)
    lpriors = np.array([r['lambda_prior_effective'] for r in inner_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(steps, data_loss, label='data_loss')
    ax.plot(steps, prior_loss, label='prior_loss')
    ax.plot(steps, total_loss, label='total_loss')
    ax.set_title('Loss Terms')
    ax.set_xlabel('inner global step')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, grad_data, label='||g_data||')
    ax.plot(steps, grad_prior, label='||g_prior||')
    ax.plot(steps, grad_total, label='||g_total||')
    ax.set_title('Gradient Norms')
    ax.set_xlabel('inner global step')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, drift, label='||drift||')
    ax.plot(steps, noise, label='||noise||')
    ax.set_title('Drift vs Langevin Noise')
    ax.set_xlabel('inner global step')
    ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, r_nd, label='noise/drift')
    ax.plot(steps, r_ng, label='noise/grad')
    ax2 = ax.twinx()
    ax2.plot(steps, lrs, '--', alpha=0.5, label='lr', color='tab:gray')
    ax2.plot(steps, lpriors, ':', alpha=0.6, label='lambda_prior', color='tab:orange')
    ax.set_title('Noise Ratios + Effective Schedules')
    ax.set_xlabel('inner global step')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    for a in axes.flat:
        for k in range(1, int(param.annealing_steps)):
            xline = k * int(param.langevin_steps) - 0.5
            a.axvline(xline, color='k', alpha=0.08, linewidth=0.8)

    fig.suptitle(
        f"DAPS Langevin Diagnose | {model_tag} | GT#{gt_idx} r{repeat_id} | "
        f"tau={param.tau:.4g}, an={param.annealing_steps}, ls={param.langevin_steps}, "
        f"smax={param.sigma_max:.4g}, smin={param.sigma_min:.4g}, sfin={param.sigma_final:.4g}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    base = (
        f"daps_langevin_curves_model-{model_tag}"
        f"_grid_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_grid_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_grid_tau-{_fmt_float_name_token(param.tau)}"
        f"_grid_annealing_steps-{int(param.annealing_steps)}"
        f"_grid_langevin_steps-{int(param.langevin_steps)}"
        f"_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.png'))
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def _save_trajectory_plot(
    output_dir: str,
    model_tag: str,
    param: LangevinParam,
    gt_idx: int,
    repeat_id: int,
    gt: torch.Tensor,
    x0: torch.Tensor,
    trajectory,
    num_snapshots: int = 10,
    mode: str = 'triplet',
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[traj_plot] matplotlib 不可用，跳过轨迹图: {e}")
        return None

    if not trajectory:
        return None

    gt4 = _to_4d(gt).float().cpu()
    x04, _ = _align_pred_to_gt(_to_4d(x0).float().cpu(), gt4)

    panels = [('GT', gt4[0, 0].numpy()), ('x0 input', x04[0, 0].numpy())]
    sel_idx = _uniform_indices(len(trajectory), num_snapshots)
    for idx in sel_idx:
        snap = trajectory[idx]
        step = int(snap['outer_step']) + 1
        sigma = float(snap['sigma'])
        if mode == 'x0hat':
            p4, _ = _align_pred_to_gt(_to_4d(snap['x0hat']).float().cpu(), gt4)
            panels.append((f"s{step}/{len(trajectory)} x0hat\nsigma={sigma:.4g}", p4[0, 0].numpy()))
        else:
            for key in ['x0hat', 'x0y', 'xt']:
                p4, _ = _align_pred_to_gt(_to_4d(snap[key]).float().cpu(), gt4)
                panels.append((f"s{step}/{len(trajectory)} {key}\nsigma={sigma:.4g}", p4[0, 0].numpy()))

    n = len(panels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for i, (title, img) in enumerate(panels):
        ax = axes.flat[i]
        ax.imshow(img, cmap=VELOCITY_CMAP, aspect='auto')
        ax.set_title(title, fontsize=8.5)

    fig.suptitle(
        f"DAPS Trajectory ({mode}) | {model_tag} | GT#{gt_idx} r{repeat_id} | "
        f"tau={param.tau:.4g}, an={param.annealing_steps}, ls={param.langevin_steps}, "
        f"smax={param.sigma_max:.4g}, smin={param.sigma_min:.4g}, sfin={param.sigma_final:.4g}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    base = (
        f"daps_langevin_traj_model-{model_tag}"
        f"_grid_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_grid_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_grid_tau-{_fmt_float_name_token(param.tau)}"
        f"_grid_annealing_steps-{int(param.annealing_steps)}"
        f"_grid_langevin_steps-{int(param.langevin_steps)}"
        f"_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.png'))
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def _save_ensemble_panel_plot(
    output_dir: str,
    model_tag: str,
    param: LangevinParam,
    gt_idx: int,
    repeat_id: int,
    gt: torch.Tensor,
    panel_rows,
    all_seed_best=None,
):
    """
    可视化 ensemble 候选：
      - 三行
      - 每行最左侧是 GT
      - 第一行: x0_hat（初始候选）
      - 第二行: DAPS 后（final x0hat）
      - 第三行左侧: 全 seed 中与 GT NRMSE 最小的 x0hat
      - 从左到右按初始数据域 misfit 升序
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ensemble_panel] matplotlib 不可用，跳过可视化: {e}")
        return None

    if not panel_rows:
        return None

    gt4 = _to_4d(gt).float().cpu()
    rows_sorted = sorted(panel_rows, key=lambda x: float(x['misfit_before']))
    n = len(rows_sorted)
    ncols = n + 1

    fig_w = max(10.0, 2.5 * ncols)
    fig_h = 9.0
    fig, axes = plt.subplots(3, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for ax in axes.flat:
        ax.axis('off')

    gt_np = gt4[0, 0].numpy()
    axes[0, 0].imshow(gt_np, cmap=VELOCITY_CMAP, aspect='auto')
    axes[0, 0].set_title('GT', fontsize=10)
    axes[1, 0].imshow(gt_np, cmap=VELOCITY_CMAP, aspect='auto')
    axes[1, 0].set_title('GT', fontsize=10)
    if all_seed_best is not None:
        best_x0, _ = _align_pred_to_gt(_to_4d(all_seed_best['x0']).float().cpu(), gt4)
        axes[2, 0].imshow(best_x0[0, 0].numpy(), cmap=VELOCITY_CMAP, aspect='auto')
        axes[2, 0].set_title(
            f"all-seed min NRMSE\nseed={int(all_seed_best['seed'])}, nrmse={float(all_seed_best['nrmse']):.4f}",
            fontsize=8.5
        )
    else:
        axes[2, 0].imshow(gt_np, cmap=VELOCITY_CMAP, aspect='auto')
        axes[2, 0].set_title("all-seed min NRMSE\nN/A", fontsize=8.5)

    for i, row in enumerate(rows_sorted, start=1):
        x0_img, _ = _align_pred_to_gt(_to_4d(row['x0']).float().cpu(), gt4)
        pred_img, _ = _align_pred_to_gt(_to_4d(row['pred_after']).float().cpu(), gt4)
        axes[0, i].imshow(x0_img[0, 0].numpy(), cmap=VELOCITY_CMAP, aspect='auto')
        axes[1, i].imshow(pred_img[0, 0].numpy(), cmap=VELOCITY_CMAP, aspect='auto')
        axes[2, i].imshow(x0_img[0, 0].numpy(), cmap=VELOCITY_CMAP, aspect='auto')

        axes[0, i].set_title(
            f"seed={int(row['seed'])}\nmisfit={float(row['misfit_before']):.4f}",
            fontsize=8.5
        )
        axes[1, i].set_title(
            f"after={float(row['misfit_after']):.4f}\nΔNRMSE={float(row['nrmse_improve']):+.4f}",
            fontsize=8.5
        )
        axes[2, i].set_title(
            f"x0 NRMSE={float(row.get('x0_gt_nrmse', float('nan'))):.4f}",
            fontsize=8.5
        )

    fig.suptitle(
        f"Ensemble Candidates | {model_tag} | GT#{gt_idx} r{repeat_id} | "
        f"sorted by init data-misfit | tau={param.tau:.4g}, an={param.annealing_steps}, "
        f"ls={param.langevin_steps}, smax={param.sigma_max:.4g}, smin={param.sigma_min:.4g}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = (
        f"daps_langevin_ensemble_panel_model-{model_tag}"
        f"_grid_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_grid_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_grid_tau-{_fmt_float_name_token(param.tau)}"
        f"_grid_annealing_steps-{int(param.annealing_steps)}"
        f"_grid_langevin_steps-{int(param.langevin_steps)}"
        f"_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.png'))
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='DAPS Langevin 机理诊断（固定 GSS 候选 x0）')
    parser.add_argument('--model_tag', type=str, default='seam_finetune',
                        choices=['seam', 'seam_finetune', 'marmousi'])
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--sm_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--eval_patches_path', type=str, default=None,
                        help='可选评估集路径（.pt/.pkl；提供后覆盖默认 SEAM 224 切片）')
    parser.add_argument('--gt_indices', type=int, nargs='+', default=[7])
    parser.add_argument('--candidate_mode', type=str, default='gss_topk',
                        choices=['gss_topk', 'all_seed', 'group_topg', 'random'])
    parser.add_argument('--gss_top_k', type=int, default=50)
    parser.add_argument('--group_top_g', type=int, default=20,
                        help='[group_topg] 每个 GT 选取 top-g 个 group，每组 1 个代表 seed')
    parser.add_argument('--all_seed_short_top_n', type=int, default=20,
                        help='[all_seed] 预筛选时按 d_obs 距离取前 N 个 seed 做短程 DAPS')
    parser.add_argument('--all_seed_short_annealing_steps', type=int, default=6,
                        help='[all_seed] 预筛选短程 DAPS 的 annealing 步数')
    parser.add_argument('--all_seed_short_langevin_steps', type=int, default=4,
                        help='[all_seed] 预筛选短程 DAPS 的 Langevin 步数')
    parser.add_argument('--ensemble_mode', action='store_true',
                        help='启用多候选并行 DAPS（UQ/多解）模式；支持 candidate_mode=all_seed/group_topg')
    parser.add_argument('--ensemble_top_n', type=int, default=20,
                        help='[ensemble_mode] 每个 GT 取前 N 个候选并行跑完整 DAPS')
    parser.add_argument('--candidate_batch_size', type=int, default=1,
                        help='[ensemble_mode] 候选并行批大小，1 表示串行')
    parser.add_argument('--force_serial_eval', action='store_true',
                        help='[ensemble_mode] 强制逐候选评估（覆盖 candidate_batch_size）')
    parser.add_argument('--ensemble_weight_temp', type=float, default=0.1,
                        help='[ensemble_mode] 按 misfit 加权时的温度系数 T')
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--master_seed', type=int, default=8)
    parser.add_argument('--seed_mode', type=str, default='repeat_dependent',
                        choices=['shared_across_repeats', 'repeat_dependent'])
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='正演算子噪声 sigma')

    # DAPS 参数（单组）
    parser.add_argument('--langevin_lr', type=float, default=3e-4)
    parser.add_argument('--langevin_steps', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lr_min_ratio', type=float, default=1.0)
    parser.add_argument('--lambda_prior', type=float, default=1.0)
    parser.add_argument('--lambda_prior_min_ratio', type=float, default=1.0)
    parser.add_argument('--annealing_steps', type=int, default=20)
    parser.add_argument('--sigma_max', type=float, default=0.3)
    parser.add_argument('--sigma_min', type=float, default=0.01)
    parser.add_argument('--sigma_final', type=float, default=0.0)
    parser.add_argument('--beta_langevin_noise', type=float, default=1.0,
                        help='Langevin inner 噪声缩放系数')
    parser.add_argument('--beta_forward_noise', type=float, default=1.0,
                        help='Outer step forward-noise 缩放系数')

    # 门控（后验标签）
    parser.add_argument('--low_freq_radius_ratio', type=float, default=0.15)
    parser.add_argument('--min_delta_rel_l2', type=float, default=0.02)
    parser.add_argument('--max_misfit_ratio', type=float, default=1.05)
    parser.add_argument('--max_high_freq_ratio', type=float, default=0.75)

    # 可视化
    parser.add_argument('--skip_curve_vis', action='store_true')
    parser.add_argument('--skip_trajectory_vis', action='store_true')
    parser.add_argument('--skip_ensemble_panel_vis', action='store_true',
                        help='[ensemble_mode] 跳过三行候选对比图（x0_hat / daps后 / all-seed最小NRMSE）')
    parser.add_argument('--trajectory_mode', type=str, default='triplet',
                        choices=['triplet', 'x0hat'])
    parser.add_argument('--vis_num_snapshots', type=int, default=10)
    parser.add_argument('--vis_max_runs', type=int, default=0,
                        help='最多保存多少个 run 的图，<=0 表示不限制')

    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n_repeats 必须 >= 1")
    if args.all_seed_short_top_n < 1:
        raise ValueError("--all_seed_short_top_n 必须 >= 1")
    if args.all_seed_short_annealing_steps < 1 or args.all_seed_short_langevin_steps < 1:
        raise ValueError("--all_seed_short_annealing_steps / --all_seed_short_langevin_steps 必须 >= 1")
    if args.group_top_g < 1:
        raise ValueError("--group_top_g 必须 >= 1")
    if args.ensemble_top_n < 1:
        raise ValueError("--ensemble_top_n 必须 >= 1")
    if args.candidate_batch_size < 1:
        raise ValueError("--candidate_batch_size 必须 >= 1")
    if args.ensemble_weight_temp <= 0:
        raise ValueError("--ensemble_weight_temp 必须 > 0")
    if args.ensemble_mode and args.candidate_mode not in ('all_seed', 'group_topg'):
        raise ValueError("--ensemble_mode 目前仅支持 candidate_mode=all_seed/group_topg")
    if args.langevin_steps < 1 or args.annealing_steps < 1:
        raise ValueError("langevin_steps / annealing_steps 必须 >= 1")
    if args.langevin_lr <= 0:
        raise ValueError("--langevin_lr 必须 > 0")
    if args.tau <= 0:
        raise ValueError("--tau 必须 > 0")
    if args.lambda_prior < 0:
        raise ValueError("--lambda_prior 必须 >= 0")
    if not (0.0 <= args.lambda_prior_min_ratio <= 1.0):
        raise ValueError("--lambda_prior_min_ratio 必须在 [0, 1] 内")
    if args.beta_langevin_noise < 0 or args.beta_forward_noise < 0:
        raise ValueError("--beta_langevin_noise / --beta_forward_noise 必须 >= 0")
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        raise ValueError("需要满足 sigma_max > sigma_min > 0")
    if args.sigma_final < 0 or args.sigma_final > args.sigma_min:
        raise ValueError("需要满足 0 <= sigma_final <= sigma_min")
    if not (0.0 < args.low_freq_radius_ratio < 1.0):
        raise ValueError("--low_freq_radius_ratio 必须在 (0, 1) 内")
    if args.max_misfit_ratio <= 0:
        raise ValueError("--max_misfit_ratio 必须 > 0")
    if not (0.0 <= args.max_high_freq_ratio <= 1.0):
        raise ValueError("--max_high_freq_ratio 必须在 [0, 1] 内")
    if args.vis_num_snapshots < 1:
        raise ValueError("--vis_num_snapshots 必须 >= 1")

    cfg = FWIConfig()
    cfg.daps.sigma = args.sigma
    device = cfg.device

    ckpt_path = _resolve_checkpoint_path(cfg, args)
    sm_info = _load_sm_info(args.sm_path, expected_model_tag=args.model_tag)
    output_dir = args.output_dir or cfg.paths.output_path
    os.makedirs(output_dir, exist_ok=True)

    param = LangevinParam(
        langevin_lr=float(args.langevin_lr),
        langevin_steps=int(args.langevin_steps),
        tau=float(args.tau),
        lr_min_ratio=float(args.lr_min_ratio),
        lambda_prior=float(args.lambda_prior),
        lambda_prior_min_ratio=float(args.lambda_prior_min_ratio),
        annealing_steps=int(args.annealing_steps),
        sigma_max=float(args.sigma_max),
        sigma_min=float(args.sigma_min),
        sigma_final=float(args.sigma_final),
        beta_langevin_noise=float(args.beta_langevin_noise),
        beta_forward_noise=float(args.beta_forward_noise),
    )

    print("=" * 72)
    print("DAPS Langevin 机理诊断（固定 GSS 候选 x0）")
    print("=" * 72)
    print(f"  model_tag: {args.model_tag}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  SM: {args.sm_path}")
    if args.eval_patches_path:
        print(f"  eval_patches_path: {args.eval_patches_path}")
    print(f"  device: {device}")
    print(f"  gt_indices: {args.gt_indices}")
    print(f"  candidate_mode: {args.candidate_mode}, gss_top_k={args.gss_top_k}")
    eff_candidate_batch_size = 1
    if args.ensemble_mode:
        eff_candidate_batch_size = 1 if args.force_serial_eval else int(args.candidate_batch_size)
        print(
            "  ensemble_mode: "
            f"top_n={args.ensemble_top_n}, candidate_batch_size={eff_candidate_batch_size}, "
            f"force_serial_eval={args.force_serial_eval}, "
            f"weight_temp={args.ensemble_weight_temp}"
        )
        if args.candidate_mode == 'group_topg':
            print(f"  group_top_g: {args.group_top_g}, group_score=m0(real operator misfit)")
    elif args.candidate_mode == 'all_seed':
        print(
            "  all_seed_multistart: "
            f"top_n={args.all_seed_short_top_n}, "
            f"short_annealing_steps={args.all_seed_short_annealing_steps}, "
            f"short_langevin_steps={args.all_seed_short_langevin_steps}"
        )
    elif args.candidate_mode == 'group_topg':
        print(f"  group_top_g: {args.group_top_g}, group_score=m0(real operator misfit)")
    print(f"  n_repeats: {args.n_repeats}, seed_mode: {args.seed_mode}, master_seed={args.master_seed}")
    print(
        "  daps: "
        f"lr={param.langevin_lr}, ls={param.langevin_steps}, tau={param.tau}, "
        f"lpr={param.lambda_prior}, lpr_min_ratio={param.lambda_prior_min_ratio}, "
        f"an={param.annealing_steps}, smax={param.sigma_max}, smin={param.sigma_min}, sfin={param.sigma_final}, "
        f"beta_lg_noise={param.beta_langevin_noise}, beta_fw_noise={param.beta_forward_noise}"
    )
    print(
        "  gate(label-only): "
        f"delta_rel_l2>={args.min_delta_rel_l2}, misfit_ratio<={args.max_misfit_ratio}, "
        f"high_freq_ratio<={args.max_high_freq_ratio}, low_freq_radius_ratio={args.low_freq_radius_ratio}"
    )

    if args.eval_patches_path:
        print("\n加载外部评估域数据...")
    else:
        print("\n加载 SEAM 评估域数据...")
    seam_data, _ = _load_eval_dataset(
        cfg,
        image_size=200,
        eval_patches_path=args.eval_patches_path,
    )
    print(f"  dataset_size: {len(seam_data)}")

    print("\n加载模型...")
    config, _ = create_sde_config(parent_dir, batch_size=1)
    base_config, lgvd_config = build_daps_configs(cfg)
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=ckpt_path
    )
    model.set_device(device)
    print("  model loaded.")

    all_seed_short_param = None
    sm_x0hat_batch = sm_info['x0hat_batch'].to(device)
    if args.candidate_mode == 'all_seed' and (not args.ensemble_mode):
        all_seed_short_param = _build_all_seed_short_param(param, args)
        print(
            "  all-seed preselect short-DAPS param: "
            f"lr={all_seed_short_param.langevin_lr:.4g}, "
            f"tau={all_seed_short_param.tau:.4g}, "
            f"an={all_seed_short_param.annealing_steps}, "
            f"ls={all_seed_short_param.langevin_steps}, "
            f"smax={all_seed_short_param.sigma_max:.4g}, "
            f"smin={all_seed_short_param.sigma_min:.4g}, "
            f"sfin={all_seed_short_param.sigma_final:.4g}, "
            f"beta_lg={all_seed_short_param.beta_langevin_noise:.4g}, "
            f"beta_fw={all_seed_short_param.beta_forward_noise:.4g}"
        )

    print("\n固定每个 GT 的候选 x0（来自 SM）...")
    x0_cache = {}
    for gt_idx in args.gt_indices:
        if gt_idx < 0 or gt_idx >= len(seam_data):
            raise IndexError(f"gt_index超范围: {gt_idx}, dataset_size={len(seam_data)}")
        gt = seam_data[gt_idx].unsqueeze(0).to(device)
        measurement = operator(gt)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(args.master_seed + gt_idx)
        x0, meta = _select_x0_candidate_from_sm(
            sm_info=sm_info,
            measurement=measurement,
            operator=operator,
            top_k=args.gss_top_k,
            group_top_g=args.group_top_g,
            candidate_mode=args.candidate_mode,
            generator=gen,
        )
        if args.ensemble_mode:
            order = torch.argsort(meta['top_candidate_distances'], dim=0)
            n_pool = max(1, min(int(args.ensemble_top_n), int(order.numel())))
            pool_pos = order[:n_pool]
            pool_indices = meta['top_indices'][pool_pos].detach().cpu()
            pool_distances = meta['top_candidate_distances'][pool_pos].detach().cpu()

            top_seed = int(pool_indices[0].item())
            x0_top = sm_x0hat_batch[top_seed].unsqueeze(0)
            with torch.no_grad():
                top_misfit_before = float(torch.norm(operator(x0_top) - measurement).item())
                top_x0_gt_nrmse = compute_nrmse(x0_top, gt)
                top_x0_gt_ssim = compute_ssim(x0_top, gt)

            x0_cache[gt_idx] = {
                'gt': gt.detach(),
                'measurement': measurement.detach(),
                'meta': meta,
                'ensemble_pool_indices': pool_indices,
                'ensemble_pool_distances': pool_distances,
            }
            all_seed_best = _find_all_seed_min_nrmse_x0hat(
                x0hat_batch=sm_x0hat_batch,
                gt=gt,
                eval_batch_size=max(64, int(eff_candidate_batch_size) * 16),
            )
            in_pool = bool((pool_indices == int(all_seed_best['seed'])).any().item())
            all_seed_best['in_pool'] = int(in_pool)
            x0_cache[gt_idx]['all_seed_best'] = all_seed_best
            group_extra = ""
            if args.candidate_mode == 'group_topg' and meta.get('selected_group_indices') is not None:
                g_list = [int(v) for v in meta['selected_group_indices'].detach().cpu().tolist()]
                show = g_list[:min(12, len(g_list))]
                suffix = "..." if len(g_list) > len(show) else ""
                group_extra = f", selected_groups={show}{suffix}"
                group_extra += f", top_m0={float(pool_distances[0].item()):.4f}"
            print(
                f"  GT#{gt_idx}: group={meta['best_group']}, ensemble_pool_n={n_pool}, "
                f"top_seed={top_seed}, top_cand_dist={float(pool_distances[0].item()):.4f}, "
                f"top_misfit_before={top_misfit_before:.4f}, "
                f"top_x0_gt_nrmse={top_x0_gt_nrmse:.4f}, top_x0_gt_ssim={top_x0_gt_ssim:.4f}, "
                f"all_seed_min_nrmse={float(all_seed_best['nrmse']):.4f} (seed={int(all_seed_best['seed'])}, in_pool={int(in_pool)})"
                f"{group_extra}"
            )
            continue

        if args.candidate_mode == 'all_seed':
            x0, short_meta = _all_seed_multistart_short_pick(
                model=model,
                operator=operator,
                measurement=measurement,
                x0hat_batch=sm_x0hat_batch,
                top_indices=meta['top_indices'],
                top_candidate_distances=meta['top_candidate_distances'],
                short_param=all_seed_short_param,
                top_n=args.all_seed_short_top_n,
                seed_base=args.master_seed + gt_idx * 100003,
            )
            meta['best_seed'] = int(short_meta['short_best_seed'])
            meta['candidate_distance'] = float(short_meta['short_best_misfit'])
            meta['short_top_n'] = int(short_meta['short_top_n'])
            meta['short_best_misfit'] = float(short_meta['short_best_misfit'])

        with torch.no_grad():
            d_x0 = operator(x0)
            misfit_before = torch.norm(d_x0 - measurement).item()
            x0_gt_nrmse = compute_nrmse(x0, gt)
            x0_gt_ssim = compute_ssim(x0, gt)
            x0_gt_psnr = compute_psnr(x0, gt)

        x0_cache[gt_idx] = {
            'gt': gt.detach(),
            'measurement': measurement.detach(),
            'x0': x0.detach(),
            'meta': meta,
            'misfit_before': float(misfit_before),
            'x0_gt_nrmse': float(x0_gt_nrmse),
            'x0_gt_ssim': float(x0_gt_ssim),
            'x0_gt_psnr': float(x0_gt_psnr),
        }
        extra = ""
        if args.candidate_mode == 'all_seed':
            extra = f", short_top_n={meta['short_top_n']}, short_best_misfit={meta['short_best_misfit']:.4f}"
        elif args.candidate_mode == 'group_topg':
            extra = f", cand_m0={meta['candidate_distance']:.4f}"
        print(
            f"  GT#{gt_idx}: group={meta['best_group']}, seed={meta['best_seed']}, "
            f"cand_dist={meta['candidate_distance']:.4f}, misfit_before={misfit_before:.4f}, "
            f"x0_gt_nrmse={x0_gt_nrmse:.4f}, x0_gt_ssim={x0_gt_ssim:.4f}{extra}"
        )

    if args.ensemble_mode:
        total_runs = 0
        for gt_idx in args.gt_indices:
            pool_n = int(x0_cache[gt_idx]['ensemble_pool_indices'].numel())
            total_runs += pool_n * args.n_repeats
    else:
        total_runs = len(args.gt_indices) * args.n_repeats
    pbar = tqdm(total=total_runs, desc='DAPS Langevin Diagnose')
    inner_all_rows = []
    outer_all_rows = []
    run_rows = []
    ensemble_rows = []
    curve_paths = []
    traj_paths = []
    panel_paths = []

    for gt_idx in args.gt_indices:
        gt = x0_cache[gt_idx]['gt'].to(device)
        measurement = x0_cache[gt_idx]['measurement'].to(device)
        meta = x0_cache[gt_idx]['meta']
        if args.ensemble_mode:
            pool_indices = x0_cache[gt_idx]['ensemble_pool_indices'].tolist()
            pool_distances = x0_cache[gt_idx]['ensemble_pool_distances'].tolist()

            for repeat_id in range(args.n_repeats):
                run_seed_base = _build_run_seed(args.master_seed, gt_idx, repeat_id, args.seed_mode)
                candidate_run_rows = []
                best_for_vis = None
                best_for_vis_misfit = float('inf')
                panel_rows = []
                batch_size = max(1, int(eff_candidate_batch_size))
                for batch_start in range(0, len(pool_indices), batch_size):
                    batch_seeds = pool_indices[batch_start: batch_start + batch_size]
                    batch_dists = pool_distances[batch_start: batch_start + batch_size]
                    curr_bs = len(batch_seeds)
                    batch_ranks = list(range(batch_start, batch_start + curr_bs))
                    batch_run_seeds = [
                        int(run_seed_base + rank * 10007 + int(seed_idx))
                        for rank, seed_idx in zip(batch_ranks, batch_seeds)
                    ]

                    x0_batch = torch.cat(
                        [sm_x0hat_batch[int(seed_idx)].unsqueeze(0) for seed_idx in batch_seeds],
                        dim=0
                    ).to(device)
                    measurement_batch = measurement.expand(curr_bs, -1, -1, -1)

                    with torch.no_grad():
                        d_x0_batch = operator(x0_batch)
                        misfit_before_vec = torch.norm(
                            (d_x0_batch - measurement_batch).reshape(curr_bs, -1),
                            dim=1
                        ).detach().cpu().numpy().tolist()
                        x0_gt_nrmse_vec = [compute_nrmse(x0_batch[i:i + 1], gt) for i in range(curr_bs)]
                        x0_gt_ssim_vec = [compute_ssim(x0_batch[i:i + 1], gt) for i in range(curr_bs)]
                        x0_gt_psnr_vec = [compute_psnr(x0_batch[i:i + 1], gt) for i in range(curr_bs)]

                    t0 = time.time()
                    diag = _run_langevin_diagnostics(
                        model=model,
                        operator=operator,
                        measurement=measurement_batch,
                        x0=x0_batch,
                        param=param,
                        sample_seeds=batch_run_seeds,
                        return_trajectory=False,
                    )
                    runtime_s = time.time() - t0
                    runtime_s_each = float(runtime_s) / max(1, curr_bs)

                    final_x0hat_batch = diag['final_x0hat'].detach()
                    final_x0y_batch = diag['final_x0y'].detach()
                    inner_rows = diag['inner_rows']
                    outer_rows = diag['outer_rows']
                    mean_noise_to_drift = _mean([r['noise_to_drift_ratio'] for r in inner_rows])
                    mean_noise_to_grad = _mean([r['noise_to_grad_ratio'] for r in inner_rows])
                    mean_prior_over_data = _mean([r['prior_over_data_loss'] for r in inner_rows])

                    row_rank = int(batch_ranks[0]) if curr_bs == 1 else -1
                    row_seed = int(batch_seeds[0]) if curr_bs == 1 else -1
                    row_dist = float(batch_dists[0]) if curr_bs == 1 else float(_mean(batch_dists))
                    row_run_seed = int(batch_run_seeds[0]) if curr_bs == 1 else int(run_seed_base + batch_start)

                    for row in inner_rows:
                        tagged = dict(row)
                        tagged.update({
                            'gt_index': int(gt_idx),
                            'repeat_id': int(repeat_id),
                            'candidate_rank': row_rank,
                            'candidate_seed': row_seed,
                            'candidate_init_distance': row_dist,
                            'run_seed': row_run_seed,
                            'best_group': int(meta['best_group']),
                            'best_seed': row_seed if row_seed >= 0 else -1,
                            'candidate_distance': row_dist,
                            'langevin_lr': float(param.langevin_lr),
                            'langevin_steps': int(param.langevin_steps),
                            'tau': float(param.tau),
                            'lr_min_ratio': float(param.lr_min_ratio),
                            'lambda_prior': float(param.lambda_prior),
                            'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                            'annealing_steps': int(param.annealing_steps),
                            'sigma_max': float(param.sigma_max),
                            'sigma_min': float(param.sigma_min),
                            'sigma_final': float(param.sigma_final),
                            'beta_langevin_noise': float(param.beta_langevin_noise),
                            'beta_forward_noise': float(param.beta_forward_noise),
                        })
                        inner_all_rows.append(tagged)

                    for row in outer_rows:
                        tagged = dict(row)
                        tagged.update({
                            'gt_index': int(gt_idx),
                            'repeat_id': int(repeat_id),
                            'candidate_rank': row_rank,
                            'candidate_seed': row_seed,
                            'candidate_init_distance': row_dist,
                            'run_seed': row_run_seed,
                            'best_group': int(meta['best_group']),
                            'best_seed': row_seed if row_seed >= 0 else -1,
                            'candidate_distance': row_dist,
                            'langevin_lr': float(param.langevin_lr),
                            'langevin_steps': int(param.langevin_steps),
                            'tau': float(param.tau),
                            'lr_min_ratio': float(param.lr_min_ratio),
                            'lambda_prior': float(param.lambda_prior),
                            'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                            'annealing_steps': int(param.annealing_steps),
                            'sigma_max': float(param.sigma_max),
                            'sigma_min': float(param.sigma_min),
                            'sigma_final': float(param.sigma_final),
                            'beta_langevin_noise': float(param.beta_langevin_noise),
                            'beta_forward_noise': float(param.beta_forward_noise),
                        })
                        outer_all_rows.append(tagged)

                    for local_i in range(curr_bs):
                        candidate_rank = int(batch_ranks[local_i])
                        seed_idx = int(batch_seeds[local_i])
                        cand_dist = float(batch_dists[local_i])
                        run_seed = int(batch_run_seeds[local_i])

                        x0_i = x0_batch[local_i: local_i + 1].detach()
                        final_x0hat = final_x0hat_batch[local_i: local_i + 1].detach()
                        final_x0y = final_x0y_batch[local_i: local_i + 1].detach()

                        misfit_before = float(misfit_before_vec[local_i])
                        x0_gt_nrmse = float(x0_gt_nrmse_vec[local_i])
                        x0_gt_ssim = float(x0_gt_ssim_vec[local_i])
                        x0_gt_psnr = float(x0_gt_psnr_vec[local_i])

                        x0hat_metrics = _compute_final_metrics(
                            pred=final_x0hat,
                            x0=x0_i,
                            gt=gt,
                            measurement=measurement,
                            operator=operator,
                            low_freq_radius_ratio=args.low_freq_radius_ratio,
                            min_delta_rel_l2=args.min_delta_rel_l2,
                            max_misfit_ratio=args.max_misfit_ratio,
                            max_high_freq_ratio=args.max_high_freq_ratio,
                            misfit_before=misfit_before,
                        )
                        x0y_metrics = _compute_final_metrics(
                            pred=final_x0y,
                            x0=x0_i,
                            gt=gt,
                            measurement=measurement,
                            operator=operator,
                            low_freq_radius_ratio=args.low_freq_radius_ratio,
                            min_delta_rel_l2=args.min_delta_rel_l2,
                            max_misfit_ratio=args.max_misfit_ratio,
                            max_high_freq_ratio=args.max_high_freq_ratio,
                            misfit_before=misfit_before,
                        )

                        run_row = {
                            'gt_index': int(gt_idx),
                            'repeat_id': int(repeat_id),
                            'candidate_rank': int(candidate_rank),
                            'candidate_seed': int(seed_idx),
                            'candidate_init_distance': float(cand_dist),
                            'run_seed': int(run_seed),
                            'best_group': int(meta['best_group']),
                            'best_seed': int(seed_idx),
                            'candidate_distance': float(cand_dist),
                            'langevin_lr': float(param.langevin_lr),
                            'langevin_steps': int(param.langevin_steps),
                            'tau': float(param.tau),
                            'lr_min_ratio': float(param.lr_min_ratio),
                            'lambda_prior': float(param.lambda_prior),
                            'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                            'annealing_steps': int(param.annealing_steps),
                            'sigma_max': float(param.sigma_max),
                            'sigma_min': float(param.sigma_min),
                            'sigma_final': float(param.sigma_final),
                            'beta_langevin_noise': float(param.beta_langevin_noise),
                            'beta_forward_noise': float(param.beta_forward_noise),
                            'x0_gt_nrmse': float(x0_gt_nrmse),
                            'x0_gt_ssim': float(x0_gt_ssim),
                            'x0_gt_psnr': float(x0_gt_psnr),
                            'misfit_before': float(misfit_before),
                            'x0hat_delta_rel_l2': float(x0hat_metrics['delta_rel_l2']),
                            'x0hat_delta_ssim': float(x0hat_metrics['delta_ssim']),
                            'x0hat_gt_nrmse': float(x0hat_metrics['gt_nrmse']),
                            'x0hat_gt_ssim': float(x0hat_metrics['gt_ssim']),
                            'x0hat_gt_psnr': float(x0hat_metrics['gt_psnr']),
                            'x0hat_gt_nrmse_improve': float(x0_gt_nrmse - x0hat_metrics['gt_nrmse']),
                            'x0hat_misfit_after': float(x0hat_metrics['meas_misfit_after']),
                            'x0hat_misfit_ratio': float(x0hat_metrics['misfit_ratio']),
                            'x0hat_high_freq_ratio': float(x0hat_metrics['delta_high_freq_ratio']),
                            'x0hat_is_structural_variant': int(x0hat_metrics['is_structural_variant']),
                            'x0y_delta_rel_l2': float(x0y_metrics['delta_rel_l2']),
                            'x0y_delta_ssim': float(x0y_metrics['delta_ssim']),
                            'x0y_gt_nrmse': float(x0y_metrics['gt_nrmse']),
                            'x0y_gt_ssim': float(x0y_metrics['gt_ssim']),
                            'x0y_gt_psnr': float(x0y_metrics['gt_psnr']),
                            'x0y_gt_nrmse_improve': float(x0_gt_nrmse - x0y_metrics['gt_nrmse']),
                            'x0y_misfit_after': float(x0y_metrics['meas_misfit_after']),
                            'x0y_misfit_ratio': float(x0y_metrics['misfit_ratio']),
                            'x0y_high_freq_ratio': float(x0y_metrics['delta_high_freq_ratio']),
                            'x0y_is_structural_variant': int(x0y_metrics['is_structural_variant']),
                            'mean_noise_to_drift_ratio': float(mean_noise_to_drift),
                            'mean_noise_to_grad_ratio': float(mean_noise_to_grad),
                            'mean_prior_over_data_loss': float(mean_prior_over_data),
                            'runtime_s': float(runtime_s_each),
                        }
                        run_rows.append(run_row)
                        candidate_run_rows.append(run_row)
                        panel_rows.append({
                            'seed': int(seed_idx),
                            'misfit_before': float(misfit_before),
                            'misfit_after': float(x0hat_metrics['meas_misfit_after']),
                            'nrmse_improve': float(x0_gt_nrmse - x0hat_metrics['gt_nrmse']),
                            'x0_gt_nrmse': float(x0_gt_nrmse),
                            'x0': x0_i.detach().cpu(),
                            'pred_after': final_x0hat.detach().cpu(),
                        })

                        if x0hat_metrics['meas_misfit_after'] < best_for_vis_misfit:
                            best_for_vis_misfit = float(x0hat_metrics['meas_misfit_after'])
                            best_for_vis = {
                                'x0': x0_i.detach().cpu(),
                                'run_seed': int(run_seed),
                            }

                    pbar.update(curr_bs)

                misfits = np.array([r['x0hat_misfit_after'] for r in candidate_run_rows], dtype=np.float64)
                min_m = float(np.min(misfits))
                logits = -(misfits - min_m) / float(args.ensemble_weight_temp)
                logits = logits - float(np.max(logits))
                weights = np.exp(logits)
                weights = weights / max(1e-12, float(np.sum(weights)))
                ess = float(1.0 / max(1e-12, float(np.sum(weights ** 2))))

                best_idx = int(np.argmin(misfits))
                oracle_idx = int(np.argmin(np.array([r['x0hat_gt_nrmse'] for r in candidate_run_rows], dtype=np.float64)))

                x0hat_gt_nrmse_mean, x0hat_gt_nrmse_std = _weighted_mean_std(
                    [r['x0hat_gt_nrmse'] for r in candidate_run_rows], weights
                )
                x0hat_imp_mean, x0hat_imp_std = _weighted_mean_std(
                    [r['x0hat_gt_nrmse_improve'] for r in candidate_run_rows], weights
                )
                x0hat_delta_mean, x0hat_delta_std = _weighted_mean_std(
                    [r['x0hat_delta_rel_l2'] for r in candidate_run_rows], weights
                )
                x0hat_mr_mean, x0hat_mr_std = _weighted_mean_std(
                    [r['x0hat_misfit_ratio'] for r in candidate_run_rows], weights
                )
                x0hat_pass_mean, _ = _weighted_mean_std(
                    [float(r['x0hat_is_structural_variant']) for r in candidate_run_rows], weights
                )

                ensemble_row = {
                    'gt_index': int(gt_idx),
                    'repeat_id': int(repeat_id),
                    'best_group': int(meta['best_group']),
                    'ensemble_top_n': int(len(candidate_run_rows)),
                    'ensemble_weight_temp': float(args.ensemble_weight_temp),
                    'ensemble_ess': float(ess),
                    'best_by_misfit_rank': int(candidate_run_rows[best_idx]['candidate_rank']),
                    'best_by_misfit_seed': int(candidate_run_rows[best_idx]['candidate_seed']),
                    'best_by_misfit_x0hat_misfit_after': float(candidate_run_rows[best_idx]['x0hat_misfit_after']),
                    'best_by_misfit_x0hat_gt_nrmse': float(candidate_run_rows[best_idx]['x0hat_gt_nrmse']),
                    'best_by_misfit_x0hat_gt_nrmse_improve': float(candidate_run_rows[best_idx]['x0hat_gt_nrmse_improve']),
                    'oracle_best_rank': int(candidate_run_rows[oracle_idx]['candidate_rank']),
                    'oracle_best_seed': int(candidate_run_rows[oracle_idx]['candidate_seed']),
                    'oracle_best_x0hat_gt_nrmse': float(candidate_run_rows[oracle_idx]['x0hat_gt_nrmse']),
                    'oracle_best_x0hat_gt_nrmse_improve': float(candidate_run_rows[oracle_idx]['x0hat_gt_nrmse_improve']),
                    'weighted_x0hat_gt_nrmse_mean': float(x0hat_gt_nrmse_mean),
                    'weighted_x0hat_gt_nrmse_std': float(x0hat_gt_nrmse_std),
                    'weighted_x0hat_gt_nrmse_improve_mean': float(x0hat_imp_mean),
                    'weighted_x0hat_gt_nrmse_improve_std': float(x0hat_imp_std),
                    'weighted_x0hat_delta_rel_l2_mean': float(x0hat_delta_mean),
                    'weighted_x0hat_delta_rel_l2_std': float(x0hat_delta_std),
                    'weighted_x0hat_misfit_ratio_mean': float(x0hat_mr_mean),
                    'weighted_x0hat_misfit_ratio_std': float(x0hat_mr_std),
                    'weighted_x0hat_gate_pass_rate': float(x0hat_pass_mean),
                    'mean_runtime_s': float(_mean([r['runtime_s'] for r in candidate_run_rows])),
                }
                ensemble_rows.append(ensemble_row)

                print(
                    f"  [ensemble] GT#{gt_idx} r{repeat_id}: "
                    f"best_misfit_seed={ensemble_row['best_by_misfit_seed']} "
                    f"(nrmse={ensemble_row['best_by_misfit_x0hat_gt_nrmse']:.4f}), "
                    f"weighted_nrmse={ensemble_row['weighted_x0hat_gt_nrmse_mean']:.4f}±"
                    f"{ensemble_row['weighted_x0hat_gt_nrmse_std']:.4f}, "
                    f"ESS={ensemble_row['ensemble_ess']:.2f}"
                )

                if best_for_vis is not None:
                    need_diag_vis = (
                        ((args.vis_max_runs <= 0) or (len(curve_paths) < args.vis_max_runs))
                        and (not args.skip_curve_vis)
                    ) or (
                        ((args.vis_max_runs <= 0) or (len(traj_paths) < args.vis_max_runs))
                        and (not args.skip_trajectory_vis)
                    )
                    if need_diag_vis:
                        vis_seed = int(best_for_vis['run_seed'])
                        torch.manual_seed(vis_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(vis_seed)
                        diag_vis = _run_langevin_diagnostics(
                            model=model,
                            operator=operator,
                            measurement=measurement,
                            x0=best_for_vis['x0'].to(device),
                            param=param,
                            sample_seeds=[vis_seed],
                            return_trajectory=(not args.skip_trajectory_vis),
                        )
                    else:
                        diag_vis = None

                    can_vis = (args.vis_max_runs <= 0) or (len(curve_paths) < args.vis_max_runs)
                    if can_vis and (not args.skip_curve_vis) and (diag_vis is not None):
                        p_curve = _save_inner_curve_plot(
                            output_dir=output_dir,
                            model_tag=args.model_tag,
                            param=param,
                            gt_idx=gt_idx,
                            repeat_id=repeat_id,
                            inner_rows=diag_vis['inner_rows'],
                        )
                        if p_curve:
                            curve_paths.append(p_curve)

                    can_vis = (args.vis_max_runs <= 0) or (len(traj_paths) < args.vis_max_runs)
                    if can_vis and (not args.skip_trajectory_vis) and (diag_vis is not None):
                        p_traj = _save_trajectory_plot(
                            output_dir=output_dir,
                            model_tag=args.model_tag,
                            param=param,
                            gt_idx=gt_idx,
                            repeat_id=repeat_id,
                            gt=gt.detach().cpu(),
                            x0=best_for_vis['x0'],
                            trajectory=diag_vis['trajectory'],
                            num_snapshots=args.vis_num_snapshots,
                            mode=args.trajectory_mode,
                        )
                        if p_traj:
                            traj_paths.append(p_traj)

                can_panel_vis = (args.vis_max_runs <= 0) or (len(panel_paths) < args.vis_max_runs)
                if can_panel_vis and (not args.skip_ensemble_panel_vis):
                    p_panel = _save_ensemble_panel_plot(
                        output_dir=output_dir,
                        model_tag=args.model_tag,
                        param=param,
                        gt_idx=gt_idx,
                        repeat_id=repeat_id,
                        gt=gt.detach().cpu(),
                        panel_rows=panel_rows,
                        all_seed_best=x0_cache[gt_idx].get('all_seed_best', None),
                    )
                    if p_panel:
                        panel_paths.append(p_panel)
            continue

        x0 = x0_cache[gt_idx]['x0'].to(device)
        misfit_before = float(x0_cache[gt_idx]['misfit_before'])
        x0_gt_nrmse = float(x0_cache[gt_idx]['x0_gt_nrmse'])
        x0_gt_ssim = float(x0_cache[gt_idx]['x0_gt_ssim'])
        x0_gt_psnr = float(x0_cache[gt_idx]['x0_gt_psnr'])

        for repeat_id in range(args.n_repeats):
            run_seed = _build_run_seed(args.master_seed, gt_idx, repeat_id, args.seed_mode)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            t0 = time.time()
            diag = _run_langevin_diagnostics(
                model=model,
                operator=operator,
                measurement=measurement,
                x0=x0,
                param=param,
                return_trajectory=(not args.skip_trajectory_vis),
            )
            runtime_s = time.time() - t0

            final_x0hat = diag['final_x0hat'].detach()
            final_x0y = diag['final_x0y'].detach()

            x0hat_metrics = _compute_final_metrics(
                pred=final_x0hat,
                x0=x0,
                gt=gt,
                measurement=measurement,
                operator=operator,
                low_freq_radius_ratio=args.low_freq_radius_ratio,
                min_delta_rel_l2=args.min_delta_rel_l2,
                max_misfit_ratio=args.max_misfit_ratio,
                max_high_freq_ratio=args.max_high_freq_ratio,
                misfit_before=misfit_before,
            )
            x0y_metrics = _compute_final_metrics(
                pred=final_x0y,
                x0=x0,
                gt=gt,
                measurement=measurement,
                operator=operator,
                low_freq_radius_ratio=args.low_freq_radius_ratio,
                min_delta_rel_l2=args.min_delta_rel_l2,
                max_misfit_ratio=args.max_misfit_ratio,
                max_high_freq_ratio=args.max_high_freq_ratio,
                misfit_before=misfit_before,
            )

            inner_rows = diag['inner_rows']
            outer_rows = diag['outer_rows']
            mean_noise_to_drift = _mean([r['noise_to_drift_ratio'] for r in inner_rows])
            mean_noise_to_grad = _mean([r['noise_to_grad_ratio'] for r in inner_rows])
            mean_prior_over_data = _mean([r['prior_over_data_loss'] for r in inner_rows])

            for row in inner_rows:
                tagged = dict(row)
                tagged.update({
                    'gt_index': int(gt_idx),
                    'repeat_id': int(repeat_id),
                    'candidate_rank': 0,
                    'candidate_seed': int(meta['best_seed']),
                    'candidate_init_distance': float(meta['candidate_distance']),
                    'run_seed': int(run_seed),
                    'best_group': int(meta['best_group']),
                    'best_seed': int(meta['best_seed']),
                    'candidate_distance': float(meta['candidate_distance']),
                    'langevin_lr': float(param.langevin_lr),
                    'langevin_steps': int(param.langevin_steps),
                    'tau': float(param.tau),
                    'lr_min_ratio': float(param.lr_min_ratio),
                    'lambda_prior': float(param.lambda_prior),
                    'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                    'annealing_steps': int(param.annealing_steps),
                    'sigma_max': float(param.sigma_max),
                    'sigma_min': float(param.sigma_min),
                    'sigma_final': float(param.sigma_final),
                    'beta_langevin_noise': float(param.beta_langevin_noise),
                    'beta_forward_noise': float(param.beta_forward_noise),
                })
                inner_all_rows.append(tagged)

            for row in outer_rows:
                tagged = dict(row)
                tagged.update({
                    'gt_index': int(gt_idx),
                    'repeat_id': int(repeat_id),
                    'candidate_rank': 0,
                    'candidate_seed': int(meta['best_seed']),
                    'candidate_init_distance': float(meta['candidate_distance']),
                    'run_seed': int(run_seed),
                    'best_group': int(meta['best_group']),
                    'best_seed': int(meta['best_seed']),
                    'candidate_distance': float(meta['candidate_distance']),
                    'langevin_lr': float(param.langevin_lr),
                    'langevin_steps': int(param.langevin_steps),
                    'tau': float(param.tau),
                    'lr_min_ratio': float(param.lr_min_ratio),
                    'lambda_prior': float(param.lambda_prior),
                    'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                    'annealing_steps': int(param.annealing_steps),
                    'sigma_max': float(param.sigma_max),
                    'sigma_min': float(param.sigma_min),
                    'sigma_final': float(param.sigma_final),
                    'beta_langevin_noise': float(param.beta_langevin_noise),
                    'beta_forward_noise': float(param.beta_forward_noise),
                })
                outer_all_rows.append(tagged)

            run_row = {
                'gt_index': int(gt_idx),
                'repeat_id': int(repeat_id),
                'candidate_rank': 0,
                'candidate_seed': int(meta['best_seed']),
                'candidate_init_distance': float(meta['candidate_distance']),
                'run_seed': int(run_seed),
                'best_group': int(meta['best_group']),
                'best_seed': int(meta['best_seed']),
                'candidate_distance': float(meta['candidate_distance']),
                'langevin_lr': float(param.langevin_lr),
                'langevin_steps': int(param.langevin_steps),
                'tau': float(param.tau),
                'lr_min_ratio': float(param.lr_min_ratio),
                'lambda_prior': float(param.lambda_prior),
                'lambda_prior_min_ratio': float(param.lambda_prior_min_ratio),
                'annealing_steps': int(param.annealing_steps),
                'sigma_max': float(param.sigma_max),
                'sigma_min': float(param.sigma_min),
                'sigma_final': float(param.sigma_final),
                'beta_langevin_noise': float(param.beta_langevin_noise),
                'beta_forward_noise': float(param.beta_forward_noise),
                'x0_gt_nrmse': float(x0_gt_nrmse),
                'x0_gt_ssim': float(x0_gt_ssim),
                'x0_gt_psnr': float(x0_gt_psnr),
                'misfit_before': float(misfit_before),
                'x0hat_delta_rel_l2': float(x0hat_metrics['delta_rel_l2']),
                'x0hat_delta_ssim': float(x0hat_metrics['delta_ssim']),
                'x0hat_gt_nrmse': float(x0hat_metrics['gt_nrmse']),
                'x0hat_gt_ssim': float(x0hat_metrics['gt_ssim']),
                'x0hat_gt_psnr': float(x0hat_metrics['gt_psnr']),
                'x0hat_gt_nrmse_improve': float(x0_gt_nrmse - x0hat_metrics['gt_nrmse']),
                'x0hat_misfit_after': float(x0hat_metrics['meas_misfit_after']),
                'x0hat_misfit_ratio': float(x0hat_metrics['misfit_ratio']),
                'x0hat_high_freq_ratio': float(x0hat_metrics['delta_high_freq_ratio']),
                'x0hat_is_structural_variant': int(x0hat_metrics['is_structural_variant']),
                'x0y_delta_rel_l2': float(x0y_metrics['delta_rel_l2']),
                'x0y_delta_ssim': float(x0y_metrics['delta_ssim']),
                'x0y_gt_nrmse': float(x0y_metrics['gt_nrmse']),
                'x0y_gt_ssim': float(x0y_metrics['gt_ssim']),
                'x0y_gt_psnr': float(x0y_metrics['gt_psnr']),
                'x0y_gt_nrmse_improve': float(x0_gt_nrmse - x0y_metrics['gt_nrmse']),
                'x0y_misfit_after': float(x0y_metrics['meas_misfit_after']),
                'x0y_misfit_ratio': float(x0y_metrics['misfit_ratio']),
                'x0y_high_freq_ratio': float(x0y_metrics['delta_high_freq_ratio']),
                'x0y_is_structural_variant': int(x0y_metrics['is_structural_variant']),
                'mean_noise_to_drift_ratio': float(mean_noise_to_drift),
                'mean_noise_to_grad_ratio': float(mean_noise_to_grad),
                'mean_prior_over_data_loss': float(mean_prior_over_data),
                'runtime_s': float(runtime_s),
            }
            run_rows.append(run_row)

            can_vis = (args.vis_max_runs <= 0) or (len(curve_paths) < args.vis_max_runs)
            if can_vis and not args.skip_curve_vis:
                p_curve = _save_inner_curve_plot(
                    output_dir=output_dir,
                    model_tag=args.model_tag,
                    param=param,
                    gt_idx=gt_idx,
                    repeat_id=repeat_id,
                    inner_rows=inner_rows,
                )
                if p_curve:
                    curve_paths.append(p_curve)

            can_vis = (args.vis_max_runs <= 0) or (len(traj_paths) < args.vis_max_runs)
            if can_vis and not args.skip_trajectory_vis:
                p_traj = _save_trajectory_plot(
                    output_dir=output_dir,
                    model_tag=args.model_tag,
                    param=param,
                    gt_idx=gt_idx,
                    repeat_id=repeat_id,
                    gt=gt.detach().cpu(),
                    x0=x0.detach().cpu(),
                    trajectory=diag['trajectory'],
                    num_snapshots=args.vis_num_snapshots,
                    mode=args.trajectory_mode,
                )
                if p_traj:
                    traj_paths.append(p_traj)

            pbar.update(1)
    pbar.close()

    # 保存 CSV
    inner_headers = [
        'gt_index', 'repeat_id', 'candidate_rank', 'candidate_seed', 'candidate_init_distance',
        'run_seed', 'best_group', 'best_seed', 'candidate_distance',
        'langevin_lr', 'langevin_steps', 'tau', 'lr_min_ratio', 'lambda_prior', 'lambda_prior_min_ratio',
        'annealing_steps', 'sigma_max', 'sigma_min', 'sigma_final',
        'beta_langevin_noise', 'beta_forward_noise',
        'outer_step', 'inner_step', 'sigma', 'ratio', 'langevin_lr_effective', 'lambda_prior_effective',
        'data_loss', 'prior_loss', 'total_loss', 'prior_over_data_loss', 'measurement_misfit',
        'grad_data_norm', 'grad_prior_norm', 'grad_total_norm', 'grad_data_prior_cos',
        'drift_norm', 'noise_norm', 'noise_to_drift_ratio', 'noise_to_grad_ratio'
    ]
    outer_headers = [
        'gt_index', 'repeat_id', 'candidate_rank', 'candidate_seed', 'candidate_init_distance',
        'run_seed', 'best_group', 'best_seed', 'candidate_distance',
        'langevin_lr', 'langevin_steps', 'tau', 'lr_min_ratio', 'lambda_prior', 'lambda_prior_min_ratio',
        'annealing_steps', 'sigma_max', 'sigma_min', 'sigma_final',
        'beta_langevin_noise', 'beta_forward_noise',
        'outer_step', 'sigma', 'next_sigma', 'langevin_lr_effective', 'lambda_prior_effective',
        'mean_data_loss', 'mean_prior_loss', 'mean_total_loss', 'mean_prior_over_data_loss', 'mean_measurement_misfit',
        'mean_grad_data_norm', 'mean_grad_prior_norm', 'mean_grad_total_norm',
        'mean_drift_norm', 'mean_noise_norm', 'mean_noise_to_drift_ratio', 'mean_noise_to_grad_ratio',
        'misfit_x0hat', 'misfit_x0y'
    ]
    run_headers = [
        'gt_index', 'repeat_id', 'candidate_rank', 'candidate_seed', 'candidate_init_distance',
        'run_seed', 'best_group', 'best_seed', 'candidate_distance',
        'langevin_lr', 'langevin_steps', 'tau', 'lr_min_ratio', 'lambda_prior', 'lambda_prior_min_ratio',
        'annealing_steps', 'sigma_max', 'sigma_min', 'sigma_final',
        'beta_langevin_noise', 'beta_forward_noise',
        'x0_gt_nrmse', 'x0_gt_ssim', 'x0_gt_psnr', 'misfit_before',
        'x0hat_delta_rel_l2', 'x0hat_delta_ssim', 'x0hat_gt_nrmse', 'x0hat_gt_ssim', 'x0hat_gt_psnr',
        'x0hat_gt_nrmse_improve', 'x0hat_misfit_after', 'x0hat_misfit_ratio', 'x0hat_high_freq_ratio', 'x0hat_is_structural_variant',
        'x0y_delta_rel_l2', 'x0y_delta_ssim', 'x0y_gt_nrmse', 'x0y_gt_ssim', 'x0y_gt_psnr',
        'x0y_gt_nrmse_improve', 'x0y_misfit_after', 'x0y_misfit_ratio', 'x0y_high_freq_ratio', 'x0y_is_structural_variant',
        'mean_noise_to_drift_ratio', 'mean_noise_to_grad_ratio', 'mean_prior_over_data_loss',
        'runtime_s'
    ]
    ensemble_headers = [
        'gt_index', 'repeat_id', 'best_group',
        'ensemble_top_n', 'ensemble_weight_temp', 'ensemble_ess',
        'best_by_misfit_rank', 'best_by_misfit_seed',
        'best_by_misfit_x0hat_misfit_after', 'best_by_misfit_x0hat_gt_nrmse',
        'best_by_misfit_x0hat_gt_nrmse_improve',
        'oracle_best_rank', 'oracle_best_seed',
        'oracle_best_x0hat_gt_nrmse', 'oracle_best_x0hat_gt_nrmse_improve',
        'weighted_x0hat_gt_nrmse_mean', 'weighted_x0hat_gt_nrmse_std',
        'weighted_x0hat_gt_nrmse_improve_mean', 'weighted_x0hat_gt_nrmse_improve_std',
        'weighted_x0hat_delta_rel_l2_mean', 'weighted_x0hat_delta_rel_l2_std',
        'weighted_x0hat_misfit_ratio_mean', 'weighted_x0hat_misfit_ratio_std',
        'weighted_x0hat_gate_pass_rate', 'mean_runtime_s'
    ]

    inner_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_langevin_inner_detail_model-{args.model_tag}', '.csv')
    )
    outer_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_langevin_outer_summary_model-{args.model_tag}', '.csv')
    )
    run_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_langevin_run_summary_model-{args.model_tag}', '.csv')
    )
    ensemble_path = None
    if args.ensemble_mode:
        ensemble_path = os.path.join(
            output_dir,
            generate_timestamped_filename(f'daps_langevin_ensemble_summary_model-{args.model_tag}', '.csv')
        )
    _save_rows_csv(inner_path, inner_all_rows, inner_headers)
    _save_rows_csv(outer_path, outer_all_rows, outer_headers)
    _save_rows_csv(run_path, run_rows, run_headers)
    if ensemble_path is not None:
        _save_rows_csv(ensemble_path, ensemble_rows, ensemble_headers)

    print("\n" + "=" * 72)
    print("Langevin 诊断完成")
    print("=" * 72)
    print(f"inner_detail_csv: {inner_path}")
    print(f"outer_summary_csv: {outer_path}")
    print(f"run_summary_csv: {run_path}")
    if ensemble_path is not None:
        print(f"ensemble_summary_csv: {ensemble_path}")

    if run_rows:
        avg_x0hat_improve = _mean([r['x0hat_gt_nrmse_improve'] for r in run_rows])
        avg_x0y_improve = _mean([r['x0y_gt_nrmse_improve'] for r in run_rows])
        avg_noise_drift = _mean([r['mean_noise_to_drift_ratio'] for r in run_rows])
        avg_noise_grad = _mean([r['mean_noise_to_grad_ratio'] for r in run_rows])
        x0hat_pass_rate = _mean([float(r['x0hat_is_structural_variant']) for r in run_rows])
        x0y_pass_rate = _mean([float(r['x0y_is_structural_variant']) for r in run_rows])

        if args.ensemble_mode:
            print("\nCandidate-level averages (across all ensemble members):")
        else:
            print("\nRun-level averages:")
        print(f"  mean x0hat_gt_nrmse_improve: {avg_x0hat_improve:.4f}")
        print(f"  mean x0y_gt_nrmse_improve:   {avg_x0y_improve:.4f}")
        print(f"  mean noise_to_drift_ratio:   {avg_noise_drift:.4f}")
        print(f"  mean noise_to_grad_ratio:    {avg_noise_grad:.4f}")
        print(f"  x0hat gate pass rate:        {x0hat_pass_rate:.3f}")
        print(f"  x0y gate pass rate:          {x0y_pass_rate:.3f}")

        if args.ensemble_mode:
            print("\nPer-candidate summary:")
        else:
            print("\nPer-run summary:")
        for r in run_rows:
            print(
                f"  GT#{r['gt_index']} r{r['repeat_id']} c{r['candidate_rank']} seed={r['candidate_seed']}: "
                f"x0hat_improve={r['x0hat_gt_nrmse_improve']:+.4f}, "
                f"x0y_improve={r['x0y_gt_nrmse_improve']:+.4f}, "
                f"x0hat_delta_rel={r['x0hat_delta_rel_l2']:.4f}, "
                f"x0y_delta_rel={r['x0y_delta_rel_l2']:.4f}, "
                f"noise/drift={r['mean_noise_to_drift_ratio']:.4f}, "
                f"noise/grad={r['mean_noise_to_grad_ratio']:.4f}, "
                f"x0hat_pass={r['x0hat_is_structural_variant']}, "
                f"x0y_pass={r['x0y_is_structural_variant']}"
            )
    if ensemble_rows:
        print("\nEnsemble summary:")
        print(f"  mean weighted_x0hat_gt_nrmse: {_mean([r['weighted_x0hat_gt_nrmse_mean'] for r in ensemble_rows]):.4f}")
        print(f"  mean weighted_x0hat_gt_nrmse_improve: {_mean([r['weighted_x0hat_gt_nrmse_improve_mean'] for r in ensemble_rows]):.4f}")
        print(f"  mean ensemble_ess: {_mean([r['ensemble_ess'] for r in ensemble_rows]):.3f}")

    if curve_paths:
        print("\n诊断曲线图:")
        for p in curve_paths:
            print(f"  {p}")
    elif args.skip_curve_vis:
        print("\n[curve_vis] skip_curve_vis=True，未保存曲线图。")
    else:
        print("\n[curve_vis] 未生成曲线图。")

    if traj_paths:
        print("\n轨迹图:")
        for p in traj_paths:
            print(f"  {p}")
    elif args.skip_trajectory_vis:
        print("\n[trajectory_vis] skip_trajectory_vis=True，未保存轨迹图。")
    else:
        print("\n[trajectory_vis] 未生成轨迹图。")

    if panel_paths:
        print("\nEnsemble 面板图:")
        for p in panel_paths:
            print(f"  {p}")
    elif args.skip_ensemble_panel_vis:
        print("\n[ensemble_panel_vis] skip_ensemble_panel_vis=True，未保存面板图。")
    elif args.ensemble_mode:
        print("\n[ensemble_panel_vis] 未生成面板图。")


if __name__ == '__main__':
    main()
