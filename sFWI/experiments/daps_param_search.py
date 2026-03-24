"""
DAPS 参数搜索实验脚本（基于 GSS 缓存候选）。

目标:
  给定一个 unconditional 模型和对应 SM 资产，先固定每个 GT 的候选 x0，
  再扫描 DAPS 参数，评估:
    1) 输出相对 x0 的偏离程度（是否产生“相近但不同”的结构）
    2) 输出相对 GT 的重建质量
    3) 输出在数据域的误差变化

典型用法:
  python sFWI/experiments/daps_param_search.py \
    --model_tag seam_finetune \
    --ckpt_dir /content/drive/MyDrive/score_sde_inverseSolving/checkpoints \
    --ckpt_file seam_finetune_checkpoint_5.pth \
    --sm_path /content/drive/MyDrive/score_sde_inverseSolving/gss_assets/sm_dataset-seam_model-seam_finetune_k200_j1500_seed8.pt \
    --gt_indices 1 11 21 31 41 \
    --gss_top_k 50 \
    --grid_langevin_lr 1e-5,1e-4,1e-3 \
    --grid_langevin_steps 10,20,50 \
    --grid_tau 0.01,0.07,0.1 \
    --grid_annealing_steps 20,50 \
    --grid_sigma_max 0.1,0.3 \
    --grid_sigma_min 0.01 \
    --master_seed 8
"""

import sys
import os
import csv
import time
import argparse
import itertools
import hashlib
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
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.utils.file_utils import generate_timestamped_filename


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """将输入统一为 [B, C, H, W]。"""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 pred/gt 对齐到同一空间尺寸后返回。
    规则: 以 gt 的空间尺寸为准，对 pred 做双线性插值。
    """
    pred4 = _to_4d(pred).float()
    gt4 = _to_4d(gt).float()

    if pred4.shape[-2:] != gt4.shape[-2:]:
        pred4 = F.interpolate(
            pred4,
            size=gt4.shape[-2:],
            mode='bilinear',
            align_corners=False
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
    pred, gt = _align_pred_to_gt(pred, gt)

    mu_x = pred.mean()
    mu_y = gt.mean()
    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x) * (gt - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).item()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    nrmse = compute_nrmse(pred, gt)
    return 10.0 * np.log10(1.0 / (nrmse ** 2 + 1e-10))


def compute_normalized_tv(x: torch.Tensor) -> float:
    """归一化总变分（越大通常代表高频抖动越强）。"""
    x4 = _to_4d(x).float()
    dx = torch.abs(x4[:, :, 1:, :] - x4[:, :, :-1, :]).mean()
    dy = torch.abs(x4[:, :, :, 1:] - x4[:, :, :, :-1]).mean()
    denom = torch.abs(x4).mean().clamp_min(1e-8)
    return float(((dx + dy) / denom).item())


def compute_delta_frequency_ratios(delta: torch.Tensor, low_radius_ratio: float = 0.15):
    """
    计算 delta 的低/高频能量占比。
    返回: (low_freq_ratio, high_freq_ratio, low_high_ratio)
    """
    d4 = _to_4d(delta).float()
    if d4.shape[-2] < 4 or d4.shape[-1] < 4:
        return 0.0, 1.0, 0.0

    # 平均到单张 2D 场，再做频谱分析
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


def _parse_float_list(text: str):
    vals = [x.strip() for x in text.split(',') if x.strip()]
    return [float(v) for v in vals]


def _parse_int_list(text: str):
    vals = [x.strip() for x in text.split(',') if x.strip()]
    return [int(v) for v in vals]


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


@dataclass
class SweepParam:
    param_id: int
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


def _build_param_grid(args):
    lrs = _parse_float_list(args.grid_langevin_lr)
    lsteps = _parse_int_list(args.grid_langevin_steps)
    taus = _parse_float_list(args.grid_tau)
    lr_min_ratios = _parse_float_list(args.grid_lr_min_ratio)
    lambda_priors = _parse_float_list(args.grid_lambda_prior)
    lambda_prior_min_ratios = _parse_float_list(args.grid_lambda_prior_min_ratio)
    asteps = _parse_int_list(args.grid_annealing_steps)
    smax = _parse_float_list(args.grid_sigma_max)
    smin = _parse_float_list(args.grid_sigma_min)
    sfinal = _parse_float_list(args.grid_sigma_final)

    grid = []
    param_id = 0
    for lr, ls, tau, lmr, lpr, lprmr, an, smx, smn, sfn in itertools.product(
        lrs, lsteps, taus, lr_min_ratios, lambda_priors, lambda_prior_min_ratios, asteps, smax, smin, sfinal
    ):
        if smn <= 0:
            continue
        if smx <= smn:
            continue
        if sfn < 0:
            continue
        if sfn > smn:
            continue
        if lpr < 0:
            continue
        if not (0.0 <= lprmr <= 1.0):
            continue
        grid.append(
            SweepParam(
                param_id=param_id,
                langevin_lr=lr,
                langevin_steps=ls,
                tau=tau,
                lr_min_ratio=lmr,
                lambda_prior=lpr,
                lambda_prior_min_ratio=lprmr,
                annealing_steps=an,
                sigma_max=smx,
                sigma_min=smn,
                sigma_final=sfn,
            )
        )
        param_id += 1
    return grid


def _load_eval_dataset(cfg, image_size=200):
    v_torch_seam = load_seam_model(cfg.paths.seam_model_path)
    data = create_velocity_dataset(v_torch_seam, image_size=image_size)
    return data, v_torch_seam


def _select_x0_candidate_from_sm(
    sm_info,
    measurement,
    operator,
    top_k=50,
    candidate_mode='gss_topk',
    generator=None,
):
    """
    给定 d_obs，使用 SM 选择候选 x0（未执行 DAPS refine）。
    """
    device = measurement.device
    sm = sm_info['similarity_matrix'].to(device)  # [k, n_seeds]
    x0hat_batch = sm_info['x0hat_batch'].to(device)  # [n_seeds, C, H, W]
    d_centroids_2d = sm_info['d_centroids_2d'].to(device)  # [k, D]
    measurement_flat = measurement.reshape(1, -1)  # [1, D]

    # Step 1: direct-match group
    centroid_distances = torch.norm(d_centroids_2d - measurement_flat, dim=1)
    best_group = int(torch.argmin(centroid_distances).item())
    centroid_distance = float(centroid_distances[best_group].item())

    # Step 2: 候选索引
    n_seeds = int(sm.shape[1])
    n_select = max(1, min(int(top_k), n_seeds))
    if candidate_mode == 'random':
        if generator is None:
            generator = torch.Generator(device='cpu')
        perm = torch.randperm(n_seeds, generator=generator, device='cpu')[:n_select]
        top_indices = perm.to(device)
    elif candidate_mode == 'all_seed':
        top_indices = torch.arange(n_seeds, device=device, dtype=torch.long)
    else:
        _, top_indices = torch.topk(sm[best_group], n_select, largest=False)

    # Step 3: 候选与 d_obs 匹配
    if 'd_samples_2d' in sm_info:
        d_samples_2d = sm_info['d_samples_2d'].to(device)  # [n_seeds, D]
        cand_flat = d_samples_2d[top_indices]
    else:
        candidates = x0hat_batch[top_indices]
        n_curr = int(top_indices.shape[0])
        candidates_high = F.interpolate(candidates, size=(128, 128), mode='bilinear', align_corners=True)
        with torch.no_grad():
            d_candidates = operator(candidates_high)
        cand_flat = d_candidates.reshape(n_curr, -1)

    candidate_distances = torch.norm(cand_flat - measurement_flat, dim=1)  # [n_select]
    best_local_idx = int(torch.argmin(candidate_distances).item())
    best_global_idx = int(top_indices[best_local_idx].item())
    best_candidate = x0hat_batch[best_global_idx].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_global_idx,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'centroid_distance': centroid_distance,
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
    }
    return best_candidate, meta


def _refine_with_custom_daps(
    model,
    operator,
    measurement,
    init_x0,
    param: SweepParam,
    return_trajectory: bool = False,
):
    """
    使用指定参数执行 DAPS-FWI refine。
    """
    from DAPS.sampler import Scheduler, DiffusionSampler, LangevinDynamics

    # 使用与当前模型一致的 schedule/timestep 族，仅替换关键超参
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

    xt = init_x0
    trajectory = []
    for step in range(annealing.num_steps):
        sigma = float(annealing.sigma_steps[step])
        diff_scheduler = Scheduler(**model.daps.diffusion_scheduler_config, sigma_max=sigma)
        diff_sampler = DiffusionSampler(diff_scheduler)
        x0hat = diff_sampler.sample(model, xt, SDE=False, verbose=False)
        if return_trajectory:
            trajectory.append({
                'step': step,
                'sigma': sigma,
                'pred': x0hat.detach().cpu(),
            })

        ratio = float(step) / float(max(1, annealing.num_steps))
        x0y = lgvd.sample(x0hat, operator, measurement, sigma, ratio)

        next_sigma = float(annealing.sigma_steps[step + 1])
        xt = x0y + torch.randn_like(x0y) * next_sigma

    if return_trajectory:
        return x0hat, trajectory
    return x0hat


def _build_all_seed_short_param(model, args):
    """构建 all-seed 预筛选用的短程 DAPS 参数。"""
    base_annealing = model.daps.annealing_scheduler
    base_lgvd = model.daps.lgvd

    sigma_max = float(getattr(base_annealing, 'sigma_max', 0.3))
    sigma_min = float(getattr(base_annealing, 'sigma_min', 0.05))
    sigma_final = float(getattr(base_annealing, 'sigma_final', 0.0))
    sigma_min = min(max(1e-6, sigma_min), max(1e-6, sigma_max - 1e-6))
    sigma_final = min(max(0.0, sigma_final), sigma_min)

    return SweepParam(
        param_id=-1,
        langevin_lr=float(getattr(base_lgvd, 'lr', 3e-4)),
        langevin_steps=int(args.all_seed_short_langevin_steps),
        tau=float(getattr(base_lgvd, 'tau', 0.07)),
        lr_min_ratio=float(getattr(base_lgvd, 'lr_min_ratio', 1.0)),
        lambda_prior=float(getattr(base_lgvd, 'lambda_prior', 1.0)),
        lambda_prior_min_ratio=float(getattr(base_lgvd, 'lambda_prior_min_ratio', 1.0)),
        annealing_steps=int(args.all_seed_short_annealing_steps),
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_final=sigma_final,
    )


def _all_seed_multistart_short_daps_pick(
    model,
    operator,
    measurement,
    x0hat_batch,
    top_indices: torch.Tensor,
    top_candidate_distances: torch.Tensor,
    short_param: SweepParam,
    top_n: int,
    seed_base: int,
):
    """
    在 all-seed 候选中，先按 d_obs 距离取 top-N，再做短程 DAPS 多起点筛选。
    返回最优 refined x0 与补充元信息。
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
        x_short = _refine_with_custom_daps(
            model=model,
            operator=operator,
            measurement=measurement,
            init_x0=init_x0,
            param=short_param,
            return_trajectory=False,
        ).detach()

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


def _save_rows_csv(path, rows, headers):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values):
    if not values:
        return float('nan')
    return sum(values) / len(values)


def _summarize(rows):
    grouped = {}
    for row in rows:
        pid = int(row['param_id'])
        grouped.setdefault(pid, []).append(row)

    summary = []
    for pid, items in grouped.items():
        base = items[0]
        mean_delta_rel = _mean([float(x['delta_rel_l2']) for x in items])
        std_delta_rel = float(np.std([float(x['delta_rel_l2']) for x in items], ddof=0))
        mean_delta_ssim = _mean([float(x['delta_ssim']) for x in items])
        mean_delta_tv = _mean([float(x['delta_tv']) for x in items])
        mean_low_freq_ratio = _mean([float(x['delta_low_freq_ratio']) for x in items])
        mean_high_freq_ratio = _mean([float(x['delta_high_freq_ratio']) for x in items])
        mean_low_high_ratio = _mean([float(x['delta_low_high_ratio']) for x in items])
        mean_nrmse = _mean([float(x['gt_nrmse']) for x in items])
        mean_ssim = _mean([float(x['gt_ssim']) for x in items])
        mean_psnr = _mean([float(x['gt_psnr']) for x in items])
        mean_x0_nrmse = _mean([float(x['x0_gt_nrmse']) for x in items])
        mean_x0_ssim = _mean([float(x['x0_gt_ssim']) for x in items])
        mean_nrmse_improve = _mean([float(x['gt_nrmse_improve']) for x in items])
        mean_ratio = _mean([float(x['misfit_ratio']) for x in items])
        pass_rate = _mean([float(x['is_structural_variant']) for x in items])
        mean_time = _mean([float(x['runtime_s']) for x in items])

        # 越大越好: 偏离更大且误差更低
        tradeoff = mean_delta_rel / max(1e-8, mean_nrmse)
        robust_tradeoff = tradeoff * pass_rate

        summary.append({
            'param_id': pid,
            'langevin_lr': base['langevin_lr'],
            'langevin_steps': base['langevin_steps'],
            'tau': base['tau'],
            'lr_min_ratio': base['lr_min_ratio'],
            'lambda_prior': base['lambda_prior'],
            'lambda_prior_min_ratio': base['lambda_prior_min_ratio'],
            'annealing_steps': base['annealing_steps'],
            'sigma_max': base['sigma_max'],
            'sigma_min': base['sigma_min'],
            'sigma_final': base['sigma_final'],
            'mean_delta_rel_l2': mean_delta_rel,
            'std_delta_rel_l2': std_delta_rel,
            'mean_delta_ssim': mean_delta_ssim,
            'mean_delta_tv': mean_delta_tv,
            'mean_delta_low_freq_ratio': mean_low_freq_ratio,
            'mean_delta_high_freq_ratio': mean_high_freq_ratio,
            'mean_delta_low_high_ratio': mean_low_high_ratio,
            'mean_gt_nrmse': mean_nrmse,
            'mean_gt_ssim': mean_ssim,
            'mean_gt_psnr': mean_psnr,
            'mean_x0_gt_nrmse': mean_x0_nrmse,
            'mean_x0_gt_ssim': mean_x0_ssim,
            'mean_gt_nrmse_improve': mean_nrmse_improve,
            'mean_misfit_after_before_ratio': mean_ratio,
            'mean_structural_pass_rate': pass_rate,
            'mean_runtime_s': mean_time,
            'tradeoff_score': tradeoff,
            'robust_tradeoff_score': robust_tradeoff,
            'n_evals': len(items),
        })
    return summary


def _build_run_seed(args, param: SweepParam, gt_idx: int, repeat_id: int) -> int:
    """
    构建可复现 seed。
    - shared_across_params: 同一 GT/repeat 跨参数共享 seed（推荐，便于公平比较）。
    - param_dependent: 兼容旧行为，seed 随 param_id 变化。
    - param_hash: seed 随参数值变化，但不依赖 param_id（跨网格更稳定）。
    """
    if args.seed_mode == 'param_dependent':
        return int(args.master_seed + param.param_id * 100003 + gt_idx * 1009 + repeat_id)
    if args.seed_mode == 'param_hash':
        key = (
            f"{param.langevin_lr:.12g}|{param.langevin_steps}|{param.tau:.12g}|"
            f"{param.lr_min_ratio:.12g}|{param.lambda_prior:.12g}|{param.lambda_prior_min_ratio:.12g}|"
            f"{param.annealing_steps}|{param.sigma_max:.12g}|{param.sigma_min:.12g}|{param.sigma_final:.12g}"
        )
        hv = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)
        return int(args.master_seed + hv + gt_idx * 1009 + repeat_id)
    # default: shared_across_params
    return int(args.master_seed + gt_idx * 1009 + repeat_id)


def _save_pass_visualizations(pass_vis_rows, output_dir, model_tag, max_per_fig=20):
    """
    可视化通过门控的反演结果（x_refined），每张子图标注 id 和 nrmse_improve。
    """
    if not pass_vis_rows:
        print("\n[pass_vis] 无通过门控结果，跳过可视化。")
        return []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[pass_vis] matplotlib 不可用，跳过可视化: {e}")
        return []

    items = sorted(
        pass_vis_rows,
        key=lambda x: (-float(x['gt_nrmse_improve']), int(x['gt_index']), int(x['param_id']), int(x['repeat_id']))
    )
    max_per_fig = int(max_per_fig)
    if max_per_fig <= 0:
        max_per_fig = len(items)

    save_paths = []
    for start in range(0, len(items), max_per_fig):
        chunk = items[start:start + max_per_fig]
        n = len(chunk)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')

        for idx, item in enumerate(chunk):
            ax = axes.flat[idx]
            pred = item['pred']
            im = ax.imshow(pred, cmap='jet', aspect='auto')
            ax.set_title(
                f"id={item['param_id']}  nrmse_improve={item['gt_nrmse_improve']:.4f}",
                fontsize=9
            )
            ax.text(
                0.02, 0.02,
                f"GT#{item['gt_index']}  r{item['repeat_id']}",
                transform=ax.transAxes,
                fontsize=8,
                color='white',
                bbox={'facecolor': 'black', 'alpha': 0.45, 'pad': 1.8, 'edgecolor': 'none'}
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        total_pages = (len(items) + max_per_fig - 1) // max_per_fig
        page_idx = start // max_per_fig + 1
        fig.suptitle(
            f"Passing Gated Inversions ({model_tag}) - page {page_idx}/{total_pages}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        suffix = f"_p{page_idx:02d}" if total_pages > 1 else ""
        filename = generate_timestamped_filename(
            f'daps_param_search_pass_vis_model-{model_tag}{suffix}', '.png'
        )
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        save_paths.append(save_path)

    return save_paths


def _fmt_float_name_token(v: float) -> str:
    """将浮点数转为文件名安全 token。"""
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


def _save_run_trajectory_visualization(
    output_dir,
    model_tag,
    param: SweepParam,
    gt_idx: int,
    repeat_id: int,
    gt: torch.Tensor,
    x0: torch.Tensor,
    trajectory,
    n_snapshots: int = 10,
    nrmse_improve: float | None = None,
    misfit_ratio: float | None = None,
    is_structural_variant: int | None = None,
):
    """
    可视化单次运行轨迹：GT + x0 + 均匀抽样的 10 张 annealing 快照。
    """
    if not trajectory:
        return None

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[all_vis] matplotlib 不可用，跳过轨迹可视化: {e}")
        return None

    gt4 = _to_4d(gt).float().cpu()
    x04, _ = _align_pred_to_gt(_to_4d(x0).float().cpu(), gt4)

    panels = [('GT', gt4[0, 0].numpy()), ('x0 input', x04[0, 0].numpy())]

    sel_idx = _uniform_indices(len(trajectory), n_snapshots)
    for idx in sel_idx:
        snap = trajectory[idx]
        pred4, _ = _align_pred_to_gt(_to_4d(snap['pred']).float().cpu(), gt4)
        panels.append(
            (
                f"step {int(snap['step']) + 1}/{len(trajectory)}\n"
                f"sigma={float(snap['sigma']):.4g}",
                pred4[0, 0].numpy(),
            )
        )

    n = len(panels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for i, (title, img) in enumerate(panels):
        ax = axes.flat[i]
        ax.imshow(img, cmap='jet', aspect='auto')
        ax.set_title(title, fontsize=9)

    meta = [
        f"id={param.param_id}",
        f"GT#{gt_idx}",
        f"repeat={repeat_id}",
        f"tau={param.tau:.4g}",
        f"an={param.annealing_steps}",
        f"ls={param.langevin_steps}",
        f"smax={param.sigma_max:.4g}",
        f"smin={param.sigma_min:.4g}",
    ]
    if nrmse_improve is not None:
        meta.append(f"nrmse_improve={float(nrmse_improve):+.4f}")
    if misfit_ratio is not None:
        meta.append(f"misfit_ratio={float(misfit_ratio):.4f}")
    if is_structural_variant is not None:
        meta.append(f"pass={int(is_structural_variant)}")
    fig.suptitle(" | ".join(meta), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = (
        f"daps_param_search_traj_model-{model_tag}"
        f"_grid_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_grid_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_grid_tau-{_fmt_float_name_token(param.tau)}"
        f"_grid_annealing_steps-{int(param.annealing_steps)}"
        f"_grid_langevin_steps-{int(param.langevin_steps)}"
        f"_pid-{int(param.param_id)}_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    filename = generate_timestamped_filename(base, '.png')
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description='DAPS 参数搜索（固定 GSS 候选 x0）')
    parser.add_argument('--model_tag', type=str, default='seam_finetune',
                        choices=['seam', 'seam_finetune', 'marmousi'],
                        help='当前评估模型标签')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='checkpoint目录（默认 project_root/checkpoints）')
    parser.add_argument('--ckpt_file', type=str, default=None,
                        help='checkpoint文件名或绝对路径')
    parser.add_argument('--sm_path', type=str, required=True,
                        help='对应模型的 SM 资产路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认 project_root/outputs）')
    parser.add_argument('--gt_indices', type=int, nargs='+', default=[1, 11, 21, 31, 41],
                        help='评估 GT 索引列表')
    parser.add_argument('--gss_top_k', type=int, default=50,
                        help='组内候选数量（candidate_mode=all_seed 时忽略）')
    parser.add_argument('--candidate_mode', type=str, default='gss_topk',
                        choices=['gss_topk', 'all_seed', 'random'],
                        help='x0 候选选择方式')
    parser.add_argument('--all_seed_short_top_n', type=int, default=20,
                        help='[all_seed] 预筛选时按 d_obs 距离取前 N 个 seed 做短程 DAPS')
    parser.add_argument('--all_seed_short_annealing_steps', type=int, default=6,
                        help='[all_seed] 预筛选短程 DAPS 的 annealing 步数')
    parser.add_argument('--all_seed_short_langevin_steps', type=int, default=4,
                        help='[all_seed] 预筛选短程 DAPS 的 Langevin 步数')
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='每组参数在每个 GT 上重复次数（不同随机噪声）')
    parser.add_argument('--master_seed', type=int, default=8,
                        help='主随机种子')
    parser.add_argument('--seed_mode', type=str, default='shared_across_params',
                        choices=['shared_across_params', 'param_dependent', 'param_hash'],
                        help='随机种子策略: shared_across_params(推荐) / param_dependent(旧行为) / param_hash')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='DAPS 正演噪声 sigma')
    parser.add_argument('--low_freq_radius_ratio', type=float, default=0.15,
                        help='频域低频圆半径比例（结构门用）')
    parser.add_argument('--min_delta_rel_l2', type=float, default=0.02,
                        help='结构变体判定: 最小相对偏移阈值')
    parser.add_argument('--max_misfit_ratio', type=float, default=1.05,
                        help='结构变体判定: 最大数据域误差比例 after/before')
    parser.add_argument('--max_high_freq_ratio', type=float, default=0.75,
                        help='结构变体判定: 最大高频能量占比')
    parser.add_argument('--skip_pass_vis', action='store_true',
                        help='跳过通过门控结果可视化')
    parser.add_argument('--pass_vis_max_per_fig', type=int, default=20,
                        help='每张图最多显示多少个通过门控结果（<=0 表示全部）')
    parser.add_argument('--vis_all_results', action='store_true',
                        help='可视化全部运行结果（不依赖门控）')
    parser.add_argument('--vis_all_num_snapshots', type=int, default=10,
                        help='all_results 可视化中均匀抽样的 annealing 快照数量')
    parser.add_argument('--vis_all_max_runs', type=int, default=0,
                        help='all_results 最多可视化的运行数（<=0 表示不限制）')

    # 网格参数（逗号分隔）
    parser.add_argument('--grid_langevin_lr', type=str, default='1e-4',
                        help='Langevin lr 列表, 逗号分隔')
    parser.add_argument('--grid_langevin_steps', type=str, default='20',
                        help='Langevin 步数列表, 逗号分隔')
    parser.add_argument('--grid_tau', type=str, default='0.07',
                        help='tau 列表, 逗号分隔')
    parser.add_argument('--grid_lr_min_ratio', type=str, default='1.0',
                        help='lr_min_ratio 列表, 逗号分隔')
    parser.add_argument('--grid_lambda_prior', type=str, default='1.0',
                        help='lambda_prior 列表, 逗号分隔')
    parser.add_argument('--grid_lambda_prior_min_ratio', type=str, default='1.0',
                        help='lambda_prior_min_ratio 列表, 逗号分隔')
    parser.add_argument('--grid_annealing_steps', type=str, default='50',
                        help='annealing 步数列表, 逗号分隔')
    parser.add_argument('--grid_sigma_max', type=str, default='0.1',
                        help='annealing sigma_max 列表, 逗号分隔')
    parser.add_argument('--grid_sigma_min', type=str, default='0.01',
                        help='annealing sigma_min 列表, 逗号分隔')
    parser.add_argument('--grid_sigma_final', type=str, default='0.0',
                        help='annealing sigma_final 列表, 逗号分隔')

    args = parser.parse_args()
    if args.n_repeats < 1:
        raise ValueError("--n_repeats 必须 >= 1")
    if args.all_seed_short_top_n < 1:
        raise ValueError("--all_seed_short_top_n 必须 >= 1")
    if args.all_seed_short_annealing_steps < 1 or args.all_seed_short_langevin_steps < 1:
        raise ValueError("--all_seed_short_annealing_steps / --all_seed_short_langevin_steps 必须 >= 1")
    if not (0.0 < args.low_freq_radius_ratio < 1.0):
        raise ValueError("--low_freq_radius_ratio 必须在 (0, 1) 内")
    if args.max_misfit_ratio <= 0:
        raise ValueError("--max_misfit_ratio 必须 > 0")
    if not (0.0 <= args.max_high_freq_ratio <= 1.0):
        raise ValueError("--max_high_freq_ratio 必须在 [0, 1] 内")
    if args.vis_all_num_snapshots < 1:
        raise ValueError("--vis_all_num_snapshots 必须 >= 1")

    cfg = FWIConfig()
    cfg.daps.sigma = args.sigma
    device = cfg.device

    ckpt_path = _resolve_checkpoint_path(cfg, args)
    sm_info = _load_sm_info(args.sm_path, expected_model_tag=args.model_tag)
    output_dir = args.output_dir or cfg.paths.output_path
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("DAPS 参数搜索（固定 GSS 候选 x0）")
    print("=" * 72)
    print(f"  model_tag: {args.model_tag}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  SM: {args.sm_path}")
    print(f"  device: {device}")
    print(f"  gt_indices: {args.gt_indices}")
    print(f"  candidate_mode: {args.candidate_mode}")
    print(f"  gss_top_k: {args.gss_top_k}")
    if args.candidate_mode == 'all_seed':
        print(
            "  all_seed_multistart: "
            f"top_n={args.all_seed_short_top_n}, "
            f"short_annealing_steps={args.all_seed_short_annealing_steps}, "
            f"short_langevin_steps={args.all_seed_short_langevin_steps}"
        )
    print(f"  n_repeats: {args.n_repeats}")
    print(f"  seed_mode: {args.seed_mode}")
    print(f"  pass_vis: {'off' if args.skip_pass_vis else 'on'} (max_per_fig={args.pass_vis_max_per_fig})")
    print(
        f"  all_vis: {'on' if args.vis_all_results else 'off'} "
        f"(snapshots={args.vis_all_num_snapshots}, max_runs={args.vis_all_max_runs})"
    )
    print(
        "  new_controls: "
        f"lambda_prior={args.grid_lambda_prior} "
        f"lambda_prior_min_ratio={args.grid_lambda_prior_min_ratio} "
        f"sigma_final={args.grid_sigma_final}"
    )
    print(
        "  structural_gate: "
        f"delta_rel_l2>={args.min_delta_rel_l2}, "
        f"misfit_ratio<={args.max_misfit_ratio}, "
        f"high_freq_ratio<={args.max_high_freq_ratio}, "
        f"low_freq_radius_ratio={args.low_freq_radius_ratio}"
    )

    param_grid = _build_param_grid(args)
    if not param_grid:
        raise ValueError("参数网格为空，请检查 grid_* 参数。")
    print(f"  param_sets: {len(param_grid)}")

    # 1) 加载评估数据（统一在 SEAM 域评估）
    print("\n加载 SEAM 评估域数据...")
    seam_data, seam_v = _load_eval_dataset(cfg, image_size=200)
    print(f"  dataset_size: {len(seam_data)}")

    # 2) 构建模型
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
    all_seed_x0hat_batch = None
    if args.candidate_mode == 'all_seed':
        all_seed_short_param = _build_all_seed_short_param(model, args)
        all_seed_x0hat_batch = sm_info['x0hat_batch'].to(device)
        print(
            "  all-seed preselect short-DAPS param: "
            f"lr={all_seed_short_param.langevin_lr:.4g}, "
            f"tau={all_seed_short_param.tau:.4g}, "
            f"an={all_seed_short_param.annealing_steps}, "
            f"ls={all_seed_short_param.langevin_steps}, "
            f"smax={all_seed_short_param.sigma_max:.4g}, "
            f"smin={all_seed_short_param.sigma_min:.4g}, "
            f"sfin={all_seed_short_param.sigma_final:.4g}"
        )

    # 3) 为每个 GT 固定 x0 候选（与参数无关）
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
            candidate_mode=args.candidate_mode,
            generator=gen,
        )
        if args.candidate_mode == 'all_seed':
            x0, short_meta = _all_seed_multistart_short_daps_pick(
                model=model,
                operator=operator,
                measurement=measurement,
                x0hat_batch=all_seed_x0hat_batch,
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
            'meas_misfit_before': misfit_before,
            'x0_gt_nrmse': x0_gt_nrmse,
            'x0_gt_ssim': x0_gt_ssim,
            'x0_gt_psnr': x0_gt_psnr,
        }
        extra = ""
        if args.candidate_mode == 'all_seed':
            extra = f", short_top_n={meta['short_top_n']}, short_best_misfit={meta['short_best_misfit']:.4f}"
        print(
            f"  GT#{gt_idx}: group={meta['best_group']}, seed={meta['best_seed']}, "
            f"cand_dist={meta['candidate_distance']:.4f}, misfit_before={misfit_before:.4f}, "
            f"x0_gt_nrmse={x0_gt_nrmse:.4f}, x0_gt_ssim={x0_gt_ssim:.4f}{extra}"
        )

    # 4) 网格搜索
    rows = []
    pass_vis_rows = []
    all_vis_paths = []
    total_runs = len(param_grid) * len(args.gt_indices) * args.n_repeats
    pbar = tqdm(total=total_runs, desc='DAPS Param Sweep')
    for param in param_grid:
        for gt_idx in args.gt_indices:
            gt = x0_cache[gt_idx]['gt'].to(device)
            measurement = x0_cache[gt_idx]['measurement'].to(device)
            x0 = x0_cache[gt_idx]['x0'].to(device)
            meta = x0_cache[gt_idx]['meta']
            misfit_before = float(x0_cache[gt_idx]['meas_misfit_before'])
            x0_gt_nrmse = float(x0_cache[gt_idx]['x0_gt_nrmse'])
            x0_gt_ssim = float(x0_cache[gt_idx]['x0_gt_ssim'])
            x0_gt_psnr = float(x0_cache[gt_idx]['x0_gt_psnr'])

            for repeat_id in range(args.n_repeats):
                # 保证不同 param/gt/repeat 的随机性可复现
                seed = _build_run_seed(args, param, gt_idx, repeat_id)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                t0 = time.time()
                if args.vis_all_results:
                    x_refined, trajectory = _refine_with_custom_daps(
                        model,
                        operator,
                        measurement,
                        x0,
                        param,
                        return_trajectory=True,
                    )
                    x_refined = x_refined.detach()
                else:
                    x_refined = _refine_with_custom_daps(
                        model,
                        operator,
                        measurement,
                        x0,
                        param,
                        return_trajectory=False,
                    ).detach()
                    trajectory = None
                dt = time.time() - t0

                # 与 x0 偏离
                delta = x_refined - x0
                delta_l2 = torch.norm(delta).item()
                delta_rel_l2 = delta_l2 / max(1e-8, torch.norm(x0).item())
                delta_nrmse = compute_nrmse(x_refined, x0)
                delta_ssim = compute_ssim(x_refined, x0)
                delta_psnr = compute_psnr(x_refined, x0)
                delta_tv = compute_normalized_tv(delta)
                delta_low_ratio, delta_high_ratio, delta_low_high_ratio = compute_delta_frequency_ratios(
                    delta, low_radius_ratio=args.low_freq_radius_ratio
                )

                # 与 GT 质量
                gt_nrmse = compute_nrmse(x_refined, gt)
                gt_ssim = compute_ssim(x_refined, gt)
                gt_psnr = compute_psnr(x_refined, gt)
                gt_nrmse_improve = x0_gt_nrmse - gt_nrmse

                # 数据域误差变化
                with torch.no_grad():
                    d_refined = operator(x_refined)
                    misfit_after = torch.norm(d_refined - measurement).item()

                misfit_ratio = misfit_after / max(1e-8, misfit_before)
                is_structural_variant = int(
                    delta_rel_l2 >= args.min_delta_rel_l2
                    and misfit_ratio <= args.max_misfit_ratio
                    and delta_high_ratio <= args.max_high_freq_ratio
                )

                rows.append({
                    'param_id': param.param_id,
                    'repeat_id': repeat_id,
                    'run_seed': seed,
                    'gt_index': gt_idx,
                    'best_group': meta['best_group'],
                    'best_seed': meta['best_seed'],
                    'candidate_distance': meta['candidate_distance'],
                    'langevin_lr': param.langevin_lr,
                    'langevin_steps': param.langevin_steps,
                    'tau': param.tau,
                    'lr_min_ratio': param.lr_min_ratio,
                    'lambda_prior': param.lambda_prior,
                    'lambda_prior_min_ratio': param.lambda_prior_min_ratio,
                    'annealing_steps': param.annealing_steps,
                    'sigma_max': param.sigma_max,
                    'sigma_min': param.sigma_min,
                    'sigma_final': param.sigma_final,
                    'delta_l2': delta_l2,
                    'delta_rel_l2': delta_rel_l2,
                    'delta_nrmse': delta_nrmse,
                    'delta_ssim': delta_ssim,
                    'delta_psnr': delta_psnr,
                    'delta_tv': delta_tv,
                    'delta_low_freq_ratio': delta_low_ratio,
                    'delta_high_freq_ratio': delta_high_ratio,
                    'delta_low_high_ratio': delta_low_high_ratio,
                    'x0_gt_nrmse': x0_gt_nrmse,
                    'x0_gt_ssim': x0_gt_ssim,
                    'x0_gt_psnr': x0_gt_psnr,
                    'gt_nrmse': gt_nrmse,
                    'gt_ssim': gt_ssim,
                    'gt_psnr': gt_psnr,
                    'gt_nrmse_improve': gt_nrmse_improve,
                    'meas_misfit_before': misfit_before,
                    'meas_misfit_after': misfit_after,
                    'misfit_ratio': misfit_ratio,
                    'is_structural_variant': is_structural_variant,
                    'runtime_s': dt,
                })

                if is_structural_variant and not args.skip_pass_vis:
                    pred_aligned, _ = _align_pred_to_gt(
                        x_refined.detach().cpu(),
                        gt.detach().cpu()
                    )
                    pass_vis_rows.append({
                        'param_id': param.param_id,
                        'repeat_id': repeat_id,
                        'gt_index': gt_idx,
                        'gt_nrmse_improve': gt_nrmse_improve,
                        'pred': pred_aligned[0, 0].numpy(),
                    })

                if args.vis_all_results:
                    can_save_all_vis = (args.vis_all_max_runs <= 0) or (len(all_vis_paths) < args.vis_all_max_runs)
                    vis_path = _save_run_trajectory_visualization(
                        output_dir=output_dir,
                        model_tag=args.model_tag,
                        param=param,
                        gt_idx=gt_idx,
                        repeat_id=repeat_id,
                        gt=gt.detach().cpu(),
                        x0=x0.detach().cpu(),
                        trajectory=trajectory,
                        n_snapshots=args.vis_all_num_snapshots,
                        nrmse_improve=gt_nrmse_improve,
                        misfit_ratio=misfit_ratio,
                        is_structural_variant=is_structural_variant,
                    ) if can_save_all_vis else None
                    if vis_path:
                        all_vis_paths.append(vis_path)
                pbar.update(1)
    pbar.close()

    # 5) 保存与汇总
    detail_headers = [
        'param_id', 'repeat_id', 'run_seed', 'gt_index', 'best_group', 'best_seed', 'candidate_distance',
        'langevin_lr', 'langevin_steps', 'tau', 'lr_min_ratio', 'lambda_prior', 'lambda_prior_min_ratio',
        'annealing_steps', 'sigma_max', 'sigma_min', 'sigma_final',
        'delta_l2', 'delta_rel_l2', 'delta_nrmse', 'delta_ssim', 'delta_psnr',
        'delta_tv', 'delta_low_freq_ratio', 'delta_high_freq_ratio', 'delta_low_high_ratio',
        'x0_gt_nrmse', 'x0_gt_ssim', 'x0_gt_psnr',
        'gt_nrmse', 'gt_ssim', 'gt_psnr',
        'gt_nrmse_improve',
        'meas_misfit_before', 'meas_misfit_after', 'misfit_ratio',
        'is_structural_variant', 'runtime_s'
    ]
    detail_name = generate_timestamped_filename(
        f'daps_param_search_detail_model-{args.model_tag}', '.csv'
    )
    detail_path = os.path.join(output_dir, detail_name)
    _save_rows_csv(detail_path, rows, detail_headers)

    summary_rows = _summarize(rows)
    summary_headers = [
        'param_id', 'langevin_lr', 'langevin_steps', 'tau', 'lr_min_ratio', 'lambda_prior', 'lambda_prior_min_ratio',
        'annealing_steps', 'sigma_max', 'sigma_min', 'sigma_final',
        'mean_delta_rel_l2', 'std_delta_rel_l2', 'mean_delta_ssim', 'mean_delta_tv',
        'mean_delta_low_freq_ratio', 'mean_delta_high_freq_ratio', 'mean_delta_low_high_ratio',
        'mean_x0_gt_nrmse', 'mean_x0_gt_ssim',
        'mean_gt_nrmse', 'mean_gt_ssim', 'mean_gt_psnr',
        'mean_gt_nrmse_improve',
        'mean_misfit_after_before_ratio', 'mean_structural_pass_rate', 'mean_runtime_s',
        'tradeoff_score', 'robust_tradeoff_score', 'n_evals'
    ]
    summary_name = generate_timestamped_filename(
        f'daps_param_search_summary_model-{args.model_tag}', '.csv'
    )
    summary_path = os.path.join(output_dir, summary_name)
    _save_rows_csv(summary_path, summary_rows, summary_headers)

    by_robust = sorted(summary_rows, key=lambda x: float(x['robust_tradeoff_score']), reverse=True)
    by_tradeoff = sorted(summary_rows, key=lambda x: float(x['tradeoff_score']), reverse=True)
    by_quality = sorted(summary_rows, key=lambda x: float(x['mean_gt_nrmse']))
    by_novelty = sorted(summary_rows, key=lambda x: float(x['mean_delta_rel_l2']), reverse=True)

    print("\n" + "=" * 72)
    print("参数搜索完成")
    print("=" * 72)
    print(f"detail_csv: {detail_path}")
    print(f"summary_csv: {summary_path}")

    def _fmt(row):
        return (
            f"id={row['param_id']} "
            f"(lr={row['langevin_lr']}, ls={row['langevin_steps']}, tau={row['tau']}, "
            f"lprior={row['lambda_prior']}, lpmr={row['lambda_prior_min_ratio']}, "
            f"an={row['annealing_steps']}, smax={row['sigma_max']}, smin={row['sigma_min']}, sfin={row['sigma_final']}) "
            f"delta_rel={float(row['mean_delta_rel_l2']):.4f}, "
            f"x0_nrmse={float(row['mean_x0_gt_nrmse']):.4f}, "
            f"gt_nrmse={float(row['mean_gt_nrmse']):.4f}, "
            f"nrmse_improve={float(row['mean_gt_nrmse_improve']):.4f}, "
            f"gt_ssim={float(row['mean_gt_ssim']):.4f}, "
            f"misfit_ratio={float(row['mean_misfit_after_before_ratio']):.4f}, "
            f"pass_rate={float(row['mean_structural_pass_rate']):.3f}, "
            f"tradeoff={float(row['tradeoff_score']):.4f}, "
            f"robust={float(row['robust_tradeoff_score']):.4f}"
        )

    print("\nTop-5 by robust_tradeoff_score (tradeoff * pass_rate):")
    for row in by_robust[:5]:
        print("  " + _fmt(row))

    print("\nTop-5 by tradeoff_score (delta_rel / gt_nrmse):")
    for row in by_tradeoff[:5]:
        print("  " + _fmt(row))

    print("\nTop-5 by best quality (lowest mean_gt_nrmse):")
    for row in by_quality[:5]:
        print("  " + _fmt(row))

    print("\nTop-5 by novelty (highest mean_delta_rel_l2):")
    for row in by_novelty[:5]:
        print("  " + _fmt(row))

    if args.skip_pass_vis:
        print("\n[pass_vis] skip_pass_vis=True，已跳过可视化。")
    else:
        vis_paths = _save_pass_visualizations(
            pass_vis_rows=pass_vis_rows,
            output_dir=output_dir,
            model_tag=args.model_tag,
            max_per_fig=args.pass_vis_max_per_fig,
        )
        if vis_paths:
            print("\n通过门控结果可视化已保存:")
            for p in vis_paths:
                print(f"  {p}")

    if args.vis_all_results:
        if all_vis_paths:
            print("\n全部结果轨迹可视化已保存:")
            for p in all_vis_paths:
                print(f"  {p}")
            if args.vis_all_max_runs > 0 and len(all_vis_paths) >= args.vis_all_max_runs:
                print(f"[all_vis] 已达到 vis_all_max_runs={args.vis_all_max_runs}，后续运行不再保存轨迹图。")
        else:
            print("\n[all_vis] vis_all_results=True，但未生成轨迹可视化。")


if __name__ == '__main__':
    main()
