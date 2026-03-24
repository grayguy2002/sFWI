"""
DAPS-FWI 过程机理探针（Experiment A 专用）。

设计目标：
1) 从 daps_langevin.py 中拆出“机理诊断最小闭环”，避免继续堆叠骨干脚本复杂度。
2) 固定候选 x0 后，记录 DAPS 内外循环的动力学量级与质量演化。
3) 输出可直接用于审稿回复的证据：CSV + 曲线PDF + 轨迹PDF。

说明：
- 不包含 ensemble 并行、不包含参数搜索。
- 可视化统一保存为 PDF（按用户要求）。
"""

from __future__ import annotations

import os
import sys
import time
import argparse

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
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.utils.file_utils import generate_timestamped_filename

# 复用已验证的核心逻辑，避免重复实现引入偏差
from sFWI.experiments.daps_langevin import (
    LangevinParam,
    _to_4d,
    _align_pred_to_gt,
    _resolve_checkpoint_path,
    _load_sm_info,
    _load_eval_dataset,
    _select_x0_candidate_from_sm,
    _build_run_seed,
    _compute_final_metrics,
    _save_rows_csv,
    _mean,
    compute_nrmse,
    compute_ssim,
    _fmt_float_name_token,
    _uniform_indices,
)

VELOCITY_CMAP = 'viridis'


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a2 = a.reshape(a.shape[0], -1)
    b2 = b.reshape(b.shape[0], -1)
    num = (a2 * b2).sum(dim=1)
    den = torch.norm(a2, dim=1) * torch.norm(b2, dim=1)
    cos = num / torch.clamp(den, min=1e-12)
    return float(cos.mean().item())


def _rmse_same_grid(pred: torch.Tensor, gt: torch.Tensor) -> float:
    err = pred - gt
    mse = torch.mean(err * err, dim=(1, 2, 3))
    return float(torch.sqrt(torch.clamp(mse, min=0.0)).mean().item())


def _run_langevin_probe_direction(
    model,
    operator,
    measurement: torch.Tensor,
    gt: torch.Tensor,
    x0: torch.Tensor,
    param: LangevinParam,
    return_trajectory: bool = True,
    update_data_weight: float = 1.0,
    update_prior_weight: float = 1.0,
):
    """
    DAPS 内外循环方向诊断版：
    - 记录更新方向与“指向 GT 方向”的余弦
    - 记录 drift/full step 对 GT-RMSE 的变化
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

    final_x0hat = xt.detach().clone()
    final_x0y = xt.detach().clone()

    for outer_step in range(annealing.num_steps):
        sigma = float(annealing.sigma_steps[outer_step])
        ratio = float(outer_step) / float(max(1, annealing.num_steps))
        lr = float(lgvd.get_lr(ratio))
        lambda_prior = float(lgvd.get_lambda_prior(ratio))
        sigma_eff = max(float(sigma), 1e-8)

        diff_scheduler = Scheduler(**model.daps.diffusion_scheduler_config, sigma_max=sigma)
        diff_sampler = DiffusionSampler(diff_scheduler)
        with torch.no_grad():
            x0hat = diff_sampler.sample(model, xt, SDE=False, verbose=False)
        x0hat_detach = x0hat.detach()

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
            grad_total_raw = grad_data + grad_prior
            grad_update = float(update_data_weight) * grad_data + float(update_prior_weight) * grad_prior

            grad_data_norm = float(torch.norm(grad_data).item())
            grad_prior_norm = float(torch.norm(grad_prior).item())
            grad_total_norm = float(torch.norm(grad_total_raw).item())
            grad_update_norm = float(torch.norm(grad_update).item())

            grad_dot = float((grad_data * grad_prior).sum().item())
            grad_cos = grad_dot / max(1e-12, grad_data_norm * grad_prior_norm)

            with torch.no_grad():
                x_before = x.detach()
                gt_small = F.interpolate(
                    _to_4d(gt).float(),
                    size=x_before.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
                to_gt = gt_small - x_before

                cos_data_to_gt = _safe_cosine(-grad_data, to_gt)
                cos_prior_to_gt = _safe_cosine(-grad_prior, to_gt)
                cos_update_to_gt = _safe_cosine(-grad_update, to_gt)

                rmse_before = _rmse_same_grid(x_before, gt_small)

                drift = -lr * grad_update
                drift_norm = float(torch.norm(drift).item())
                x_after_drift = x_before + drift
                rmse_after_drift = _rmse_same_grid(x_after_drift, gt_small)

                epsilon = torch.randn_like(x_before)
                noise = param.beta_langevin_noise * np.sqrt(2.0 * lr) * epsilon
                noise_norm = float(torch.norm(noise).item())

                x_after = x_after_drift + noise
                rmse_after_full = _rmse_same_grid(x_after, gt_small)

            data_loss_mean = float(data_loss_vec.mean().item())
            prior_loss_mean = float(prior_loss_vec.mean().item())
            total_loss_mean = float((data_loss_vec + prior_loss_vec).mean().item())
            meas_misfit_mean = float(torch.sqrt(torch.clamp(err_sq_vec, min=0.0)).mean().item())

            delta_rmse_drift = float(rmse_after_drift - rmse_before)
            delta_rmse_full = float(rmse_after_full - rmse_before)
            drift_toward_gt = int(delta_rmse_drift <= 0.0)
            full_toward_gt = int(delta_rmse_full <= 0.0)

            row = {
                'outer_step': int(outer_step),
                'inner_step': int(inner_step),
                'sigma': float(sigma),
                'ratio': float(ratio),
                'langevin_lr_effective': float(lr),
                'lambda_prior_effective': float(lambda_prior),
                'update_data_weight': float(update_data_weight),
                'update_prior_weight': float(update_prior_weight),
                'tau': float(param.tau),
                'data_loss': data_loss_mean,
                'prior_loss': prior_loss_mean,
                'total_loss': total_loss_mean,
                'prior_over_data_loss': float(prior_loss_mean / max(1e-12, data_loss_mean)),
                'measurement_misfit': meas_misfit_mean,
                'grad_data_norm': grad_data_norm,
                'grad_prior_norm': grad_prior_norm,
                'grad_total_norm': grad_total_norm,
                'grad_update_norm': grad_update_norm,
                'grad_data_prior_cos': float(grad_cos),
                'cos_data_to_gt': float(cos_data_to_gt),
                'cos_prior_to_gt': float(cos_prior_to_gt),
                'cos_update_to_gt': float(cos_update_to_gt),
                'gt_rmse_before': float(rmse_before),
                'gt_rmse_after_drift': float(rmse_after_drift),
                'gt_rmse_after_full': float(rmse_after_full),
                'delta_gt_rmse_drift': float(delta_rmse_drift),
                'delta_gt_rmse_full': float(delta_rmse_full),
                'drift_toward_gt': int(drift_toward_gt),
                'full_toward_gt': int(full_toward_gt),
                'drift_norm': float(drift_norm),
                'noise_norm': float(noise_norm),
                'noise_to_drift_ratio': float(noise_norm / max(1e-12, drift_norm)),
                'noise_to_grad_ratio': float(noise_norm / max(1e-12, grad_update_norm)),
            }
            inner_rows.append(row)
            per_outer.append(row)

            x = x_after

            if torch.isnan(x).any():
                x = torch.zeros_like(x)
                break

        x0y = x.detach()
        next_sigma = float(annealing.sigma_steps[outer_step + 1])
        with torch.no_grad():
            fw_eps = torch.randn_like(x0y)
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
            'mean_grad_update_norm': _mean([r['grad_update_norm'] for r in per_outer]),
            'mean_cos_data_to_gt': _mean([r['cos_data_to_gt'] for r in per_outer]),
            'mean_cos_prior_to_gt': _mean([r['cos_prior_to_gt'] for r in per_outer]),
            'mean_cos_update_to_gt': _mean([r['cos_update_to_gt'] for r in per_outer]),
            'mean_delta_gt_rmse_drift': _mean([r['delta_gt_rmse_drift'] for r in per_outer]),
            'mean_delta_gt_rmse_full': _mean([r['delta_gt_rmse_full'] for r in per_outer]),
            'frac_drift_toward_gt': _mean([r['drift_toward_gt'] for r in per_outer]),
            'frac_full_toward_gt': _mean([r['full_toward_gt'] for r in per_outer]),
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


def _save_inner_curve_pdf(
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
        print(f"[curve_vis] matplotlib 不可用，跳过曲线图: {e}")
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
    grad_update = np.array([r.get('grad_update_norm', r['grad_total_norm']) for r in inner_rows], dtype=np.float64)
    drift = np.array([r['drift_norm'] for r in inner_rows], dtype=np.float64)
    noise = np.array([r['noise_norm'] for r in inner_rows], dtype=np.float64)
    r_nd = np.array([r['noise_to_drift_ratio'] for r in inner_rows], dtype=np.float64)
    r_ng = np.array([r['noise_to_grad_ratio'] for r in inner_rows], dtype=np.float64)
    lrs = np.array([r['langevin_lr_effective'] for r in inner_rows], dtype=np.float64)
    lpriors = np.array([r['lambda_prior_effective'] for r in inner_rows], dtype=np.float64)
    cos_data = np.array([r.get('cos_data_to_gt', np.nan) for r in inner_rows], dtype=np.float64)
    cos_prior = np.array([r.get('cos_prior_to_gt', np.nan) for r in inner_rows], dtype=np.float64)
    cos_update = np.array([r.get('cos_update_to_gt', np.nan) for r in inner_rows], dtype=np.float64)
    d_rmse_drift = np.array([r.get('delta_gt_rmse_drift', np.nan) for r in inner_rows], dtype=np.float64)
    d_rmse_full = np.array([r.get('delta_gt_rmse_full', np.nan) for r in inner_rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

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
    ax.plot(steps, grad_total, label='||g_total(raw)||')
    ax.plot(steps, grad_update, label='||g_update||', alpha=0.8)
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
    ax.plot(steps, r_ng, label='noise/grad_update')
    ax2 = ax.twinx()
    ax2.plot(steps, lrs, '--', alpha=0.5, label='lr', color='tab:gray')
    ax2.plot(steps, lpriors, ':', alpha=0.6, label='lambda_prior', color='tab:orange')
    ax.set_title('Noise Ratios + Effective Schedules')
    ax.set_xlabel('inner global step')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    ax = axes[2, 0]
    ax.plot(steps, cos_data, label='cos(-g_data, to_GT)')
    ax.plot(steps, cos_prior, label='cos(-g_prior, to_GT)')
    ax.plot(steps, cos_update, label='cos(-g_update, to_GT)')
    ax.axhline(0.0, color='k', alpha=0.3, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('Direction Alignment to GT')
    ax.set_xlabel('inner global step')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(steps, d_rmse_drift, label='delta GT-RMSE (after drift)')
    ax.plot(steps, d_rmse_full, label='delta GT-RMSE (after full step)')
    ax.axhline(0.0, color='k', alpha=0.3, linewidth=0.8)
    ax.set_title('Per-step GT-RMSE Change')
    ax.set_xlabel('inner global step')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    for a in axes.flat:
        for k in range(1, int(param.annealing_steps)):
            xline = k * int(param.langevin_steps) - 0.5
            a.axvline(xline, color='k', alpha=0.08, linewidth=0.8)

    fig.suptitle(
        f"DAPS Mechanism Curves | {model_tag} | GT#{gt_idx} r{repeat_id} | "
        f"tau={param.tau:.4g}, an={param.annealing_steps}, ls={param.langevin_steps}, "
        f"smax={param.sigma_max:.4g}, smin={param.sigma_min:.4g}, sfin={param.sigma_final:.4g}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    base = (
        f"daps_mech_curves_model-{model_tag}"
        f"_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_tau-{_fmt_float_name_token(param.tau)}"
        f"_annealing_steps-{int(param.annealing_steps)}"
        f"_langevin_steps-{int(param.langevin_steps)}"
        f"_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.pdf'))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _save_trajectory_pdf(
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
        print(f"[trajectory_vis] matplotlib 不可用，跳过轨迹图: {e}")
        return None

    if not trajectory:
        return None

    gt4 = _to_4d(gt).float().cpu()
    x04, _ = _align_pred_to_gt(_to_4d(x0).float().cpu(), gt4)

    # 第一列：GT 和输入 x0
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
        f"DAPS Mechanism Trajectory ({mode}) | {model_tag} | GT#{gt_idx} r{repeat_id} | "
        f"tau={param.tau:.4g}, an={param.annealing_steps}, ls={param.langevin_steps}, "
        f"smax={param.sigma_max:.4g}, smin={param.sigma_min:.4g}, sfin={param.sigma_final:.4g}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    base = (
        f"daps_mech_traj_model-{model_tag}"
        f"_sigma_max-{_fmt_float_name_token(param.sigma_max)}"
        f"_sigma_min-{_fmt_float_name_token(param.sigma_min)}"
        f"_tau-{_fmt_float_name_token(param.tau)}"
        f"_annealing_steps-{int(param.annealing_steps)}"
        f"_langevin_steps-{int(param.langevin_steps)}"
        f"_gt-{int(gt_idx)}_r-{int(repeat_id)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.pdf'))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _save_group_topg_panel_pdf(
    output_dir: str,
    model_tag: str,
    gt_idx: int,
    gt: torch.Tensor,
    x0hat_batch: torch.Tensor,
    top_indices: torch.Tensor | None,
    top_candidate_distances: torch.Tensor | None,
    selected_group_indices: torch.Tensor | None,
    best_seed: int,
    best_group: int,
    top_n: int = 20,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[group_topg_vis] matplotlib 不可用，跳过可视化: {e}")
        return None

    if top_indices is None or top_candidate_distances is None:
        return None

    if int(top_indices.numel()) == 0:
        return None

    gt4 = _to_4d(gt).float().cpu()
    n_total = int(top_indices.numel())
    n_show = max(1, min(int(top_n), n_total))

    order = torch.argsort(top_candidate_distances, dim=0)
    pick = order[:n_show]

    cand_seeds = top_indices[pick].detach().cpu().long()
    cand_m0 = top_candidate_distances[pick].detach().cpu().float()
    if selected_group_indices is not None:
        cand_groups = selected_group_indices[pick].detach().cpu().long()
    else:
        cand_groups = torch.full((n_show,), -1, dtype=torch.long)

    panels = [('GT', gt4[0, 0].numpy())]
    imgs = [gt4[0, 0].numpy()]

    for i in range(n_show):
        seed = int(cand_seeds[i].item())
        group = int(cand_groups[i].item())
        m0 = float(cand_m0[i].item())
        x = x0hat_batch[seed:seed + 1].detach().cpu()
        x4, _ = _align_pred_to_gt(_to_4d(x).float(), gt4)
        img = x4[0, 0].numpy()
        is_best = (seed == int(best_seed))
        prefix = "[BEST] " if is_best else ""
        if group >= 0:
            title = f"{prefix}#{i + 1} g={group} s={seed}\nm0={m0:.4f}"
        else:
            title = f"{prefix}#{i + 1} s={seed}\nm0={m0:.4f}"
        panels.append((title, img))
        imgs.append(img)

    all_vals = np.concatenate([im.reshape(-1) for im in imgs], axis=0)
    vmin = float(np.percentile(all_vals, 1.0))
    vmax = float(np.percentile(all_vals, 99.0))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-6

    n = len(panels)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for i, (title, img) in enumerate(panels):
        ax = axes.flat[i]
        ax.imshow(img, cmap=VELOCITY_CMAP, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8.0)

    fig.suptitle(
        f"GroupTopG Candidates | {model_tag} | GT#{gt_idx} | "
        f"best_group={best_group}, best_seed={best_seed}, top_n={n_show}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = (
        f"daps_mech_group_topg_panel_model-{model_tag}"
        f"_gt-{int(gt_idx)}"
        f"_topn-{int(n_show)}"
    )
    path = os.path.join(output_dir, generate_timestamped_filename(base, '.pdf'))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='DAPS-FWI 过程机理探针（Experiment A）')
    parser.add_argument('--model_tag', type=str, default='seam_finetune',
                        choices=['seam', 'seam_finetune', 'marmousi'])
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--sm_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--eval_patches_path', type=str, default=None,
                        help='可选评估集路径（.pt/.pkl；提供后覆盖默认 SEAM 224 切片）')
    parser.add_argument('--gt_indices', type=int, nargs='+', default=[7])

    parser.add_argument('--candidate_mode', type=str, default='group_topg',
                        choices=['gss_topk', 'all_seed', 'group_topg', 'random', 'fixed_seed'])
    parser.add_argument('--gss_top_k', type=int, default=50)
    parser.add_argument('--group_top_g', type=int, default=20)
    parser.add_argument('--fixed_seed', type=int, default=None,
                        help='candidate_mode=fixed_seed 时必须提供')

    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--master_seed', type=int, default=8)
    parser.add_argument('--seed_mode', type=str, default='repeat_dependent',
                        choices=['shared_across_repeats', 'repeat_dependent'])
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='正演算子中的噪声/归一化配置参数（cfg.daps.sigma）')

    # DAPS 单组参数
    parser.add_argument('--langevin_lr', type=float, default=3.2e-4)
    parser.add_argument('--langevin_steps', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lr_min_ratio', type=float, default=1.0)
    parser.add_argument('--lambda_prior', type=float, default=1.0)
    parser.add_argument('--lambda_prior_min_ratio', type=float, default=0.1)
    parser.add_argument('--annealing_steps', type=int, default=20)
    parser.add_argument('--sigma_max', type=float, default=0.3)
    parser.add_argument('--sigma_min', type=float, default=0.05)
    parser.add_argument('--sigma_final', type=float, default=0.0)
    parser.add_argument('--beta_langevin_noise', type=float, default=0.1)
    parser.add_argument('--beta_forward_noise', type=float, default=0.1)
    parser.add_argument('--update_data_weight', type=float, default=1.0,
                        help='inner drift 中 data gradient 的权重')
    parser.add_argument('--update_prior_weight', type=float, default=1.0,
                        help='inner drift 中 prior gradient 的权重')

    # 门控（仅后验打标签）
    parser.add_argument('--low_freq_radius_ratio', type=float, default=0.15)
    parser.add_argument('--min_delta_rel_l2', type=float, default=0.02)
    parser.add_argument('--max_misfit_ratio', type=float, default=1.05)
    parser.add_argument('--max_high_freq_ratio', type=float, default=0.75)

    # 可视化
    parser.add_argument('--skip_curve_vis', action='store_true')
    parser.add_argument('--skip_trajectory_vis', action='store_true')
    parser.add_argument('--skip_group_topg_vis', action='store_true',
                        help='candidate_mode=group_topg 时，跳过 top-g 候选面板图')
    parser.add_argument('--group_topg_vis_top_n', type=int, default=20,
                        help='group_topg 候选面板中显示前 N 个（按 m0 升序）')
    parser.add_argument('--group_topg_vis_max_gts', type=int, default=0,
                        help='最多为多少个 GT 保存 group_topg 面板图，<=0 表示不限制')
    parser.add_argument('--trajectory_mode', type=str, default='triplet', choices=['triplet', 'x0hat'])
    parser.add_argument('--vis_num_snapshots', type=int, default=10)
    parser.add_argument('--vis_max_runs', type=int, default=0,
                        help='最多保存多少个 run 的图，<=0 表示不限制')

    args = parser.parse_args()

    # 参数检查
    if args.n_repeats < 1:
        raise ValueError('--n_repeats 必须 >= 1')
    if args.group_top_g < 1:
        raise ValueError('--group_top_g 必须 >= 1')
    if args.gss_top_k < 1:
        raise ValueError('--gss_top_k 必须 >= 1')
    if args.candidate_mode == 'fixed_seed' and args.fixed_seed is None:
        raise ValueError('candidate_mode=fixed_seed 时必须提供 --fixed_seed')
    if args.langevin_steps < 1 or args.annealing_steps < 1:
        raise ValueError('langevin_steps / annealing_steps 必须 >= 1')
    if args.langevin_lr <= 0:
        raise ValueError('--langevin_lr 必须 > 0')
    if args.tau <= 0:
        raise ValueError('--tau 必须 > 0')
    if args.lambda_prior < 0:
        raise ValueError('--lambda_prior 必须 >= 0')
    if not (0.0 <= args.lambda_prior_min_ratio <= 1.0):
        raise ValueError('--lambda_prior_min_ratio 必须在 [0,1]')
    if args.beta_langevin_noise < 0 or args.beta_forward_noise < 0:
        raise ValueError('--beta_langevin_noise / --beta_forward_noise 必须 >= 0')
    if args.update_data_weight < 0 or args.update_prior_weight < 0:
        raise ValueError('--update_data_weight / --update_prior_weight 必须 >= 0')
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        raise ValueError('需要满足 sigma_max > sigma_min > 0')
    if args.sigma_final < 0 or args.sigma_final > args.sigma_min:
        raise ValueError('需要满足 0 <= sigma_final <= sigma_min')
    if args.vis_num_snapshots < 1:
        raise ValueError('--vis_num_snapshots 必须 >= 1')
    if args.group_topg_vis_top_n < 1:
        raise ValueError('--group_topg_vis_top_n 必须 >= 1')

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

    print('=' * 72)
    print('DAPS-FWI 过程机理探针（Experiment A）')
    print('=' * 72)
    print(f'  model_tag: {args.model_tag}')
    print(f'  checkpoint: {ckpt_path}')
    print(f'  SM: {args.sm_path}')
    if args.eval_patches_path:
        print(f'  eval_patches_path: {args.eval_patches_path}')
    print(f'  output_dir: {output_dir}')
    print(f'  device: {device}')
    print(f'  gt_indices: {args.gt_indices}')
    print(f'  candidate_mode: {args.candidate_mode}, gss_top_k={args.gss_top_k}, group_top_g={args.group_top_g}')
    if args.candidate_mode == 'fixed_seed':
        print(f'  fixed_seed: {args.fixed_seed}')
    if args.candidate_mode == 'group_topg':
        print(
            f'  group_topg_vis: skip={args.skip_group_topg_vis}, '
            f'top_n={args.group_topg_vis_top_n}, max_gts={args.group_topg_vis_max_gts}'
        )
    print(f'  n_repeats: {args.n_repeats}, seed_mode={args.seed_mode}, master_seed={args.master_seed}')
    print(
        '  daps: '
        f'lr={param.langevin_lr}, ls={param.langevin_steps}, tau={param.tau}, '
        f'an={param.annealing_steps}, smax={param.sigma_max}, smin={param.sigma_min}, sfin={param.sigma_final}, '
        f'lambda_prior={param.lambda_prior}, lambda_prior_min_ratio={param.lambda_prior_min_ratio}, '
        f'beta_langevin_noise={param.beta_langevin_noise}, beta_forward_noise={param.beta_forward_noise}, '
        f'update_data_weight={args.update_data_weight}, update_prior_weight={args.update_prior_weight}'
    )
    print(
        '  gate(label-only): '
        f'delta_rel_l2>={args.min_delta_rel_l2}, misfit_ratio<={args.max_misfit_ratio}, '
        f'high_freq_ratio<={args.max_high_freq_ratio}, low_freq_radius_ratio={args.low_freq_radius_ratio}'
    )

    if args.eval_patches_path:
        print('\n加载外部评估域数据...')
    else:
        print('\n加载评估域数据...')
    seam_data, _ = _load_eval_dataset(
        cfg,
        image_size=200,
        eval_patches_path=args.eval_patches_path,
    )
    print(f'  dataset_size: {len(seam_data)}')

    print('\n初始化模型与正演算子...')
    model_config, _ = create_sde_config(parent_dir, batch_size=1)
    base_config, lgvd_config = build_daps_configs(cfg)
    operator = DAPSSeismicOperator(model_config, image_size=200, sigma=cfg.daps.sigma)
    model = NCSNpp_DAPS(
        model_config=model_config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=ckpt_path,
    )
    model.set_device(device)
    model.eval()

    x0hat_batch = sm_info['x0hat_batch'].to(device).float()

    # 预缓存每个 GT 的 x0 候选与基本指标
    x0_cache = {}
    print('\n固定每个 GT 的候选 x0（来自 SM）...')
    for gt_idx in args.gt_indices:
        if gt_idx < 0 or gt_idx >= len(seam_data):
            raise IndexError(f'gt_idx 越界: {gt_idx}, dataset_size={len(seam_data)}')

        gt = seam_data[gt_idx].unsqueeze(0).to(device)
        with torch.no_grad():
            measurement = operator(gt)

        if args.candidate_mode == 'fixed_seed':
            seed = int(args.fixed_seed)
            if seed < 0 or seed >= int(x0hat_batch.shape[0]):
                raise IndexError(f'fixed_seed 越界: {seed}, n_seeds={int(x0hat_batch.shape[0])}')
            x0 = x0hat_batch[seed:seed + 1]
            with torch.no_grad():
                misfit_before = float(torch.norm(operator(x0) - measurement).item())
            meta = {
                'best_group': -1,
                'best_seed': seed,
                'candidate_distance': float(misfit_before),
                'centroid_distance': float('nan'),
            }
        else:
            x0, meta = _select_x0_candidate_from_sm(
                sm_info=sm_info,
                measurement=measurement,
                operator=operator,
                top_k=args.gss_top_k,
                group_top_g=args.group_top_g,
                candidate_mode=args.candidate_mode,
            )
            with torch.no_grad():
                misfit_before = float(torch.norm(operator(x0) - measurement).item())

        x0_gt_nrmse = compute_nrmse(x0, gt)
        x0_gt_ssim = compute_ssim(x0, gt)

        x0_cache[gt_idx] = {
            'x0': x0.detach(),
            'gt': gt.detach(),
            'measurement': measurement.detach(),
            'meta': meta,
            'misfit_before': float(misfit_before),
            'x0_gt_nrmse': float(x0_gt_nrmse),
            'x0_gt_ssim': float(x0_gt_ssim),
        }

        print(
            f"  GT#{gt_idx}: group={meta['best_group']}, seed={meta['best_seed']}, "
            f"cand_m0={meta['candidate_distance']:.4f}, misfit_before={misfit_before:.4f}, "
            f"x0_gt_nrmse={x0_gt_nrmse:.4f}, x0_gt_ssim={x0_gt_ssim:.4f}"
        )

    group_topg_paths = []
    if args.candidate_mode == 'group_topg' and (not args.skip_group_topg_vis):
        print('\n保存 group_topg 候选面板图...')
        for gt_idx in args.gt_indices:
            if args.group_topg_vis_max_gts > 0 and len(group_topg_paths) >= args.group_topg_vis_max_gts:
                break
            item = x0_cache[gt_idx]
            meta = item['meta']
            p_group = _save_group_topg_panel_pdf(
                output_dir=output_dir,
                model_tag=args.model_tag,
                gt_idx=int(gt_idx),
                gt=item['gt'],
                x0hat_batch=x0hat_batch,
                top_indices=meta.get('top_indices'),
                top_candidate_distances=meta.get('top_candidate_distances'),
                selected_group_indices=meta.get('selected_group_indices'),
                best_seed=int(meta['best_seed']),
                best_group=int(meta['best_group']),
                top_n=int(args.group_topg_vis_top_n),
            )
            if p_group:
                group_topg_paths.append(p_group)

    # 主循环
    total_runs = len(args.gt_indices) * args.n_repeats
    inner_all_rows = []
    outer_all_rows = []
    run_rows = []
    curve_paths = []
    traj_paths = []

    pbar = tqdm(total=total_runs, desc='DAPS Mechanism Probe', dynamic_ncols=True)

    for gt_idx in args.gt_indices:
        item = x0_cache[gt_idx]
        gt = item['gt']
        x0 = item['x0']
        measurement = item['measurement']
        meta = item['meta']
        misfit_before = float(item['misfit_before'])
        x0_gt_nrmse = float(item['x0_gt_nrmse'])

        for repeat_id in range(args.n_repeats):
            run_seed = _build_run_seed(args.master_seed, gt_idx, repeat_id, args.seed_mode)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            t0 = time.time()
            diag = _run_langevin_probe_direction(
                model=model,
                operator=operator,
                measurement=measurement,
                gt=gt,
                x0=x0,
                param=param,
                return_trajectory=(not args.skip_trajectory_vis),
                update_data_weight=float(args.update_data_weight),
                update_prior_weight=float(args.update_prior_weight),
            )
            elapsed = float(time.time() - t0)

            pred_x0hat = diag['final_x0hat']
            pred_x0y = diag['final_x0y']
            inner_rows = diag['inner_rows']
            outer_rows = diag['outer_rows']

            m_x0hat = _compute_final_metrics(
                pred=pred_x0hat,
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
            m_x0y = _compute_final_metrics(
                pred=pred_x0y,
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

            x0hat_improve = float(x0_gt_nrmse - m_x0hat['gt_nrmse'])
            x0y_improve = float(x0_gt_nrmse - m_x0y['gt_nrmse'])

            mean_noise_to_drift = _mean([r['noise_to_drift_ratio'] for r in inner_rows])
            mean_noise_to_grad = _mean([r['noise_to_grad_ratio'] for r in inner_rows])
            mean_prior_over_data = _mean([r['prior_over_data_loss'] for r in inner_rows])
            mean_cos_data_to_gt = _mean([r['cos_data_to_gt'] for r in inner_rows])
            mean_cos_prior_to_gt = _mean([r['cos_prior_to_gt'] for r in inner_rows])
            mean_cos_update_to_gt = _mean([r['cos_update_to_gt'] for r in inner_rows])
            mean_delta_gt_rmse_drift = _mean([r['delta_gt_rmse_drift'] for r in inner_rows])
            mean_delta_gt_rmse_full = _mean([r['delta_gt_rmse_full'] for r in inner_rows])
            frac_drift_toward_gt = _mean([r['drift_toward_gt'] for r in inner_rows])
            frac_full_toward_gt = _mean([r['full_toward_gt'] for r in inner_rows])

            # 追加逐步日志
            for r in inner_rows:
                rr = dict(r)
                rr.update({
                    'gt_idx': int(gt_idx),
                    'repeat_id': int(repeat_id),
                    'seed': int(run_seed),
                    'candidate_seed': int(meta['best_seed']),
                    'candidate_group': int(meta['best_group']),
                })
                inner_all_rows.append(rr)

            for r in outer_rows:
                rr = dict(r)
                rr.update({
                    'gt_idx': int(gt_idx),
                    'repeat_id': int(repeat_id),
                    'seed': int(run_seed),
                    'candidate_seed': int(meta['best_seed']),
                    'candidate_group': int(meta['best_group']),
                })
                outer_all_rows.append(rr)

            row = {
                'gt_idx': int(gt_idx),
                'repeat_id': int(repeat_id),
                'seed': int(run_seed),
                'candidate_mode': str(args.candidate_mode),
                'candidate_group': int(meta['best_group']),
                'candidate_seed': int(meta['best_seed']),
                'candidate_m0': float(meta['candidate_distance']),
                'centroid_distance': float(meta['centroid_distance']),
                'misfit_before': float(misfit_before),
                'x0_gt_nrmse': float(x0_gt_nrmse),
                'x0_gt_ssim': float(item['x0_gt_ssim']),
                'x0hat_gt_nrmse': float(m_x0hat['gt_nrmse']),
                'x0hat_gt_ssim': float(m_x0hat['gt_ssim']),
                'x0hat_gt_psnr': float(m_x0hat['gt_psnr']),
                'x0hat_gt_nrmse_improve': float(x0hat_improve),
                'x0hat_delta_rel_l2': float(m_x0hat['delta_rel_l2']),
                'x0hat_misfit_after': float(m_x0hat['meas_misfit_after']),
                'x0hat_misfit_ratio': float(m_x0hat['misfit_ratio']),
                'x0hat_structural_pass': int(m_x0hat['is_structural_variant']),
                'x0y_gt_nrmse': float(m_x0y['gt_nrmse']),
                'x0y_gt_ssim': float(m_x0y['gt_ssim']),
                'x0y_gt_psnr': float(m_x0y['gt_psnr']),
                'x0y_gt_nrmse_improve': float(x0y_improve),
                'x0y_delta_rel_l2': float(m_x0y['delta_rel_l2']),
                'x0y_misfit_after': float(m_x0y['meas_misfit_after']),
                'x0y_misfit_ratio': float(m_x0y['misfit_ratio']),
                'x0y_structural_pass': int(m_x0y['is_structural_variant']),
                'mean_noise_to_drift_ratio': float(mean_noise_to_drift),
                'mean_noise_to_grad_ratio': float(mean_noise_to_grad),
                'mean_prior_over_data_loss': float(mean_prior_over_data),
                'mean_cos_data_to_gt': float(mean_cos_data_to_gt),
                'mean_cos_prior_to_gt': float(mean_cos_prior_to_gt),
                'mean_cos_update_to_gt': float(mean_cos_update_to_gt),
                'mean_delta_gt_rmse_drift': float(mean_delta_gt_rmse_drift),
                'mean_delta_gt_rmse_full': float(mean_delta_gt_rmse_full),
                'frac_drift_toward_gt': float(frac_drift_toward_gt),
                'frac_full_toward_gt': float(frac_full_toward_gt),
                'run_seconds': float(elapsed),
            }
            run_rows.append(row)

            can_vis = (args.vis_max_runs <= 0) or (len(curve_paths) < args.vis_max_runs)
            if can_vis and (not args.skip_curve_vis):
                p_curve = _save_inner_curve_pdf(
                    output_dir=output_dir,
                    model_tag=args.model_tag,
                    param=param,
                    gt_idx=gt_idx,
                    repeat_id=repeat_id,
                    inner_rows=inner_rows,
                )
                if p_curve:
                    curve_paths.append(p_curve)

            if can_vis and (not args.skip_trajectory_vis):
                p_traj = _save_trajectory_pdf(
                    output_dir=output_dir,
                    model_tag=args.model_tag,
                    param=param,
                    gt_idx=gt_idx,
                    repeat_id=repeat_id,
                    gt=gt,
                    x0=x0,
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
        'gt_idx', 'repeat_id', 'seed', 'candidate_seed', 'candidate_group',
        'outer_step', 'inner_step', 'sigma', 'ratio', 'langevin_lr_effective',
        'lambda_prior_effective', 'update_data_weight', 'update_prior_weight',
        'tau', 'data_loss', 'prior_loss', 'total_loss',
        'prior_over_data_loss', 'measurement_misfit',
        'grad_data_norm', 'grad_prior_norm', 'grad_total_norm', 'grad_update_norm', 'grad_data_prior_cos',
        'cos_data_to_gt', 'cos_prior_to_gt', 'cos_update_to_gt',
        'gt_rmse_before', 'gt_rmse_after_drift', 'gt_rmse_after_full',
        'delta_gt_rmse_drift', 'delta_gt_rmse_full', 'drift_toward_gt', 'full_toward_gt',
        'drift_norm', 'noise_norm', 'noise_to_drift_ratio', 'noise_to_grad_ratio',
    ]
    outer_headers = [
        'gt_idx', 'repeat_id', 'seed', 'candidate_seed', 'candidate_group',
        'outer_step', 'sigma', 'next_sigma', 'langevin_lr_effective', 'lambda_prior_effective',
        'mean_data_loss', 'mean_prior_loss', 'mean_total_loss', 'mean_prior_over_data_loss',
        'mean_measurement_misfit', 'mean_grad_data_norm', 'mean_grad_prior_norm',
        'mean_grad_total_norm', 'mean_grad_update_norm',
        'mean_cos_data_to_gt', 'mean_cos_prior_to_gt', 'mean_cos_update_to_gt',
        'mean_delta_gt_rmse_drift', 'mean_delta_gt_rmse_full',
        'frac_drift_toward_gt', 'frac_full_toward_gt',
        'mean_drift_norm', 'mean_noise_norm',
        'mean_noise_to_drift_ratio', 'mean_noise_to_grad_ratio',
        'misfit_x0hat', 'misfit_x0y',
    ]
    run_headers = [
        'gt_idx', 'repeat_id', 'seed', 'candidate_mode', 'candidate_group', 'candidate_seed',
        'candidate_m0', 'centroid_distance', 'misfit_before', 'x0_gt_nrmse', 'x0_gt_ssim',
        'x0hat_gt_nrmse', 'x0hat_gt_ssim', 'x0hat_gt_psnr', 'x0hat_gt_nrmse_improve',
        'x0hat_delta_rel_l2', 'x0hat_misfit_after', 'x0hat_misfit_ratio', 'x0hat_structural_pass',
        'x0y_gt_nrmse', 'x0y_gt_ssim', 'x0y_gt_psnr', 'x0y_gt_nrmse_improve',
        'x0y_delta_rel_l2', 'x0y_misfit_after', 'x0y_misfit_ratio', 'x0y_structural_pass',
        'mean_noise_to_drift_ratio', 'mean_noise_to_grad_ratio', 'mean_prior_over_data_loss',
        'mean_cos_data_to_gt', 'mean_cos_prior_to_gt', 'mean_cos_update_to_gt',
        'mean_delta_gt_rmse_drift', 'mean_delta_gt_rmse_full',
        'frac_drift_toward_gt', 'frac_full_toward_gt',
        'run_seconds',
    ]

    inner_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_mech_inner_detail_model-{args.model_tag}', '.csv')
    )
    outer_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_mech_outer_summary_model-{args.model_tag}', '.csv')
    )
    run_path = os.path.join(
        output_dir,
        generate_timestamped_filename(f'daps_mech_run_summary_model-{args.model_tag}', '.csv')
    )

    _save_rows_csv(inner_path, inner_all_rows, inner_headers)
    _save_rows_csv(outer_path, outer_all_rows, outer_headers)
    _save_rows_csv(run_path, run_rows, run_headers)

    print('\n' + '=' * 72)
    print('DAPS 机理探针完成')
    print('=' * 72)
    print(f'inner_detail_csv: {inner_path}')
    print(f'outer_summary_csv: {outer_path}')
    print(f'run_summary_csv: {run_path}')

    if run_rows:
        print('\nRun-level averages:')
        print(f"  mean x0hat_gt_nrmse_improve: {_mean([r['x0hat_gt_nrmse_improve'] for r in run_rows]):.4f}")
        print(f"  mean x0y_gt_nrmse_improve:   {_mean([r['x0y_gt_nrmse_improve'] for r in run_rows]):.4f}")
        print(f"  mean noise_to_drift_ratio:   {_mean([r['mean_noise_to_drift_ratio'] for r in run_rows]):.4f}")
        print(f"  mean noise_to_grad_ratio:    {_mean([r['mean_noise_to_grad_ratio'] for r in run_rows]):.4f}")
        print(f"  mean cos(data->GT):          {_mean([r['mean_cos_data_to_gt'] for r in run_rows]):.4f}")
        print(f"  mean cos(prior->GT):         {_mean([r['mean_cos_prior_to_gt'] for r in run_rows]):.4f}")
        print(f"  mean cos(update->GT):        {_mean([r['mean_cos_update_to_gt'] for r in run_rows]):.4f}")
        print(f"  mean delta_gt_rmse_drift:    {_mean([r['mean_delta_gt_rmse_drift'] for r in run_rows]):+.6f}")
        print(f"  mean delta_gt_rmse_full:     {_mean([r['mean_delta_gt_rmse_full'] for r in run_rows]):+.6f}")
        print(f"  frac drift toward GT:        {_mean([r['frac_drift_toward_gt'] for r in run_rows]):.3f}")
        print(f"  frac full-step toward GT:    {_mean([r['frac_full_toward_gt'] for r in run_rows]):.3f}")
        print(f"  x0hat gate pass rate:        {_mean([r['x0hat_structural_pass'] for r in run_rows]):.3f}")
        print(f"  x0y gate pass rate:          {_mean([r['x0y_structural_pass'] for r in run_rows]):.3f}")

    if curve_paths:
        print('\n曲线图（PDF）:')
        for p in curve_paths:
            print(f'  {p}')
    elif args.skip_curve_vis:
        print('\n[curve_vis] skip_curve_vis=True，未保存曲线图。')
    else:
        print('\n[curve_vis] 未生成曲线图。')

    if traj_paths:
        print('\n轨迹图（PDF）:')
        for p in traj_paths:
            print(f'  {p}')
    elif args.skip_trajectory_vis:
        print('\n[trajectory_vis] skip_trajectory_vis=True，未保存轨迹图。')
    else:
        print('\n[trajectory_vis] 未生成轨迹图。')

    if group_topg_paths:
        print('\ngroup_topg 候选图（PDF）:')
        for p in group_topg_paths:
            print(f'  {p}')
    elif args.candidate_mode == 'group_topg' and args.skip_group_topg_vis:
        print('\n[group_topg_vis] skip_group_topg_vis=True，未保存候选面板图。')
    elif args.candidate_mode == 'group_topg':
        print('\n[group_topg_vis] 未生成候选面板图。')


if __name__ == '__main__':
    main()
