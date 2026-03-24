"""
SM 候选重合度审计脚本。

目标:
  从数据角度审计 “GT 是否已经被 SM 候选基本覆盖/重合”：
    1) 按 GSS 逻辑选出的 best x0 与 GT 的接近程度
    2) top-k 候选中最接近 GT 的程度
    3) 全部 seeds 中最接近 GT 的程度（可选）
    4) centroid patch 与 GT 的重合程度（200 groups vs 224 patches）

典型用法:
  python sFWI/experiments/sm_candidate_overlap_audit.py \
    --model_tag seam_finetune \
    --sm_path /content/drive/MyDrive/score_sde_inverseSolving/gss_assets/sm_dataset-seam_model-seam_finetune_k200_j1500_seed8.pt \
    --output_dir /content/drive/MyDrive/score_sde_inverseSolving/outputs \
    --gss_top_k 50 \
    --master_seed 8
"""

import sys
import os
import csv
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

from sFWI.config import FWIConfig
from sFWI.models.sde_setup import create_sde_config
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.utils.file_utils import generate_timestamped_filename


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
        pred4 = F.interpolate(pred4, size=gt4.shape[-2:], mode='bilinear', align_corners=False)
    return pred4, gt4


def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor, fallback_range: float | None = None) -> float:
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    mse = F.mse_loss(pred4, gt4).item()
    rmse = np.sqrt(mse)
    gt_range = (gt4.max() - gt4.min()).item()
    if gt_range < 1e-8:
        if fallback_range is not None and fallback_range > 1e-8:
            return rmse / float(fallback_range)
        return float('nan')
    return rmse / gt_range


def compute_rel_l2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    num = torch.norm((pred4 - gt4).reshape(-1)).item()
    den = torch.norm(gt4.reshape(-1)).item()
    return num / max(1e-8, den)


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


def _batch_nrmse_rel(
    pred_batch: torch.Tensor,
    gt: torch.Tensor,
    fallback_range: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred4, gt4 = _align_pred_to_gt(pred_batch, gt)
    diff = pred4 - gt4
    mse = diff.flatten(1).pow(2).mean(dim=1)
    rmse = torch.sqrt(mse)
    gt_range = float((gt4.max() - gt4.min()).item())
    if gt_range < 1e-8:
        if fallback_range is not None and fallback_range > 1e-8:
            denom = float(fallback_range)
        else:
            denom = 1e-8
    else:
        denom = gt_range
    nrmse = rmse / denom
    rel_l2 = diff.flatten(1).norm(dim=1) / gt4.flatten(1).norm().clamp_min(1e-8)
    return nrmse, rel_l2


def _parse_float_list(text: str):
    vals = [x.strip() for x in text.split(',') if x.strip()]
    return [float(v) for v in vals]


def _mean(vals):
    if not vals:
        return float('nan')
    arr = [float(v) for v in vals if np.isfinite(float(v))]
    if not arr:
        return float('nan')
    return float(sum(arr) / len(arr))


def _std(vals):
    if not vals:
        return float('nan')
    arr = np.array([float(v) for v in vals if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    return float(np.std(arr))


def _quantiles(vals, qs=(0.25, 0.5, 0.75)):
    if not vals:
        return [float('nan')] * len(qs)
    arr = np.array([float(v) for v in vals if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return [float('nan')] * len(qs)
    return [float(np.quantile(arr, q)) for q in qs]


def _sanitize_thr(v: float) -> str:
    return f"{v:.6g}".replace('-', 'm').replace('.', 'p')


def _save_rows_csv(path, rows, headers):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dataset_global_minmax(data):
    """
    兼容 VelocityDataset / Tensor / list-like，返回全局 min/max。
    """
    # VelocityDataset 常见路径：直接访问其缓存张量
    if hasattr(data, 'velocity_models'):
        t = getattr(data, 'velocity_models')
        if isinstance(t, torch.Tensor):
            return float(torch.amin(t).item()), float(torch.amax(t).item())

    # Tensor 数据路径
    if isinstance(data, torch.Tensor):
        return float(torch.amin(data).item()), float(torch.amax(data).item())

    # 通用回退：遍历样本
    dmin = float('inf')
    dmax = float('-inf')
    for i in range(len(data)):
        x = data[i]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        xv_min = float(torch.amin(x).item())
        xv_max = float(torch.amax(x).item())
        dmin = min(dmin, xv_min)
        dmax = max(dmax, xv_max)
    return dmin, dmax


def _load_sm_info(sm_path, expected_model_tag=None):
    if not os.path.isfile(sm_path):
        raise FileNotFoundError(f"SM 文件不存在: {sm_path}")
    sm_data = torch.load(sm_path, weights_only=False)
    required_keys = ['similarity_matrix', 'k', 'centroid_indices', 'x0hat_batch', 'd_centroids_2d']
    for key in required_keys:
        if key not in sm_data:
            raise KeyError(f"SM 文件缺少字段: {key}")
    model_tag = sm_data.get('model_tag')
    if expected_model_tag and model_tag and model_tag != expected_model_tag:
        raise ValueError(f"SM 模型标签不匹配: expected={expected_model_tag}, actual={model_tag}")
    return sm_data


def _load_eval_dataset(cfg, model_tag: str, image_size: int, marmousi_data_path: str | None):
    if model_tag in ('seam', 'seam_finetune'):
        v_torch = load_seam_model(cfg.paths.seam_model_path)
        dataset = create_velocity_dataset(v_torch, image_size=image_size)
        dataset_tag = 'seam'
        return dataset, dataset_tag

    if model_tag == 'marmousi':
        pkl_path = marmousi_data_path or cfg.paths.marmousi_dataset_path
        v_torch, _ = load_marmousi_from_pkl(pkl_path, image_size=200)
        dataset = create_velocity_dataset(v_torch, image_size=image_size)
        dataset_tag = 'marmousi'
        return dataset, dataset_tag

    raise ValueError(f"不支持的 model_tag: {model_tag}")


def _select_x0_candidate_from_sm(
    sm: torch.Tensor,
    x0hat_batch: torch.Tensor,
    d_centroids_2d: torch.Tensor,
    d_samples_2d: torch.Tensor | None,
    measurement: torch.Tensor,
    operator: DAPSSeismicOperator,
    top_k: int = 50,
    candidate_mode: str = 'gss_topk',
    generator: torch.Generator | None = None,
    group_top_m: int = 8,
    group_margin_ratio: float = 0.15,
    group_top_m_max: int = 16,
    per_group_top_k: int = 25,
    pre_rerank_top_n: int = 200,
    group_proxy_beta: float = 0.7,
):
    measurement_flat = measurement.reshape(1, -1)
    centroid_distances = torch.norm(d_centroids_2d - measurement_flat, dim=1)
    best_group = int(torch.argmin(centroid_distances).item())
    centroid_distance = float(centroid_distances[best_group].item())

    n_seeds = int(sm.shape[1])
    n_groups = int(sm.shape[0])
    n_select = max(1, min(int(top_k), n_seeds))
    selected_groups = torch.tensor([best_group], device=sm.device, dtype=torch.long)
    if candidate_mode == 'random':
        if generator is None:
            generator = torch.Generator(device='cpu')
        perm = torch.randperm(n_seeds, generator=generator, device='cpu')[:n_select]
        top_indices = perm.to(sm.device)
    elif candidate_mode == 'multi_group_rerank':
        m = max(1, min(int(group_top_m), n_groups))
        group_top = torch.topk(centroid_distances, k=m, largest=False).indices
        best_centroid_dist = float(centroid_distances[group_top[0]].item())

        margin_ratio = max(0.0, float(group_margin_ratio))
        if margin_ratio > 0.0:
            margin_thr = best_centroid_dist * (1.0 + margin_ratio)
            margin_groups = torch.nonzero(centroid_distances <= margin_thr, as_tuple=False).squeeze(1)
        else:
            margin_groups = torch.empty((0,), device=sm.device, dtype=torch.long)

        selected_groups = torch.unique(torch.cat([group_top, margin_groups], dim=0))
        if selected_groups.numel() == 0:
            selected_groups = group_top[:1]
        selected_groups = selected_groups[torch.argsort(centroid_distances[selected_groups])]
        m_max = max(1, min(int(group_top_m_max), n_groups))
        selected_groups = selected_groups[:m_max]

        per_group = max(1, min(int(per_group_top_k), n_seeds))
        beta = float(group_proxy_beta)
        norm_denom = max(1e-8, best_centroid_dist)
        seed_proxy = {}
        for g in selected_groups.tolist():
            _, idx = torch.topk(sm[g], per_group, largest=False)
            group_penalty = float(centroid_distances[g].item()) / norm_denom
            proxy_vals = sm[g, idx] + beta * group_penalty
            for sid, score in zip(idx.tolist(), proxy_vals.tolist()):
                prev = seed_proxy.get(sid)
                if prev is None or score < prev:
                    seed_proxy[sid] = float(score)

        if seed_proxy:
            sorted_items = sorted(seed_proxy.items(), key=lambda kv: kv[1])
            pre_top_n = min(len(sorted_items), max(int(pre_rerank_top_n), n_select))
            pre_indices = torch.tensor(
                [sid for sid, _ in sorted_items[:pre_top_n]],
                device=sm.device,
                dtype=torch.long,
            )

            if d_samples_2d is not None:
                pre_flat = d_samples_2d[pre_indices]
            else:
                candidates = x0hat_batch[pre_indices]
                candidates_high = F.interpolate(candidates, size=(128, 128), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    d_candidates = operator(candidates_high)
                pre_flat = d_candidates.reshape(pre_indices.shape[0], -1)
            pre_dist = torch.norm(pre_flat - measurement_flat, dim=1)
            rerank = torch.argsort(pre_dist)
            top_indices = pre_indices[rerank[:n_select]]
        else:
            _, top_indices = torch.topk(sm[best_group], n_select, largest=False)
    else:
        _, top_indices = torch.topk(sm[best_group], n_select, largest=False)

    if d_samples_2d is not None:
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
    best_seed = int(top_indices[best_local_idx].item())
    best_candidate = x0hat_batch[best_seed].unsqueeze(0)

    meta = {
        'best_group': best_group,
        'best_seed': best_seed,
        'candidate_distance': float(candidate_distances[best_local_idx].item()),
        'top_indices': top_indices,
        'top_candidate_distances': candidate_distances,
        'centroid_distance': centroid_distance,
        'selected_group_count': int(selected_groups.numel()),
        'selected_groups': [int(x) for x in selected_groups.tolist()],
    }
    return best_candidate, meta


def _min_metric_over_candidates(gt, candidates, chunk_size=128, fallback_range: float | None = None):
    min_nrmse = float('inf')
    min_rel = float('inf')
    argmin_nrmse = -1
    argmin_rel = -1
    total = int(candidates.shape[0])
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            batch = candidates[start:end]
            nrmse, rel = _batch_nrmse_rel(batch, gt, fallback_range=fallback_range)
            bmin_nrmse, bidx_n = torch.min(nrmse, dim=0)
            bmin_rel, bidx_r = torch.min(rel, dim=0)
            bmin_nrmse = float(bmin_nrmse.item())
            bmin_rel = float(bmin_rel.item())
            if bmin_nrmse < min_nrmse:
                min_nrmse = bmin_nrmse
                argmin_nrmse = start + int(bidx_n.item())
            if bmin_rel < min_rel:
                min_rel = bmin_rel
                argmin_rel = start + int(bidx_r.item())
    return min_nrmse, argmin_nrmse, min_rel, argmin_rel


def _entropy_from_counts(counts: Counter, k: int):
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    norm = float(entropy / np.log(max(2, k)))
    return entropy, norm


def main():
    parser = argparse.ArgumentParser(description='SM 候选重合度审计')
    parser.add_argument('--model_tag', type=str, default='seam_finetune',
                        choices=['seam', 'seam_finetune', 'marmousi'])
    parser.add_argument('--sm_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gt_indices', type=int, nargs='+', default=None,
                        help='默认审计全部样本')
    parser.add_argument('--gss_top_k', type=int, default=50)
    parser.add_argument('--candidate_mode', type=str, default='gss_topk',
                        choices=['gss_topk', 'multi_group_rerank', 'random'])
    parser.add_argument('--group_top_m', type=int, default=8)
    parser.add_argument('--group_margin_ratio', type=float, default=0.15)
    parser.add_argument('--group_top_m_max', type=int, default=16)
    parser.add_argument('--per_group_top_k', type=int, default=25)
    parser.add_argument('--pre_rerank_top_n', type=int, default=200)
    parser.add_argument('--group_proxy_beta', type=float, default=0.7)
    parser.add_argument('--master_seed', type=int, default=8)
    parser.add_argument('--near_nrmse', type=str, default='0.01,0.02,0.05')
    parser.add_argument('--near_rel_l2', type=str, default='0.01,0.02,0.05')
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--skip_all_seed_scan', action='store_true',
                        help='跳过全 1500 seeds 最近邻扫描（提速）')
    parser.add_argument('--image_size', type=int, default=200)
    parser.add_argument('--operator_sigma', type=float, default=0.3)
    parser.add_argument('--marmousi_data_path', type=str, default=None,
                        help='仅 model_tag=marmousi 时使用')
    parser.add_argument('--skip_invalid_gt', action='store_true',
                        help='GT 正演失败时跳过并继续（推荐开启）')
    args = parser.parse_args()

    if args.gss_top_k < 1:
        raise ValueError('--gss_top_k 必须 >= 1')
    if args.group_top_m < 1 or args.group_top_m_max < 1:
        raise ValueError('--group_top_m / --group_top_m_max 必须 >= 1')
    if args.per_group_top_k < 1 or args.pre_rerank_top_n < 1:
        raise ValueError('--per_group_top_k / --pre_rerank_top_n 必须 >= 1')
    if args.group_margin_ratio < 0:
        raise ValueError('--group_margin_ratio 必须 >= 0')
    if args.chunk_size < 1:
        raise ValueError('--chunk_size 必须 >= 1')

    nrmse_thrs = _parse_float_list(args.near_nrmse)
    rel_thrs = _parse_float_list(args.near_rel_l2)

    cfg = FWIConfig()
    cfg.daps.sigma = args.operator_sigma
    device = cfg.device
    output_dir = args.output_dir or cfg.paths.output_path
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("SM 候选重合度审计")
    print("=" * 72)
    print(f"  model_tag: {args.model_tag}")
    print(f"  sm_path: {args.sm_path}")
    print(f"  device: {device}")
    print(f"  candidate_mode: {args.candidate_mode}")
    print(f"  gss_top_k: {args.gss_top_k}")
    if args.candidate_mode == 'multi_group_rerank':
        print(
            "  retrieval(multi_group_rerank): "
            f"group_top_m={args.group_top_m}, "
            f"group_margin_ratio={args.group_margin_ratio}, "
            f"group_top_m_max={args.group_top_m_max}, "
            f"per_group_top_k={args.per_group_top_k}, "
            f"pre_rerank_top_n={args.pre_rerank_top_n}, "
            f"group_proxy_beta={args.group_proxy_beta}"
        )
    print(f"  skip_all_seed_scan: {args.skip_all_seed_scan}")
    print(f"  thresholds(nrmse): {nrmse_thrs}")
    print(f"  thresholds(rel_l2): {rel_thrs}")

    print("\n加载评估域数据...")
    data, dataset_tag = _load_eval_dataset(
        cfg=cfg,
        model_tag=args.model_tag,
        image_size=args.image_size,
        marmousi_data_path=args.marmousi_data_path,
    )
    total_samples = len(data)
    gt_indices = args.gt_indices if args.gt_indices else list(range(total_samples))
    for gi in gt_indices:
        if gi < 0 or gi >= total_samples:
            raise IndexError(f'gt_index 越界: {gi}, dataset_size={total_samples}')
    print(f"  dataset_tag: {dataset_tag}")
    print(f"  dataset_size: {total_samples}")
    print(f"  n_eval_gt: {len(gt_indices)}")
    with torch.no_grad():
        data_min, data_max = _dataset_global_minmax(data)
    dataset_global_range = max(1e-8, data_max - data_min)
    print(f"  dataset_range(global): {dataset_global_range:.6f} (min={data_min:.3f}, max={data_max:.3f})")

    print("\n加载 SM 资产...")
    sm_info = _load_sm_info(args.sm_path, expected_model_tag=args.model_tag)
    sm = sm_info['similarity_matrix'].to(device)  # [k, n_seeds]
    x0hat_batch = sm_info['x0hat_batch'].to(device)  # [n_seeds,1,32,32]
    d_centroids_2d = sm_info['d_centroids_2d'].to(device)  # [k,D]
    d_samples_2d = sm_info.get('d_samples_2d')
    if d_samples_2d is not None:
        d_samples_2d = d_samples_2d.to(device)
    centroid_indices = list(sm_info['centroid_indices'])
    k = int(sm_info['k'])
    n_seeds = int(sm.shape[1])
    print(f"  k={k}, n_seeds={n_seeds}, centroid_count={len(centroid_indices)}")
    print(f"  d_samples_2d: {'yes' if d_samples_2d is not None else 'no (fallback operator)'}")

    # 准备 operator 与 centroid patches（用于 group 选择、centroid 重合度审计）
    print("\n初始化正演算子...")
    config, _ = create_sde_config(parent_dir, batch_size=1)
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)

    print("缓存 centroid patches...")
    centroid_patches = []
    for idx in centroid_indices:
        if 0 <= idx < total_samples:
            centroid_patches.append(data[idx].unsqueeze(0))
        else:
            centroid_patches.append(None)
    valid_centroid_mask = [x is not None for x in centroid_patches]
    if any(valid_centroid_mask):
        centroid_patch_tensor = torch.cat([x for x in centroid_patches if x is not None], dim=0).to(device)
        valid_centroid_map = [i for i, ok in enumerate(valid_centroid_mask) if ok]
    else:
        centroid_patch_tensor = None
        valid_centroid_map = []

    rows = []
    best_group_counter = Counter()
    failed_rows = []
    flat_gt_count = 0

    pbar = tqdm(gt_indices, desc='Audit GT')
    for gt_idx in pbar:
        gt = data[gt_idx].unsqueeze(0).to(device)
        gt_range = float((gt.max() - gt.min()).item())
        gt_is_flat = int(gt_range < 1e-8)
        if gt_is_flat:
            flat_gt_count += 1
        try:
            with torch.no_grad():
                measurement = operator(gt)
        except Exception as e:
            if args.skip_invalid_gt:
                failed_rows.append({
                    'gt_index': gt_idx,
                    'error': str(e),
                })
                continue
            raise

        gen = torch.Generator(device='cpu')
        gen.manual_seed(args.master_seed + gt_idx)
        best_x0, meta = _select_x0_candidate_from_sm(
            sm=sm,
            x0hat_batch=x0hat_batch,
            d_centroids_2d=d_centroids_2d,
            d_samples_2d=d_samples_2d,
            measurement=measurement,
            operator=operator,
            top_k=args.gss_top_k,
            candidate_mode=args.candidate_mode,
            generator=gen,
            group_top_m=args.group_top_m,
            group_margin_ratio=args.group_margin_ratio,
            group_top_m_max=args.group_top_m_max,
            per_group_top_k=args.per_group_top_k,
            pre_rerank_top_n=args.pre_rerank_top_n,
            group_proxy_beta=args.group_proxy_beta,
        )
        best_group = int(meta['best_group'])
        best_seed = int(meta['best_seed'])
        best_group_counter[best_group] += 1

        # 1) best x0 与 GT
        best_nrmse = compute_nrmse(best_x0, gt, fallback_range=dataset_global_range)
        best_rel = compute_rel_l2(best_x0, gt)
        best_ssim = compute_ssim(best_x0, gt)

        # 2) top-k 最近邻（图像域）
        top_indices = meta['top_indices']
        topk_candidates = x0hat_batch[top_indices]
        with torch.no_grad():
            topk_nrmse, topk_rel = _batch_nrmse_rel(topk_candidates, gt, fallback_range=dataset_global_range)
        topk_min_nrmse_val, topk_min_nrmse_pos = torch.min(topk_nrmse, dim=0)
        topk_min_rel_val, topk_min_rel_pos = torch.min(topk_rel, dim=0)
        topk_min_nrmse = float(topk_min_nrmse_val.item())
        topk_min_rel = float(topk_min_rel_val.item())
        topk_min_nrmse_seed = int(top_indices[int(topk_min_nrmse_pos.item())].item())
        topk_min_rel_seed = int(top_indices[int(topk_min_rel_pos.item())].item())

        # 3) 全 seeds 最近邻（图像域）
        if args.skip_all_seed_scan:
            all_min_nrmse = float('nan')
            all_min_nrmse_seed = -1
            all_min_rel = float('nan')
            all_min_rel_seed = -1
        else:
            all_min_nrmse, all_min_nrmse_seed, all_min_rel, all_min_rel_seed = _min_metric_over_candidates(
                gt=gt,
                candidates=x0hat_batch,
                chunk_size=args.chunk_size,
                fallback_range=dataset_global_range,
            )

        # 4) centroid patch 重合（best group）
        centroid_patch_idx = int(centroid_indices[best_group]) if 0 <= best_group < len(centroid_indices) else -1
        if 0 <= centroid_patch_idx < total_samples:
            centroid_patch = data[centroid_patch_idx].unsqueeze(0).to(device)
            centroid_nrmse = compute_nrmse(centroid_patch, gt, fallback_range=dataset_global_range)
            centroid_rel = compute_rel_l2(centroid_patch, gt)
        else:
            centroid_nrmse = float('nan')
            centroid_rel = float('nan')

        centroid_same_gt = int(centroid_patch_idx == gt_idx)

        # 5) centroid patches 中图像最近邻（独立于 measurement 分组）
        if centroid_patch_tensor is not None:
            with torch.no_grad():
                c_nrmse, c_rel = _batch_nrmse_rel(centroid_patch_tensor, gt, fallback_range=dataset_global_range)
            c_min_nrmse_val, c_min_nrmse_pos = torch.min(c_nrmse, dim=0)
            c_min_rel_val, c_min_rel_pos = torch.min(c_rel, dim=0)
            c_min_nrmse = float(c_min_nrmse_val.item())
            c_min_rel = float(c_min_rel_val.item())
            mapped_group_by_nrmse = valid_centroid_map[int(c_min_nrmse_pos.item())]
            mapped_group_by_rel = valid_centroid_map[int(c_min_rel_pos.item())]
            centroid_nn_group_by_nrmse = int(mapped_group_by_nrmse)
            centroid_nn_group_by_rel = int(mapped_group_by_rel)
            centroid_nn_patch_by_nrmse = int(centroid_indices[centroid_nn_group_by_nrmse])
            centroid_nn_patch_by_rel = int(centroid_indices[centroid_nn_group_by_rel])
            centroid_nn_patch_is_gt = int(centroid_nn_patch_by_nrmse == gt_idx)
        else:
            c_min_nrmse = float('nan')
            c_min_rel = float('nan')
            centroid_nn_group_by_nrmse = -1
            centroid_nn_group_by_rel = -1
            centroid_nn_patch_by_nrmse = -1
            centroid_nn_patch_by_rel = -1
            centroid_nn_patch_is_gt = 0

        rows.append({
            'gt_index': gt_idx,
            'gt_range': gt_range,
            'gt_is_flat': gt_is_flat,
            'best_group': best_group,
            'best_seed': best_seed,
            'candidate_distance': float(meta['candidate_distance']),
            'centroid_distance': float(meta['centroid_distance']),
            'selected_group_count': int(meta['selected_group_count']),
            'best_x0_nrmse': best_nrmse,
            'best_x0_rel_l2': best_rel,
            'best_x0_ssim': best_ssim,
            'topk_min_nrmse': topk_min_nrmse,
            'topk_min_nrmse_seed': topk_min_nrmse_seed,
            'topk_min_rel_l2': topk_min_rel,
            'topk_min_rel_l2_seed': topk_min_rel_seed,
            'all_seed_min_nrmse': all_min_nrmse,
            'all_seed_min_nrmse_seed': all_min_nrmse_seed,
            'all_seed_min_rel_l2': all_min_rel,
            'all_seed_min_rel_l2_seed': all_min_rel_seed,
            'best_group_centroid_patch_index': centroid_patch_idx,
            'best_group_centroid_nrmse': centroid_nrmse,
            'best_group_centroid_rel_l2': centroid_rel,
            'best_group_centroid_is_same_gt': centroid_same_gt,
            'centroid_nn_group_by_nrmse': centroid_nn_group_by_nrmse,
            'centroid_nn_patch_by_nrmse': centroid_nn_patch_by_nrmse,
            'centroid_nn_nrmse': c_min_nrmse,
            'centroid_nn_group_by_rel_l2': centroid_nn_group_by_rel,
            'centroid_nn_patch_by_rel_l2': centroid_nn_patch_by_rel,
            'centroid_nn_rel_l2': c_min_rel,
            'centroid_nn_patch_is_same_gt': centroid_nn_patch_is_gt,
        })

    # 保存 detail
    if not rows:
        raise RuntimeError(
            "没有可用 GT 完成审计。可尝试加上 --skip_invalid_gt 跳过异常样本。"
        )

    detail_headers = list(rows[0].keys()) if rows else []
    detail_name = generate_timestamped_filename(
        f'sm_candidate_overlap_detail_dataset-{dataset_tag}_model-{args.model_tag}', '.csv'
    )
    detail_path = os.path.join(output_dir, detail_name)
    _save_rows_csv(detail_path, rows, detail_headers)

    failed_path = None
    if failed_rows:
        failed_name = generate_timestamped_filename(
            f'sm_candidate_overlap_failed_dataset-{dataset_tag}_model-{args.model_tag}', '.csv'
        )
        failed_path = os.path.join(output_dir, failed_name)
        _save_rows_csv(failed_path, failed_rows, ['gt_index', 'error'])

    # 汇总
    best_nrmse_vals = [float(r['best_x0_nrmse']) for r in rows]
    topk_nrmse_vals = [float(r['topk_min_nrmse']) for r in rows]
    all_nrmse_vals = [float(r['all_seed_min_nrmse']) for r in rows if not np.isnan(float(r['all_seed_min_nrmse']))]
    best_rel_vals = [float(r['best_x0_rel_l2']) for r in rows]
    topk_rel_vals = [float(r['topk_min_rel_l2']) for r in rows]
    all_rel_vals = [float(r['all_seed_min_rel_l2']) for r in rows if not np.isnan(float(r['all_seed_min_rel_l2']))]
    centroid_nn_nrmse_vals = [float(r['centroid_nn_nrmse']) for r in rows if not np.isnan(float(r['centroid_nn_nrmse']))]
    centroid_nn_rel_vals = [float(r['centroid_nn_rel_l2']) for r in rows if not np.isnan(float(r['centroid_nn_rel_l2']))]

    q_best = _quantiles(best_nrmse_vals)
    q_topk = _quantiles(topk_nrmse_vals)
    q_all = _quantiles(all_nrmse_vals)
    q_centroid = _quantiles(centroid_nn_nrmse_vals)

    entropy, entropy_norm = _entropy_from_counts(best_group_counter, k)
    group_coverage = len(best_group_counter) / max(1, k)

    summary = {
        'dataset_tag': dataset_tag,
        'model_tag': args.model_tag,
        'sm_path': args.sm_path,
        'dataset_size': total_samples,
        'n_eval_gt': len(gt_indices),
        'n_valid_gt': len(rows),
        'n_failed_gt': len(failed_rows),
        'n_flat_gt': flat_gt_count,
        'k': k,
        'n_noise_seeds': n_seeds,
        'gss_top_k': args.gss_top_k,
        'candidate_mode': args.candidate_mode,
        'group_top_m': args.group_top_m,
        'group_margin_ratio': args.group_margin_ratio,
        'group_top_m_max': args.group_top_m_max,
        'per_group_top_k': args.per_group_top_k,
        'pre_rerank_top_n': args.pre_rerank_top_n,
        'group_proxy_beta': args.group_proxy_beta,
        'skip_all_seed_scan': int(args.skip_all_seed_scan),
        'mean_best_x0_nrmse': _mean(best_nrmse_vals),
        'std_best_x0_nrmse': _std(best_nrmse_vals),
        'p25_best_x0_nrmse': q_best[0],
        'p50_best_x0_nrmse': q_best[1],
        'p75_best_x0_nrmse': q_best[2],
        'mean_topk_min_nrmse': _mean(topk_nrmse_vals),
        'std_topk_min_nrmse': _std(topk_nrmse_vals),
        'p25_topk_min_nrmse': q_topk[0],
        'p50_topk_min_nrmse': q_topk[1],
        'p75_topk_min_nrmse': q_topk[2],
        'mean_all_seed_min_nrmse': _mean(all_nrmse_vals),
        'std_all_seed_min_nrmse': _std(all_nrmse_vals),
        'p25_all_seed_min_nrmse': q_all[0],
        'p50_all_seed_min_nrmse': q_all[1],
        'p75_all_seed_min_nrmse': q_all[2],
        'mean_centroid_nn_nrmse': _mean(centroid_nn_nrmse_vals),
        'p50_centroid_nn_nrmse': q_centroid[1],
        'mean_best_x0_rel_l2': _mean(best_rel_vals),
        'mean_topk_min_rel_l2': _mean(topk_rel_vals),
        'mean_all_seed_min_rel_l2': _mean(all_rel_vals),
        'mean_centroid_nn_rel_l2': _mean(centroid_nn_rel_vals),
        'best_group_centroid_is_same_gt_rate': _mean([float(r['best_group_centroid_is_same_gt']) for r in rows]),
        'centroid_nn_patch_is_same_gt_rate': _mean([float(r['centroid_nn_patch_is_same_gt']) for r in rows]),
        'group_coverage_ratio': group_coverage,
        'group_entropy': entropy,
        'group_entropy_normalized': entropy_norm,
    }

    for thr in nrmse_thrs:
        t = _sanitize_thr(thr)
        summary[f'hit_best_x0_nrmse_le_{t}'] = _mean([float(v <= thr) for v in best_nrmse_vals])
        summary[f'hit_topk_min_nrmse_le_{t}'] = _mean([float(v <= thr) for v in topk_nrmse_vals])
        summary[f'hit_all_seed_min_nrmse_le_{t}'] = _mean([float(v <= thr) for v in all_nrmse_vals]) if all_nrmse_vals else float('nan')
        summary[f'hit_centroid_nn_nrmse_le_{t}'] = _mean([float(v <= thr) for v in centroid_nn_nrmse_vals]) if centroid_nn_nrmse_vals else float('nan')

    for thr in rel_thrs:
        t = _sanitize_thr(thr)
        summary[f'hit_best_x0_rel_l2_le_{t}'] = _mean([float(v <= thr) for v in best_rel_vals])
        summary[f'hit_topk_min_rel_l2_le_{t}'] = _mean([float(v <= thr) for v in topk_rel_vals])
        summary[f'hit_all_seed_min_rel_l2_le_{t}'] = _mean([float(v <= thr) for v in all_rel_vals]) if all_rel_vals else float('nan')
        summary[f'hit_centroid_nn_rel_l2_le_{t}'] = _mean([float(v <= thr) for v in centroid_nn_rel_vals]) if centroid_nn_rel_vals else float('nan')

    summary_name = generate_timestamped_filename(
        f'sm_candidate_overlap_summary_dataset-{dataset_tag}_model-{args.model_tag}', '.csv'
    )
    summary_path = os.path.join(output_dir, summary_name)
    _save_rows_csv(summary_path, [summary], list(summary.keys()))

    # 控制台摘要
    print("\n" + "=" * 72)
    print("审计完成")
    print("=" * 72)
    print(f"detail_csv:  {detail_path}")
    print(f"summary_csv: {summary_path}")
    if failed_path:
        print(f"failed_csv:  {failed_path}")

    print("\n核心统计:")
    print(f"  mean(best_x0_nrmse):      {summary['mean_best_x0_nrmse']:.6f}")
    print(f"  mean(topk_min_nrmse):     {summary['mean_topk_min_nrmse']:.6f}")
    print(f"  mean(all_seed_min_nrmse): {summary['mean_all_seed_min_nrmse']:.6f}")
    print(f"  mean(centroid_nn_nrmse):  {summary['mean_centroid_nn_nrmse']:.6f}")
    print(f"  n_flat_gt:                {summary['n_flat_gt']}")
    print(f"  centroid_nn_same_gt_rate: {summary['centroid_nn_patch_is_same_gt_rate']:.3f}")
    print(f"  best_group_same_gt_rate:  {summary['best_group_centroid_is_same_gt_rate']:.3f}")
    print(f"  group_coverage_ratio:     {summary['group_coverage_ratio']:.3f}")
    print(f"  group_entropy_norm:       {summary['group_entropy_normalized']:.3f}")

    print("\n阈值命中率（NRMSE）:")
    for thr in nrmse_thrs:
        t = _sanitize_thr(thr)
        b = summary[f'hit_best_x0_nrmse_le_{t}']
        k_hit = summary[f'hit_topk_min_nrmse_le_{t}']
        a = summary[f'hit_all_seed_min_nrmse_le_{t}']
        c = summary[f'hit_centroid_nn_nrmse_le_{t}']
        print(f"  <= {thr:g}: best_x0={b:.3f}, topk_min={k_hit:.3f}, all_seed_min={a:.3f}, centroid_nn={c:.3f}")

    print("\n阈值命中率（rel_l2）:")
    for thr in rel_thrs:
        t = _sanitize_thr(thr)
        b = summary[f'hit_best_x0_rel_l2_le_{t}']
        k_hit = summary[f'hit_topk_min_rel_l2_le_{t}']
        a = summary[f'hit_all_seed_min_rel_l2_le_{t}']
        c = summary[f'hit_centroid_nn_rel_l2_le_{t}']
        print(f"  <= {thr:g}: best_x0={b:.3f}, topk_min={k_hit:.3f}, all_seed_min={a:.3f}, centroid_nn={c:.3f}")


if __name__ == '__main__':
    main()
