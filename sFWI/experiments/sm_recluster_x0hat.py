#!/usr/bin/env python3
"""
基于 x0hat 空间重分组并重建 SM 资产。

支持两种输入:
1) 读取已有 SM 资产 (--sm_path)，直接使用其中 x0hat_batch（可选复用 d_samples_2d）
2) 读取 unconditional checkpoint 重新采样 x0hat_batch

输出:
- 新 SM 文件（兼容现有字段，不覆盖旧文件）
- 分组摘要 CSV（每组大小、代表 seed）

分组模式:
1) kmeans: 指定 --k，固定组数
2) threshold: 指定 --distance_threshold，基于 complete-linkage 阈值自动决定组数
"""

from __future__ import annotations

import sys
import os
import csv
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

from sFWI.config import FWIConfig, build_daps_configs
from sFWI.models.sde_setup import create_sde_config
from sFWI.models.score_model import NCSNpp_DAPS
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.utils.file_utils import generate_timestamped_filename


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


def _get_asset_dir(cfg, args):
    out_dir = args.asset_dir or os.path.join(cfg.paths.project_root, 'gss_assets')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _infer_dataset_tag(model_tag: str) -> str:
    if model_tag in ('seam', 'seam_finetune'):
        return 'seam'
    return 'marmousi'


def _load_sm_for_x0hat(sm_path: str):
    if not os.path.isfile(sm_path):
        raise FileNotFoundError(f"SM 文件不存在: {sm_path}")
    sm_data = torch.load(sm_path, weights_only=False, map_location='cpu')
    if 'x0hat_batch' not in sm_data:
        raise KeyError(f"SM 文件缺少字段: x0hat_batch, file={sm_path}")
    return sm_data


def _sample_x0hat_from_model(cfg, args, device, checkpoint_path):
    print("\n创建 unconditional 模型并采样 x0hat...")
    config, _ = create_sde_config(parent_dir, batch_size=1)
    base_config, lgvd_config = build_daps_configs(cfg)
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=checkpoint_path
    )
    model.set_device(device)

    torch.manual_seed(args.master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.master_seed)

    from score_sde_pytorch import sampling as sde_sampling

    shape = (args.n_noise_seeds, 1, cfg.image_size, cfg.image_size)
    sampling_fn = sde_sampling.get_pc_sampler(
        sde=model.daps.sde,
        shape=shape,
        predictor=sde_sampling.ReverseDiffusionPredictor,
        corrector=sde_sampling.LangevinCorrector,
        inverse_scaler=model.daps.inverse_scaler,
        snr=0.16,
        n_steps=1,
        probability_flow=False,
        continuous=True,
        eps=model.daps.sampling_eps,
        device=device,
        seed=None,
    )
    x0hat_batch, _ = sampling_fn(model)
    print(f"✓ 采样完成: {tuple(x0hat_batch.shape)}")
    return x0hat_batch.detach().cpu()


def _compute_d_samples_2d(operator, x0hat_batch: torch.Tensor, device, forward_batch_size: int):
    """
    计算所有 seed 的数据域表示 d_samples_2d: [N, D]
    """
    n = int(x0hat_batch.shape[0])
    all_rows = []
    for start in range(0, n, forward_batch_size):
        end = min(n, start + forward_batch_size)
        x_chunk = x0hat_batch[start:end].to(device)
        x_high = F.interpolate(x_chunk, size=(128, 128), mode='bilinear', align_corners=True)
        with torch.no_grad():
            d_chunk = operator(x_high)
        all_rows.append(d_chunk.reshape(end - start, -1).detach().cpu())
    return torch.cat(all_rows, dim=0)


def _fmt_float_token(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace('-', 'm').replace('.', 'p')


def _labels_to_groups(feats: np.ndarray, labels_raw: np.ndarray):
    """
    将任意标签映射到 [0, k-1]，并为每组选择代表 seed（离组均值最近）。
    """
    labels_raw = np.asarray(labels_raw).astype(np.int64)
    uniq = sorted(np.unique(labels_raw).tolist())
    label_map = {old: new for new, old in enumerate(uniq)}
    labels = np.array([label_map[int(v)] for v in labels_raw], dtype=np.int64)

    k = len(uniq)
    group_members = []
    rep_seed_indices = []
    rep_center_dist = []

    for g in range(k):
        members = np.where(labels == g)[0]
        local = feats[members]
        center = local.mean(axis=0, keepdims=True)
        d = np.linalg.norm(local - center, axis=1)
        best_local = int(np.argmin(d))
        rep = int(members[best_local])

        group_members.append([int(v) for v in members.tolist()])
        rep_seed_indices.append(rep)
        rep_center_dist.append(float(d[best_local]))

    return labels, rep_seed_indices, group_members, rep_center_dist


def _cluster_kmeans(feats: np.ndarray, k: int, random_state: int, max_iter: int):
    n = int(feats.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"k 必须在 [1, {n}] 范围内，当前 k={k}")
    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        max_iter=max_iter,
    )
    labels_raw = km.fit_predict(feats)
    return _labels_to_groups(feats, labels_raw)


def _cluster_threshold_complete(feats: np.ndarray, distance_threshold: float):
    """
    基于 complete-linkage + 距离阈值的层次聚类。
    """
    n = int(feats.shape[0])
    if distance_threshold <= 0:
        raise ValueError(f"distance_threshold 必须 > 0，当前为 {distance_threshold}")
    if n == 1:
        labels_raw = np.array([0], dtype=np.int64)
        return _labels_to_groups(feats, labels_raw)

    condensed = pdist(feats, metric='euclidean')
    Z = linkage(condensed, method='complete')
    labels_raw = fcluster(Z, t=float(distance_threshold), criterion='distance')
    return _labels_to_groups(feats, labels_raw)


def _save_group_summary_csv(path: str, group_members, rep_seed_indices, rep_center_dist):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'group_id', 'group_size', 'rep_seed', 'rep_center_dist',
                'top_members'
            ]
        )
        writer.writeheader()
        for gid, members in enumerate(group_members):
            row = {
                'group_id': int(gid),
                'group_size': int(len(members)),
                'rep_seed': int(rep_seed_indices[gid]),
                'rep_center_dist': float(rep_center_dist[gid]),
                'top_members': ' '.join(str(v) for v in members[:20]),
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='基于 x0hat 重分组并重建 SM 资产')

    # 输入模式
    parser.add_argument('--sm_path', type=str, default=None,
                        help='已有 SM 路径；提供后可直接读取 x0hat_batch')
    parser.add_argument('--model_tag', type=str, default='seam_finetune',
                        choices=['seam', 'seam_finetune', 'marmousi'],
                        help='当不提供 --sm_path 时，用于解析 checkpoint 默认名')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--n_noise_seeds', type=int, default=1500,
                        help='仅采样模式使用')

    # 分组与输出
    parser.add_argument('--cluster_mode', type=str, default='kmeans',
                        choices=['kmeans', 'threshold'],
                        help='重分组模式: kmeans(固定组数) 或 threshold(阈值自动组数)')
    parser.add_argument('--k', type=int, default=None,
                        help='kmeans 模式下的组数')
    parser.add_argument('--distance_threshold', type=float, default=None,
                        help='threshold 模式下的距离阈值（complete-linkage）')
    parser.add_argument('--threshold_space', type=str, default='x0hat',
                        choices=['x0hat', 'd_samples'],
                        help='threshold 模式下在哪个空间做阈值聚类')
    parser.add_argument('--master_seed', type=int, default=8)
    parser.add_argument('--kmeans_max_iter', type=int, default=300)
    parser.add_argument('--sigma', type=float, default=0.3, help='正演算子 sigma')
    parser.add_argument('--forward_batch_size', type=int, default=64,
                        help='计算 d_samples_2d 时的批大小')
    parser.add_argument('--asset_dir', type=str, default=None)
    parser.add_argument('--recompute_d_samples', action='store_true',
                        help='即使 SM 已有 d_samples_2d 也强制重算')

    args = parser.parse_args()

    if args.forward_batch_size < 1:
        raise ValueError("--forward_batch_size 必须 >= 1")
    if args.kmeans_max_iter < 1:
        raise ValueError("--kmeans_max_iter 必须 >= 1")
    if args.n_noise_seeds < 1:
        raise ValueError("--n_noise_seeds 必须 >= 1")
    if args.cluster_mode == 'kmeans':
        if args.k is None:
            raise ValueError("kmeans 模式必须提供 --k")
        if args.k < 1:
            raise ValueError("--k 必须 >= 1")
    else:
        if args.distance_threshold is None:
            raise ValueError("threshold 模式必须提供 --distance_threshold")
        if args.distance_threshold <= 0:
            raise ValueError("--distance_threshold 必须 > 0")
        if args.k is not None:
            print("[WARN] threshold 模式下将忽略 --k。")

    cfg = FWIConfig()
    cfg.daps.sigma = args.sigma
    device = cfg.device
    out_dir = _get_asset_dir(cfg, args)

    print("=" * 72)
    print("SM 重分组（x0hat 空间）")
    print("=" * 72)
    print(f"  device: {device}")
    print(f"  output_dir: {out_dir}")
    print(f"  cluster_mode: {args.cluster_mode}")
    if args.cluster_mode == 'kmeans':
        print(f"  k: {args.k}")
    else:
        print(f"  distance_threshold: {args.distance_threshold}")
        print(f"  threshold_space: {args.threshold_space}")
    print(f"  master_seed: {args.master_seed}")

    checkpoint_path: Optional[str] = None
    source_sm_path: Optional[str] = None

    if args.sm_path:
        sm_data_in = _load_sm_for_x0hat(args.sm_path)
        source_sm_path = args.sm_path
        x0hat_batch = sm_data_in['x0hat_batch'].detach().cpu().float()
        model_tag = str(sm_data_in.get('model_tag', args.model_tag))
        dataset_tag = str(sm_data_in.get('dataset_tag', _infer_dataset_tag(model_tag)))
        checkpoint_path = sm_data_in.get('checkpoint_path', None)
        print("\n输入模式: 读取已有 SM")
        print(f"  sm_path: {args.sm_path}")
        print(f"  model_tag(inferred): {model_tag}")
        print(f"  dataset_tag(inferred): {dataset_tag}")
    else:
        sm_data_in = {}
        model_tag = args.model_tag
        dataset_tag = _infer_dataset_tag(model_tag)
        checkpoint_path = _resolve_checkpoint_path(cfg, args)
        print("\n输入模式: unconditional 采样")
        print(f"  model_tag: {model_tag}")
        print(f"  checkpoint: {checkpoint_path}")
        x0hat_batch = _sample_x0hat_from_model(cfg, args, device, checkpoint_path)

    n_seeds = int(x0hat_batch.shape[0])
    if args.cluster_mode == 'kmeans' and int(args.k) > n_seeds:
        raise ValueError(f"k={args.k} 不能大于样本数 n_seeds={n_seeds}")
    print(f"\n样本信息: x0hat_batch={tuple(x0hat_batch.shape)}")

    # 仅在需要时提前准备 d_samples_2d（threshold + d_samples 空间）
    d_samples_2d = None
    if args.cluster_mode == 'threshold' and args.threshold_space == 'd_samples':
        print("\nStep 1a: threshold_space=d_samples，先准备 d_samples_2d...")
        if (not args.recompute_d_samples) and ('d_samples_2d' in sm_data_in):
            d_samples_2d = sm_data_in['d_samples_2d'].detach().cpu().float()
            if int(d_samples_2d.shape[0]) != n_seeds:
                raise ValueError(
                    f"已有 d_samples_2d 样本数({d_samples_2d.shape[0]})与 x0hat_batch({n_seeds})不一致，"
                    f"请加 --recompute_d_samples"
                )
            print("✓ 复用已有 d_samples_2d")
        else:
            config, _ = create_sde_config(parent_dir, batch_size=1)
            operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
            d_samples_2d = _compute_d_samples_2d(
                operator=operator,
                x0hat_batch=x0hat_batch,
                device=device,
                forward_batch_size=args.forward_batch_size,
            )
            print(f"✓ d_samples_2d 重算完成: {tuple(d_samples_2d.shape)}")

    print("\nStep 1: 重分组...")
    if args.cluster_mode == 'kmeans':
        feats = x0hat_batch.reshape(n_seeds, -1).numpy().astype(np.float32)
        labels, rep_seed_indices, group_members, rep_center_dist = _cluster_kmeans(
            feats=feats,
            k=int(args.k),
            random_state=args.master_seed,
            max_iter=args.kmeans_max_iter,
        )
        grouping_space = 'x0hat'
        grouping_method = 'kmeans'
    else:
        if args.threshold_space == 'x0hat':
            feats = x0hat_batch.reshape(n_seeds, -1).numpy().astype(np.float32)
        else:
            if d_samples_2d is None:
                raise RuntimeError("内部错误: threshold_space=d_samples 但 d_samples_2d 未准备好")
            feats = d_samples_2d.numpy().astype(np.float32)
        labels, rep_seed_indices, group_members, rep_center_dist = _cluster_threshold_complete(
            feats=feats,
            distance_threshold=float(args.distance_threshold),
        )
        grouping_space = args.threshold_space
        grouping_method = 'threshold_complete'

    k_out = int(len(group_members))
    group_sizes = [len(m) for m in group_members]
    print("✓ 重分组完成")
    print(f"  group_count: {k_out}")
    print(f"  group_size(min/mean/max): {min(group_sizes)}/{np.mean(group_sizes):.2f}/{max(group_sizes)}")
    print(f"  rep_seed_indices (前10): {rep_seed_indices[:10]}")

    print("\nStep 2: 构建数据域表示 d_samples_2d...")
    if d_samples_2d is None:
        if (not args.recompute_d_samples) and ('d_samples_2d' in sm_data_in):
            d_samples_2d = sm_data_in['d_samples_2d'].detach().cpu().float()
            if int(d_samples_2d.shape[0]) != n_seeds:
                raise ValueError(
                    f"已有 d_samples_2d 样本数({d_samples_2d.shape[0]})与 x0hat_batch({n_seeds})不一致，"
                    f"请加 --recompute_d_samples"
                )
            print("✓ 复用已有 d_samples_2d")
        else:
            config, _ = create_sde_config(parent_dir, batch_size=1)
            operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
            d_samples_2d = _compute_d_samples_2d(
                operator=operator,
                x0hat_batch=x0hat_batch,
                device=device,
                forward_batch_size=args.forward_batch_size,
            )
            print(f"✓ d_samples_2d 重算完成: {tuple(d_samples_2d.shape)}")

    print("\nStep 3: 构建 d_centroids_2d 与 similarity_matrix...")
    rep_seed_tensor = torch.tensor(rep_seed_indices, dtype=torch.long)
    d_centroids_2d = d_samples_2d[rep_seed_tensor]  # [k_out, D]
    similarity_matrix = torch.cdist(d_centroids_2d, d_samples_2d, p=2)  # [k_out, N]
    print(f"✓ similarity_matrix: {tuple(similarity_matrix.shape)}")

    print("\nStep 4: 保存新 SM 资产...")
    if args.cluster_mode == 'kmeans':
        cluster_suffix = f"recluster-kmeans-k{k_out}"
    else:
        thr = _fmt_float_token(args.distance_threshold)
        cluster_suffix = f"recluster-thr-{thr}-space-{grouping_space}"
    sm_basename = (
        f"sm_dataset-{dataset_tag}_model-{model_tag}"
        f"_k{k_out}_j{n_seeds}_seed{args.master_seed}_{cluster_suffix}"
    )
    sm_filename = generate_timestamped_filename(sm_basename, '.pt')
    sm_path = os.path.join(out_dir, sm_filename)

    save_obj = {
        'similarity_matrix': similarity_matrix.cpu(),
        'k': int(k_out),
        # 兼容字段: 原语义为训练 patch centroid 索引；此处存 group 代表 seed 索引
        'centroid_indices': [int(v) for v in rep_seed_indices],
        'centroid_seed_indices': [int(v) for v in rep_seed_indices],
        'group_labels': torch.tensor(labels, dtype=torch.long),
        'group_members': group_members,
        'rep_center_dist': [float(v) for v in rep_center_dist],
        'n_noise_seeds': int(n_seeds),
        'master_seed': int(args.master_seed),
        'x0hat_batch': x0hat_batch.cpu(),
        'd_samples_2d': d_samples_2d.cpu(),
        'd_centroids_2d': d_centroids_2d.cpu(),
        'dataset_tag': dataset_tag,
        'model_tag': model_tag,
        'checkpoint_path': checkpoint_path,
        'sigma': float(cfg.daps.sigma),
        'image_size': int(cfg.image_size),
        'grouping_space': grouping_space,
        'grouping_method': grouping_method,
        'cluster_mode': args.cluster_mode,
        'distance_threshold': float(args.distance_threshold) if args.distance_threshold is not None else None,
        'source_sm_path': source_sm_path,
    }
    torch.save(save_obj, sm_path)

    group_csv_basename = (
        f"sm_group_summary_dataset-{dataset_tag}_model-{model_tag}"
        f"_k{k_out}_j{n_seeds}_seed{args.master_seed}_{cluster_suffix}"
    )
    group_csv_name = generate_timestamped_filename(group_csv_basename, '.csv')
    group_csv_path = os.path.join(out_dir, group_csv_name)
    _save_group_summary_csv(
        path=group_csv_path,
        group_members=group_members,
        rep_seed_indices=rep_seed_indices,
        rep_center_dist=rep_center_dist,
    )

    print("✓ 新 SM 已保存")
    print(f"  sm_path: {sm_path}")
    print(f"  group_csv: {group_csv_path}")
    print("\n说明:")
    print("  - 新文件名带时间戳，不会覆盖旧 SM。")
    print("  - centroid_indices 在该文件中表示“代表 seed 索引”（非训练 patch 索引）。")
    if args.cluster_mode == 'threshold':
        print(f"  - threshold 模式自动得到组数 k={k_out}。")


if __name__ == '__main__':
    main()
