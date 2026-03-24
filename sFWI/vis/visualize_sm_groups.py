#!/usr/bin/env python3
"""
从 GSS 相似度矩阵资产可视化 group 代表和组内样本。

功能:
1) 可视化 n 个 group 代表（每组 1 个 seed）
2) 可视化 n 个 group，每组不少于 2 个速度模型（可配置 members_per_group）

分组规则:
- similarity_matrix 形状 [k, n_seeds]
- 对每个 seed j，group(j) = argmin_i similarity_matrix[i, j]
- 某组代表 seed = 该组内使 similarity_matrix[group, seed] 最小的 seed
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _to_2d(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]
    return x.detach().cpu().float().numpy()


def _load_sm(sm_path: str):
    if not os.path.isfile(sm_path):
        raise FileNotFoundError(f"SM 文件不存在: {sm_path}")
    sm_data = torch.load(sm_path, weights_only=False, map_location='cpu')
    required = ['similarity_matrix', 'x0hat_batch']
    for key in required:
        if key not in sm_data:
            raise KeyError(f"SM 文件缺少字段: {key}")
    return sm_data


def _build_group_index(
    similarity_matrix: torch.Tensor,
    group_members=None,
    rep_seed_indices=None,
):
    """
    返回:
      group_to_seeds: dict[group_id] -> list[(seed_idx, score)] 按 score 升序
      grouping_source: str
    """
    k, n_seeds = similarity_matrix.shape
    # 新资产优先: 直接使用保存的 group_members
    if group_members is not None:
        gm = group_members
        if isinstance(gm, torch.Tensor):
            gm = gm.detach().cpu().tolist()
        if isinstance(gm, (list, tuple)):
            if len(gm) == k:
                group_to_seeds = {}
                for g in range(k):
                    members = [int(v) for v in gm[g]]
                    entries = []
                    for seed in members:
                        if 0 <= seed < n_seeds:
                            score = float(similarity_matrix[g, seed].item())
                            entries.append((seed, score))
                    # 若该组为空，尝试用代表 seed 回填
                    if not entries and rep_seed_indices is not None and g < len(rep_seed_indices):
                        rep = int(rep_seed_indices[g])
                        if 0 <= rep < n_seeds:
                            entries.append((rep, float(similarity_matrix[g, rep].item())))
                    # 将指定代表 seed 放到首位（如果存在）
                    if rep_seed_indices is not None and g < len(rep_seed_indices):
                        rep = int(rep_seed_indices[g])
                        entries.sort(key=lambda x: x[1])
                        pos = None
                        for i, (s, _) in enumerate(entries):
                            if s == rep:
                                pos = i
                                break
                        if pos is not None and pos != 0:
                            rep_item = entries.pop(pos)
                            entries.insert(0, rep_item)
                    else:
                        entries.sort(key=lambda x: x[1])
                    group_to_seeds[g] = entries
                return group_to_seeds, 'group_members'

    # 回退旧逻辑: 按列 argmin 分配组
    group_of_seed = torch.argmin(similarity_matrix, dim=0)  # [n_seeds]
    group_to_seeds = {g: [] for g in range(k)}
    for seed in range(n_seeds):
        g = int(group_of_seed[seed].item())
        score = float(similarity_matrix[g, seed].item())
        group_to_seeds[g].append((seed, score))
    for g in range(k):
        group_to_seeds[g].sort(key=lambda x: x[1])
    return group_to_seeds, 'argmin(similarity)'


def _select_groups(group_to_seeds, n_groups: int, min_members: int, mode: str, master_seed: int):
    valid_groups = [g for g, lst in group_to_seeds.items() if len(lst) >= min_members]
    if not valid_groups:
        raise RuntimeError(f"没有 group 满足 min_members={min_members}")

    if mode == 'largest':
        valid_groups.sort(key=lambda g: len(group_to_seeds[g]), reverse=True)
    elif mode == 'random':
        rng = np.random.default_rng(master_seed)
        rng.shuffle(valid_groups)
    elif mode == 'index':
        valid_groups.sort()
    else:
        raise ValueError(f"未知 group_select_mode: {mode}")

    selected = valid_groups[: min(n_groups, len(valid_groups))]
    return selected


def _compute_vrange(x0hat_batch: torch.Tensor, selected_groups, group_to_seeds, members_per_group: int):
    vals = []
    for g in selected_groups:
        top = group_to_seeds[g][:members_per_group]
        for seed, _ in top:
            vals.append(_to_2d(x0hat_batch[seed]))
    arr = np.concatenate([v.reshape(-1) for v in vals], axis=0)
    vmin = float(np.percentile(arr, 1.0))
    vmax = float(np.percentile(arr, 99.0))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-6
    return vmin, vmax


def _save_representatives_plot(
    x0hat_batch: torch.Tensor,
    selected_groups,
    group_to_seeds,
    output_path: str,
    title_prefix: str,
    vmin: float,
    vmax: float,
):
    n = len(selected_groups)
    ncols = min(6, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for idx, g in enumerate(selected_groups):
        ax = axes.flat[idx]
        rep_seed, rep_score = group_to_seeds[g][0]
        img = _to_2d(x0hat_batch[rep_seed])
        ax.imshow(img, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"group={g}\nseed={rep_seed}, score={rep_score:.4f}", fontsize=8.5)

    fig.suptitle(f"{title_prefix} | Group Representatives", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _save_group_members_plot(
    x0hat_batch: torch.Tensor,
    selected_groups,
    group_to_seeds,
    members_per_group: int,
    output_path: str,
    title_prefix: str,
    vmin: float,
    vmax: float,
):
    n_groups = len(selected_groups)
    ncols = members_per_group
    nrows = n_groups
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.8 * ncols, 2.4 * nrows),
        squeeze=False
    )

    for row_idx, g in enumerate(selected_groups):
        members = group_to_seeds[g][:members_per_group]
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.axis('off')
            if col_idx >= len(members):
                continue
            seed, score = members[col_idx]
            img = _to_2d(x0hat_batch[seed])
            ax.imshow(img, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
            if col_idx == 0:
                ax.set_title(f"group={g}\nseed={seed}, score={score:.4f}", fontsize=8.5)
            else:
                ax.set_title(f"seed={seed}, score={score:.4f}", fontsize=8.0)

    fig.suptitle(
        f"{title_prefix} | Group Members (top-{members_per_group} per group)",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='可视化 SM 资产中的 group 代表与组内样本')
    parser.add_argument('--sm_path', type=str, required=True,
                        help='sm_dataset-*.pt 文件路径')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出目录')
    parser.add_argument('--n_groups', type=int, default=12,
                        help='可视化 group 数量')
    parser.add_argument('--members_per_group', type=int, default=3,
                        help='每组展示样本数量，需 >=2')
    parser.add_argument('--group_select_mode', type=str, default='largest',
                        choices=['largest', 'random', 'index'],
                        help='group 选择策略: largest|random|index')
    parser.add_argument('--master_seed', type=int, default=8,
                        help='random 模式下随机种子')
    args = parser.parse_args()

    if args.n_groups < 1:
        raise ValueError("--n_groups 必须 >= 1")
    if args.members_per_group < 2:
        raise ValueError("--members_per_group 必须 >= 2")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("SM Group 可视化")
    print("=" * 72)
    print(f"  sm_path: {args.sm_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  n_groups: {args.n_groups}")
    print(f"  members_per_group: {args.members_per_group}")
    print(f"  group_select_mode: {args.group_select_mode}")

    sm_data = _load_sm(args.sm_path)
    similarity_matrix = sm_data['similarity_matrix'].float().cpu()
    x0hat_batch = sm_data['x0hat_batch'].float().cpu()
    k, n_seeds = similarity_matrix.shape
    print(f"  matrix shape: {tuple(similarity_matrix.shape)}")
    print(f"  x0hat_batch shape: {tuple(x0hat_batch.shape)}")

    rep_seed_indices = sm_data.get('centroid_seed_indices', None)
    group_members = sm_data.get('group_members', None)
    group_to_seeds, grouping_source = _build_group_index(
        similarity_matrix=similarity_matrix,
        group_members=group_members,
        rep_seed_indices=rep_seed_indices,
    )
    print(f"  grouping_source: {grouping_source}")
    selected_groups = _select_groups(
        group_to_seeds=group_to_seeds,
        n_groups=args.n_groups,
        min_members=args.members_per_group,
        mode=args.group_select_mode,
        master_seed=args.master_seed,
    )

    print(f"  selected groups ({len(selected_groups)}): {selected_groups}")
    for g in selected_groups:
        rep_seed, rep_score = group_to_seeds[g][0]
        print(f"    group {g}: size={len(group_to_seeds[g])}, rep_seed={rep_seed}, rep_score={rep_score:.4f}")

    vmin, vmax = _compute_vrange(x0hat_batch, selected_groups, group_to_seeds, args.members_per_group)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = str(sm_data.get('model_tag', 'unknown'))
    dataset_tag = str(sm_data.get('dataset_tag', 'unknown'))
    grouping_method = str(sm_data.get('grouping_method', grouping_source))
    title_prefix = (
        f"dataset={dataset_tag}, model={model_tag}, k={k}, seeds={n_seeds}, "
        f"grouping={grouping_method}"
    )

    rep_path = os.path.join(
        args.output_dir,
        f"sm_group_representatives_dataset-{dataset_tag}_model-{model_tag}_n{len(selected_groups)}_{ts}.png"
    )
    members_path = os.path.join(
        args.output_dir,
        f"sm_group_members_dataset-{dataset_tag}_model-{model_tag}_n{len(selected_groups)}_m{args.members_per_group}_{ts}.png"
    )

    _save_representatives_plot(
        x0hat_batch=x0hat_batch,
        selected_groups=selected_groups,
        group_to_seeds=group_to_seeds,
        output_path=rep_path,
        title_prefix=title_prefix,
        vmin=vmin,
        vmax=vmax,
    )
    _save_group_members_plot(
        x0hat_batch=x0hat_batch,
        selected_groups=selected_groups,
        group_to_seeds=group_to_seeds,
        members_per_group=args.members_per_group,
        output_path=members_path,
        title_prefix=title_prefix,
        vmin=vmin,
        vmax=vmax,
    )

    print("\n输出文件:")
    print(f"  representatives: {rep_path}")
    print(f"  group_members:   {members_path}")


if __name__ == '__main__':
    main()
