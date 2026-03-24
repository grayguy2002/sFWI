"""
构建 SEAM 平移测试集（用于替换默认 gt_indices 的 224 训练切片）。

策略：
1) 先加载 SEAM 224 个 200x200 切片。
2) 对每个样本执行一次随机方向平移（up/down/left/right，默认 10 像素）。
3) 若目标数量大于 224（如 1000），再从 224 中有放回采样并做同样平移，补齐到目标数量。
4) 保存:
   - .pt: test_v_patches + 元信息
   - .csv: 每个样本的 base_idx / 方向 / 位移

后续可直接在 daps_langevin.py / daps_mechanism_probe.py 中使用:
  --eval_patches_path <this_pt_path>
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
from datetime import datetime

import numpy as np
import torch

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.config import FWIConfig
from sFWI.data.loaders import load_seam_model
from sFWI.utils.file_utils import generate_timestamped_filename


DIRS = ('up', 'down', 'left', 'right')


def _reflect_indices(idx: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(idx, dtype=np.int64)
    period = 2 * size - 2
    imod = np.mod(idx, period)
    return np.where(imod < size, imod, period - imod).astype(np.int64)


def _shift_patch(
    patch: np.ndarray,
    dy: int,
    dx: int,
    pad_mode: str = 'edge',
    constant_value: float = 0.0,
) -> np.ndarray:
    h, w = patch.shape
    if pad_mode == 'constant':
        out = np.full((h, w), float(constant_value), dtype=np.float32)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = patch[src_y0:src_y1, src_x0:src_x1]
        return out

    y_idx = np.arange(h, dtype=np.int64) - int(dy)
    x_idx = np.arange(w, dtype=np.int64) - int(dx)
    if pad_mode == 'edge':
        y_src = np.clip(y_idx, 0, h - 1)
        x_src = np.clip(x_idx, 0, w - 1)
    elif pad_mode == 'reflect':
        y_src = _reflect_indices(y_idx, h)
        x_src = _reflect_indices(x_idx, w)
    else:
        raise ValueError(f"不支持的 pad_mode: {pad_mode}")
    return patch[np.ix_(y_src, x_src)].astype(np.float32)


def _sample_direction(rng: np.random.Generator, shift_pixels: int):
    d = int(rng.integers(0, 4))
    if d == 0:
        return 'up', -shift_pixels, 0
    if d == 1:
        return 'down', shift_pixels, 0
    if d == 2:
        return 'left', 0, -shift_pixels
    return 'right', 0, shift_pixels


def _build_plan(
    n_base: int,
    target_size: int,
    shift_pixels: int,
    rng: np.random.Generator,
):
    plan = []

    if target_size <= 0:
        raise ValueError("--target_size 必须 > 0")

    if target_size >= n_base:
        for i in range(n_base):
            direction, dy, dx = _sample_direction(rng, shift_pixels)
            plan.append({
                'base_idx': int(i),
                'direction': direction,
                'dy': int(dy),
                'dx': int(dx),
                'is_coverage': 1,
            })
        extra = target_size - n_base
        for _ in range(extra):
            base_idx = int(rng.integers(0, n_base))
            direction, dy, dx = _sample_direction(rng, shift_pixels)
            plan.append({
                'base_idx': base_idx,
                'direction': direction,
                'dy': int(dy),
                'dx': int(dx),
                'is_coverage': 0,
            })
    else:
        chosen = rng.choice(n_base, size=target_size, replace=False)
        for i in chosen.tolist():
            direction, dy, dx = _sample_direction(rng, shift_pixels)
            plan.append({
                'base_idx': int(i),
                'direction': direction,
                'dy': int(dy),
                'dx': int(dx),
                'is_coverage': 1,
            })
    return plan


def main():
    parser = argparse.ArgumentParser(description='构建 SEAM 平移测试集（1000 切片等）')
    parser.add_argument('--seam_model_path', type=str, default=None,
                        help='SEAM .sgy 路径；默认使用 FWIConfig')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录；默认使用 FWIConfig.paths.output_path')
    parser.add_argument('--target_size', type=int, default=1000,
                        help='目标测试集大小（默认 1000）')
    parser.add_argument('--shift_pixels', type=int, default=10,
                        help='平移像素数（默认 10）')
    parser.add_argument('--pad_mode', type=str, default='edge',
                        choices=['edge', 'reflect', 'constant'],
                        help='平移后边界填充模式')
    parser.add_argument('--constant_value', type=float, default=0.0,
                        help='pad_mode=constant 时的填充值')
    parser.add_argument('--master_seed', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--base_name', type=str, default='seam_shifted_testset')
    args = parser.parse_args()

    if args.shift_pixels < 1:
        raise ValueError('--shift_pixels 必须 >= 1')
    if args.patch_size < 2:
        raise ValueError('--patch_size 必须 >= 2')
    if args.stride < 1:
        raise ValueError('--stride 必须 >= 1')

    cfg = FWIConfig()
    seam_model_path = args.seam_model_path or cfg.paths.seam_model_path
    output_dir = args.output_dir or cfg.paths.output_path
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 72)
    print('SEAM 平移测试集构建')
    print('=' * 72)
    print(f'  seam_model_path: {seam_model_path}')
    print(f'  output_dir:      {output_dir}')
    print(f'  target_size:     {args.target_size}')
    print(f'  shift_pixels:    {args.shift_pixels}')
    print(f'  pad_mode:        {args.pad_mode}')
    print(f'  master_seed:     {args.master_seed}')

    base = load_seam_model(
        seam_model_path,
        patch_size_h=args.patch_size,
        patch_size_w=args.patch_size,
        stride_h=args.stride,
        stride_w=args.stride,
    )
    if base.dim() != 3:
        raise ValueError(f'SEAM 切片维度异常，期望 [N,H,W]，当前: {tuple(base.shape)}')
    base = base.detach().cpu().float()
    n_base = int(base.shape[0])
    print(f'  base_patches:    {tuple(base.shape)}')

    rng = np.random.default_rng(int(args.master_seed))
    plan = _build_plan(
        n_base=n_base,
        target_size=int(args.target_size),
        shift_pixels=int(args.shift_pixels),
        rng=rng,
    )

    shifted = []
    rows = []
    for i, item in enumerate(plan):
        base_idx = int(item['base_idx'])
        dy = int(item['dy'])
        dx = int(item['dx'])
        p = base[base_idx].numpy()
        p_shift = _shift_patch(
            patch=p,
            dy=dy,
            dx=dx,
            pad_mode=args.pad_mode,
            constant_value=args.constant_value,
        )
        shifted.append(torch.from_numpy(p_shift))
        rows.append({
            'sample_idx': int(i),
            'base_idx': int(base_idx),
            'direction': str(item['direction']),
            'dy': int(dy),
            'dx': int(dx),
            'is_coverage': int(item['is_coverage']),
        })

    test_v_patches = torch.stack(shifted, dim=0).float()
    n_test = int(test_v_patches.shape[0])

    direction_counts = {d: 0 for d in DIRS}
    for r in rows:
        direction_counts[r['direction']] += 1
    unique_base = len(set(int(r['base_idx']) for r in rows))
    cover_count = int(sum(int(r['is_coverage']) for r in rows))

    token = (
        f"{args.base_name}_n{n_test}"
        f"_shift{int(args.shift_pixels)}"
        f"_seed{int(args.master_seed)}"
        f"_pad{args.pad_mode}"
    )
    pt_name = generate_timestamped_filename(token, '.pt')
    csv_name = generate_timestamped_filename(token + '_meta', '.csv')
    pt_path = os.path.join(output_dir, pt_name)
    csv_path = os.path.join(output_dir, csv_name)

    payload = {
        'test_v_patches': test_v_patches,
        'base_indices': torch.tensor([r['base_idx'] for r in rows], dtype=torch.long),
        'shift_dy': torch.tensor([r['dy'] for r in rows], dtype=torch.long),
        'shift_dx': torch.tensor([r['dx'] for r in rows], dtype=torch.long),
        'directions': [r['direction'] for r in rows],
        'is_coverage': torch.tensor([r['is_coverage'] for r in rows], dtype=torch.long),
        'source_dataset': 'seam',
        'source_seam_model_path': seam_model_path,
        'source_base_size': int(n_base),
        'target_size': int(n_test),
        'shift_pixels': int(args.shift_pixels),
        'pad_mode': str(args.pad_mode),
        'constant_value': float(args.constant_value),
        'master_seed': int(args.master_seed),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    torch.save(payload, pt_path)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['sample_idx', 'base_idx', 'direction', 'dy', 'dx', 'is_coverage'],
        )
        writer.writeheader()
        writer.writerows(rows)

    print('\n构建完成:')
    print(f'  test_v_patches: {tuple(test_v_patches.shape)}')
    print(f'  unique_base:    {unique_base}/{n_base}')
    print(f'  coverage_rows:  {cover_count}')
    print(
        '  direction_count: '
        f"up={direction_counts['up']}, down={direction_counts['down']}, "
        f"left={direction_counts['left']}, right={direction_counts['right']}"
    )
    print(f'  pt_path:        {pt_path}')
    print(f'  csv_path:       {csv_path}')


if __name__ == '__main__':
    main()
