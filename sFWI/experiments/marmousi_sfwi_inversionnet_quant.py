#!/usr/bin/env python3
"""
sFWI vs InversionNet on Marmousi: quantitative comparison helper.

用途:
1) 读取 daps_langevin 的 run_summary.csv 作为 sFWI 指标来源
2) 在同一批 GT 索引上运行 InversionNet (seismic -> velocity)
3) 输出 per-GT 对比表 + summary 表 (+ 可选 LaTeX 表)

典型用法:
  python sFWI/experiments/marmousi_sfwi_inversionnet_quant.py \
    --sfwi_run_csv "/path/to/daps_langevin_run_summary_model-marmousi_*.csv" \
    --inversionnet_ckpt /path/to/inversionnet_best.pth \
    --dataset_path /path/to/seismic_dataset.pkl \
    --gt_from_summary /path/to/gt_format_match_summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # .../code
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.config import FWIConfig
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.models.inversionnet import InversionNetSFWI, load_inversionnet_state_dict
from sFWI.utils.file_utils import generate_timestamped_filename


def _resolve_paths(specs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for spec in specs:
        matched = sorted(glob.glob(spec))
        if matched:
            out.extend(matched)
        elif os.path.isfile(spec):
            out.append(spec)
    # dedup + keep order
    seen = set()
    uniq = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred4 = _to_4d(pred).float()
    gt4 = _to_4d(gt).float()
    if pred4.shape[-2:] != gt4.shape[-2:]:
        pred4 = F.interpolate(pred4, size=gt4.shape[-2:], mode='bilinear', align_corners=False)
    return pred4, gt4


def compute_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred4, gt4 = _align_pred_to_gt(pred, gt)
    mse = F.mse_loss(pred4, gt4).item()
    rmse = math.sqrt(mse)
    gt_range = float((gt4.max() - gt4.min()).item())
    if gt_range < 1e-8:
        return float('inf')
    return float(rmse / gt_range)


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
    return float((num / den).item())


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    nrmse = compute_nrmse(pred, gt)
    return float(10.0 * np.log10(1.0 / (nrmse ** 2 + 1e-10)))


def _safe_float(v, default=float('nan')) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float('nan')
    return float(sum(vals) / len(vals))


def _std(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float('nan')
    if len(vals) == 1:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=np.float64)))


def _apply_norm(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    mode = (stats or {}).get('mode', 'none')
    if mode == 'none':
        return x
    if mode == 'zscore':
        mean = torch.tensor(float(stats['mean']), dtype=x.dtype, device=x.device)
        std = torch.tensor(float(stats['std']), dtype=x.dtype, device=x.device).clamp_min(1e-8)
        return (x - mean) / std
    if mode == 'minmax_01':
        mn = torch.tensor(float(stats['min']), dtype=x.dtype, device=x.device)
        mx = torch.tensor(float(stats['max']), dtype=x.dtype, device=x.device)
        return (x - mn) / (mx - mn).clamp_min(1e-8)
    if mode == 'minmax_m11':
        mn = torch.tensor(float(stats['min']), dtype=x.dtype, device=x.device)
        mx = torch.tensor(float(stats['max']), dtype=x.dtype, device=x.device)
        x01 = (x - mn) / (mx - mn).clamp_min(1e-8)
        return x01 * 2.0 - 1.0
    raise ValueError(f'未知归一化模式: {mode}')


def _undo_norm(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    mode = (stats or {}).get('mode', 'none')
    if mode == 'none':
        return x
    if mode == 'zscore':
        mean = torch.tensor(float(stats['mean']), dtype=x.dtype, device=x.device)
        std = torch.tensor(float(stats['std']), dtype=x.dtype, device=x.device)
        return x * std + mean
    if mode == 'minmax_01':
        mn = torch.tensor(float(stats['min']), dtype=x.dtype, device=x.device)
        mx = torch.tensor(float(stats['max']), dtype=x.dtype, device=x.device)
        return x * (mx - mn) + mn
    if mode == 'minmax_m11':
        mn = torch.tensor(float(stats['min']), dtype=x.dtype, device=x.device)
        mx = torch.tensor(float(stats['max']), dtype=x.dtype, device=x.device)
        x01 = (x + 1.0) * 0.5
        return x01 * (mx - mn) + mn
    raise ValueError(f'未知归一化模式: {mode}')


def _sync_if_cuda(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _load_sfwi_rows(csv_paths: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for p in csv_paths:
        with open(p, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = dict(row)
                r['_source_csv'] = p
                rows.append(r)
    return rows


def _choose_row(rows: Sequence[Dict], reduce_mode: str, metric_col: str, misfit_col: str) -> Dict:
    if not rows:
        return {}
    if reduce_mode == 'mean':
        # mean 模式不需要单行选择，这里返回第一行占位
        return rows[0]
    if reduce_mode == 'best_nrmse':
        return min(rows, key=lambda r: _safe_float(r.get(metric_col), float('inf')))
    if reduce_mode == 'best_misfit':
        if misfit_col in rows[0]:
            return min(rows, key=lambda r: _safe_float(r.get(misfit_col), float('inf')))
        return min(rows, key=lambda r: _safe_float(r.get(metric_col), float('inf')))
    raise ValueError(f'未知 reduce_mode: {reduce_mode}')


def _aggregate_sfwi_by_gt(
    rows: Sequence[Dict],
    gt_indices: Sequence[int],
    prefix: str,
    reduce_mode: str,
) -> Dict[int, Dict]:
    metric_col = f'{prefix}_gt_nrmse'
    ssim_col = f'{prefix}_gt_ssim'
    psnr_col = f'{prefix}_gt_psnr'
    misfit_col = f'{prefix}_misfit_after'
    time_col = 'runtime_s'

    grouped = defaultdict(list)
    for r in rows:
        gi_raw = r.get('gt_index', r.get('gt_idx'))
        if gi_raw is None:
            continue
        gi = int(gi_raw)
        grouped[gi].append(r)

    out: Dict[int, Dict] = {}
    for gi in gt_indices:
        rs = grouped.get(int(gi), [])
        if not rs:
            continue

        if reduce_mode == 'mean':
            nrmse = _mean([_safe_float(x.get(metric_col)) for x in rs])
            ssim = _mean([_safe_float(x.get(ssim_col)) for x in rs])
            psnr = _mean([_safe_float(x.get(psnr_col)) for x in rs])
            time_s = _mean([_safe_float(x.get(time_col)) for x in rs])
            chosen = rs[0]
        else:
            chosen = _choose_row(rs, reduce_mode=reduce_mode, metric_col=metric_col, misfit_col=misfit_col)
            nrmse = _safe_float(chosen.get(metric_col))
            ssim = _safe_float(chosen.get(ssim_col))
            psnr = _safe_float(chosen.get(psnr_col))
            time_s = _safe_float(chosen.get(time_col))

        out[int(gi)] = {
            'NRMSE': float(nrmse),
            'SSIM': float(ssim),
            'PSNR': float(psnr),
            'time_s': float(time_s),
            'n_runs': int(len(rs)),
            'best_group': int(_safe_float(chosen.get('best_group'), -1)),
            'best_seed': int(_safe_float(chosen.get('best_seed'), -1)),
            'candidate_seed': int(_safe_float(chosen.get('candidate_seed'), -1)),
        }
    return out


def _indices_from_gt_summary(summary_path: str) -> List[int]:
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best = data.get('best_per_panel', {})
    out: List[int] = []
    for k in sorted(best.keys(), key=lambda x: int(x)):
        rec = best[k]
        if isinstance(rec, dict) and ('dataset_index' in rec):
            out.append(int(rec['dataset_index']))
    # dedup
    out = sorted(set(out))
    return out


def _resolve_dataset_path(path_arg: str) -> str:
    if path_arg:
        return path_arg
    cfg = FWIConfig()
    return cfg.paths.marmousi_dataset_path


def _resolve_output_dir(path_arg: str) -> str:
    if path_arg:
        return path_arg
    cfg = FWIConfig()
    return os.path.join(cfg.paths.output_path, 'quant_marmousi_sfwi_vs_inversionnet')


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg and device_arg.lower() != 'auto':
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _write_csv(path: str, rows: Sequence[Dict], headers: Sequence[str]):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_latex_table(path: str, rows: Sequence[Dict]):
    headers = ['Method', 'NRMSE', 'SSIM', 'PSNR(dB)', 'Time(s)']
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{sFWI vs InversionNet on Marmousi (same GT set)}')
    lines.append(r'\label{tab:sfwi_vs_inversionnet_marmousi}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\hline')
    lines.append(' & '.join(headers) + r' \\')
    lines.append(r'\hline')
    for r in rows:
        method = str(r['method'])
        nrmse = f"{float(r['nrmse_mean']):.4f} ± {float(r['nrmse_std']):.4f}"
        ssim = f"{float(r['ssim_mean']):.4f} ± {float(r['ssim_std']):.4f}"
        psnr = f"{float(r['psnr_mean']):.2f} ± {float(r['psnr_std']):.2f}"
        t = f"{float(r['time_mean_s']):.3f}"
        lines.append(f"{method} & {nrmse} & {ssim} & {psnr} & {t} \\\\")
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Marmousi: sFWI vs InversionNet 定量对比')
    parser.add_argument('--sfwi_run_csv', type=str, nargs='+', required=True,
                        help='daps_langevin_run_summary_model-*.csv 路径或 glob (可多个)')
    parser.add_argument('--inversionnet_ckpt', type=str, required=True,
                        help='InversionNet checkpoint 路径')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='seismic_dataset.pkl 路径，默认自动推断')
    parser.add_argument('--out_dir', type=str, default='',
                        help='输出目录，默认 code/outputs/quant_marmousi_sfwi_vs_inversionnet')
    parser.add_argument('--gt_indices', type=int, nargs='*', default=None,
                        help='指定 GT 索引；不指定时优先用 --gt_from_summary，否则取 sFWI CSV 中全部 GT')
    parser.add_argument('--gt_from_summary', type=str, default='',
                        help='gt_format_match_summary.json 路径，用于自动提取 GT 索引')
    parser.add_argument('--sfwi_prefix', type=str, default='x0hat', choices=['x0hat', 'x0y'],
                        help='使用 sFWI CSV 中哪组指标前缀')
    parser.add_argument('--sfwi_reduce', type=str, default='mean',
                        choices=['mean', 'best_nrmse', 'best_misfit'],
                        help='同一 GT 多次 run 的聚合方式')
    parser.add_argument('--batch_size', type=int, default=16, help='InversionNet 推理 batch size')
    parser.add_argument('--device', type=str, default='auto', help='cuda / cpu / auto')
    parser.add_argument('--strict_load', action='store_true',
                        help='严格加载 InversionNet checkpoint（默认非严格）')
    parser.add_argument('--write_latex', action='store_true', help='额外保存 LaTeX 表格')
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError('--batch_size 必须 >= 1')

    csv_paths = _resolve_paths(args.sfwi_run_csv)
    if not csv_paths:
        raise FileNotFoundError('未解析到任何 --sfwi_run_csv 文件')
    sfwi_rows = _load_sfwi_rows(csv_paths)
    if not sfwi_rows:
        raise RuntimeError('sFWI CSV 为空')

    if args.gt_indices:
        gt_indices = sorted(set(int(x) for x in args.gt_indices))
        gt_source = '--gt_indices'
    elif args.gt_from_summary:
        if not os.path.isfile(args.gt_from_summary):
            raise FileNotFoundError(f'--gt_from_summary 文件不存在: {args.gt_from_summary}')
        gt_indices = _indices_from_gt_summary(args.gt_from_summary)
        gt_source = '--gt_from_summary'
    else:
        gt_indices = sorted(
            set(
                int(r.get('gt_index', r.get('gt_idx')))
                for r in sfwi_rows
                if (r.get('gt_index', r.get('gt_idx')) is not None)
            )
        )
        gt_source = 'sfwi_run_csv'

    if not gt_indices:
        raise RuntimeError('未解析到任何 GT 索引')

    sfwi_by_gt = _aggregate_sfwi_by_gt(
        rows=sfwi_rows,
        gt_indices=gt_indices,
        prefix=args.sfwi_prefix,
        reduce_mode=args.sfwi_reduce,
    )

    dataset_path = _resolve_dataset_path(args.dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'dataset 不存在: {dataset_path}')
    if not os.path.isfile(args.inversionnet_ckpt):
        raise FileNotFoundError(f'checkpoint 不存在: {args.inversionnet_ckpt}')

    out_dir = _resolve_output_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = _resolve_device(args.device)

    print('=' * 72)
    print('Marmousi Quantitative Comparison: sFWI vs InversionNet')
    print('=' * 72)
    print(f'  sFWI CSV count: {len(csv_paths)}')
    print(f'  GT source: {gt_source}, n_gt={len(gt_indices)}')
    print(f'  sfwi_prefix={args.sfwi_prefix}, sfwi_reduce={args.sfwi_reduce}')
    print(f'  dataset: {dataset_path}')
    print(f'  inversionnet_ckpt: {args.inversionnet_ckpt}')
    print(f'  device: {device}')
    print(f'  out_dir: {out_dir}')

    velocity, seismic = load_marmousi_from_pkl(dataset_path)
    velocity = velocity.float().cpu()
    seismic = None if seismic is None else seismic.float().cpu()
    if seismic is None:
        raise RuntimeError("Marmousi 数据不包含 seismic，无法评估 InversionNet。")
    if velocity.dim() == 3:
        velocity = velocity.unsqueeze(1)
    if seismic.dim() == 3:
        seismic = seismic.unsqueeze(1)

    n_total = int(min(velocity.shape[0], seismic.shape[0]))
    velocity = velocity[:n_total]
    seismic = seismic[:n_total]

    gt_in_range = [gi for gi in gt_indices if 0 <= gi < n_total]
    missing_range = sorted(set(gt_indices) - set(gt_in_range))
    if missing_range:
        print(f'[warn] 以下 GT 索引超出数据范围 [0, {n_total - 1}]，将忽略: {missing_range}')

    gt_final = [gi for gi in gt_in_range if gi in sfwi_by_gt]
    miss_sfwi = sorted(set(gt_in_range) - set(gt_final))
    if miss_sfwi:
        print(f'[warn] 以下 GT 在 sFWI CSV 中无记录，已忽略: {miss_sfwi}')
    if not gt_final:
        raise RuntimeError('没有可比较的 GT（需同时存在于数据集与 sFWI CSV）')

    ckpt = torch.load(args.inversionnet_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    input_shape = tuple(ckpt.get('input_shape', tuple(seismic.shape[1:]))) if isinstance(ckpt, dict) else tuple(seismic.shape[1:])
    output_shape = tuple(ckpt.get('output_shape', tuple(velocity.shape[-2:]))) if isinstance(ckpt, dict) else tuple(velocity.shape[-2:])
    if len(input_shape) == 2:
        input_shape = (1,) + tuple(input_shape)
    if len(output_shape) == 3:
        output_shape = tuple(output_shape[-2:])
    if len(output_shape) != 2:
        output_shape = tuple(velocity.shape[-2:])

    model = InversionNetSFWI(
        input_shape=input_shape,
        output_shape=output_shape,
        dim1=int(ckpt_args.get('dim1', 32)),
        dim2=int(ckpt_args.get('dim2', 64)),
        dim3=int(ckpt_args.get('dim3', 128)),
        dim4=int(ckpt_args.get('dim4', 256)),
        dim5=int(ckpt_args.get('dim5', 512)),
        norm=str(ckpt_args.get('norm_type', 'bn')),
        output_crop=int(ckpt_args.get('output_crop', 5)),
    ).to(device)
    incompatible = load_inversionnet_state_dict(model, ckpt, strict=bool(args.strict_load))
    model.eval()
    print(f'  inversionnet load: missing={len(incompatible.missing_keys)}, '
          f'unexpected={len(incompatible.unexpected_keys)}')

    seismic_stats = ckpt.get('seismic_stats', {'mode': 'none'}) if isinstance(ckpt, dict) else {'mode': 'none'}
    velocity_stats = ckpt.get('velocity_stats', {'mode': 'none'}) if isinstance(ckpt, dict) else {'mode': 'none'}
    print(f"  inversionnet norm: seismic={seismic_stats.get('mode', 'none')}, "
          f"velocity={velocity_stats.get('mode', 'none')}")

    inv_by_gt: Dict[int, Dict] = {}
    batch_size = int(args.batch_size)

    for st in range(0, len(gt_final), batch_size):
        ed = min(len(gt_final), st + batch_size)
        idx_batch = gt_final[st:ed]
        sx = seismic[idx_batch].to(device)      # [B,1,100,300]
        gt = velocity[idx_batch].to(device)     # [B,1,200,200]

        sx_n = _apply_norm(sx, seismic_stats)
        _sync_if_cuda(device)
        t0 = time.time()
        with torch.no_grad():
            pred_n = model(sx_n)
        _sync_if_cuda(device)
        elapsed = float(time.time() - t0)
        each_time = elapsed / max(1, len(idx_batch))

        pred = _undo_norm(pred_n, velocity_stats)

        for i, gi in enumerate(idx_batch):
            pred_i = pred[i:i + 1].detach()
            gt_i = gt[i:i + 1].detach()
            inv_by_gt[int(gi)] = {
                'NRMSE': compute_nrmse(pred_i, gt_i),
                'SSIM': compute_ssim(pred_i, gt_i),
                'PSNR': compute_psnr(pred_i, gt_i),
                'time_s': float(each_time),
            }

    per_gt_rows = []
    for gi in gt_final:
        s = sfwi_by_gt[int(gi)]
        b = inv_by_gt[int(gi)]
        row = {
            'gt_index': int(gi),
            'sfwi_nrmse': float(s['NRMSE']),
            'sfwi_ssim': float(s['SSIM']),
            'sfwi_psnr': float(s['PSNR']),
            'sfwi_time_s': float(s['time_s']),
            'sfwi_n_runs': int(s['n_runs']),
            'inversionnet_nrmse': float(b['NRMSE']),
            'inversionnet_ssim': float(b['SSIM']),
            'inversionnet_psnr': float(b['PSNR']),
            'inversionnet_time_s': float(b['time_s']),
            'delta_nrmse_inv_minus_sfwi': float(b['NRMSE'] - s['NRMSE']),
            'delta_ssim_inv_minus_sfwi': float(b['SSIM'] - s['SSIM']),
            'delta_psnr_inv_minus_sfwi': float(b['PSNR'] - s['PSNR']),
        }
        per_gt_rows.append(row)

    sfwi_nrmse = [r['sfwi_nrmse'] for r in per_gt_rows]
    sfwi_ssim = [r['sfwi_ssim'] for r in per_gt_rows]
    sfwi_psnr = [r['sfwi_psnr'] for r in per_gt_rows]
    sfwi_time = [r['sfwi_time_s'] for r in per_gt_rows]

    inv_nrmse = [r['inversionnet_nrmse'] for r in per_gt_rows]
    inv_ssim = [r['inversionnet_ssim'] for r in per_gt_rows]
    inv_psnr = [r['inversionnet_psnr'] for r in per_gt_rows]
    inv_time = [r['inversionnet_time_s'] for r in per_gt_rows]

    summary_rows = [
        {
            'method': 'sFWI',
            'n_samples': int(len(per_gt_rows)),
            'nrmse_mean': _mean(sfwi_nrmse),
            'nrmse_std': _std(sfwi_nrmse),
            'ssim_mean': _mean(sfwi_ssim),
            'ssim_std': _std(sfwi_ssim),
            'psnr_mean': _mean(sfwi_psnr),
            'psnr_std': _std(sfwi_psnr),
            'time_mean_s': _mean(sfwi_time),
            'time_total_s': float(sum(sfwi_time)),
        },
        {
            'method': 'InversionNet',
            'n_samples': int(len(per_gt_rows)),
            'nrmse_mean': _mean(inv_nrmse),
            'nrmse_std': _std(inv_nrmse),
            'ssim_mean': _mean(inv_ssim),
            'ssim_std': _std(inv_ssim),
            'psnr_mean': _mean(inv_psnr),
            'psnr_std': _std(inv_psnr),
            'time_mean_s': _mean(inv_time),
            'time_total_s': float(sum(inv_time)),
        },
    ]

    win_nrmse_sfwi = int(sum(1 for r in per_gt_rows if r['sfwi_nrmse'] < r['inversionnet_nrmse']))
    win_ssim_sfwi = int(sum(1 for r in per_gt_rows if r['sfwi_ssim'] > r['inversionnet_ssim']))
    win_psnr_sfwi = int(sum(1 for r in per_gt_rows if r['sfwi_psnr'] > r['inversionnet_psnr']))

    cmp_row = {
        'method': 'sFWI - InversionNet',
        'n_samples': int(len(per_gt_rows)),
        'nrmse_mean': float(_mean(sfwi_nrmse) - _mean(inv_nrmse)),
        'nrmse_std': float('nan'),
        'ssim_mean': float(_mean(sfwi_ssim) - _mean(inv_ssim)),
        'ssim_std': float('nan'),
        'psnr_mean': float(_mean(sfwi_psnr) - _mean(inv_psnr)),
        'psnr_std': float('nan'),
        'time_mean_s': float(_mean(sfwi_time) - _mean(inv_time)),
        'time_total_s': float(sum(sfwi_time) - sum(inv_time)),
        'sfwi_win_nrmse': int(win_nrmse_sfwi),
        'sfwi_win_ssim': int(win_ssim_sfwi),
        'sfwi_win_psnr': int(win_psnr_sfwi),
    }

    stamp = generate_timestamped_filename('marmousi_sfwi_vs_inversionnet', '').split('_')[-2:]
    stamp = '_'.join(stamp)
    per_gt_path = os.path.join(out_dir, f'quant_per_gt_{stamp}.csv')
    summary_path = os.path.join(out_dir, f'quant_summary_{stamp}.csv')
    meta_path = os.path.join(out_dir, f'quant_meta_{stamp}.json')

    per_gt_headers = [
        'gt_index',
        'sfwi_nrmse', 'sfwi_ssim', 'sfwi_psnr', 'sfwi_time_s', 'sfwi_n_runs',
        'inversionnet_nrmse', 'inversionnet_ssim', 'inversionnet_psnr', 'inversionnet_time_s',
        'delta_nrmse_inv_minus_sfwi', 'delta_ssim_inv_minus_sfwi', 'delta_psnr_inv_minus_sfwi',
    ]
    _write_csv(per_gt_path, per_gt_rows, per_gt_headers)

    summary_headers = [
        'method', 'n_samples',
        'nrmse_mean', 'nrmse_std',
        'ssim_mean', 'ssim_std',
        'psnr_mean', 'psnr_std',
        'time_mean_s', 'time_total_s',
        'sfwi_win_nrmse', 'sfwi_win_ssim', 'sfwi_win_psnr',
    ]
    summary_out = list(summary_rows) + [cmp_row]
    _write_csv(summary_path, summary_out, summary_headers)

    meta = {
        'sfwi_run_csv': csv_paths,
        'inversionnet_ckpt': os.path.abspath(args.inversionnet_ckpt),
        'dataset_path': os.path.abspath(dataset_path),
        'gt_source': gt_source,
        'gt_indices': gt_final,
        'sfwi_prefix': args.sfwi_prefix,
        'sfwi_reduce': args.sfwi_reduce,
        'device': str(device),
        'batch_size': int(batch_size),
        'strict_load': bool(args.strict_load),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print('\nSummary (mean over GT):')
    for r in summary_rows:
        print(f"  {r['method']:<12} "
              f"NRMSE={r['nrmse_mean']:.4f}±{r['nrmse_std']:.4f}, "
              f"SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f}, "
              f"PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f}, "
              f"time={r['time_mean_s']:.3f}s")

    print('\nWin counts (sFWI better):')
    print(f'  NRMSE: {win_nrmse_sfwi}/{len(per_gt_rows)}')
    print(f'  SSIM:  {win_ssim_sfwi}/{len(per_gt_rows)}')
    print(f'  PSNR:  {win_psnr_sfwi}/{len(per_gt_rows)}')

    print('\nSaved:')
    print(f'  per_gt_csv:   {per_gt_path}')
    print(f'  summary_csv:  {summary_path}')
    print(f'  meta_json:    {meta_path}')

    if args.write_latex:
        tex_path = os.path.join(out_dir, f'quant_summary_{stamp}.tex')
        _write_latex_table(tex_path, summary_rows)
        print(f'  latex_table:  {tex_path}')


if __name__ == '__main__':
    main()
