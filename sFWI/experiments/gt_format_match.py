"""
Figure GT panel -> Marmousi sample 反查工具。

用途:
  1) 从论文图中读取 GT panel（png/jpg/pdf）
  2) 与 `seismic_dataset.pkl` 的 velocity 样本做结构匹配
  3) 识别最可能的数据索引与空间变换（翻转/旋转/转置）
  4) 产出 CSV/JSON/预览图，辅助确认论文 GT 的“数据格式”

核心匹配思想:
  - 对图像做零均值单位方差标准化后用余弦相似度
  - 同时比较原图与梯度幅值图（更稳健，弱化 colormap 差异）
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # .../code
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.config import FWIConfig
from sFWI.data.marmousi_loader import load_marmousi_from_pkl


@dataclass
class MatchRow:
    panel_id: int
    panel_index: int
    rank: int
    dataset_index: int
    transform: str
    score_combined: float
    score_raw: float
    score_grad: float


def _resolve_dataset_path(path_arg: str) -> str:
    if path_arg:
        return path_arg
    colab_default = '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'
    if os.path.isfile(colab_default):
        return colab_default
    cfg = FWIConfig()
    return cfg.paths.marmousi_dataset_path


def _ensure_float_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        raise ValueError(f'Unsupported image ndim: {arr.ndim}')
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def _trim_white_border(img_gray: np.ndarray, white_thresh: float = 0.985) -> np.ndarray:
    mask = img_gray < white_thresh
    if not mask.any():
        return img_gray
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img_gray[y0:y1, x0:x1]


def _parse_crop(crop: str) -> Tuple[int, int, int, int]:
    vals = [int(v.strip()) for v in crop.split(',')]
    if len(vals) != 4:
        raise ValueError('--crop 格式应为 x0,y0,x1,y1')
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        raise ValueError('crop 坐标非法: 要求 x1>x0 且 y1>y0')
    return x0, y0, x1, y1


def _rasterize_if_pdf(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext != '.pdf':
        return image_path

    out_png = os.path.join('/tmp', f'gt_match_{os.getpid()}.png')
    cmd = ['sips', '-s', 'format', 'png', image_path, '--out', out_png]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or (not os.path.isfile(out_png)):
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f'PDF 栅格化失败（需要 macOS sips）: {image_path}\n{msg}'
        )
    return out_png


def _load_panels(
    image_path: str,
    crop: str,
    auto_trim: bool,
    grid_rows: int,
    grid_cols: int,
    panel_indices: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    raster_path = _rasterize_if_pdf(image_path)
    img = Image.open(raster_path).convert('RGBA')
    arr = np.array(img)
    gray = _ensure_float_gray(arr)

    if crop:
        x0, y0, x1, y1 = _parse_crop(crop)
        gray = gray[y0:y1, x0:x1]

    if auto_trim:
        gray = _trim_white_border(gray)

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError('--grid_rows 和 --grid_cols 必须为正整数')

    H, W = gray.shape
    panels = []
    indices_out = []
    all_cells = grid_rows * grid_cols
    if not panel_indices:
        panel_indices = list(range(all_cells))

    for pidx in panel_indices:
        if pidx < 0 or pidx >= all_cells:
            raise ValueError(f'panel index 越界: {pidx}, grid={grid_rows}x{grid_cols}')
        r = pidx // grid_cols
        c = pidx % grid_cols
        y0 = int(round(r * H / grid_rows))
        y1 = int(round((r + 1) * H / grid_rows))
        x0 = int(round(c * W / grid_cols))
        x1 = int(round((c + 1) * W / grid_cols))
        panel = gray[y0:y1, x0:x1]
        panel = _trim_white_border(panel) if auto_trim else panel
        panels.append(panel)
        indices_out.append(pidx)

    return gray, panels, indices_out


def _expand_panel_paths(panel_paths: Sequence[str]) -> List[str]:
    """
    Expand manual panel paths.
    Supports:
      - explicit file paths
      - directory paths (auto collect png/jpg/jpeg/webp)
      - glob patterns
    """
    exts = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.tif', '*.tiff', '*.bmp')
    out = []
    for p in panel_paths:
        if os.path.isdir(p):
            for ext in exts:
                out.extend(sorted(glob.glob(os.path.join(p, ext))))
            continue
        hit = sorted(glob.glob(p))
        if hit:
            out.extend(hit)
        elif os.path.isfile(p):
            out.append(p)
    # 去重保序
    seen = set()
    dedup = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            dedup.append(ap)
    return dedup


def _load_panels_from_paths(
    panel_paths: Sequence[str],
    auto_trim: bool,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    paths = _expand_panel_paths(panel_paths)
    if not paths:
        raise ValueError('未找到任何 panel 图片，请检查 --panel_paths')

    panels = []
    labels = []
    for p in paths:
        arr = np.array(Image.open(p).convert('RGBA'))
        gray = _ensure_float_gray(arr)
        if auto_trim:
            gray = _trim_white_border(gray)
        panels.append(gray)
        labels.append(os.path.basename(p))

    indices = list(range(len(panels)))
    return panels, indices, labels


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N,1,H,W] or [1,1,H,W]
    return: same shape, each sample zero-mean unit-std
    """
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (x - m) / s


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N,1,H,W]
    return: [N,1,H,W]
    """
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def _apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    x: [N,1,H,W]
    """
    if name == 'identity':
        return x
    if name == 'hflip':
        return torch.flip(x, dims=(-1,))
    if name == 'vflip':
        return torch.flip(x, dims=(-2,))
    if name == 'hvflip':
        return torch.flip(x, dims=(-2, -1))
    if name == 'rot90':
        return torch.rot90(x, k=1, dims=(-2, -1))
    if name == 'rot180':
        return torch.rot90(x, k=2, dims=(-2, -1))
    if name == 'rot270':
        return torch.rot90(x, k=3, dims=(-2, -1))
    if name == 'transpose':
        return x.transpose(-2, -1)
    raise ValueError(f'Unknown transform: {name}')


def _flatten_unit(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N,1,H,W] -> [N,K], each row unit-norm
    """
    flat = x.flatten(start_dim=1)
    n = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-8)
    return flat / n


def _similarity_scores(
    sample_maps: torch.Tensor,   # [N,1,S,S]
    panel_map: torch.Tensor,     # [1,1,S,S]
    transforms: Sequence[str],
    alpha_raw: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    return:
      scores[transform] = {
        'combined': [N],
        'raw': [N],
        'grad': [N],
      }
    """
    panel_n = _normalize_map(panel_map)
    panel_g = _normalize_map(_grad_mag(panel_map))
    panel_raw_vec = _flatten_unit(panel_n)[0]   # [K]
    panel_grad_vec = _flatten_unit(panel_g)[0]  # [K]

    scores = {}
    for tname in transforms:
        x = _apply_transform(sample_maps, tname)
        x_n = _normalize_map(x)
        x_g = _normalize_map(_grad_mag(x))
        raw_vec = _flatten_unit(x_n)   # [N,K]
        grad_vec = _flatten_unit(x_g)  # [N,K]

        raw_score = raw_vec @ panel_raw_vec
        grad_score = grad_vec @ panel_grad_vec
        combined = alpha_raw * raw_score + (1.0 - alpha_raw) * grad_score
        scores[tname] = {
            'combined': combined,
            'raw': raw_score,
            'grad': grad_score,
        }
    return scores


def _topk_global(
    scores_by_tf: Dict[str, Dict[str, torch.Tensor]],
    topk: int,
) -> List[Tuple[str, int, float, float, float]]:
    """
    Return list of tuples:
      (transform, dataset_index, combined, raw, grad)
    """
    all_rows = []
    for tf_name, d in scores_by_tf.items():
        combined = d['combined']
        k = min(topk, combined.numel())
        vals, idx = torch.topk(combined, k=k, largest=True, sorted=True)
        for i in range(k):
            di = int(idx[i].item())
            all_rows.append(
                (
                    tf_name,
                    di,
                    float(vals[i].item()),
                    float(d['raw'][di].item()),
                    float(d['grad'][di].item()),
                )
            )
    all_rows.sort(key=lambda x: x[2], reverse=True)
    return all_rows[:topk]


def _save_preview(
    out_png: str,
    panels: List[np.ndarray],
    panel_indices: List[int],
    velocity: torch.Tensor,  # [N,1,H,W]
    matches: Dict[int, List[MatchRow]],
    panel_labels: Optional[Sequence[str]] = None,
):
    if not HAS_MPL:
        print('[preview] matplotlib 不可用，跳过预览图保存。')
        return

    n_panels = len(panels)
    n_cols = 4  # panel + top3
    fig, axes = plt.subplots(n_panels, n_cols, figsize=(3.6 * n_cols, 3.2 * n_panels))
    if n_panels == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(n_panels):
        panel = panels[r]
        pid = panel_indices[r]
        ax0 = axes[r, 0]
        ax0.imshow(panel, cmap='gray', aspect='auto')
        if panel_labels is not None and r < len(panel_labels):
            ax0.set_title(panel_labels[r])
        else:
            ax0.set_title(f'Panel #{pid}')
        ax0.axis('off')

        top_rows = matches.get(pid, [])[:3]
        for c in range(1, n_cols):
            ax = axes[r, c]
            if c - 1 < len(top_rows):
                row = top_rows[c - 1]
                sample = velocity[row.dataset_index:row.dataset_index + 1]
                sample = _apply_transform(sample, row.transform)[0, 0].cpu().numpy()
                ax.imshow(sample, cmap='seismic', aspect='auto')
                ax.set_title(
                    f'#{row.rank} idx={row.dataset_index}\n{row.transform}\n'
                    f's={row.score_combined:.3f}'
                )
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _save_panel_crops(
    out_dir: str,
    panels: List[np.ndarray],
    panel_indices: List[int],
):
    panel_dir = os.path.join(out_dir, 'panels')
    os.makedirs(panel_dir, exist_ok=True)
    for panel, pidx in zip(panels, panel_indices):
        panel_u8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(panel_u8)
        img.save(os.path.join(panel_dir, f'panel_{pidx:03d}.png'))
    print(f'[save] panel crops -> {panel_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='从论文图反查 Marmousi GT 样本索引与格式（方向/变换）'
    )
    parser.add_argument('--image_path', type=str, default='',
                        help='论文图路径（png/jpg/pdf）')
    parser.add_argument('--panel_paths', type=str, nargs='*', default=None,
                        help='手工提取后的 GT panel 图片路径/目录/glob（提供后优先，忽略 grid 切分）')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='seismic_dataset.pkl 路径，默认自动推断')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录，默认 code/outputs/gt_format_match')

    parser.add_argument('--crop', type=str, default='',
                        help='可选裁剪: x0,y0,x1,y1')
    parser.add_argument('--no_auto_trim', action='store_true',
                        help='关闭白边自动裁切')
    parser.add_argument('--grid_rows', type=int, default=1,
                        help='将（裁剪后）图像按行切分')
    parser.add_argument('--grid_cols', type=int, default=1,
                        help='将（裁剪后）图像按列切分')
    parser.add_argument('--panel_indices', type=int, nargs='*', default=None,
                        help='要匹配的 panel 下标（行优先），默认全部')

    parser.add_argument('--max_samples', type=int, default=0,
                        help='仅使用前 N 个数据样本匹配（0 表示全量）')
    parser.add_argument('--match_size', type=int, default=96,
                        help='匹配时统一缩放尺寸（建议 64~128）')
    parser.add_argument('--topk', type=int, default=10,
                        help='每个 panel 输出前 K 个候选')
    parser.add_argument('--alpha_raw', type=float, default=0.4,
                        help='combined = alpha*raw + (1-alpha)*grad')
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'dataset 不存在: {dataset_path}')

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(parent_dir, 'outputs', 'gt_format_match')
    os.makedirs(out_dir, exist_ok=True)

    panel_labels: List[str] = []
    full_gray = None
    if args.panel_paths:
        panels, panel_indices, panel_labels = _load_panels_from_paths(
            panel_paths=args.panel_paths,
            auto_trim=(not args.no_auto_trim),
        )
        print(f'[info] manual panel mode: {len(panels)} panels')
        for i, name in enumerate(panel_labels):
            print(f'  panel[{i}] <- {name}')
    else:
        if not args.image_path:
            raise ValueError('请提供 --image_path 或 --panel_paths')
        full_gray, panels, panel_indices = _load_panels(
            image_path=args.image_path,
            crop=args.crop,
            auto_trim=(not args.no_auto_trim),
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            panel_indices=args.panel_indices,
        )
        panel_labels = [f'panel_{idx:03d}' for idx in panel_indices]
        print(f'[info] full image shape (after crop/trim): {full_gray.shape}')
        print(f'[info] selected panels: {panel_indices}')
    _save_panel_crops(out_dir, panels, panel_indices)

    velocity, _seismic = load_marmousi_from_pkl(dataset_path)
    velocity = velocity.float().cpu()
    if velocity.dim() == 3:
        velocity = velocity.unsqueeze(1)
    if args.max_samples > 0:
        velocity = velocity[:args.max_samples]
    print(f'[info] candidate velocity set: {tuple(velocity.shape)}')

    S = int(args.match_size)
    vel_resized = F.interpolate(
        velocity, size=(S, S), mode='bilinear', align_corners=False
    )

    transforms = [
        'identity', 'hflip', 'vflip', 'hvflip',
        'rot90', 'rot180', 'rot270', 'transpose',
    ]

    rows_all: List[MatchRow] = []
    matches_by_panel: Dict[int, List[MatchRow]] = {}

    for local_i, pidx in enumerate(panel_indices):
        panel_np = panels[local_i]
        panel_t = torch.from_numpy(panel_np).float()[None, None]  # [1,1,h,w]
        panel_t = F.interpolate(panel_t, size=(S, S), mode='bilinear', align_corners=False)

        scores = _similarity_scores(
            sample_maps=vel_resized,
            panel_map=panel_t,
            transforms=transforms,
            alpha_raw=float(args.alpha_raw),
        )
        top = _topk_global(scores, topk=int(args.topk))

        panel_rows: List[MatchRow] = []
        for rank, (tf_name, d_idx, s_comb, s_raw, s_grad) in enumerate(top, start=1):
            row = MatchRow(
                panel_id=local_i,
                panel_index=pidx,
                rank=rank,
                dataset_index=d_idx,
                transform=tf_name,
                score_combined=s_comb,
                score_raw=s_raw,
                score_grad=s_grad,
            )
            panel_rows.append(row)
            rows_all.append(row)
        matches_by_panel[pidx] = panel_rows

        if panel_rows:
            b = panel_rows[0]
            panel_name = panel_labels[local_i] if local_i < len(panel_labels) else f'panel_{pidx:03d}'
            print(
                f'[panel {pidx} {panel_name}] best idx={b.dataset_index}, tf={b.transform}, '
                f'score={b.score_combined:.4f}'
            )

    # CSV
    csv_path = os.path.join(out_dir, 'gt_format_match_topk.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'panel_id', 'panel_index', 'panel_label', 'rank', 'dataset_index', 'transform',
            'score_combined', 'score_raw', 'score_grad',
        ])
        for r in rows_all:
            panel_name = panel_labels[r.panel_id] if r.panel_id < len(panel_labels) else f'panel_{r.panel_index:03d}'
            writer.writerow([
                r.panel_id, r.panel_index, panel_name, r.rank, r.dataset_index, r.transform,
                f'{r.score_combined:.8f}', f'{r.score_raw:.8f}', f'{r.score_grad:.8f}',
            ])
    print(f'[save] {csv_path}')

    # JSON summary
    panel_label_map = {}
    for i, pidx in enumerate(panel_indices):
        if i < len(panel_labels):
            panel_label_map[pidx] = panel_labels[i]
        else:
            panel_label_map[pidx] = f'panel_{pidx:03d}'

    best_per_panel = {}
    for pidx in panel_indices:
        if not matches_by_panel.get(pidx):
            best_per_panel[str(pidx)] = {}
            continue
        best = matches_by_panel[pidx][0]
        sample = velocity[best.dataset_index]
        best_per_panel[str(pidx)] = {
            'panel_label': panel_label_map.get(pidx, f'panel_{pidx:03d}'),
            'dataset_index': int(best.dataset_index),
            'transform': best.transform,
            'score_combined': float(best.score_combined),
            'score_raw': float(best.score_raw),
            'score_grad': float(best.score_grad),
            'sample_min': float(sample.min().item()),
            'sample_max': float(sample.max().item()),
        }

    summary = {
        'image_path': args.image_path,
        'panel_paths': args.panel_paths if args.panel_paths else [],
        'panel_labels': panel_labels,
        'dataset_path': dataset_path,
        'full_shape_after_crop_trim': list(full_gray.shape) if full_gray is not None else [],
        'grid': {'rows': args.grid_rows, 'cols': args.grid_cols},
        'panel_indices': panel_indices,
        'match_size': S,
        'alpha_raw': float(args.alpha_raw),
        'n_candidates': int(velocity.shape[0]),
        'best_per_panel': best_per_panel,
    }
    json_path = os.path.join(out_dir, 'gt_format_match_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'[save] {json_path}')

    # Preview
    preview_path = os.path.join(out_dir, 'gt_format_match_preview.png')
    _save_preview(
        out_png=preview_path,
        panels=panels,
        panel_indices=panel_indices,
        velocity=velocity,
        matches=matches_by_panel,
        panel_labels=panel_labels,
    )
    if HAS_MPL:
        print(f'[save] {preview_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
