"""
InversionNet 监督训练脚本（seismic -> velocity）。

目标:
  - 使用 Marmousi `seismic_dataset.pkl` 训练 InversionNetSFWI baseline
  - 产出可用于 Figure 3 公平性增强实验的模型权重

默认输入输出:
  - 输入 seismic:  [B, 1, 100, 300]
  - 输出 velocity: [B, 1, 200, 200]

用法:
  # Colab
  %run sFWI/training/inversionnet_train.py -- --epochs 80 --batch_size 16

  # 本地
  python sFWI/training/inversionnet_train.py \
    --data_path /path/to/seismic_dataset.pkl \
    --workdir /path/to/output_root
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable


# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # .../code
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sFWI.config import FWIConfig
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.models.inversionnet import InversionNetSFWI


@dataclass
class NormStats:
    mode: str
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_norm_stats(x: torch.Tensor, mode: str) -> NormStats:
    mode = mode.lower()
    if mode == 'none':
        return NormStats(mode=mode)
    if mode in ('minmax_01', 'minmax_m11'):
        return NormStats(
            mode=mode,
            min=float(x.min().item()),
            max=float(x.max().item()),
        )
    if mode == 'zscore':
        return NormStats(
            mode=mode,
            mean=float(x.mean().item()),
            std=float(x.std().item()),
        )
    raise ValueError(f"未知归一化模式: {mode}")


def _apply_norm(x: torch.Tensor, stats: NormStats) -> torch.Tensor:
    mode = stats.mode
    if mode == 'none':
        return x
    if mode == 'zscore':
        std = torch.tensor(stats.std, dtype=x.dtype, device=x.device).clamp_min(1e-8)
        mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
        return (x - mean) / std
    if mode == 'minmax_01':
        min_v = torch.tensor(stats.min, dtype=x.dtype, device=x.device)
        max_v = torch.tensor(stats.max, dtype=x.dtype, device=x.device)
        den = (max_v - min_v).clamp_min(1e-8)
        return (x - min_v) / den
    if mode == 'minmax_m11':
        min_v = torch.tensor(stats.min, dtype=x.dtype, device=x.device)
        max_v = torch.tensor(stats.max, dtype=x.dtype, device=x.device)
        den = (max_v - min_v).clamp_min(1e-8)
        x01 = (x - min_v) / den
        return x01 * 2.0 - 1.0
    raise ValueError(f"不支持的归一化模式: {mode}")


class SeismicVelocityDataset(Dataset):
    """按给定 index 子集读取 seismic/velocity，并在读取时执行归一化。"""

    def __init__(
        self,
        seismic: torch.Tensor,
        velocity: torch.Tensor,
        indices: Sequence[int],
        seismic_stats: NormStats,
        velocity_stats: NormStats,
    ):
        super().__init__()
        self.seismic = seismic
        self.velocity = velocity
        self.indices = list(indices)
        self.seismic_stats = seismic_stats
        self.velocity_stats = velocity_stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        sx = self.seismic[i].clone().float()    # [C, H, W]
        vy = self.velocity[i].clone().float()   # [C, H, W]
        sx = _apply_norm(sx, self.seismic_stats)
        vy = _apply_norm(vy, self.velocity_stats)
        return sx, vy


class InversionNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(args.device)
        self.workdir = self._resolve_workdir(args.workdir)

        self.checkpoint_dir = os.path.join(self.workdir, 'checkpoints')
        self.meta_ckpt_path = os.path.join(
            self.workdir, 'checkpoints-meta', f'{args.checkpoint_prefix}.pth'
        )
        self.tb_dir = os.path.join(self.workdir, 'tensorboard', 'inversionnet')
        self.stats_json_path = os.path.join(
            self.checkpoint_dir, f'{args.checkpoint_prefix}_norm_stats.json'
        )
        self.history_json_path = os.path.join(
            self.checkpoint_dir, f'{args.checkpoint_prefix}_history.json'
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_ckpt_path), exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.tb_dir)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        self.start_epoch = 0
        self.global_step = 0
        self.best_val = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

        self.seismic_stats: Optional[NormStats] = None
        self.velocity_stats: Optional[NormStats] = None
        self.input_shape = None
        self.output_shape = None

    @staticmethod
    def _resolve_device(device_arg: str) -> torch.device:
        if device_arg and device_arg.lower() != 'auto':
            return torch.device(device_arg)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _resolve_workdir(workdir_arg: Optional[str]) -> str:
        if workdir_arg:
            return workdir_arg
        # Colab默认输出目录
        colab_dir = '/content/drive/MyDrive/score_sde_inverseSolving'
        if os.path.exists('/content') and os.path.isdir('/content'):
            return colab_dir
        # 本地回退
        cfg = FWIConfig()
        return cfg.paths.project_root

    @staticmethod
    def _resolve_data_path(data_path_arg: Optional[str]) -> str:
        if data_path_arg:
            return data_path_arg
        # Colab默认路径
        colab_path = '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'
        if os.path.isfile(colab_path):
            return colab_path
        cfg = FWIConfig()
        return cfg.paths.marmousi_dataset_path

    def _prepare_data(self):
        data_path = self._resolve_data_path(self.args.data_path)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        self.logger.info(f"加载数据: {data_path}")
        velocity, seismic = load_marmousi_from_pkl(data_path)
        if seismic is None:
            raise RuntimeError("数据集中不包含 'seismic' 字段，无法训练 InversionNet。")

        velocity = velocity.float().cpu()
        seismic = seismic.float().cpu()

        # 统一维度
        if seismic.dim() == 3:
            seismic = seismic.unsqueeze(1)
        if velocity.dim() == 3:
            velocity = velocity.unsqueeze(1)

        if seismic.dim() != 4 or velocity.dim() != 4:
            raise ValueError(
                f"输入维度异常: seismic={tuple(seismic.shape)}, velocity={tuple(velocity.shape)}"
            )

        n_samples = min(seismic.shape[0], velocity.shape[0])
        seismic = seismic[:n_samples]
        velocity = velocity[:n_samples]

        # 可选子集（用于快速调试）
        if self.args.max_samples and self.args.max_samples > 0:
            n_samples = min(n_samples, self.args.max_samples)
            seismic = seismic[:n_samples]
            velocity = velocity[:n_samples]
            self.logger.info(f"启用 max_samples={self.args.max_samples}, 实际样本数={n_samples}")

        self.input_shape = tuple(seismic.shape[1:])   # (C, H, W)
        self.output_shape = tuple(velocity.shape[2:])  # (H, W)
        self.logger.info(
            f"数据形状: seismic={tuple(seismic.shape)}, velocity={tuple(velocity.shape)}"
        )

        # 划分 train/val
        n_total = seismic.shape[0]
        n_val = int(round(n_total * self.args.val_ratio))
        n_val = max(1, n_val)
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train

        g = torch.Generator()
        g.manual_seed(self.args.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]

        # 用 train split 统计归一化参数
        seismic_train = seismic[train_idx]
        velocity_train = velocity[train_idx]
        self.seismic_stats = _build_norm_stats(seismic_train, self.args.seismic_norm)
        self.velocity_stats = _build_norm_stats(velocity_train, self.args.velocity_norm)

        train_ds = SeismicVelocityDataset(
            seismic=seismic,
            velocity=velocity,
            indices=train_idx,
            seismic_stats=self.seismic_stats,
            velocity_stats=self.velocity_stats,
        )
        val_ds = SeismicVelocityDataset(
            seismic=seismic,
            velocity=velocity,
            indices=val_idx,
            seismic_stats=self.seismic_stats,
            velocity_stats=self.velocity_stats,
        )

        num_workers = (
            self.args.num_workers
            if self.args.num_workers is not None
            else int(os.environ.get('SFWI_DATALOADER_WORKERS', '0'))
        )
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.logger.info(
            f"样本划分: train={len(train_ds)}, val={len(val_ds)}, workers={num_workers}"
        )
        return train_loader, val_loader

    def _setup_model_and_optim(self):
        self.model = InversionNetSFWI(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            dim1=self.args.dim1,
            dim2=self.args.dim2,
            dim3=self.args.dim3,
            dim4=self.args.dim4,
            dim5=self.args.dim5,
            norm=self.args.norm_type,
            output_crop=self.args.output_crop,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.args.epochs)
        )
        use_amp = torch.cuda.is_available() and (not self.args.no_amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.logger.info(
            f"模型初始化完成: InversionNetSFWI, device={self.device}, amp={use_amp}"
        )

    def _get_loss_fn(self):
        if self.args.loss == 'mse':
            return nn.MSELoss()
        if self.args.loss == 'l1':
            return nn.L1Loss()
        if self.args.loss == 'smoothl1':
            return nn.SmoothL1Loss(beta=1.0)
        raise ValueError(f"未知 loss 类型: {self.args.loss}")

    def _save_norm_stats(self):
        payload = {
            'seismic_stats': asdict(self.seismic_stats),
            'velocity_stats': asdict(self.velocity_stats),
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }
        with open(self.stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_history(self):
        with open(self.history_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _pack_checkpoint(self, epoch: int):
        return {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val': self.best_val,
            'history': self.history,
            'args': vars(self.args),
            'seismic_stats': asdict(self.seismic_stats),
            'velocity_stats': asdict(self.velocity_stats),
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        }

    def _save_checkpoint(self, path: str, epoch: int):
        ckpt = self._pack_checkpoint(epoch)
        torch.save(ckpt, path)

    def _try_resume(self):
        resume_path = None
        if self.args.resume_ckpt:
            resume_path = self.args.resume_ckpt
        elif self.args.auto_resume and os.path.isfile(self.meta_ckpt_path):
            resume_path = self.meta_ckpt_path

        if not resume_path:
            return
        if not os.path.isfile(resume_path):
            self.logger.warning(f"resume checkpoint 不存在: {resume_path}")
            return

        self.logger.info(f"恢复训练: {resume_path}")
        ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = int(ckpt.get('epoch', -1)) + 1
        self.global_step = int(ckpt.get('global_step', 0))
        self.best_val = float(ckpt.get('best_val', float('inf')))
        self.history = ckpt.get('history', self.history)
        self.logger.info(
            f"恢复完成: start_epoch={self.start_epoch}, global_step={self.global_step}, "
            f"best_val={self.best_val:.6f}"
        )

    def train(self):
        train_loader, val_loader = self._prepare_data()
        self._setup_model_and_optim()
        self._try_resume()
        self._save_norm_stats()

        criterion = self._get_loss_fn()
        amp_enabled = self.scaler.is_enabled()

        self.logger.info(
            f"开始训练: epochs={self.args.epochs}, batch_size={self.args.batch_size}, "
            f"lr={self.args.lr}, loss={self.args.loss}"
        )
        self.logger.info(f"归一化: seismic={self.args.seismic_norm}, velocity={self.args.velocity_norm}")

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}", leave=False)
            for batch_idx, (sx, vy) in enumerate(pbar):
                sx = sx.to(self.device, non_blocking=True)
                vy = vy.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred = self.model(sx)
                    loss = criterion(pred, vy)

                self.scaler.scale(loss).backward()
                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bsz = sx.shape[0]
                train_loss_sum += float(loss.item()) * bsz
                train_count += bsz
                self.global_step += 1

                if batch_idx % self.args.log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4e}")
                    self.writer.add_scalar('train/loss_step', float(loss.item()), self.global_step)

            train_loss = train_loss_sum / max(1, train_count)

            # 验证
            self.model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for sx, vy in val_loader:
                    sx = sx.to(self.device, non_blocking=True)
                    vy = vy.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        pred = self.model(sx)
                        loss = criterion(pred, vy)
                    bsz = sx.shape[0]
                    val_loss_sum += float(loss.item()) * bsz
                    val_count += bsz
            val_loss = val_loss_sum / max(1, val_count)

            lr = float(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            self.writer.add_scalar('train/loss_epoch', train_loss, epoch + 1)
            self.writer.add_scalar('val/loss_epoch', val_loss, epoch + 1)
            self.writer.add_scalar('train/lr', lr, epoch + 1)

            self.logger.info(
                f"[epoch {epoch+1:03d}] train_loss={train_loss:.6e} "
                f"val_loss={val_loss:.6e} lr={lr:.3e}"
            )

            # best checkpoint
            if val_loss < self.best_val:
                self.best_val = val_loss
                best_path = os.path.join(
                    self.checkpoint_dir, f'{self.args.checkpoint_prefix}_best.pth'
                )
                self._save_checkpoint(best_path, epoch)
                self.logger.info(f"保存 best checkpoint: {best_path}")

            # periodic checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                epoch_ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f'{self.args.checkpoint_prefix}_epoch_{epoch+1:03d}.pth'
                )
                self._save_checkpoint(epoch_ckpt_path, epoch)
                self.logger.info(f"保存周期 checkpoint: {epoch_ckpt_path}")

            # meta checkpoint for resume
            self._save_checkpoint(self.meta_ckpt_path, epoch)
            self._save_history()

        final_path = os.path.join(
            self.checkpoint_dir, f'{self.args.checkpoint_prefix}_final.pth'
        )
        self._save_checkpoint(final_path, self.args.epochs - 1)
        self._save_history()
        self.writer.close()

        self.logger.info("训练完成。")
        self.logger.info(f"Final checkpoint: {final_path}")
        self.logger.info(f"Best val loss: {self.best_val:.6e}")
        self.logger.info(f"Normalization stats: {self.stats_json_path}")


def build_parser():
    parser = argparse.ArgumentParser(description='InversionNet 训练脚本')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Marmousi seismic_dataset.pkl 路径')
    parser.add_argument('--workdir', type=str, default=None,
                        help='输出目录根路径（包含 checkpoints/ 与 tensorboard/）')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备，例如 cuda:0 / cpu / auto')

    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--loss', type=str, default='smoothl1',
                        choices=['mse', 'l1', 'smoothl1'], help='损失函数')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（0-1）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers，默认读取环境变量 SFWI_DATALOADER_WORKERS 或 0')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='仅使用前 N 个样本（调试用，0 表示不限制）')

    parser.add_argument('--seismic_norm', type=str, default='minmax_m11',
                        choices=['none', 'zscore', 'minmax_01', 'minmax_m11'],
                        help='seismic 归一化模式')
    parser.add_argument('--velocity_norm', type=str, default='minmax_m11',
                        choices=['none', 'zscore', 'minmax_01', 'minmax_m11'],
                        help='velocity 归一化模式')

    parser.add_argument('--dim1', type=int, default=32, help='InversionNet dim1')
    parser.add_argument('--dim2', type=int, default=64, help='InversionNet dim2')
    parser.add_argument('--dim3', type=int, default=128, help='InversionNet dim3')
    parser.add_argument('--dim4', type=int, default=256, help='InversionNet dim4')
    parser.add_argument('--dim5', type=int, default=512, help='InversionNet dim5')
    parser.add_argument('--norm_type', type=str, default='bn',
                        choices=['bn', 'in', 'ln'], help='网络归一化层类型')
    parser.add_argument('--output_crop', type=int, default=5,
                        help='解码末端裁边像素（与官方实现一致）')

    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪阈值，<=0 表示不裁剪')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每 N 个 epoch 保存一次周期 checkpoint')
    parser.add_argument('--log_every', type=int, default=20,
                        help='每 N 个 batch 记录一次 step loss')
    parser.add_argument('--checkpoint_prefix', type=str, default='inversionnet',
                        help='checkpoint 文件名前缀')

    parser.add_argument('--no_amp', action='store_true',
                        help='禁用 AMP 混合精度')
    parser.add_argument('--auto_resume', action='store_true',
                        help='若存在 checkpoints-meta/<prefix>.pth 则自动续训')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='指定恢复训练的 checkpoint 路径（优先级高于 auto_resume）')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio 必须在 (0, 1) 区间。")

    set_seed(args.seed)
    trainer = InversionNetTrainer(args)
    trainer.train()
    print(f"\nInversionNet 训练完成！检查点保存在: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
