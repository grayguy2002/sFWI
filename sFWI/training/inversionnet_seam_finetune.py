"""
SEAM 数据上的 InversionNet 微调脚本（从 Marmousi 预训练继续）。

功能:
  1) 从 Marmousi 训练好的 InversionNet checkpoint 加载权重
  2) 读取 SEAM 速度 patch，并通过 DAPS 正演算子生成 synthetic seismic
  3) 以监督方式微调 InversionNet (seismic -> velocity)
  4) 使用独立 checkpoint 前缀，避免覆盖 Marmousi 预训练权重

默认命名:
  - checkpoints/<prefix>_best.pth
  - checkpoints/<prefix>_final.pth
  - checkpoints-meta/<prefix>.pth

示例:
  python sFWI/training/inversionnet_seam_finetune.py \
    --workdir /content/drive/MyDrive/score_sde_inverseSolving \
    --pretrain_ckpt /content/drive/MyDrive/score_sde_inverseSolving/checkpoints/inversionnet_best.pth
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
import torch.nn.functional as F
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

from sFWI.models.sde_setup import setup_score_sde_path, create_sde_config
setup_score_sde_path(parent_dir)

from sFWI.config import FWIConfig
from sFWI.data.loaders import load_seam_model
from sFWI.models.inversionnet import InversionNetSFWI, load_inversionnet_state_dict
from sFWI.operators.daps_operator import DAPSSeismicOperator


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


def _parse_hw_token(hw: str) -> Optional[Tuple[int, int]]:
    if not hw:
        return None
    vals = [int(v.strip()) for v in hw.split(',')]
    if len(vals) != 2:
        raise ValueError("--seismic_resize_hw 格式应为 H,W")
    h, w = vals
    if h < 1 or w < 1:
        raise ValueError("--seismic_resize_hw 必须为正整数")
    return h, w


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


class InversionNetSEAMFineTuneTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(args.device)
        self.workdir = self._resolve_workdir(args.workdir)

        self.checkpoint_dir = os.path.join(self.workdir, 'checkpoints')
        self.meta_ckpt_path = os.path.join(
            self.workdir, 'checkpoints-meta', f'{args.checkpoint_prefix}.pth'
        )
        self.tb_dir = os.path.join(self.workdir, 'tensorboard', 'inversionnet_seam_finetune')
        self.cache_path = self._resolve_cache_path(args.cache_path)
        self.stats_json_path = os.path.join(
            self.checkpoint_dir, f'{args.checkpoint_prefix}_norm_stats.json'
        )
        self.history_json_path = os.path.join(
            self.checkpoint_dir, f'{args.checkpoint_prefix}_history.json'
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_ckpt_path), exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

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
        self.seam_model_path = self._resolve_seam_model_path(args.seam_model_path)
        self.pretrain_ckpt = self._resolve_pretrain_ckpt(args.pretrain_ckpt)

    @staticmethod
    def _resolve_device(device_arg: str) -> torch.device:
        if device_arg and device_arg.lower() != 'auto':
            return torch.device(device_arg)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _resolve_workdir(workdir_arg: Optional[str]) -> str:
        if workdir_arg:
            return workdir_arg
        colab_dir = '/content/drive/MyDrive/score_sde_inverseSolving'
        if os.path.exists('/content') and os.path.isdir('/content'):
            return colab_dir
        cfg = FWIConfig()
        return cfg.paths.project_root

    def _resolve_cache_path(self, cache_path_arg: Optional[str]) -> str:
        if cache_path_arg:
            return cache_path_arg
        return os.path.join(self.workdir, 'cache', 'seam_inversionnet_pairs.pt')

    @staticmethod
    def _resolve_seam_model_path(path_arg: Optional[str]) -> str:
        if path_arg:
            return path_arg
        cfg = FWIConfig()
        return cfg.paths.seam_model_path

    def _resolve_pretrain_ckpt(self, ckpt_arg: Optional[str]) -> Optional[str]:
        if ckpt_arg:
            return ckpt_arg

        candidates = [
            os.path.join(self.workdir, 'checkpoints', 'inversionnet_best.pth'),
            os.path.join(self.workdir, 'checkpoints', 'inversionnet_final.pth'),
        ]
        cfg = FWIConfig()
        candidates.extend([
            os.path.join(cfg.paths.project_root, 'checkpoints', 'inversionnet_best.pth'),
            os.path.join(cfg.paths.project_root, 'checkpoints', 'inversionnet_final.pth'),
        ])

        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _simulate_seismic(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        由 SEAM velocity 生成 synthetic seismic。

        输入:
          velocity: [N, 1, 200, 200]
        输出:
          seismic:  [N, 1, H, W]，默认 H=W=128（来自 DAPS operator）
        """
        cfg, _ = create_sde_config(parent_dir, batch_size=1)
        operator = DAPSSeismicOperator(
            cfg,
            image_size=200,
            sigma=float(self.args.operator_sigma),
        )
        resize_hw = _parse_hw_token(self.args.seismic_resize_hw)

        n = int(velocity.shape[0])
        batch_size = int(self.args.forward_batch_size)
        out = []
        self.logger.info(
            f"开始正演生成 SEAM seismic: n={n}, forward_batch_size={batch_size}, sigma={self.args.operator_sigma}"
        )
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            v_chunk = velocity[start:end].to(self.device)
            with torch.no_grad():
                s_chunk = operator(v_chunk).detach().cpu().float()
            if resize_hw is not None and tuple(s_chunk.shape[-2:]) != tuple(resize_hw):
                s_chunk = F.interpolate(
                    s_chunk,
                    size=resize_hw,
                    mode='bilinear',
                    align_corners=False,
                )
            out.append(s_chunk)
            if (start // batch_size) % 5 == 0 or end == n:
                self.logger.info(f"  seismic progress: {end}/{n}")

        seismic = torch.cat(out, dim=0)
        return seismic

    def _load_or_build_pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if os.path.isfile(self.cache_path) and (not self.args.regenerate_cache):
            self.logger.info(f"加载缓存样本对: {self.cache_path}")
            blob = torch.load(self.cache_path, map_location='cpu', weights_only=False)
            if not isinstance(blob, dict) or ('velocity' not in blob) or ('seismic' not in blob):
                raise RuntimeError(f"缓存格式非法: {self.cache_path}")
            velocity = blob['velocity'].float().cpu()
            seismic = blob['seismic'].float().cpu()
            self.logger.info(
                f"缓存读取完成: velocity={tuple(velocity.shape)}, seismic={tuple(seismic.shape)}"
            )
            return velocity, seismic

        if not os.path.isfile(self.seam_model_path):
            raise FileNotFoundError(f"SEAM 模型不存在: {self.seam_model_path}")

        self.logger.info(f"加载 SEAM velocity: {self.seam_model_path}")
        velocity = load_seam_model(self.seam_model_path).float().cpu()  # [N,H,W]
        if velocity.dim() == 3:
            velocity = velocity.unsqueeze(1)  # [N,1,H,W]
        if velocity.dim() != 4:
            raise ValueError(f"SEAM velocity 维度异常: {tuple(velocity.shape)}")

        seismic = self._simulate_seismic(velocity)
        if seismic.dim() != 4:
            raise ValueError(f"生成 seismic 维度异常: {tuple(seismic.shape)}")
        if seismic.shape[0] != velocity.shape[0]:
            raise ValueError(
                f"样本数不一致: velocity={velocity.shape[0]}, seismic={seismic.shape[0]}"
            )

        payload = {
            'velocity': velocity.cpu(),
            'seismic': seismic.cpu(),
            'meta': {
                'seam_model_path': self.seam_model_path,
                'operator_sigma': float(self.args.operator_sigma),
                'seismic_resize_hw': self.args.seismic_resize_hw,
                'forward_batch_size': int(self.args.forward_batch_size),
            },
        }
        torch.save(payload, self.cache_path)
        self.logger.info(f"已保存缓存样本对: {self.cache_path}")
        return velocity, seismic

    def _prepare_data(self):
        velocity, seismic = self._load_or_build_pairs()

        if self.args.max_samples and self.args.max_samples > 0:
            n_samples = min(int(self.args.max_samples), int(velocity.shape[0]))
            velocity = velocity[:n_samples]
            seismic = seismic[:n_samples]
            self.logger.info(f"启用 max_samples={self.args.max_samples}, 实际样本数={n_samples}")

        if velocity.shape[0] < 2:
            raise RuntimeError("样本数量过少，至少需要 2 个样本。")

        self.input_shape = tuple(seismic.shape[1:])    # (C, H, W)
        self.output_shape = tuple(velocity.shape[2:])  # (H, W)
        self.logger.info(
            f"数据形状: seismic={tuple(seismic.shape)}, velocity={tuple(velocity.shape)}"
        )

        n_total = int(seismic.shape[0])
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

    def _save_norm_stats(self):
        payload = {
            'seismic_stats': asdict(self.seismic_stats),
            'velocity_stats': asdict(self.velocity_stats),
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'seam_model_path': self.seam_model_path,
            'cache_path': self.cache_path,
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
            'seam_model_path': self.seam_model_path,
            'cache_path': self.cache_path,
            'pretrain_ckpt': self.pretrain_ckpt,
        }

    def _save_checkpoint(self, path: str, epoch: int):
        ckpt = self._pack_checkpoint(epoch)
        torch.save(ckpt, path)

    def _try_resume(self) -> bool:
        resume_path = None
        if self.args.resume_ckpt:
            resume_path = self.args.resume_ckpt
        elif self.args.auto_resume and os.path.isfile(self.meta_ckpt_path):
            resume_path = self.meta_ckpt_path

        if not resume_path:
            return False
        if not os.path.isfile(resume_path):
            self.logger.warning(f"resume checkpoint 不存在: {resume_path}")
            return False

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
        return True

    def _load_pretrain(self):
        if not self.pretrain_ckpt:
            self.logger.warning("未检测到 pretrain_ckpt，将从随机初始化开始。")
            return
        if not os.path.isfile(self.pretrain_ckpt):
            self.logger.warning(f"pretrain_ckpt 不存在: {self.pretrain_ckpt}，将从随机初始化开始。")
            return

        self.logger.info(f"加载预训练权重: {self.pretrain_ckpt}")
        ckpt = torch.load(self.pretrain_ckpt, map_location=self.device, weights_only=False)
        incompatible = load_inversionnet_state_dict(
            self.model,
            ckpt,
            strict=bool(self.args.strict_pretrain),
        )
        self.logger.info(
            f"预训练权重加载完成: missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}, strict={self.args.strict_pretrain}"
        )

    def _get_loss_fn(self):
        if self.args.loss == 'mse':
            return nn.MSELoss()
        if self.args.loss == 'l1':
            return nn.L1Loss()
        if self.args.loss == 'smoothl1':
            return nn.SmoothL1Loss(beta=1.0)
        raise ValueError(f"未知 loss 类型: {self.args.loss}")

    def train(self):
        train_loader, val_loader = self._prepare_data()
        self._setup_model_and_optim()
        resumed = self._try_resume()
        if not resumed:
            self._load_pretrain()
        self._save_norm_stats()

        criterion = self._get_loss_fn()
        amp_enabled = self.scaler.is_enabled()

        self.logger.info(
            f"开始微调: epochs={self.args.epochs}, batch_size={self.args.batch_size}, "
            f"lr={self.args.lr}, loss={self.args.loss}"
        )
        self.logger.info(
            f"归一化: seismic={self.args.seismic_norm}, velocity={self.args.velocity_norm}"
        )

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

            if val_loss < self.best_val:
                self.best_val = val_loss
                best_path = os.path.join(
                    self.checkpoint_dir, f'{self.args.checkpoint_prefix}_best.pth'
                )
                self._save_checkpoint(best_path, epoch)
                self.logger.info(f"保存 best checkpoint: {best_path}")

            if (epoch + 1) % self.args.save_every == 0:
                epoch_ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f'{self.args.checkpoint_prefix}_epoch_{epoch+1:03d}.pth'
                )
                self._save_checkpoint(epoch_ckpt_path, epoch)
                self.logger.info(f"保存周期 checkpoint: {epoch_ckpt_path}")

            self._save_checkpoint(self.meta_ckpt_path, epoch)
            self._save_history()

        final_path = os.path.join(
            self.checkpoint_dir, f'{self.args.checkpoint_prefix}_final.pth'
        )
        self._save_checkpoint(final_path, self.args.epochs - 1)
        self._save_history()
        self.writer.close()

        self.logger.info("SEAM 微调完成。")
        self.logger.info(f"Final checkpoint: {final_path}")
        self.logger.info(f"Best val loss: {self.best_val:.6e}")
        self.logger.info(f"Normalization stats: {self.stats_json_path}")


def build_parser():
    parser = argparse.ArgumentParser(description='InversionNet 在 SEAM 上的微调脚本')
    parser.add_argument('--workdir', type=str, default=None,
                        help='输出目录根路径（包含 checkpoints/ 与 tensorboard/）')
    parser.add_argument('--seam_model_path', type=str, default=None,
                        help='SEAM 速度模型 .sgy 路径')
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                        help='Marmousi 预训练 InversionNet 权重路径')
    parser.add_argument('--checkpoint_prefix', type=str,
                        default='inversionnet_seam_finetune_checkpoint',
                        help='checkpoint 文件名前缀')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备，例如 cuda:0 / cpu / auto')

    parser.add_argument('--cache_path', type=str, default=None,
                        help='SEAM synthetic 样本对缓存路径 (.pt)')
    parser.add_argument('--regenerate_cache', action='store_true',
                        help='强制重建 SEAM synthetic 样本对缓存')
    parser.add_argument('--operator_sigma', type=float, default=0.3,
                        help='生成 seismic 时 DAPS operator 的 sigma')
    parser.add_argument('--forward_batch_size', type=int, default=8,
                        help='正演生成 seismic 时的批大小')
    parser.add_argument('--seismic_resize_hw', type=str, default='100,300',
                        help='对 synthetic seismic 重采样到 H,W（默认 100,300）')

    parser.add_argument('--epochs', type=int, default=40, help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='训练批大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--loss', type=str, default='smoothl1',
                        choices=['mse', 'l1', 'smoothl1'], help='损失函数')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（0-1）')
    parser.add_argument('--seed', type=int, default=8, help='随机种子')
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
    parser.add_argument('--strict_pretrain', action='store_true',
                        help='严格加载预训练权重（默认非严格）')

    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪阈值，<=0 表示不裁剪')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每 N 个 epoch 保存一次周期 checkpoint')
    parser.add_argument('--log_every', type=int, default=20,
                        help='每 N 个 batch 记录一次 step loss')

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
    if args.forward_batch_size < 1:
        raise ValueError("--forward_batch_size 必须 >= 1。")

    set_seed(args.seed)
    trainer = InversionNetSEAMFineTuneTrainer(args)
    trainer.train()
    print(f"\nInversionNet SEAM 微调完成！检查点保存在: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
