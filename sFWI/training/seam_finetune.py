"""
SEAM数据集上的Score SDE fine-tune脚本

功能:
  1) 从Marmousi预训练checkpoint加载权重
  2) 在SEAM速度patch上进行fine-tune
  3) 以独立命名保存checkpoint，避免覆盖marmousi/旧seam模型

默认checkpoint命名:
  - seam_finetune_checkpoint_{i}.pth
  - seam_finetune_checkpoint_final.pth

用法:
  python sFWI/training/seam_finetune.py \
    --workdir /content/drive/MyDrive/score_sde_inverseSolving \
    --pretrain_ckpt /content/drive/MyDrive/score_sde_inverseSolving/checkpoints/marmousi_checkpoint_5.pth
"""
import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 设置score_sde路径
from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

# 导入score_sde模块
from configs.ve import marmousi_ncsnpp_continuous as configs
from models import ncsnpp  # noqa: F401  # 触发模型注册
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE
from models import utils as mutils
from losses import get_optimizer, optimization_manager, get_step_fn
from utils import restore_checkpoint, save_checkpoint
import datasets

# 导入sFWI模块
from sFWI.config import FWIConfig
from sFWI.data.loaders import load_seam_model


class SEAMFineTuneTrainer:
    """SEAM数据集fine-tune训练器。"""

    def __init__(
        self,
        config,
        workdir,
        seam_model_path=None,
        pretrain_ckpt=None,
        checkpoint_prefix='seam_finetune_checkpoint',
    ):
        self.config = config
        self.workdir = workdir
        self.seam_model_path = seam_model_path or self._get_default_seam_model_path()
        self.pretrain_ckpt = pretrain_ckpt or self._get_default_pretrain_ckpt()
        self.checkpoint_prefix = checkpoint_prefix

        self.sample_dir = os.path.join(workdir, "samples")
        self.tb_dir = os.path.join(workdir, "tensorboard")
        self.checkpoint_dir = os.path.join(workdir, "checkpoints")
        self.checkpoint_meta_path = os.path.join(
            workdir, "checkpoints-meta", f"{self.checkpoint_prefix}.pth"
        )

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_meta_path), exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.tb_dir)

        self.sde = None
        self.score_model = None
        self.optimizer = None
        self.ema = None
        self.state = None
        self.initial_step = 0

        self._setup_sde()
        self._setup_model()

    def _get_default_seam_model_path(self):
        cfg = FWIConfig()
        return cfg.paths.seam_model_path

    def _get_default_pretrain_ckpt(self):
        cfg = FWIConfig()
        return cfg.paths.marmousi_checkpoint_path

    def _setup_sde(self):
        if self.config.training.sde.lower() == 'vesde':
            self.sde = VESDE(
                sigma_min=self.config.model.sigma_min,
                sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} not implemented.")

        if hasattr(self.sde, 'discrete_sigmas'):
            self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

        self.logger.info(f"SDE {self.config.training.sde} setup complete.")

    def _load_pretrained_weights(self):
        if not self.pretrain_ckpt:
            self.logger.warning("未提供pretrain checkpoint，将从随机初始化开始。")
            return

        if not os.path.isfile(self.pretrain_ckpt):
            self.logger.warning(f"pretrain checkpoint不存在: {self.pretrain_ckpt}，将从随机初始化开始。")
            return

        self.logger.info(f"Loading pretrain checkpoint: {self.pretrain_ckpt}")
        loaded_state = torch.load(
            self.pretrain_ckpt, map_location=self.config.device, weights_only=False
        )
        model_state = loaded_state.get('model', loaded_state)
        incompatible = self.score_model.load_state_dict(model_state, strict=False)
        missing = len(incompatible.missing_keys)
        unexpected = len(incompatible.unexpected_keys)
        self.logger.info(f"Pretrain weights loaded (missing={missing}, unexpected={unexpected})")

        if 'ema' in loaded_state:
            try:
                self.ema.load_state_dict(loaded_state['ema'])
                self.ema.copy_to(self.score_model.parameters())
                self.logger.info("EMA state loaded from pretrain checkpoint.")
            except Exception as exc:
                self.logger.warning(f"EMA加载失败，继续训练: {exc}")

    def _setup_model(self):
        self.score_model = mutils.create_model(self.config)
        self.optimizer = get_optimizer(self.config, self.score_model.parameters())
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=self.config.model.ema_rate)
        self.state = dict(
            optimizer=self.optimizer,
            model=self.score_model,
            ema=self.ema,
            step=0
        )

        if os.path.exists(self.checkpoint_meta_path):
            try:
                self.logger.info(f"Found fine-tune checkpoint: {self.checkpoint_meta_path}")
                self.state = restore_checkpoint(
                    self.checkpoint_meta_path,
                    self.state,
                    self.config.device
                )
                self.initial_step = int(self.state['step'])
                self.logger.info(f"Resumed fine-tune from step {self.initial_step}")
                return
            except Exception as exc:
                self.logger.warning(f"恢复fine-tune checkpoint失败: {exc}")
                self.initial_step = 0

        self._load_pretrained_weights()
        self.initial_step = 0

    def _create_velocity_dataset(self, velocity_patches):
        class VelocityDataset(torch.utils.data.Dataset):
            def __init__(self, velocity_patches, image_size=32):
                super().__init__()
                if not isinstance(velocity_patches, torch.Tensor):
                    velocity_patches = torch.tensor(velocity_patches, dtype=torch.float32)

                self.v_mean = velocity_patches.mean()
                self.v_std = velocity_patches.std()
                velocity = (velocity_patches - self.v_mean) / self.v_std

                if velocity.dim() == 3:
                    velocity = velocity.unsqueeze(1)

                if velocity.shape[2] != image_size or velocity.shape[3] != image_size:
                    velocity = torch.nn.functional.interpolate(
                        velocity, size=(image_size, image_size),
                        mode='bilinear', align_corners=False
                    )

                self.velocity = velocity.contiguous().cpu()

            def __len__(self):
                return len(self.velocity)

            def __getitem__(self, idx):
                return self.velocity[idx]

        return VelocityDataset(velocity_patches, image_size=self.config.data.image_size)

    def create_data_loader(self, velocity_patches):
        if isinstance(velocity_patches, torch.Tensor):
            velocity_patches = velocity_patches.detach().to(device='cpu', dtype=torch.float32)
        else:
            velocity_patches = torch.tensor(velocity_patches, dtype=torch.float32)

        dataset = self._create_velocity_dataset(velocity_patches)

        num_workers = int(os.environ.get("SFWI_DATALOADER_WORKERS", "0"))
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': True if torch.cuda.is_available() else False,
            'drop_last': True,
        }
        return DataLoader(dataset, **loader_kwargs)

    def train(self, velocity_patches=None):
        self.logger.info(f"Starting SEAM fine-tune at step {self.initial_step}")
        self.score_model = self.score_model.to(self.config.device)

        if velocity_patches is None:
            self.logger.info(f"Loading SEAM data from {self.seam_model_path}")
            velocity_patches = load_seam_model(self.seam_model_path)

        train_loader = self.create_data_loader(velocity_patches)
        scaler = datasets.get_data_scaler(self.config)

        optimize_fn = optimization_manager(self.config)
        train_step_fn = get_step_fn(
            self.sde, train=True,
            optimize_fn=optimize_fn,
            reduce_mean=self.config.training.reduce_mean,
            continuous=self.config.training.continuous,
            likelihood_weighting=self.config.training.likelihood_weighting
        )

        train_iter = iter(train_loader)
        n_iters = self.config.training.n_iters

        pbar = range(self.initial_step, n_iters + 1)
        if sys.stdout.isatty():
            pbar = tqdm(pbar, desc="SEAM Fine-tune (loss: N/A)", dynamic_ncols=True)

        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(self.config.device)
            batch = scaler(batch)

            loss = train_step_fn(self.state, batch)

            if step % self.config.training.log_freq == 0:
                self.logger.info(f"step: {step}, training_loss: {loss.item():.5e}")
                self.writer.add_scalar("training_loss", loss.item(), step)

            if step != 0 and step % self.config.training.snapshot_freq == 0:
                save_step = step // self.config.training.snapshot_freq
                ckpt_name = f'{self.checkpoint_prefix}_{save_step}.pth'
                save_checkpoint(os.path.join(self.checkpoint_dir, ckpt_name), self.state)
                save_checkpoint(self.checkpoint_meta_path, self.state)
                self.logger.info(f"Saved checkpoint at step {step}: {ckpt_name}")

        final_name = f'{self.checkpoint_prefix}_final.pth'
        save_checkpoint(os.path.join(self.checkpoint_dir, final_name), self.state)
        save_checkpoint(self.checkpoint_meta_path, self.state)
        self.writer.close()
        self.logger.info(f"SEAM fine-tune complete! Final step: {step}")
        self.logger.info(f"Final checkpoint: {os.path.join(self.checkpoint_dir, final_name)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SEAM Score SDE fine-tune脚本')
    parser.add_argument('--workdir', type=str, default=None,
                        help='工作目录（用于保存checkpoint和tensorboard）')
    parser.add_argument('--seam_model_path', type=str, default=None,
                        help='SEAM速度模型路径（.sgy）')
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                        help='Marmousi预训练checkpoint路径')
    parser.add_argument('--checkpoint_prefix', type=str, default='seam_finetune_checkpoint',
                        help='checkpoint文件名前缀')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--n_iters', type=int, default=5001,
                        help='训练迭代次数')
    parser.add_argument('--image_size', type=int, default=32,
                        help='训练图像尺寸')

    args = parser.parse_args()

    config = configs.get_config()
    config.training.batch_size = args.batch_size
    config.training.n_iters = args.n_iters
    config.data.image_size = args.image_size
    target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(config.device, torch.device):
        config.device = target_device
    else:
        config.device = str(target_device)

    cfg_runtime = FWIConfig()
    workdir = args.workdir or '/content/drive/MyDrive/score_sde_inverseSolving'
    if not os.path.exists(workdir) and not args.workdir:
        workdir = cfg_runtime.paths.project_root

    trainer = SEAMFineTuneTrainer(
        config=config,
        workdir=workdir,
        seam_model_path=args.seam_model_path,
        pretrain_ckpt=args.pretrain_ckpt,
        checkpoint_prefix=args.checkpoint_prefix,
    )
    trainer.train()
    print(f"\nSEAM fine-tune完成！检查点保存在: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
