"""
Marmousi数据集上的Score SDE训练脚本

用于训练用于OOD泛化实验的预训练模型。

数据来源: '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'

用法:
  # Colab环境
  %run sFWI/training/marmousi_train.py

  # 本地环境（需要修改路径）
  python sFWI/training/marmousi_train.py --data_path <path_to_seismic_dataset.pkl> --workdir <output_directory>
"""
import sys
import os
import time
import logging
import torch
import numpy as np
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
from models import ncsnpp
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE
from models import utils as mutils
from losses import get_optimizer, optimization_manager, get_step_fn
from utils import restore_checkpoint, save_checkpoint
import datasets

# 导入sFWI数据加载器
from sFWI.data.marmousi_loader import load_marmousi_from_pkl


# ================================================================
#  训练器类
# ================================================================

class MarmousiTrainer:
    """Marmousi数据集训练器。"""

    def __init__(self, config, workdir, data_path=None):
        """
        初始化训练器。

        参数:
            config: ml_collections.ConfigDict, 配置对象
            workdir: str, 工作目录（用于保存checkpoint和tensorboard）
            data_path: str, Marmousi pickle数据路径（可选）
        """
        self.config = config
        self.workdir = workdir
        self.data_path = data_path or self._get_default_data_path()

        # 设置目录
        self.sample_dir = os.path.join(workdir, "samples")
        self.tb_dir = os.path.join(workdir, "tensorboard")
        self.checkpoint_dir = os.path.join(workdir, "checkpoints")
        self.checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_meta_dir), exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.tb_dir)

        # 设置SDE和模型
        self.sde = None
        self.score_model = None
        self.optimizer = None
        self.ema = None
        self.state = None
        self.initial_step = 0

        self._setup_sde()
        self._setup_model()

    def _get_default_data_path(self):
        """获取默认数据路径。"""
        # Colab环境
        colab_path = '/content/drive/MyDrive/solving_inverse_in_SGM/dataset/seismic_dataset.pkl'
        if os.path.exists(colab_path):
            return colab_path
        # 本地环境
        local_path = os.path.join(
            os.path.dirname(parent_dir),
            'solving_inverse_in_SGM',
            'dataset',
            'seismic_dataset.pkl'
        )
        return local_path

    def _setup_sde(self):
        """设置VESDE。"""
        if self.config.training.sde.lower() == 'vesde':
            self.sde = VESDE(
                sigma_min=self.config.model.sigma_min,
                sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} not implemented.")

        # 确保离散sigma在正确的设备上
        if hasattr(self.sde, 'discrete_sigmas'):
            self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

        self.logger.info(f"SDE {self.config.training.sde} setup complete.")

    def _setup_model(self):
        """初始化模型、优化器和EMA。"""
        self.score_model = mutils.create_model(self.config)
        self.optimizer = get_optimizer(self.config, self.score_model.parameters())
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(),
            decay=self.config.model.ema_rate
        )
        self.state = dict(
            optimizer=self.optimizer,
            model=self.score_model,
            ema=self.ema,
            step=0
        )

        # 检查并恢复检查点
        if os.path.exists(self.checkpoint_meta_dir):
            try:
                self.logger.info(f"Found checkpoint: {self.checkpoint_meta_dir}")
                self.state = restore_checkpoint(
                    self.checkpoint_meta_dir,
                    self.state,
                    self.config.device
                )
                self.initial_step = int(self.state['step'])
                self.logger.info(f"Restored from step {self.initial_step}")
            except Exception as e:
                self.logger.warning(f"Failed to restore checkpoint: {e}")
                self.initial_step = 0
        else:
            self.logger.info("No checkpoint found, starting from scratch.")

    def create_data_loader(self, velocity_patches, is_training=True):
        """创建PyTorch DataLoader。"""
        # 统一放在CPU，避免DataLoader worker触发CUDA初始化错误
        if isinstance(velocity_patches, torch.Tensor):
            velocity_patches = velocity_patches.detach().to(device='cpu', dtype=torch.float32)
        else:
            velocity_patches = torch.tensor(velocity_patches, dtype=torch.float32)

        # 创建VelocityDataset（使用与预训练脚本相同的格式）
        dataset = self._create_velocity_dataset(velocity_patches)

        # Colab下多进程worker常与CUDA上下文冲突，默认关闭；可通过环境变量覆盖
        num_workers = int(os.environ.get("SFWI_DATALOADER_WORKERS", "0"))
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'shuffle': is_training,
            'num_workers': num_workers,
            'pin_memory': True if torch.cuda.is_available() else False,
            'drop_last': True,
        }

        return DataLoader(dataset, **loader_kwargs)

    def _create_velocity_dataset(self, velocity_patches):
        """创建VelocityDataset实例。"""
        class VelocityDataset(torch.utils.data.Dataset):
            def __init__(self, velocity_patches, image_size=32):
                super().__init__()
                if not isinstance(velocity_patches, torch.Tensor):
                    velocity_patches = torch.tensor(velocity_patches, dtype=torch.float32)

                self.v_mean = velocity_patches.mean()
                self.v_std = velocity_patches.std()

                # 标准化
                velocity = (velocity_patches - self.v_mean) / self.v_std

                # 添加通道维度
                if velocity.dim() == 3:
                    velocity = velocity.unsqueeze(1)

                # 调整尺寸
                if velocity.shape[2] != image_size or velocity.shape[3] != image_size:
                    velocity = torch.nn.functional.interpolate(
                        velocity, size=(image_size, image_size),
                        mode='bilinear', align_corners=False
                    )

                # 保证底层数据驻留CPU，避免worker进程触发CUDA初始化
                self.velocity = velocity.contiguous().cpu()

            def __len__(self):
                return len(self.velocity)

            def __getitem__(self, idx):
                return self.velocity[idx]

        return VelocityDataset(velocity_patches, image_size=self.config.data.image_size)

    def train(self, velocity_patches=None):
        """主训练循环。"""
        self.logger.info(f"Starting training at step {self.initial_step}")
        self.score_model = self.score_model.to(self.config.device)

        # 加载数据
        if velocity_patches is None:
            self.logger.info(f"Loading data from {self.data_path}")
            velocity_patches, _ = load_marmousi_from_pkl(self.data_path)

        train_loader = self.create_data_loader(velocity_patches, is_training=True)
        scaler = datasets.get_data_scaler(self.config)

        # 设置训练步骤函数
        optimize_fn = optimization_manager(self.config)
        train_step_fn = get_step_fn(
            self.sde, train=True,
            optimize_fn=optimize_fn,
            reduce_mean=self.config.training.reduce_mean,
            continuous=self.config.training.continuous,
            likelihood_weighting=self.config.training.likelihood_weighting
        )

        # 训练循环
        train_iter = iter(train_loader)
        n_iters = self.config.training.n_iters

        pbar = range(self.initial_step, n_iters + 1)
        if sys.stdout.isatty():
            pbar = tqdm(pbar, desc=f"Training (loss: N/A)", dynamic_ncols=True)

        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(self.config.device)
            batch = scaler(batch)

            # 训练步骤
            loss = train_step_fn(self.state, batch)

            # 记录日志
            if step % self.config.training.log_freq == 0:
                self.logger.info(f"step: {step}, training_loss: {loss.item():.5e}")
                self.writer.add_scalar("training_loss", loss.item(), step)

            # 定期保存检查点
            if step != 0 and step % self.config.training.snapshot_freq == 0:
                save_step = step // self.config.training.snapshot_freq
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, f'checkpoint_{save_step}.pth'),
                    self.state
                )
                self.logger.info(f"Saved checkpoint at step {step}")

        # 保存最终检查点
        save_checkpoint(
            os.path.join(self.checkpoint_dir, 'checkpoint_final.pth'),
            self.state
        )
        self.writer.close()
        self.logger.info(f"Training complete! Final step: {step}")


# ================================================================
#  CLI入口
# ================================================================

def main():
    """主入口。"""
    import argparse

    parser = argparse.ArgumentParser(description='Marmousi Score SDE训练脚本')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Marmousi pickle数据路径')
    parser.add_argument('--workdir', type=str, default=None,
                        help='工作目录（用于保存checkpoint）')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--n_iters', type=int, default=5001,
                        help='训练迭代次数')
    parser.add_argument('--image_size', type=int, default=32,
                        help='图像尺寸')

    args = parser.parse_args()

    # 获取配置
    config = configs.get_config()
    config.training.batch_size = args.batch_size
    config.training.n_iters = args.n_iters
    config.data.image_size = args.image_size
    target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(config.device, torch.device):
        config.device = target_device
    else:
        config.device = str(target_device)

    # 设置工作目录
    workdir = args.workdir or '/content/drive/MyDrive/score_sde_inverseSolving'
    if not os.path.exists(workdir) and not args.workdir:
        # 本地回退
        workdir = os.path.join(os.path.dirname(parent_dir), 'sFWI_MLST')

    # 创建训练器并开始训练
    trainer = MarmousiTrainer(config, workdir, args.data_path)
    trainer.train()

    print(f"\n训练完成！检查点保存在: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
