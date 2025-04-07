main_path = 'C:/Users/盖/Desktop/sFWI_repository/'
import os
import sys
from setup_environment import base_path, f_path, install_dependencies

# Verify paths
print(f"Base path: {base_path}")
print(f"Working directory: {f_path}")

# Run specific logic for script1
print("Running script1...")


## prepare dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pickle
import random
from torchvision import transforms

#@title load pickle dataset##################################################################################
class SeismicPatchDataset(Dataset):
    def __init__(self, velocity_patches, seismic_patches, source_locations, receiver_locations):
        # Remove the torch.stack since inputs are already tensors
        self.velocity_models = velocity_patches
        self.seismic_data = seismic_patches
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

        # Normalize velocity and seismic data
        self.v_mean = self.velocity_models.mean()
        self.v_std = self.velocity_models.std()
        self.s_mean = self.seismic_data.mean()
        self.s_std = self.seismic_data.std()

        self.velocity_models = (self.velocity_models - self.v_mean) / self.v_std
        self.seismic_data = (self.seismic_data - self.s_mean) / self.s_std

    def __len__(self):
        return len(self.velocity_models)

    def __getitem__(self, idx):
        return {
            'velocity': self.velocity_models[idx],
            'seismic': self.seismic_data[idx],
            'source_locations': self.source_locations[idx],
            'receiver_locations': self.receiver_locations[idx]
        }

    def normalize_velocity(self, velocity):
        """Denormalize velocity data"""
        self.velocity_models_n = (velocity - self.v_mean) / self.v_std
        return self.velocity_models_n


class Rotate90Transform:
    """随机将图像旋转 N*90 度的变换类"""
    def __init__(self, p=0.5):
        """
        参数:
            p (float): 执行旋转的概率，默认为 0.5
        """
        self.p = p

    def __call__(self, x):
        """
        参数:
            x (Tensor): 形状为 (C, H, W) 的输入张量
        返回:
            Tensor: 旋转后的张量
        """
        if random.random() < self.p:
            k = random.randint(0, 3)
            return torch.rot90(x, k=k, dims=(1, 2))
        return x

class VelocityScoreSDEDataset(Dataset):
    def __init__(self, velocity_patches, image_size=32, transform=None, device='cpu'):
        super().__init__()

        # 确保输入是 PyTorch 张量
        if not isinstance(velocity_patches, torch.Tensor):
            velocity_patches = torch.tensor(velocity_patches)

        # 将数据移到指定设备并转换为 float32
        self.velocity_models = velocity_patches.to(device, dtype=torch.float32)

        # 设置变换
        self.transform = transform
        self.image_size = image_size
        self.device = device

        # 计算并存储统计信息
        with torch.no_grad():
            self.v_mean = self.velocity_models.mean()
            self.v_std = self.velocity_models.std()

            # 预处理数据
            self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据：标准化、添加通道维度（如果需要）、调整大小"""
        with torch.no_grad():
            try:
                # 标准化
                self.velocity_models = (self.velocity_models - self.v_mean) / self.v_std

                # 添加通道维度（如果需要）
                if self.velocity_models.dim() == 3:  # (N, H, W)
                    self.velocity_models = self.velocity_models.unsqueeze(1)  # (N, 1, H, W)

                # 如果需要调整大小
                if (self.velocity_models.shape[2] != self.image_size or
                    self.velocity_models.shape[3] != self.image_size):
                    self.velocity_models = torch.nn.functional.interpolate(
                        self.velocity_models,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    )

                # 将处理后的数据移到 CPU，以防止 CUDA 内存问题
                self.velocity_models = self.velocity_models.cpu()

            except RuntimeError as e:
                print(f"数据预处理错误: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 尝试分批处理
                    self._batch_preprocess_data()
                else:
                    raise e

    def _batch_preprocess_data(self, batch_size=32):
        """分批处理大型数据集"""
        with torch.no_grad():
            n_samples = len(self.velocity_models)
            processed_data = []

            for i in range(0, n_samples, batch_size):
                batch = self.velocity_models[i:i+batch_size]

                # 标准化
                batch = (batch - self.v_mean) / self.v_std

                # 添加通道维度（如果需要）
                if batch.dim() == 3:
                    batch = batch.unsqueeze(1)

                # 调整大小
                if (batch.shape[2] != self.image_size or
                    batch.shape[3] != self.image_size):
                    batch = torch.nn.functional.interpolate(
                        batch,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    )

                processed_data.append(batch.cpu())

                # 清理 GPU 内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 合并所有批次
            self.velocity_models = torch.cat(processed_data, dim=0)

    def __getitem__(self, idx):
        try:
            # 从 CPU 内存加载数据
            velocity = self.velocity_models[idx].clone()

            if self.transform is not None:
                velocity = self.transform(velocity)

            # 如果需要，将数据移到指定设备
            if self.device != 'cpu':
                velocity = velocity.to(self.device)

            return velocity

        except Exception as e:
            print(f"加载数据项 {idx} 时出错: {str(e)}")
            raise e

    def __len__(self):
        return len(self.velocity_models)

    def denormalize_velocity(self, velocity):
        """反标准化速度数据"""
        with torch.no_grad():
            return velocity * self.v_std + self.v_mean

    def to(self, device):
        """将数据集移动到指定设备"""
        self.device = device
        self.v_mean = self.v_mean.to(device)
        self.v_std = self.v_std.to(device)
        return self

    def cpu(self):
        """将数据集移动到 CPU"""
        return self.to('cpu')

    def cuda(self, device=None):
        """将数据集移动到 CUDA 设备"""
        if device is None:
            device = torch.cuda.current_device()
        return self.to(f'cuda:{device}')



# Define transforms
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # 将单通道转换为三通道
])

# Load data
save_path = main_path+'dataset/seismic_dataset-001.pkl'
with open(save_path, 'rb') as f:
    dataset_origin = pickle.load(f)


# Stack velocity data
velocity = []
seismic = []
for i in range(len(dataset_origin)):
    velocity.append(dataset_origin[i]['velocity'])
    seismic.append(dataset_origin[i]['seismic'])
v_torch = torch.stack(velocity) #(10_000,200,200)
s_torch = torch.stack(seismic) #(10_00,1,100,300)

#@title  autoload all modules##################################################################################
import sys
import os
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_gan as tfgan
import tqdm
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set_theme(font_scale=2)
sns.set_theme(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets

# Set up paths
base_path = os.path.abspath("./score_sde_pytorch")  # Path to the score_sde_pytorch folder
sys.path.append(base_path)  # Add score_sde_pytorch to Python path

# Verify paths
print(f"Score SDE PyTorch path: {base_path}")

# Initialize seaborn settings
sns.set_theme(font_scale=2)
sns.set_theme(style="whitegrid")

print("Environment setup completed!")

#@title Load the score-based##################################################################################
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import cifar10_ncsnpp_continuous as configs
  my_file_path = main_path
  ckpt_filename = my_file_path+'checkpoints/32_checkpoint_5_cifar10_ncsnpp_continuous.pth'
  config = configs.get_config()
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sde.discrete_sigmas = sde.discrete_sigmas.to(config.device)
  sampling_eps = 1e-5


batch_size =   64#@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

#@title DAPS Repository##################################################################################
# 设置当前工作目录为本地的 SFWI_REPOSITORY 文件夹
import os

# 替换为您的本地 SFWI_REPOSITORY 文件夹路径
local_repo_path = "C:/Users/盖/Desktop/sFWI_repository"

# 切换到本地 SFWI_REPOSITORY 文件夹
os.chdir(local_repo_path)
print(f"Current working directory: {os.getcwd()}")

# 确保 DAPS 文件夹存在
daps_path = os.path.join(local_repo_path, "DAPS")
if os.path.exists(daps_path):
    print(f"DAPS directory found: {daps_path}")
else:
    raise FileNotFoundError(f"DAPS directory not found at {daps_path}")

# 安装 omegaconf
#os.system("pip install omegaconf")

#@title 0.module import##########################################################################################
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.distributions import Categorical
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

#@title 1.Dataset#########################################################################################
from DAPS.data import DiffusionData, register_dataset
import torch.nn.functional as F
from torchvision import transforms

@register_dataset(name='velocity_dataset')
class VelocityDataset(DiffusionData):
    def __init__(self, velocity_patches, image_size=32, transform=None, device='cpu'):
        """
        Args:
            velocity_patches: 速度模型数据 (N, H, W)
            image_size: 目标图像大小
            transform: 数据变换
            device: 计算设备
        """
        super().__init__()

        # 确保输入是 PyTorch 张量
        if not isinstance(velocity_patches, torch.Tensor):
            velocity_patches = torch.tensor(velocity_patches)

        # 数据预处理
        self.device = device
        self.image_size = image_size

        # 计算统计信息
        with torch.no_grad():
            self.v_mean = velocity_patches.mean()
            self.v_std = velocity_patches.std()

            # 标准化
            self.velocity_models = (velocity_patches - self.v_mean) / self.v_std

            # 添加通道维度（如果需要）
            if self.velocity_models.dim() == 3:  # (N, H, W)
                self.velocity_models = self.velocity_models.unsqueeze(1)  # (N, 1, H, W)

            # 调整大小
            if (self.velocity_models.shape[2] != image_size or
                self.velocity_models.shape[3] != image_size):
                self.velocity_models = F.interpolate(
                    self.velocity_models,
                    size=(image_size, image_size),
                    mode='bilinear',
                    align_corners=False
                )

            # # 转换为三通道（如果需要）
            # if self.velocity_models.shape[1] == 1:
            #     self.velocity_models = self.velocity_models.repeat(1, 3, 1, 1)

        # 将数据移到 CPU 以节省显存
        self.velocity_models = self.velocity_models.cpu()
        self.transform = transform

    def __getitem__(self, idx):
        """获取单个数据样本"""
        velocity = self.velocity_models[idx].clone()

        if self.transform is not None:
            velocity = self.transform(velocity)

        # 确保返回正确的数据类型和形状
        return velocity.to(self.device)

    def __len__(self):
        """返回数据集长度"""
        return len(self.velocity_models)

    def get_shape(self):
        """返回数据形状"""
        return self.velocity_models.shape[1:]  # 返回 (C, H, W)

    def denormalize(self, velocity):
        """反标准化方法"""
        return velocity * self.v_std + self.v_mean

# 使用示例：
def create_velocity_dataset(v_torch, image_size=32):
    # 定义变换
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()),  # 确保数据类型为 float
    ])

    # 创建数据集
    dataset = VelocityDataset(
        velocity_patches=v_torch,  #  速度数据 (10_000, 200, 200)
        image_size=image_size,     # 目标图像大小
        transform=transform,
        device='cpu'
    )

    return dataset

#@title 2.Score Model Initialization#########################################################################################
# %cd /content/DAPS
from DAPS.model import DiffusionModel, register_model

@register_model(name='my_model')
class NCSNpp_DAPS(DiffusionModel):
    def __init__(self, model_config=None):
        super().__init__()

        # 使用默认配置或传入的配置
        if model_config is None:
            from configs.ve import cifar10_ncsnpp_continuous as configs
            self.config = configs.get_config()
        else:
            self.config = model_config

        # 初始化SDE
        self.sde = VESDE(
            sigma_min=self.config.model.sigma_min,
            sigma_max=self.config.model.sigma_max,
            N=self.config.model.num_scales
        )
        self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

        # 初始化数据处理工具
        self.sigmas = mutils.get_sigmas(self.config)
        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        # 创建和加载模型
        self.score_model = mutils.create_model(self.config)
        self._initialize_model()

        # 创建sampler配置并添加latent参数
        sampler_kwargs = {
            'annealing_scheduler_config': base_config.annealing_scheduler_config,
            'diffusion_scheduler_config': base_config.diffusion_scheduler_config,
            'lgvd_config': lgvd_config,
            'sde': self.sde,
            'inverse_scaler': self.inverse_scaler,
            'sampling_eps': 1e-5,
            'latent': False
        }
        # 使用get_sampler创建DAPS实例
        self.daps = get_sampler(**sampler_kwargs)

    def forward(self, x, t):
        """
        前向传播函数，直接调用内部score_model的forward方法

        Args:
            x (torch.Tensor): 输入数据
            t (torch.Tensor): 时间步/噪声水平

        Returns:
            torch.Tensor: 计算得到的score
        """
        # 确保数据在正确的设备上
        x = x.to(self.config.device)
        t = t.to(self.config.device)

        # 直接使用内部的score_model进行计算
        return self.score_model(x, t)

    def _initialize_model(self, ckpt_filename=None):
        """初始化模型，加载检查点"""
        if ckpt_filename is None:
            ckpt_filename = f_path+'checkpoints/32_checkpoint_5_cifar10_ncsnpp_continuous.pth'

        # 初始化优化器和EMA
        optimizer = get_optimizer(self.config, self.score_model.parameters())
        ema = ExponentialMovingAverage(
            self.score_model.parameters(),
            decay=self.config.model.ema_rate
        )

        # 创建状态字典
        state = dict(
            step=0,
            optimizer=optimizer,
            model=self.score_model,
            ema=ema
        )

        # 加载检查点
        state = restore_checkpoint(ckpt_filename, state, self.config.device)
        ema.copy_to(self.score_model.parameters())

    def score(self, x, sigma):
        """
        计算score function

        参数:
            x: torch.Tensor([B, *data.shape]) - 输入数据
            sigma: float or torch.Tensor - 噪声水平
        """
        if isinstance(sigma, float):
            sigma = torch.ones(x.shape[0], device=x.device) * sigma

        # 确保数据在正确的设备上
        x = x.to(self.config.device)
        sigma = sigma.to(self.config.device)

        # 使用score模型计算score
        score = self.score_model(x, sigma)
        return score

    def set_device(self, device):
        """设置模型设备"""
        self.config.device = torch.device(device)
        self.score_model = self.score_model.to(self.config.device)
        self.sde.discrete_sigmas = self.sde.discrete_sigmas.to(self.config.device)

    def load_checkpoint(self, ckpt_filename):
        """加载新的检查点"""
        self._initialize_model(ckpt_filename)

#@title 3.Operator#########################################################################################
from DAPS.forward_operator import *

# 安装deepwave Version: 0.0.20 
#os.system("pip install deepwave")

import torch.nn.functional as F


@register_operator(name='seismic_fo')
class SeismicForwardOperator(Operator):
    def __init__(self, config, image_size=128,sigma=0.05):
        super().__init__(sigma=sigma)
        """
        初始化地震正演算子的参数

        参数：
        image_size: 高密度网格边长

        """
        super().__init__()
        self.config = config
        # 图像尺寸参数
        self.image_size = image_size

        # 网格参数
        self.dx = 2.

        # 采集系统参数
        self.n_shots = 1
        self.n_sources_per_shot = 1
        self.n_receivers_per_shot = 100
        self.d_receiver = 2
        self.first_receiver = 0
        self.source_depth = 2  # Keep sources near surface
        # 震源参数
        self.freq = 25
        self.dt = 0.002
        self.nt = 300
        self.peak_time = 1.5 / self.freq

        # 确保所有参数都在正确的设备上
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_acquisition_geometry(self, patch_size):
        """设置采集系统几何布置"""
        # 震源位置 (固定在模型中间上方)
        source_locations = torch.zeros(self.n_shots, self.n_sources_per_shot, 2,
                                     dtype=torch.long, device=self.device)
        source_locations[..., 1] = self.source_depth  # Keep source depth fixed near surface
        source_locations[:, 0, 0] = torch.clamp(    # torch.arange(0, n_shots) * d_source + first_source,
            torch.arange(1,2)*(patch_size/2), #mid point source
            0, patch_size-1
        )

        # 检波器位置
        receiver_locations = torch.zeros(self.n_shots, self.n_receivers_per_shot, 2,
                                       dtype=torch.long, device=self.device)
        receiver_locations[..., 1] = 2  # 检波器深度
        receiver_locations[:, :, 0] = torch.clamp(
            (torch.arange(self.n_receivers_per_shot) * self.d_receiver + self.first_receiver)
            .repeat(self.n_shots, 1),
            0, patch_size-1
        )

        # 震源子波
        source_amplitudes = (
            deepwave.wavelets.ricker(self.freq, self.nt, self.dt, self.peak_time)
            .repeat(self.n_shots, self.n_sources_per_shot, 1)
            .to(self.device)
        )

        return source_locations, receiver_locations, source_amplitudes

    def _toVelocityShape(self, seismic_data):
        """


        Args:
            seismic_data: shape [B, n_receivers, nt]

        Returns:
            reshaped_data: shape [B, 1, image_size, image_size]
        """
        batch_size = seismic_data.shape[0]

        # 首先将数据重塑为 [B, 1, n_receivers, nt]
        data = seismic_data.unsqueeze(1)

        # 使用双线性插值将数据调整为目标尺寸
        reshaped_data = F.interpolate(
            data,
            size=(128,128),
            mode='bilinear',
            align_corners=True
        )

        return reshaped_data

    def __call__(self, x):
        """
        计算速度模型对应的地震记录并重塑为相同尺寸

        Args:
            x: 速度模型, torch.Tensor([B, 1, H, W])

        Returns:
            y: 重塑后的地震记录, torch.Tensor([B, 1, H, W])
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)

        # 如果输入尺寸小于高密度网格边长，先进行上采样
        if x.shape[2] < self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=True
            )

        # 1. 首先将输入数据规范化到[0,1]范围
        x_normalized = (x - x.min()) / (x.max() - x.min())

        # 2. 然后映射到实际的速度范围
        v_min, v_max = 1500.0, 5500.0
        x = v_min + (v_max - v_min) * x_normalized

        # # 验证速度值
        # print(f"速度范围: [{x.min().item():.2f}, {x.max().item():.2f}] m/s")

        # # 检查采样条件
        # wavelength = x.min().item() / self.freq
        # points_per_wavelength = wavelength / self.dx

        # print(f"每波长的网格点数: {points_per_wavelength:.2f}")
        # assert points_per_wavelength >= 6, (
        #     f"采样不足！每波长需要至少6个网格点，当前只有{points_per_wavelength:.2f}个点"
        # )

        # 计算批次大小
        batch_size = x.shape[0]

        # 去除channel维度，因为deepwave期望的输入形状是[H, W]
        x = x.squeeze(1)

        # 准备存储所有批次的输出
        all_receiver_amplitudes = []

        # 为每个批次设置采集几何系统
        source_locations, receiver_locations, source_amplitudes = self.setup_acquisition_geometry(x.shape[-1])

        # 对每个批次进行正演模拟
        for i in range(batch_size):
            velocity = x[i]  # 提取单个速度模型

            # 进行正演模拟
            out = deepwave.scalar(
                velocity,
                self.dx,
                self.dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=8,
                pml_freq=self.freq
            )

            # 提取接收器记录
            receiver_amplitudes = out[-1]  # shape: [n_shots, n_receivers_per_shot, nt]

            # 重塑数据以匹配期望的输出格式
            receiver_amplitudes = receiver_amplitudes.reshape(-1, self.nt)  # [n_shots*n_receivers_per_shot, nt]
            all_receiver_amplitudes.append(receiver_amplitudes)

        # 将所有批次的结果堆叠在一起
        seismic_data = torch.stack(all_receiver_amplitudes, dim=0)  # [B, n_shots*n_receivers_per_shot, nt]

        # 将地震记录重塑为目标图像尺寸
        y = self._toVelocityShape(seismic_data)

        # 对整个批次数据一起归一化
        y_normalized = (y - y.min()) / (y.max() - y.min())

        return y_normalized

class RHcalculator_0112:
    def __init__(self, seismic_operator):
        """
        初始化雅可比矩阵计算器

        Parameters:
        -----------
        seismic_operator : SeismicForwardOperator
            地震正演算子实例
        """
        self.operator = seismic_operator
        self.device = seismic_operator.device

    def calculate_H_matrix(self, velocity_model):
        """
        使用批量计算优化 H 矩阵计算
        """
        velocity_model.requires_grad_(True)
        seismic_data = self.operator(velocity_model)

        # 批量展开数据，构建 loss
        n_receivers, nt = seismic_data.shape[2:]
        n_model_params = velocity_model.numel()
        H = torch.zeros((n_receivers * nt, n_model_params), device=velocity_model.device) #状态空间--映射-->测量空间（等维度映射）

        seismic_data_flat = seismic_data.view(-1)  # 展平所有观测数据
        indices = torch.arange(seismic_data_flat.size(0), device=velocity_model.device)

        # 使用 retain_graph 计算批量梯度
        gradients = []
        for idx in indices:
            if velocity_model.grad is not None:
                velocity_model.grad.zero_()
            seismic_data_flat[idx].backward(retain_graph=True)
            gradients.append(velocity_model.grad.clone().flatten())

        H = torch.stack(gradients, dim=0)
        return H

    def calculate_H_efficient(self, velocity_model, observed_data):
        """
        使用伴随状态方法更高效地计算H矩阵

        Parameters:
        -----------
        velocity_model : torch.Tensor
            速度模型 [1, 1, H, W]
        observed_data : torch.Tensor
            观测数据 [1, 1, n_receivers, nt]

        Returns:
        --------
        H : torch.Tensor
            观测矩阵（以隐式形式）
        """
        velocity_model = velocity_model.clone().detach().requires_grad_(True)

        # 计算正演
        predicted_data = self.operator(velocity_model)

        # 计算残差
        residual = predicted_data - observed_data

        # 计算梯度（伴随状态方法）
        loss = 0.5 * torch.sum(residual ** 2)
        loss.backward()

        # 获取梯度（这实际上是H^T * residual）
        gradient = velocity_model.grad.clone()

        return gradient

    def setup_noise_covariance(self, velocity_model, snr_db=20):
          """
          使用稀疏矩阵优化噪声协方差
          """
          from scipy.sparse import diags
          seismic_data = self.operator(velocity_model)
          n_receivers, nt = seismic_data.shape[2:]
          n_data = n_receivers * nt

          noise_var = 10 ** (-snr_db / 10)
          R_diag = np.ones(n_data) * noise_var

          # 构建稀疏相关矩阵（只考虑局部相关性）
          space_corr = diags([1, 0.5], [0, 1], shape=(n_data, n_data)).toarray()
          time_corr = diags([1, 0.7], [0, 1], shape=(n_data, n_data)).toarray()
          R = diags(R_diag).toarray() * space_corr * time_corr

          return R

    # def calculate_H_matrix(self, velocity_model):
    #     """
    #     计算完整的观测矩阵H

    #     Parameters:
    #     -----------
    #     velocity_model : torch.Tensor
    #         速度模型

    #     Returns:
    #     --------
    #     H : torch.Tensor
    #         观测矩阵
    #     """
    #     # 确保velocity_model需要梯度
    #     velocity_model.requires_grad_(True)  # 添加这行
    #     # 获取模型参数数量
    #     n_model_params = torch.numel(velocity_model)

    #     # 获取接收器数量和时间步数
    #     seismic_data = self.operator(velocity_model)
    #     n_receivers = seismic_data.shape[2]  # 接收器数量
    #     nt = seismic_data.shape[3]          # 时间步数

    #     # 初始化H矩阵
    #     H = torch.zeros((n_receivers * nt, n_model_params),
    #                    device=velocity_model.device)

    #     # 计算每个观测点对应的梯度
    #     for receiver_idx in range(n_receivers):
    #         for time_idx in range(nt):
    #             # 清除之前的梯度
    #             if velocity_model.grad is not None:
    #                 velocity_model.grad.zero_()

    #             # 确保seismic_data需要梯度
    #             if not seismic_data.requires_grad:
    #                 seismic_data.requires_grad_(True)

    #             # 计算当前数据点对应的梯度
    #             try:
    #                 current_value = seismic_data[0, 0, receiver_idx, time_idx]
    #                 current_value.backward(retain_graph=True)

    #                 # 将梯度存储到H矩阵中
    #                 row_idx = receiver_idx * nt + time_idx
    #                 H[row_idx] = velocity_model.grad.flatten()
    #             except IndexError as e:
    #                 print(f"Error accessing index: receiver_idx={receiver_idx}, time_idx={time_idx}")
    #                 print(f"Seismic data shape: {seismic_data.shape}")
    #                 raise e

    #     return H

        # def setup_noise_covariance(self, velocity_model, snr_db=20):
    #     """
    #     设置观测噪声协方差矩阵

    #     Parameters:
    #     -----------
    #     velocity_model : torch.Tensor
    #         速度模型 [1, 1, H, W]
    #     snr_db : float
    #         信噪比（分贝）

    #     Returns:
    #     --------
    #     R : np.ndarray
    #         噪声协方差矩阵
    #     """
    #     # 获取接收器数量和时间步数
    #     seismic_data = self.operator(velocity_model)
    #     n_receivers = seismic_data.shape[2]  # 接收器数量
    #     nt = seismic_data.shape[3]          # 时间步数
    #     n_data = n_receivers * nt

    #     # 基于信噪比计算噪声方差
    #     noise_var = 10 ** (-snr_db / 10)

    #     # 创建对角协方差矩阵
    #     R = np.eye(n_data) * noise_var

    #     # 添加空间-时间相关性（可选）
    #     correlation_length_space = 2  # 接收点之间的相关长度
    #     correlation_length_time = 5   # 时间采样点之间的相关长度

    #     for i in range(n_data):
    #         rec_i = i // nt
    #         time_i = i % nt
    #         for j in range(n_data):
    #             rec_j = j // nt
    #             time_j = j % nt

    #             # 计算空间和时间相关性
    #             space_corr = np.exp(-((rec_i - rec_j)**2) / (2 * correlation_length_space**2))
    #             time_corr = np.exp(-((time_i - time_j)**2) / (2 * correlation_length_time**2))

    #             R[i, j] *= space_corr * time_corr

    #     return R


#@title 4.Evaluation#########################################################################################
'''
We use POT to compute Wasserstein distance between sampled posterior and true posterior.
'''
# 安装pot和piq
#os.system("pip install pot") #pot-0.9.5
#os.system("pip install piq") #piq-0.8.0

from DAPS.eval import EvalFn, register_eval_fn
import ot

def check(samples):
    idx = torch.isnan(samples)
    samples[idx] = torch.zeros_like(samples[idx])
    return samples

def get_gt_posterior(gt, operator, y, oversample=1, sigma=0.05):
    '''
    gt: 实际地震记录
    '''
    idx = []
    for _ in range(oversample):
        likelihood = operator.likelihood(gt, y)
        # print(f'likelihood={likelihood}')
        resampling = torch.multinomial(likelihood/likelihood.sum(), len(gt), replacement=True).to(gt.device)# A tensor containing the indices of the drawn samples.
        idx.append(resampling)
    idx = torch.cat(idx)
    return gt[idx]

def wasserstein(sample1, sample2):
    sample1, sample2 = check(sample1), check(sample2)
    a = np.ones((sample1.shape[0],)) / sample1.shape[0]
    b = np.ones((sample2.shape[0],)) / sample2.shape[0]
    C = ot.dist(sample1.numpy(), sample2.numpy(), metric='euclidean')
    w = ot.emd2(a, b, C)
    return w

@register_eval_fn(name='w2dist')
class Wasserstein(EvalFn):
    cmp = 'min'
    def __init__(self, operator):
        self.operator = operator
        self.requires_gt = True  # 有监督评估函数
    def __call__(self, gt, measurement, sample, reduction='mean'):
        '''
            gt           : reference ground truth, torch.Tensor([B, *data.shape])
            measurement  : noisy measurement, torch.Tensor([B, *measurement.shape])#实际地震记录
            sample       : posterior samples, torch.Tensor([B, *data.shape])
        '''
        gt_sample = get_gt_posterior(gt, operator, measurement)
        gt_prob = np.ones((gt_sample.shape[0],)) / gt_sample.shape[0]

        pred_prob = np.ones((sample.shape[0],)) / sample.shape[0]

        # Reshape gt_sample and sample to 2D
        gt_sample_2d = gt_sample.detach().cpu().numpy().reshape(gt_sample.shape[0], -1)  # Flatten spatial and channel dimensions
        sample_2d = sample.detach().cpu().numpy().reshape(sample.shape[0], -1)  # Flatten spatial and channel dimensions

        dist = ot.dist(gt_sample_2d, sample_2d, metric='euclidean') # Calculate distance using reshaped tensors
        w = ot.emd2(gt_prob, pred_prob, dist)

        if reduction == 'none':
            w = torch.tensor([w]*gt.shape[0], device=gt.device)
        else:
            w = torch.tensor([[w]], device=gt.device)
        return w

@register_eval_fn(name='w2dist_unsupervised')
class Wasserstein_us(EvalFn):
    cmp = 'min'

    def __init__(self, operator):
        self.operator = operator
        self.requires_gt = False  # 无监督评估函数
    def __call__(self, gt, measurement, sample, reduction='mean'):
        '''
            gt: None
            measurement  : noisy measurement, torch.Tensor([B, *measurement.shape]) # 实际地震记录
            sample       : posterior samples, torch.Tensor([B, *data.shape])       # 后验样本
        '''
        # Apply the operator to the posterior samples
        pred = self.operator(sample)  # Predicted data from the operator
        pred_prob = np.ones((pred.shape[0],)) / pred.shape[0]  # Uniform probability for predicted samples

        # Prepare measurement probabilities
        measurement_prob = np.ones((measurement.shape[0],)) / measurement.shape[0]

        # Reshape pred and measurement to 2D for distance computation
        pred_2d = pred.detach().cpu().numpy().reshape(pred.shape[0], -1)  # Flatten spatial and channel dimensions
        measurement_2d = measurement.detach().cpu().numpy().reshape(measurement.shape[0], -1)  # Flatten spatial and channel dimensions

        # Compute the pairwise distance between pred and measurement
        dist = ot.dist(pred_2d, measurement_2d, metric='euclidean')  # Euclidean distance matrix
        w = ot.emd2(pred_prob, measurement_prob, dist)  # Wasserstein distance

        # Return the Wasserstein distance (with or without reduction)
        if reduction == 'none':
            w = torch.tensor([w] * sample.shape[0], device=sample.device)
        else:
            w = torch.tensor([[w]], device=sample.device)
        return w

#@title DAPS Hyperparameter#########################################################################################
from omegaconf import OmegaConf
def load_conf(file_path):
    conf = OmegaConf.load(file_path)
    return conf

# for pixel sapce diffusion model
base_config_file = "edm_daps" # @param ["edm_daps"]
base_config = load_conf(main_path+"DAPS/config/sampler/{base_config_file}.yaml")

# annealing scheduler
annealing_steps = 50 # @param {type:"integer"}
base_config.annealing_scheduler_config.num_steps = annealing_steps

annealing_sigma_max = 0.1 #@param {"type": "number"}
base_config.annealing_scheduler_config.sigma_max = annealing_sigma_max

annealing_sigma_min = 0.01 #@param {"type": "number"}
base_config.annealing_scheduler_config.sigma_min = annealing_sigma_min

# diffusion scheduler
diffusion_steps = 20 # @param {type:"integer"} #10
base_config.diffusion_scheduler_config.num_steps = diffusion_steps

# lgvd config
lgvd_config = OmegaConf.create()
langevin_steps = 20 # @param {type:"integer"} #100
lgvd_config.num_steps = langevin_steps

lr = 1e-4 #@param {"type": "number"}
lgvd_config.lr = lr

tau = 0.07 #@param {"type": "number"}
lgvd_config.tau = tau

lr_min_ratio = 1 #@param {"type": "number"}
lgvd_config.lr_min_ratio = lr_min_ratio

batch_size = 1 # @param {type:"number"}
sigma = 0.3 # @param {type:"number"}

#@title daps module import#########################################################################################
from sde_lib_daps import VESDE, VPSDE, subVPSDE
from sampling_daps import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)

#@title DAPS#########################################################################################
from DAPS.sampler import get_sampler, DAPS
from DAPS.eval import Evaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data
data = create_velocity_dataset(v_torch, image_size=32)

# model
from score_sde_pytorch.models.ncsnpp import NCSNpp
# 创建模型实例
model = NCSNpp_DAPS(config)

# 如果需要，设置设备
model.set_device('cuda')
# 加载特定检查点（如果默认路径不正确）
model.load_checkpoint(f_path+'checkpoints/32_checkpoint_5_cifar10_ncsnpp_continuous.pth')

# 初始化算子
operator = SeismicForwardOperator(config,image_size=200,sigma=sigma)
eval_fn = Wasserstein(operator)
eval_us_fn = Wasserstein_us(operator)
# reference ground truth points
seed_index = 0           ######RSS17已经完成###########
gt = data.get_data(batch_size, 0, seed=seed_index) # SM raw
measurement = operator(gt.to(device))

evaluator = Evaluator((eval_fn,))
evaluator_us = Evaluator((eval_us_fn,))

#@title Random Score Search#########################################################################################
# seed: SM columnl gt: SM raw
sample = model.daps.explicit_sample(model, model.daps.get_start(gt.to(device)), operator, measurement, evaluator_us, evaluator, record=True, verbose=False, seed=27, gt=gt)
trajectory = model.daps.trajectory.compile()

visualize_data(sample.detach().cpu())
visualize_data(gt.detach().cpu())


#@title check wess distance to list the time-varying model##########################################################
def compute_wd(gt1, gt2):
    """
    计算两个gt之间的Wasserstein距离

    Parameters:
    -----------
    gt1, gt2 : torch.Tensor
        需要比较的两个ground truth tensor

    Returns:
    --------
    float
        Wasserstein距离
    """
    # 确保输入是tensor并且在CPU上
    if torch.is_tensor(gt1):
        gt1 = gt1.detach().cpu()
    if torch.is_tensor(gt2):
        gt2 = gt2.detach().cpu()

    # 将tensor展平为2D
    gt1_2d = gt1.reshape(gt1.shape[0], -1).numpy()
    gt2_2d = gt2.reshape(gt2.shape[0], -1).numpy()

    # 计算概率分布（均匀分布）
    gt1_prob = np.ones((gt1.shape[0],)) / gt1.shape[0]
    gt2_prob = np.ones((gt2.shape[0],)) / gt2.shape[0]

    # 计算距离矩阵
    dist = ot.dist(gt1_2d, gt2_2d, metric='euclidean')

    # 计算Wasserstein距离
    w = ot.emd2(gt1_prob, gt2_prob, dist)

    return w

from tqdm.auto import tqdm
from collections import defaultdict
def find_similar_gts(data, batch_size, num_seeds=100, threshold=5, base_seeds=None):
    """
    寻找所有相似的gt组

    Parameters:
    -----------
    data: 数据生成器
    batch_size: int
        批次大小
    num_seeds: int
        要检查的seed数量
    threshold: float
        Wasserstein距离的阈值，低于此值认为相似
    base_seeds: list or range, optional
        如果指定，则只寻找与这些seeds生成的gt相似的其他gt

    Returns:
    --------
    similar_groups: dict
        key为seed，value为(相似gt的seed列表，对应的WD值列表)
    """
    # 首先生成所有gt
    print("Generating GTs...")
    gts = {}
    for seed in tqdm(range(num_seeds), desc="Generating GTs"):
        gts[seed] = data.get_data(batch_size, 0, seed=seed)

    similar_groups = defaultdict(lambda: ([], []))

    # 如果指定了base_seeds，使用指定的seeds列表
    if base_seeds is not None:
        seeds_to_check = base_seeds
    else:
        seeds_to_check = range(num_seeds)

    # 计算总比较次数
    # 对于每个base_seed，我们需要与所有大于它的seed比较
    total_comparisons = 0
    for seed1 in seeds_to_check:
        total_comparisons += sum(1 for seed2 in range(num_seeds) if seed1 < seed2)

    # 使用tqdm创建进度条
    with tqdm(total=total_comparisons, desc="Computing WD distances") as pbar:
        # 计算距离
        for seed1 in seeds_to_check:
            gt1 = gts[seed1]
            for seed2 in range(num_seeds):
                if seed1 >= seed2:  # 避免重复计算
                    continue

                gt2 = gts[seed2]
                wd = compute_wd(gt1, gt2)

                # 如果距离小于阈值，记录这对gt
                if wd < threshold:
                    similar_groups[seed1][0].append(seed2)
                    similar_groups[seed1][1].append(wd)
                    # 对称性
                    similar_groups[seed2][0].append(seed1)
                    similar_groups[seed2][1].append(wd)

                    print(f"\nFound similar pair: ({seed1}, {seed2}) with WD = {wd:.4f}")

                pbar.update(1)

    return similar_groups

# 使用示例
def print_similar_groups(similar_groups, min_group_size=2):
    """打印结果，只显示至少有min_group_size个相似gt的组"""
    print("\n=== Similar GT Groups ===")
    for seed, (similar_seeds, wds) in similar_groups.items():
        if len(similar_seeds) >= min_group_size - 1:  # -1是因为不包括自己
            print(f"\nBase seed {seed} has {len(similar_seeds)} similar GTs:")
            for sim_seed, wd in zip(similar_seeds, wds):
                print(f"  - Seed {sim_seed}: WD = {wd:.4f}")

def get_sorted_similar_groups(similar_groups, min_group_size=2):
    """
    返回排序后的相似seed组列表，处理带权重值的字典

    Parameters:
    -----------
    similar_groups: dict
        格式为 {seed: ([similar_seeds], [weights])}
    min_group_size: int
        最小组大小（包括seed本身）

    Returns:
    --------
    sorted_groups: list of list
        每个子列表包含一组相似的seeds，按从小到大排序
    """
    processed_seeds = set()
    sorted_groups = []

    for seed, (similar_seeds, _) in similar_groups.items():
        # 如果seed已处理过则跳过
        if seed in processed_seeds:
            continue

        # 创建当前组（包含seed本身和相似seeds）
        current_group = {seed} | set(similar_seeds)

        # 扩展组：检查组内每个seed的相似seeds
        expanded_group = current_group.copy()
        for s in current_group:
            if s in similar_groups:
                expanded_group.update(similar_groups[s][0])

        # 如果组大小满足要求
        if len(expanded_group) >= min_group_size:
            # 转为列表并排序
            sorted_group = sorted(list(expanded_group))
            sorted_groups.append(sorted_group)

            # 标记所有seeds为已处理
            processed_seeds.update(expanded_group)

    # 按第一个元素排序所有组
    sorted_groups.sort(key=lambda x: x[0])
    return sorted_groups


# 使用示例
threshold = 10  # 设置阈值
num_seeds = 100  # 检查前100个seed
base_seeds = range(100)

# 运行原来的函数
similar_groups = find_similar_gts(data, batch_size, num_seeds=num_seeds,
                                threshold=threshold, base_seeds=base_seeds)

# 打印原来的详细输出
print_similar_groups(similar_groups, min_group_size=3)

# 获取并打印排序后的相似组
sorted_groups = get_sorted_similar_groups(similar_groups, min_group_size=2)
print("\n=== Sorted Similar Groups ===")
for group in sorted_groups:
    print(group)

#@title Group Score Search#########################################################################################
seed_4_SimilarityMatrix = range(50)
x_former = 40

# 首先读取已有的2×50矩阵
load_existing_matrix = True
if load_existing_matrix:
    existing_matrix = torch.load(main_path+f'similarity_matrix_{x_former}x50.pt')
else:
    existing_matrix = torch.zeros((x_former, 50), device=device)

# 设定目标大小 x（比如x=50）
x = 50

# 初始化新的x×50矩阵，前两行复制已有数据
new_similarity_matrix = torch.zeros((x, 50), device=device)
new_similarity_matrix[:x_former] = existing_matrix[:x_former]

# 获取已计算的相似组
sorted_groups = get_sorted_similar_groups(similar_groups, min_group_size=2)

# 创建一个集合来跟踪需要跳过的seed
skip_seeds = set()
for group in sorted_groups:
    min_seed = min(group)
    skip_seeds.update(seed for seed in group if seed != min_seed)

from tqdm import tqdm
# 从索引x_former开始继续计算
for seed_index in tqdm(range(x_former, x), desc="Building Similarity Matrix"):
    # 如果当前seed在skip_seeds中，跳过计算
    if seed_index in skip_seeds:
        # 找到这个seed所在的组
        for group in sorted_groups:
            if seed_index in group:
                # 复制组中最小seed的相似度行
                min_seed = min(group)
                new_similarity_matrix[seed_index] = new_similarity_matrix[min_seed]
                break
        continue

    # 获取ground truth和对应的measurement
    gt = data.get_data(batch_size, 0, seed=seed_index)
    measurement = operator(gt.to(device))

    # 计算当前ground truth对应的所有50个采样点的Wasserstein距离
    w_distances = model.daps.sample(
        model,
        model.daps.get_start(gt.to(device)),
        operator,
        measurement,
        evaluator_us,
        evaluator,
        record=True,
        verbose=False,
        seed=seed_4_SimilarityMatrix,
        gt=gt
    )

    # 将结果存入相似度矩阵的对应行
    new_similarity_matrix[seed_index] = w_distances

# 保存结果
torch.save(new_similarity_matrix, f_path+f'similarity_matrix_{x}x50.pt')

# 打印矩阵的形状确认
print(f"New Similarity Matrix shape: {new_similarity_matrix.shape}")
torch.set_printoptions(precision=4)
print(f"New Similarity Matrix:\n{new_similarity_matrix}")

#@title metrics check#########################################################################################
# 访问数据
# 张量数据
all_xt = trajectory.tensor_data['xt']  # 所有xt的记录
all_x0y = trajectory.tensor_data['x0y']  # 所有x0y的记录
all_x0hat = trajectory.tensor_data['x0hat']  # 所有x0hat的记录

# 标量数据
sigma_history = trajectory.value_data['sigma']

# 如果有评估指标数据
x0hat_metrics = {k: v for k, v in trajectory.value_data.items() if k.startswith('x0hat_')}
x0y_metrics = {k: v for k, v in trajectory.value_data.items() if k.startswith('x0y_')}

# 确保保存路径存在
save_dir = os.path.join(f_path, 'samples')
os.makedirs(save_dir, exist_ok=True)

# 显示某个时间步的结果
def show_step_results(trajectory, step):
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(trajectory.tensor_data['xt'][step].squeeze())
    plt.title(f'xt at step {step}')

    plt.subplot(132)
    plt.imshow(trajectory.tensor_data['x0hat'][step].squeeze())
    plt.title(f'x0hat at step {step}')

    plt.subplot(133)
    plt.imshow(trajectory.tensor_data['x0y'][step].squeeze())
    plt.title(f'x0y at step {step}')

    # 保存为PDF
    save_path = os.path.join(save_dir, f'Step_{step}_seed{seed_index}.pdf')
    # plt.savefig(save_path)
    plt.close()

# 为每个时间步创建图像
for step in range(len(trajectory.tensor_data['xt'])):
    show_step_results(trajectory, step)

# 显示sigma变化
fig = plt.figure()
plt.plot(trajectory.value_data['sigma'])
plt.title(f'Sigma History_seed{seed_index}')
save_path = os.path.join(save_dir, f'Sigma_History_seed{seed_index}.pdf')
plt.savefig(save_path)
plt.close()

# 显示评估指标（如果有的话）
for metric_name in x0hat_metrics:
    fig = plt.figure()
    plt.plot(x0hat_metrics[metric_name], label='x0hat')
    plt.plot(x0y_metrics[metric_name.replace('x0hat_', 'x0y_')], label='x0y')
    plt.title(f'{metric_name} History_seed{seed_index}')
    plt.legend()
    save_path = os.path.join(save_dir, f'{metric_name}_History_seed{seed_index}.pdf')
    plt.savefig(save_path)
    plt.close()

# 打印可用数据信息
print("Available tensor data:", trajectory.tensor_data.keys())
print("Available value data:", trajectory.value_data.keys())

