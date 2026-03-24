"""
地震数据集类

来源: fwi_common.py:26-216
包含 SeismicPatchDataset, Rotate90Transform, VelocityScoreSDEDataset
"""

import random
import torch
from torch.utils.data import Dataset


class SeismicPatchDataset(Dataset):
    def __init__(self, velocity_patches, seismic_patches, source_locations, receiver_locations):
        self.velocity_models = velocity_patches
        self.seismic_data = seismic_patches
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

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
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            k = random.randint(0, 3)
            return torch.rot90(x, k=k, dims=(1, 2))
        return x


class VelocityScoreSDEDataset(Dataset):
    def __init__(self, velocity_patches, image_size=32, transform=None, device='cpu'):
        super().__init__()

        if not isinstance(velocity_patches, torch.Tensor):
            velocity_patches = torch.tensor(velocity_patches)

        self.velocity_models = velocity_patches.to(device, dtype=torch.float32)
        self.transform = transform
        self.image_size = image_size
        self.device = device

        with torch.no_grad():
            self.v_mean = self.velocity_models.mean()
            self.v_std = self.velocity_models.std()
            self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据：标准化、添加通道维度（如果需要）、调整大小"""
        with torch.no_grad():
            try:
                self.velocity_models = (self.velocity_models - self.v_mean) / self.v_std

                if self.velocity_models.dim() == 3:
                    self.velocity_models = self.velocity_models.unsqueeze(1)

                if (self.velocity_models.shape[2] != self.image_size or
                    self.velocity_models.shape[3] != self.image_size):
                    self.velocity_models = torch.nn.functional.interpolate(
                        self.velocity_models,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    )

                self.velocity_models = self.velocity_models.cpu()

            except RuntimeError as e:
                print(f"数据预处理错误: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
                batch = (batch - self.v_mean) / self.v_std

                if batch.dim() == 3:
                    batch = batch.unsqueeze(1)

                if (batch.shape[2] != self.image_size or
                    batch.shape[3] != self.image_size):
                    batch = torch.nn.functional.interpolate(
                        batch,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    )

                processed_data.append(batch.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.velocity_models = torch.cat(processed_data, dim=0)

    def __getitem__(self, idx):
        try:
            velocity = self.velocity_models[idx].clone()

            if self.transform is not None:
                velocity = self.transform(velocity)

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
        return self.to('cpu')

    def cuda(self, device=None):
        if device is None:
            device = torch.cuda.current_device()
        return self.to(f'cuda:{device}')
