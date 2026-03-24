"""
DAPS 数据集适配器

来源: fwi_common.py:546-633
注意：导入此模块会触发 DAPS 全局注册表的副作用（@register_dataset）。
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


def _register_velocity_dataset():
    """延迟注册，仅在调用时导入 DAPS"""
    from DAPS.data import DiffusionData, register_dataset

    @register_dataset(name='velocity_dataset')
    class VelocityDataset(DiffusionData):
        def __init__(self, velocity_patches, image_size=32, transform=None, device='cpu'):
            super().__init__()

            if not isinstance(velocity_patches, torch.Tensor):
                velocity_patches = torch.tensor(velocity_patches)

            self.device = device
            self.image_size = image_size

            with torch.no_grad():
                self.v_mean = velocity_patches.mean()
                self.v_std = velocity_patches.std()

                self.velocity_models = (velocity_patches - self.v_mean) / self.v_std

                if self.velocity_models.dim() == 3:
                    self.velocity_models = self.velocity_models.unsqueeze(1)

                if (self.velocity_models.shape[2] != image_size or
                    self.velocity_models.shape[3] != image_size):
                    self.velocity_models = F.interpolate(
                        self.velocity_models,
                        size=(image_size, image_size),
                        mode='bilinear',
                        align_corners=False
                    )

            self.velocity_models = self.velocity_models.cpu()
            self.transform = transform

        def __getitem__(self, idx):
            velocity = self.velocity_models[idx].clone()
            if self.transform is not None:
                velocity = self.transform(velocity)
            return velocity.to(self.device)

        def __len__(self):
            return len(self.velocity_models)

        def get_shape(self):
            return self.velocity_models.shape[1:]

        def denormalize(self, velocity):
            return velocity * self.v_std + self.v_mean

    return VelocityDataset


# 模块级变量，缓存注册后的类
_VelocityDataset = None


def create_velocity_dataset(v_torch, image_size=32):
    """
    创建 VelocityDataset 实例的工厂函数。
    首次调用时会触发 DAPS 注册。

    参数:
        v_torch: torch.Tensor, 速度数据 (N, H, W)
        image_size: int, 目标图像大小

    返回:
        VelocityDataset 实例
    """
    global _VelocityDataset
    if _VelocityDataset is None:
        _VelocityDataset = _register_velocity_dataset()

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
    ])

    dataset = _VelocityDataset(
        velocity_patches=v_torch,
        image_size=image_size,
        transform=transform,
        device='cpu'
    )

    return dataset
