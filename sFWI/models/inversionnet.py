"""
InversionNet models for seismic inversion baselines.

Source:
- OpenFWI official implementation (`lanl/OpenFWI`, `network.py`)
  from Los Alamos National Laboratory.

This module keeps the original InversionNet topology and adds a small
sFWI adapter (`InversionNetSFWI`) for current project I/O shapes.
"""

from collections import OrderedDict
from math import ceil
from typing import Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NORM_LAYERS = {
    "bn": nn.BatchNorm2d,
    "in": nn.InstanceNorm2d,
    "ln": nn.LayerNorm,
}


def replace_legacy_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Map legacy OpenFWI block names to the current module field names."""
    items = []
    for key, value in state_dict.items():
        new_key = (
            key.replace("Conv2DwithBN", "layers")
            .replace("Conv2DwithBN_Tanh", "layers")
            .replace("Deconv2DwithBN", "layers")
            .replace("ResizeConv2DwithBN", "layers")
        )
        items.append((new_key, value))
    return OrderedDict(items)


def _conv2d_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _infer_terminal_kernel(input_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Infer convblock8 kernel size so encoder terminal feature map becomes 1x1."""
    h, w = input_hw

    # Height path: stride-2 conv appears in convblock1,2_1,3_1,4_1,5_1,6_1,7_1
    h = _conv2d_out_dim(h, 7, 2, 3)
    for _ in range(6):
        h = _conv2d_out_dim(h, 3, 2, 1)

    # Width path: first four blocks keep width, then 5_1,6_1,7_1 downsample by 2
    for _ in range(3):
        w = _conv2d_out_dim(w, 3, 2, 1)

    return max(h, 1), max(w, 1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size=3,
        stride=1,
        padding=1,
        norm: Optional[str] = "bn",
        relu_slop: float = 0.2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            p = float(dropout) if isinstance(dropout, (float, int)) else 0.8
            layers.append(nn.Dropout2d(p))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlockTanh(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size=3,
        stride=1,
        padding=1,
        norm: Optional[str] = "bn",
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        norm: Optional[str] = "bn",
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InversionNet(nn.Module):
    """
    OpenFWI InversionNet (official topology).

    Default I/O:
    - input:  [B, 5, 1000, 70]
    - output: [B, 1, 70, 70]

    Notes:
    - `sample_spatial` is kept from the official code path.
    - `latent_kernel_size` can override the final encoder kernel for non-OpenFWI
      input sizes while preserving the rest of the architecture.
    """

    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        sample_spatial: float = 1.0,
        in_channels: int = 5,
        norm: str = "bn",
        latent_kernel_size: Optional[Tuple[int, int]] = None,
        output_size: Optional[Tuple[int, int]] = None,
        output_crop: int = 5,
    ):
        super().__init__()
        if latent_kernel_size is None:
            kernel_h = 8
            kernel_w = ceil(70 * sample_spatial / 8)
        else:
            kernel_h = max(int(latent_kernel_size[0]), 1)
            kernel_w = max(int(latent_kernel_size[1]), 1)

        self.output_size = output_size
        self.output_crop = int(output_crop)

        # Encoder
        self.convblock1 = ConvBlock(
            in_channels,
            dim1,
            kernel_size=(7, 1),
            stride=(2, 1),
            padding=(3, 0),
            norm=norm,
        )
        self.convblock2_1 = ConvBlock(
            dim1,
            dim2,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            norm=norm,
        )
        self.convblock2_2 = ConvBlock(
            dim2, dim2, kernel_size=(3, 1), padding=(1, 0), norm=norm
        )
        self.convblock3_1 = ConvBlock(
            dim2,
            dim2,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            norm=norm,
        )
        self.convblock3_2 = ConvBlock(
            dim2, dim2, kernel_size=(3, 1), padding=(1, 0), norm=norm
        )
        self.convblock4_1 = ConvBlock(
            dim2,
            dim3,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            norm=norm,
        )
        self.convblock4_2 = ConvBlock(
            dim3, dim3, kernel_size=(3, 1), padding=(1, 0), norm=norm
        )
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2, norm=norm)
        self.convblock5_2 = ConvBlock(dim3, dim3, norm=norm)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2, norm=norm)
        self.convblock6_2 = ConvBlock(dim4, dim4, norm=norm)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2, norm=norm)
        self.convblock7_2 = ConvBlock(dim4, dim4, norm=norm)
        self.convblock8 = ConvBlock(
            dim4,
            dim5,
            kernel_size=(kernel_h, kernel_w),
            padding=0,
            norm=norm,
        )

        # Decoder
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5, norm=norm)
        self.deconv1_2 = ConvBlock(dim5, dim5, norm=norm)
        self.deconv2_1 = DeconvBlock(
            dim5, dim4, kernel_size=4, stride=2, padding=1, norm=norm
        )
        self.deconv2_2 = ConvBlock(dim4, dim4, norm=norm)
        self.deconv3_1 = DeconvBlock(
            dim4, dim3, kernel_size=4, stride=2, padding=1, norm=norm
        )
        self.deconv3_2 = ConvBlock(dim3, dim3, norm=norm)
        self.deconv4_1 = DeconvBlock(
            dim3, dim2, kernel_size=4, stride=2, padding=1, norm=norm
        )
        self.deconv4_2 = ConvBlock(dim2, dim2, norm=norm)
        self.deconv5_1 = DeconvBlock(
            dim2, dim1, kernel_size=4, stride=2, padding=1, norm=norm
        )
        self.deconv5_2 = ConvBlock(dim1, dim1, norm=norm)
        self.deconv6 = ConvBlockTanh(dim1, 1, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)

        # Decoder
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)

        if self.output_crop > 0:
            c = self.output_crop
            x = F.pad(x, (-c, -c, -c, -c), mode="constant", value=0)

        x = self.deconv6(x)

        if self.output_size is not None and tuple(x.shape[-2:]) != tuple(self.output_size):
            x = F.interpolate(
                x, size=self.output_size, mode="bilinear", align_corners=False
            )
        return x


class InversionNetSFWI(nn.Module):
    """
    sFWI adapter of InversionNet.

    Default I/O:
    - input:  [B, 1, 100, 300]
    - output: [B, 1, 200, 200]
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 100, 300),
        output_shape: Tuple[int, int] = (200, 200),
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        norm: str = "bn",
        output_crop: int = 5,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (C, H, W).")

        in_channels, in_h, in_w = input_shape
        terminal_kernel = _infer_terminal_kernel((in_h, in_w))

        self.model = InversionNet(
            dim1=dim1,
            dim2=dim2,
            dim3=dim3,
            dim4=dim4,
            dim5=dim5,
            in_channels=in_channels,
            norm=norm,
            latent_kernel_size=terminal_kernel,
            output_size=output_shape,
            output_crop=output_crop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_inversionnet_state_dict(
    model: nn.Module,
    checkpoint: MutableMapping[str, torch.Tensor],
    strict: bool = False,
):
    """
    Load a checkpoint with OpenFWI legacy-key compatibility.

    Returns:
        (missing_keys, unexpected_keys) from `load_state_dict`.
    """
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        state_dict = dict(checkpoint["state_dict"])
    elif "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        state_dict = dict(checkpoint["model"])
    else:
        state_dict = dict(checkpoint)

    state_dict = replace_legacy_state_dict_keys(state_dict)

    # DDP/DataParallel compatibility.
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = OrderedDict((k[len("module."):], v) for k, v in state_dict.items())

    # Wrapper-prefix compatibility between InversionNet and InversionNetSFWI.
    model_keys = list(model.state_dict().keys())
    if model_keys and state_dict:
        expects_model_prefix = model_keys[0].startswith("model.")
        has_model_prefix = next(iter(state_dict.keys())).startswith("model.")
        if expects_model_prefix and not has_model_prefix:
            state_dict = OrderedDict((f"model.{k}", v) for k, v in state_dict.items())
        elif not expects_model_prefix and has_model_prefix:
            state_dict = OrderedDict((k[len("model."):], v) for k, v in state_dict.items())

    # Non-strict mode: keep only keys with matching tensor shape.
    # This avoids RuntimeError when loading checkpoints trained with a
    # different input/output geometry.
    if not strict:
        model_state = model.state_dict()
        filtered = OrderedDict()
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
        state_dict = filtered

    return model.load_state_dict(state_dict, strict=strict)
