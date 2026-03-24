"""
Baseline 模型类定义。

从 conditional_autoencoder_0_1b.py 提取, 去除 Colab 副作用代码。
仅保留模型结构定义, 供 evaluation_exp.py 的 Baseline 接口调用。

模型列表:
  - Autoencoder:   基础卷积自编码器
  - ModernUNet:    带 SE + ResBlock 的 U-Net
  - VAE:           变分自编码器
  - CustomDiffAE:  扩散自编码器 (带时间条件)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  1. Autoencoder
# ================================================================

class Autoencoder(nn.Module):
    """基础卷积自编码器。

    输入: seismic [B, 1, 100, 300]
    输出: velocity [B, 200, 200]
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # (8, 50, 150)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # (16, 25, 75)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 25 * 75, 512),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 16 * 25 * 75),
            nn.ReLU(),
            nn.Unflatten(1, (16, 25, 75)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # (8, 50, 150)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # (1, 100, 300)
            nn.ReLU()
        )
        # Final mapping to velocity
        self.fc_out = nn.Sequential(
            nn.Linear(100 * 300, 200 * 200),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.fc_out(decoded.view(decoded.size(0), -1))
        return output.view(-1, 200, 200)


# ================================================================
#  2. ModernUNet (带 SE + ResBlock)
# ================================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class ModernUNet(nn.Module):
    """带 SE-ResBlock 的 U-Net。

    输入: seismic [B, 1, 100, 300]
    输出: velocity [B, 200, 200]  (内部 squeeze)
    """

    def __init__(self):
        super(ModernUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(ResBlock(1, 64), ResBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(ResBlock(64, 128), ResBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(ResBlock(128, 256), ResBlock(256, 256))

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(ResBlock(256, 128), ResBlock(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(ResBlock(128, 64), ResBlock(64, 64))

        # Final output
        self.final_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.final_up = nn.Upsample(size=(200, 200), mode='bilinear',
                                     align_corners=True)
        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = self.final_up(out)
        out = self.final_layer(out)
        return out.squeeze(1)  # [B, 200, 200]


# ================================================================
#  3. VAE
# ================================================================

class VAE(nn.Module):
    """变分自编码器。

    输入: seismic [B, 1, 100, 300]
    输出: (recon [B, 200, 200], mu, logvar)
    """

    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_size = 16 * 25 * 75
        self.fc_mu = nn.Linear(self.fc_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_size, latent_dim)

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, self.fc_size),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU()
        )

        self.fc_out = nn.Sequential(
            nn.Linear(100 * 300, 200 * 200),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_linear(z)
        batch_size = x.size(0)
        x = x.view(batch_size, 16, 25, 75)
        x = self.decoder_conv(x)
        x = x.view(batch_size, -1)
        x = self.fc_out(x)
        return x.view(-1, 200, 200)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ================================================================
#  4. CustomDiffAE (扩散自编码器)
# ================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """带时间条件的残差块。"""

    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                padding=1)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        residual = self.shortcut(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[..., None, None]
        x = x + time_emb

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(out) + residual


class CustomDiffAE(nn.Module):
    """扩散自编码器: 适应非对称输入输出维度。

    输入: seismic [B, 1, 100, 300], time (optional)
    输出: (recon [B, 1, 200, 200], z, mean, log_var)
    """

    def __init__(
        self,
        input_shape=(1, 100, 300),
        output_shape=(200, 200),
        latent_dim=256,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
        use_attention=True,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        input_channels = input_shape[0]

        # 时间嵌入
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 编码器
        self.encoder_in_conv = nn.Conv2d(input_channels, base_channels,
                                          kernel_size=3, padding=1)

        self.encoder_down_blocks = nn.ModuleList()
        current_channels = base_channels

        for mult in channel_mults:
            out_channels = base_channels * mult
            self.encoder_down_blocks.append(
                ResidualBlock(current_channels, out_channels, time_emb_dim)
            )
            current_channels = out_channels

            if use_attention and mult >= 4:
                self.encoder_down_blocks.append(
                    AttentionBlock(current_channels)
                )

            if mult != channel_mults[-1]:
                self.encoder_down_blocks.append(nn.Sequential(
                    nn.GroupNorm(8, current_channels),
                    nn.SiLU(),
                    nn.Conv2d(current_channels, current_channels,
                              kernel_size=4, stride=2, padding=1)
                ))

        final_h = input_shape[1] // (2 ** (len(channel_mults) - 1))
        final_w = input_shape[2] // (2 ** (len(channel_mults) - 1))

        self.encoder_out = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(current_channels * final_h * final_w, latent_dim * 2)
        )

        # 解码器
        decoder_init_h = output_shape[0] // (2 ** (len(channel_mults) - 1))
        decoder_init_w = output_shape[1] // (2 ** (len(channel_mults) - 1))

        self.decoder_in = nn.Linear(
            latent_dim,
            current_channels * decoder_init_h * decoder_init_w
        )
        self.decoder_init_size = (decoder_init_h, decoder_init_w)

        self.decoder_up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult
            self.decoder_up_blocks.append(
                ResidualBlock(current_channels, out_channels, time_emb_dim)
            )
            current_channels = out_channels

            if use_attention and mult >= 4:
                self.decoder_up_blocks.append(
                    AttentionBlock(current_channels)
                )

            if i < len(channel_mults) - 1:
                self.decoder_up_blocks.append(nn.Sequential(
                    nn.GroupNorm(8, current_channels),
                    nn.SiLU(),
                    nn.ConvTranspose2d(current_channels,
                                       current_channels // 2,
                                       kernel_size=4, stride=2, padding=1)
                ))
                current_channels = current_channels // 2

        self.decoder_out = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, 1, kernel_size=3, padding=1)
        )

        self.final_resize = nn.Upsample(size=output_shape, mode='bilinear',
                                          align_corners=False)

    def encode(self, x, time=None):
        if time is None:
            time = torch.zeros(x.shape[0], device=x.device)

        time_emb = self.time_embedder(time)
        time_emb = self.time_mlp(time_emb)

        h = self.encoder_in_conv(x)

        for block in self.encoder_down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:
                h = block(h)

        h = self.encoder_out(h)
        mean, log_var = torch.chunk(h, 2, dim=1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z, mean, log_var

    def decode(self, z, time=None):
        if time is None:
            time = torch.zeros(z.shape[0], device=z.device)

        time_emb = self.time_embedder(time)
        time_emb = self.time_mlp(time_emb)

        h = self.decoder_in(z)
        h = h.view(
            -1,
            h.shape[1] // (self.decoder_init_size[0] *
                            self.decoder_init_size[1]),
            self.decoder_init_size[0],
            self.decoder_init_size[1],
        )

        for block in self.decoder_up_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:
                h = block(h)

        h = self.decoder_out(h)
        output = self.final_resize(h)
        return output

    def forward(self, x, time=None, noise_level=0.0):
        z, mean, log_var = self.encode(x, time)

        if noise_level > 0:
            noise = torch.randn_like(z) * noise_level
            z = z + noise

        recon = self.decode(z, time)
        return recon, z, mean, log_var
