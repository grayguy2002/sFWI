# coding=utf-8
"""Training NCSN++ on Marmousi with VE SDE."""
from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    training.batch_size = 64  # Marmousi专用批次大小
    training.n_iters = 5001  # 训练迭代次数
    training.log_freq = 50
    training.eval_freq = 100
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 10000

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # model - 使用与CIFAR10相同的架构
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    # data - Marmousi specific
    data = config.data
    data.dataset = 'Velocity'
    data.image_size = 32  # 与CIFAR10相同
    data.num_channels = 1  # 单通道速度模型
    data.centered = False
    data.uniform_dequantization = False

    # optimization
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42  # Marmousi默认种子
    config.device = 'cuda' if config.device == 'cuda:0' else config.device

    return config
