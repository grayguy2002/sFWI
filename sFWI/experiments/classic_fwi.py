"""
经典多频 FWI 实验脚本

来源: conditional_ncsn++_daps_fwi_with_conventionalfwi.py (classicFWI multi-frequency 部分)
改造：使用模块化的 sFWI 包，使用 L-BFGS-B 优化器，SEAM 真实模型。
"""

import sys
import os
import torch
import numpy as np
import argparse
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 添加父目录到路径以导入 sFWI 包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 必须在导入 DAPS 相关模块之前设置 score_sde_pytorch 路径
from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

# 导入 sFWI 模块
from sFWI.config import FWIConfig
from sFWI.operators.classic_operator import ClassicSeismicOperator
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.utils.file_utils import generate_timestamped_filename


# ==============================================================================
# ClassicFWI: L-BFGS-B 优化器（与源码一致）
# ==============================================================================
class ClassicFWI:
    """
    经典全波形反演类，使用 L-BFGS-B 优化器。

    与源码完全一致：使用 scipy.optimize.minimize(method='L-BFGS-B')，
    支持速度约束 bounds。
    """

    def __init__(self, operator, d_obs, initial_model, max_iter=20,
                 v_min=1500.0, v_max=4500.0):
        """
        Args:
            operator: ClassicSeismicOperator, 正演算子
            d_obs: torch.Tensor, 观测数据
            initial_model: torch.Tensor, 初始速度模型
            max_iter: int, L-BFGS-B 最大迭代次数
            v_min: float, 速度下界 (m/s)
            v_max: float, 速度上界 (m/s)
        """
        self.operator = operator
        self.d_obs = d_obs
        self.initial_model = initial_model.clone().detach()
        self.device = operator.device
        self.max_iter = max_iter
        self.v_min = v_min
        self.v_max = v_max
        self.initial_params = self.initial_model.cpu().numpy().flatten()
        self.loss_history = []
        self.iter_count = 0

    def _objective_function(self, params):
        """L-BFGS-B 目标函数：返回 (loss, gradient)"""
        model = torch.from_numpy(params).float().to(self.device).reshape(
            self.operator.model_shape
        )
        model.requires_grad_()
        d_syn = self.operator.forward(model)
        loss = 0.5 * torch.sum((d_syn - self.d_obs) ** 2)
        loss.backward()
        grad = model.grad.cpu().numpy().flatten()
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        print(f"  Iteration: {self.iter_count:3d}, Misfit: {loss_val:.4e}")
        self.iter_count += 1
        return loss_val, grad

    def run(self):
        """运行 L-BFGS-B 反演，返回反演结果模型"""
        print(f"-> Starting L-BFGS-B optimization for f_peak={self.operator.f_peak} Hz...")
        self.iter_count = 0
        bounds = [(self.v_min, self.v_max) for _ in range(len(self.initial_params))]
        result = scipy.optimize.minimize(
            fun=self._objective_function, x0=self.initial_params,
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': self.max_iter, 'disp': True}
        )
        inverted_model = torch.from_numpy(result.x).float().reshape(
            self.operator.model_shape
        )
        print(f"-> Optimization finished for f_peak={self.operator.f_peak} Hz.")
        return inverted_model.to(self.device)


# ==============================================================================
# 可视化函数（与源码一致）
# ==============================================================================
def plot_results(models, titles, vmin, vmax, filename=None):
    """
    在一张图上展示整个反演过程。

    Args:
        models: list of torch.Tensor, 模型列表
        titles: list of str, 标题列表
        vmin: float, 颜色映射最小值
        vmax: float, 颜色映射最大值
        filename: str, 保存路径（可选，支持 PDF）
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (model, title) in enumerate(zip(models, titles)):
        im = axes[i].imshow(model.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel("X (m)")
        if i == 0:
            axes[i].set_ylabel("Z (m)")

    fig.subplots_adjust(right=0.9)
    cbar = fig.colorbar(im)
    cbar.set_label('Velocity (m/s)')

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Plot successfully saved to: {filename}")

    plt.show()


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    """经典多频 FWI 实验主函数"""

    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description='经典多频FWI实验脚本')
    parser.add_argument('--seed_index', type=int, default=40,
                        help='SEAM 数据集中的 GT 样本索引，默认: 40')
    parser.add_argument('--frequencies', type=float, nargs='+',
                        default=[5.0, 10.0, 15.0, 25.0],
                        help='多频反演的频率列表 (Hz)，默认: 5 10 15 25')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='每个频率阶段的 L-BFGS-B 最大迭代次数，默认: 20')
    parser.add_argument('--n_shots', type=int, default=16,
                        help='炮数，默认: 16')
    parser.add_argument('--n_receivers_per_shot', type=int, default=200,
                        help='每炮接收器数，默认: 200')
    parser.add_argument('--nt', type=int, default=500,
                        help='时间步数，默认: 500')
    parser.add_argument('--gaussian_sigma', type=float, default=10.0,
                        help='初始模型高斯模糊的 sigma，默认: 10.0')
    parser.add_argument('--v_min', type=float, default=1500.0,
                        help='速度下界 (m/s)，默认: 1500')
    parser.add_argument('--v_max', type=float, default=4500.0,
                        help='速度上界 (m/s)，默认: 4500')
    parser.add_argument('--image_size', type=int, default=32,
                        help='低分辨率图像尺寸，默认: 32')
    parser.add_argument('--high_res', type=int, default=200,
                        help='高分辨率模型尺寸，默认: 200')

    try:
        if any('ipykernel' in arg or 'jupyter' in arg for arg in sys.argv):
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
    except:
        args = parser.parse_args([])

    print("=" * 60)
    print("经典多频 FWI 实验")
    print("=" * 60)

    # 1. 基础配置
    cfg = FWIConfig()
    device = torch.device(cfg.device)
    HIGH_RES_SHAPE = (args.high_res, args.high_res)

    fwi_config = {
        'device': device,
        'model_shape': HIGH_RES_SHAPE,
        'dx': 2.0,
        'dt': 0.002,
        'nt': args.nt,
        'n_shots': args.n_shots,
        'n_receivers_per_shot': args.n_receivers_per_shot,
        'src_depth': 2,
        'rec_depth': 2,
    }

    print(f"\n配置信息:")
    print(f"  设备: {device}")
    print(f"  高分辨率模型尺寸: {HIGH_RES_SHAPE}")
    print(f"  低分辨率图像尺寸: {args.image_size}")
    print(f"  频率列表: {args.frequencies}")
    print(f"  每频迭代次数: {args.max_iter}")
    print(f"  炮数: {args.n_shots}, 接收器: {args.n_receivers_per_shot}")
    print(f"  速度约束: [{args.v_min}, {args.v_max}] m/s")
    print(f"  初始模型高斯 sigma: {args.gaussian_sigma}")
    print(f"  Seed index: {args.seed_index}")

    # 2. 加载 SEAM 数据并准备真实模型（与源码一致的流程）
    print(f"\n--- 正在适配 sFWI 的数据 ---")
    v_torch_seam = load_seam_model(cfg.paths.seam_model_path)
    sde_data = create_velocity_dataset(v_torch_seam, image_size=args.image_size)

    # 获取归一化的低分辨率样本
    gt_low_res_norm = sde_data.get_data(size=1, sigma=0, seed=args.seed_index)

    # 反归一化得到物理速度值
    gt_low_res_denorm = sde_data.denormalize(gt_low_res_norm)

    # 插值到高分辨率
    model_true_4d = F.interpolate(
        gt_low_res_denorm,
        size=HIGH_RES_SHAPE,
        mode='bilinear',
        align_corners=True
    )
    true_model = model_true_4d[0, 0].to(device)
    print(f"✓ 真实模型已加载 (seed={args.seed_index}), 形状: {true_model.shape}")

    # 3. 准备初始模型：对真实模型做高斯模糊
    initial_model = torch.from_numpy(
        scipy.ndimage.gaussian_filter(true_model.cpu().numpy(), sigma=args.gaussian_sigma)
    ).to(device)
    print(f"✓ 初始模型已创建 (高斯模糊 sigma={args.gaussian_sigma})")

    # 4. 多频 FWI 主循环
    print(f"\n" + "=" * 60)
    print("开始多频 FWI 反演")
    print("=" * 60)

    current_model = initial_model
    inversion_results = [initial_model.cpu()]
    result_titles = ["Initial Model"]

    for f_peak in args.frequencies:
        print(f"\n========================================================")
        print(f"Processing Frequency Stage: {f_peak} Hz")
        print(f"========================================================")

        forward_operator = ClassicSeismicOperator(fwi_config, f_peak=f_peak)

        print("Generating observed data for this stage...")
        with torch.no_grad():
            observed_data = forward_operator.forward(true_model)
        print("Done.")

        fwi_solver = ClassicFWI(
            operator=forward_operator,
            d_obs=observed_data,
            initial_model=current_model,
            max_iter=args.max_iter,
            v_min=args.v_min,
            v_max=args.v_max
        )
        inverted_model = fwi_solver.run()
        current_model = inverted_model

        inversion_results.append(inverted_model.cpu())
        result_titles.append(f"Result {f_peak} Hz")

        # 每个频率阶段的临时可视化
        plot_results(
            [initial_model.cpu(), inverted_model.cpu(), true_model.cpu()],
            ["Initial Model", f"Current Result ({f_peak} Hz)", "Ground Truth"],
            vmin=args.v_min, vmax=args.v_max
        )

    # 5. 最终可视化
    print(f"\nMulti-Scale FWI Finished. Displaying final results...")
    inversion_results.append(true_model.cpu())
    result_titles.append("Ground Truth")
    vmin_val, vmax_val = true_model.min().item(), true_model.max().item()

    output_filename = generate_timestamped_filename(
        f'fwi_final_results_model{args.seed_index}', '.pdf'
    )
    output_path = os.path.join(cfg.paths.f_path, output_filename)
    plot_results(inversion_results, result_titles,
                 vmin=vmin_val, vmax=vmax_val, filename=output_path)

    print(f"\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
