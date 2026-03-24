"""
RSS (Random Seed Sampling) 和 DMS (Distance Matrix Similarity) 实验脚本

来源: exp_rss_dms.py
改造：使用模块化的 sFWI 包，消除 exec() 和全局变量依赖。
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# 添加父目录到路径以导入 sFWI 包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 必须在导入 DAPS 相关模块之前设置 score_sde_pytorch 路径
from sFWI.models.sde_setup import setup_score_sde_path
setup_score_sde_path(parent_dir)

# 导入 sFWI 模块
from sFWI.config import FWIConfig, build_daps_configs
from sFWI.models.sde_setup import create_sde_config
from sFWI.models.score_model import NCSNpp_DAPS
from sFWI.data.daps_adapter import create_velocity_dataset
from sFWI.data.loaders import load_seam_model
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.evaluation.wasserstein import Wasserstein, Wasserstein_us
from sFWI.utils.file_utils import generate_timestamped_filename, generate_timestamped_path

# 导入 DAPS 相关模块
from DAPS.sampler import get_sampler, DAPS
from DAPS.eval import Evaluator


def main():
    """RSS 和 DMS 实验主函数"""

    print("=" * 60)
    print("RSS 和 DMS 实验")
    print("=" * 60)

    # 1. 创建配置
    cfg = FWIConfig()
    cfg.daps.batch_size = 1
    cfg.daps.sigma = 0.3

    print(f"\n配置信息:")
    print(f"  设备: {cfg.device}")
    print(f"  图像尺寸: {cfg.image_size}")
    print(f"  DAPS sigma: {cfg.daps.sigma}")

    # 2. 设置 Score SDE 路径
    code_dir = cfg.paths.code_dir
    setup_score_sde_path(code_dir)
    print(f"\n✓ Score SDE 路径已设置")

    # 3. 创建 SDE 配置
    config, sde = create_sde_config(code_dir, batch_size=cfg.daps.batch_size)
    print(f"✓ SDE 配置已创建")

    # 4. 构建 DAPS 配置
    base_config, lgvd_config = build_daps_configs(cfg)
    print(f"✓ DAPS 配置已构建")

    # 5. 加载数据
    print(f"\n加载数据...")
    v_torch_seam = load_seam_model(cfg.paths.seam_model_path)
    data = create_velocity_dataset(v_torch_seam, image_size=cfg.image_size)
    print(f"✓ 数据集已加载: {len(data)} 个样本")

    # 6. 创建模型
    print(f"\n创建模型...")
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=cfg.paths.checkpoint_path
    )
    model.set_device(cfg.device)
    print(f"✓ 模型已创建并加载到 {cfg.device}")

    # 7. 初始化算子和评估器
    print(f"\n初始化算子和评估器...")
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
    eval_fn = Wasserstein(operator)
    eval_us_fn = Wasserstein_us(operator)
    evaluator = Evaluator((eval_fn,))
    evaluator_us = Evaluator((eval_us_fn,))
    print(f"✓ 算子和评估器已初始化")

    # 8. RSS 和 DMS 实验参数
    print(f"\n" + "="*60)
    print("RSS 和 DMS 实验")
    print("="*60)

    MASTER_SEED = 8
    NUM_GT_SAMPLES = 10  # i 参数：GT样本数量
    NUM_SAMPLING_SEEDS = 50  # j 参数：每个GT的采样种子数量

    print(f"\n实验参数:")
    print(f"  MASTER_SEED: {MASTER_SEED}")
    print(f"  GT样本数量 (i): {NUM_GT_SAMPLES}")
    print(f"  采样种子数量 (j): {NUM_SAMPLING_SEEDS}")

    # 9. 构建相似度矩阵
    print(f"\n构建相似度矩阵...")
    similarity_matrix = torch.zeros((NUM_GT_SAMPLES, NUM_SAMPLING_SEEDS), device=cfg.device)

    seed_4_SimilarityMatrix = range(NUM_SAMPLING_SEEDS)

    for gt_idx in tqdm(range(NUM_GT_SAMPLES), desc="处理GT样本"):
        # 获取当前GT
        gt = data.get_data(cfg.daps.batch_size, 0, seed=gt_idx)
        measurement = operator(gt.to(cfg.device))

        # 对当前GT进行多次采样
        w_distances = model.daps.sample(
            model,
            model.daps.get_start(gt.to(cfg.device)),
            operator,
            measurement,
            evaluator_us,
            evaluator,
            record=False,
            verbose=False,
            seed=seed_4_SimilarityMatrix,
            gt=gt
        )

        # 存储到相似度矩阵
        similarity_matrix[gt_idx] = w_distances

    print(f"✓ 相似度矩阵构建完成: {similarity_matrix.shape}")

    # 10. 保存相似度矩阵
    output_filename = generate_timestamped_filename(
        f'SEAM_similarity_matrix_i{NUM_GT_SAMPLES}_j{NUM_SAMPLING_SEEDS}_master_seed_{MASTER_SEED}',
        '.pt'
    )
    output_path = os.path.join(cfg.paths.f_path, output_filename)
    torch.save(similarity_matrix, output_path)
    print(f"✓ 相似度矩阵已保存: {output_filename}")

    # 11. 打印统计信息
    print(f"\n相似度矩阵统计:")
    print(f"  形状: {similarity_matrix.shape}")
    print(f"  最小值: {similarity_matrix.min().item():.4f}")
    print(f"  最大值: {similarity_matrix.max().item():.4f}")
    print(f"  平均值: {similarity_matrix.mean().item():.4f}")

    print(f"\n" + "="*60)
    print("实验完成")
    print("="*60)


if __name__ == '__main__':
    main()

