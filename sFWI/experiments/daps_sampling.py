"""
DAPS 采样实验脚本

来源: exp_daps_sampling.py
改造：使用模块化的 sFWI 包，消除 exec() 和全局变量依赖。
"""

import sys
import os
import torch
import numpy as np
import argparse
from tqdm.auto import tqdm
from collections import defaultdict

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
import ot


# 辅助函数
def compute_wd(gt1, gt2):
    """计算两个gt之间的Wasserstein距离"""
    gt1_2d = gt1.detach().cpu().reshape(gt1.shape[0], -1).numpy()
    gt2_2d = gt2.detach().cpu().reshape(gt2.shape[0], -1).numpy()
    gt1_prob = np.ones((gt1.shape[0],)) / gt1.shape[0]
    gt2_prob = np.ones((gt2.shape[0],)) / gt2.shape[0]
    dist_matrix = ot.dist(gt1_2d, gt2_2d, metric='euclidean')
    w_dist = ot.emd2(gt1_prob, gt2_prob, dist_matrix)
    return w_dist


def find_similar_gts(data_generator, batch_size, num_seeds=100, threshold=5, base_seeds=None):
    """通过计算成对的Wasserstein距离来寻找所有相似的gt组"""
    print("步骤1: 生成所有用于比较的 Ground Truths...")
    gts = {seed: data_generator.get_data(batch_size, 0, seed=seed) for seed in tqdm(range(num_seeds), desc="生成GTs")}

    similar_groups = defaultdict(lambda: ([], []))
    seeds_to_check = base_seeds if base_seeds is not None else range(num_seeds)

    total_comparisons = sum(1 for i, seed1 in enumerate(seeds_to_check) for seed2 in range(i + 1, num_seeds))

    print("步骤2: 计算成对距离以寻找相似组...")
    with tqdm(total=total_comparisons, desc="计算GT间Wasserstein距离") as pbar:
        for i, seed1 in enumerate(seeds_to_check):
            for seed2 in range(i + 1, num_seeds):
                wd = compute_wd(gts[seed1], gts[seed2])
                if wd < threshold:
                    similar_groups[seed1][0].append(seed2)
                    similar_groups[seed1][1].append(wd)
                    similar_groups[seed2][0].append(seed1)
                    similar_groups[seed2][1].append(wd)
                pbar.update(1)
    return similar_groups


def get_sorted_similar_groups(similar_groups, min_group_size=2):
    """将发现的相似对合并成完整的、不重复的组"""
    processed_seeds = set()
    sorted_groups = []
    for seed, (similar_seeds, _) in similar_groups.items():
        if seed in processed_seeds:
            continue
        current_group = {seed} | set(similar_seeds)
        while True:
            new_additions = set()
            for s in current_group:
                if s in similar_groups:
                    new_additions.update(similar_groups[s][0])
            if new_additions.issubset(current_group):
                break
            current_group.update(new_additions)

        if len(current_group) >= min_group_size:
            sorted_groups.append(sorted(list(current_group)))
            processed_seeds.update(current_group)
    sorted_groups.sort(key=lambda x: x[0])
    return sorted_groups


def main():
    """DAPS 采样实验主函数"""

    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description='DAPS采样实验脚本')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='DAPS 批次大小，默认: 1')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='DAPS 噪声水平，默认: 0.3')
    parser.add_argument('--seed_index', type=int, default=1,
                        help='Ground truth seed 索引，默认: 1')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='相似性判定的 Wasserstein 距离阈值，默认: 1.0')
    parser.add_argument('--num_seeds', type=int, default=100,
                        help='参与比较的 GT 数量，默认: 100')
    parser.add_argument('--min_group_size', type=int, default=2,
                        help='相似组的最小成员数，默认: 2')

    try:
        if any('ipykernel' in arg or 'jupyter' in arg for arg in sys.argv):
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
    except:
        args = parser.parse_args([])

    print("=" * 60)
    print("DAPS 采样实验")
    print("=" * 60)

    # 1. 创建配置
    cfg = FWIConfig()
    cfg.daps.batch_size = args.batch_size
    cfg.daps.sigma = args.sigma
    cfg.seed_index = args.seed_index

    print(f"\n配置信息:")
    print(f"  设备: {cfg.device}")
    print(f"  图像尺寸: {cfg.image_size}")
    print(f"  Seed: {cfg.seed_index}")
    print(f"  DAPS sigma: {cfg.daps.sigma}")
    print(f"  相似性阈值: {args.threshold}")
    print(f"  比较GT数量: {args.num_seeds}")
    print(f"  最小组大小: {args.min_group_size}")

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

    # 8. 获取 ground truth 和 measurement
    print(f"\n获取 ground truth...")
    gt = data.get_data(cfg.daps.batch_size, 0, seed=cfg.seed_index)
    measurement = operator(gt.to(cfg.device))
    print(f"✓ Ground truth 已获取 (seed={cfg.seed_index})")

    # 9. 寻找相似的 GT 组
    print(f"\n" + "="*60)
    print("寻找相似的 Ground Truth 组")
    print("="*60)

    THRESHOLD = args.threshold
    NUM_SEEDS = args.num_seeds

    similar_groups_dict = find_similar_gts(
        data, cfg.daps.batch_size,
        num_seeds=NUM_SEEDS,
        threshold=THRESHOLD
    )
    sorted_similar_groups = get_sorted_similar_groups(similar_groups_dict, min_group_size=args.min_group_size)

    print(f"\n=== 发现的相似GT组 ===")
    for group in sorted_similar_groups:
        print(f"相似组: {group}")

    print(f"\n" + "="*60)
    print("实验完成")
    print("="*60)


if __name__ == '__main__':
    main()

