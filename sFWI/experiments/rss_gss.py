"""
RSS (Random Score Search) 和 GSS (Group Score Search) 实验脚本

功能:
  Phase 1 (GSS): 对训练域速度模型 patch 进行聚类，得到 group 和 centroid
  Phase 2 (GSS): 以 centroid 为基准构建相似度矩阵 M[i,j] = W2(F(centroid_i), F(sample_j))
  Phase 3 (GSS): 给定观测数据，匹配最优 group，在组内执行 RSS（待后续实现）
  RSS: 从相似度矩阵中选取最优 seed + 复现验证 + MSE 排序可视化
"""

import sys
import os
import torch
import numpy as np
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
from sFWI.data.marmousi_loader import load_marmousi_from_pkl
from sFWI.operators.daps_operator import DAPSSeismicOperator
from sFWI.evaluation.wasserstein import Wasserstein, Wasserstein_us
from sFWI.utils.file_utils import generate_timestamped_filename
from sFWI.utils.visualization import visualize_data
from sFWI.utils.clustering import (
    cluster_velocity_patches, get_centroids,
    save_clustering_results, load_clustering_results,
    visualize_clustering
)

# 导入 DAPS 相关模块
from DAPS.sampler import get_sampler, DAPS
from DAPS.eval import Evaluator


def _resolve_checkpoint_path(cfg, args):
    """解析并校验模型checkpoint路径。"""
    default_by_tag = {
        'seam': 'checkpoint_5.pth',
        'seam_finetune': 'seam_finetune_checkpoint_5.pth',
        'marmousi': 'marmousi_checkpoint_5.pth',
    }
    ckpt_dir = args.ckpt_dir or os.path.join(cfg.paths.project_root, 'checkpoints')

    if args.ckpt_file:
        if os.path.isabs(args.ckpt_file):
            ckpt_path = args.ckpt_file
        else:
            ckpt_path = os.path.join(ckpt_dir, args.ckpt_file)
    else:
        ckpt_path = os.path.join(ckpt_dir, default_by_tag[args.model_tag])

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint不存在: {ckpt_path}\n"
            f"请检查 --ckpt_dir / --ckpt_file / --model_tag 参数。"
        )
    return ckpt_path


def _get_asset_dir(cfg, args):
    """获取GSS资产输出目录。"""
    asset_dir = args.asset_dir or os.path.join(cfg.paths.project_root, 'gss_assets')
    os.makedirs(asset_dir, exist_ok=True)
    return asset_dir


def _load_training_velocity_patches(cfg, args):
    """根据模型标签加载用于GSS构建的训练域速度patch。"""
    if args.model_tag in ('seam', 'seam_finetune'):
        if args.model_tag == 'seam_finetune':
            print(f"\n加载 SEAM fine-tune 训练域数据...")
        else:
            print(f"\n加载 SEAM 训练域数据...")
        v_torch = load_seam_model(cfg.paths.seam_model_path)
        dataset_tag = 'seam'
    else:
        marmousi_path = args.marmousi_data_path or cfg.paths.marmousi_dataset_path
        print(f"\n加载 Marmousi 训练域数据: {marmousi_path}")
        v_torch, _ = load_marmousi_from_pkl(marmousi_path)
        dataset_tag = 'marmousi'

    # 对超大数据集做可复现子采样，避免 O(N^2) 的SSIM矩阵过大
    original_n = int(v_torch.shape[0])
    if args.cluster_max_samples > 0 and original_n > args.cluster_max_samples:
        generator = torch.Generator()
        generator.manual_seed(args.master_seed)
        selected_indices = torch.randperm(
            v_torch.shape[0], generator=generator
        )[:args.cluster_max_samples]
        v_torch = v_torch[selected_indices]
        print(
            f"  [subset] 从 {original_n} 个样本中"
            f" 子采样到 {int(args.cluster_max_samples)} 个用于聚类/centroid构建"
        )

    if isinstance(v_torch, torch.Tensor):
        v_torch = v_torch.detach().to(device='cpu', dtype=torch.float32).contiguous()
    else:
        v_torch = torch.tensor(v_torch, dtype=torch.float32)

    print(f"✓ 训练域速度patch已加载: {tuple(v_torch.shape)}")
    return dataset_tag, v_torch


def main():
    """RSS 和 GSS 实验主函数"""

    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description='RSS/GSS 实验脚本')

    # 模型与路径参数
    parser.add_argument('--model_tag', type=str, default='seam',
                        choices=['seam', 'seam_finetune', 'marmousi'],
                        help='当前加载模型标签，用于checkpoint解析与资产命名')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='checkpoint目录（默认: {project_root}/checkpoints）')
    parser.add_argument('--ckpt_file', type=str, default=None,
                        help='checkpoint文件名或绝对路径（优先级高于model_tag默认映射）')
    parser.add_argument('--asset_dir', type=str, default=None,
                        help='GSS资产输出目录（默认: {project_root}/gss_assets）')
    parser.add_argument('--marmousi_data_path', type=str, default=None,
                        help='Marmousi数据路径(.pkl)，仅 model_tag=marmousi 时使用')
    parser.add_argument('--cluster_max_samples', type=int, default=1000,
                        help='聚类最大样本数（<=0表示不限制；默认1000）')

    # GSS 相似度矩阵参数
    parser.add_argument('--master_seed', type=int, default=8,
                        help='主种子，控制实验可复现性，默认: 8')
    parser.add_argument('--n_noise_seeds', type=int, default=50,
                        help='S_noise 数量（矩阵列数），默认: 50')
    parser.add_argument('--sigma', type=float, default=0.3,
                        help='DAPS 噪声水平，默认: 0.3')

    # 聚类参数
    parser.add_argument('--k_min', type=int, default=5,
                        help='聚类 k 值搜索下界，默认: 5')
    parser.add_argument('--k_max', type=int, default=20,
                        help='聚类 k 值搜索上界，默认: 20')
    parser.add_argument('--k', type=int, default=None,
                        help='直接指定聚类 k 值（跳过自动选择）')

    # RSS 选择参数
    parser.add_argument('--top_k', type=int, default=50,
                        help='RSS 从矩阵中选取的候选数量，默认: 50')
    parser.add_argument('--target_group_index', type=int, default=0,
                        help='要复现的目标 group 索引，默认: 0')

    # Direct match 模式
    parser.add_argument('--direct_match', action='store_true',
                        help='直接在数据域匹配 d_obs 与 F(centroid)，跳过相似度矩阵构建')
    parser.add_argument('--d_obs_index', type=int, default=None,
                        help='direct_match 模式下用作 d_obs 的 patch 索引（测试用）')

    # 控制流
    parser.add_argument('--skip_clustering', action='store_true',
                        help='跳过聚类（加载已有聚类结果）')
    parser.add_argument('--skip_cluster_vis', action='store_true',
                        help='跳过聚类可视化（仅保存聚类结果）')
    parser.add_argument('--clustering_only', action='store_true',
                        help='只做聚类，不构建矩阵（方便先看聚类结果）')
    parser.add_argument('--skip_gss', action='store_true',
                        help='跳过 GSS 矩阵构建（直接加载已有矩阵）')
    parser.add_argument('--skip_rss', action='store_true',
                        help='跳过 RSS 选择和复现验证')

    try:
        if any('ipykernel' in arg or 'jupyter' in arg for arg in sys.argv):
            args = parser.parse_args([])
        else:
            args = parser.parse_args()
    except:
        args = parser.parse_args([])

    print("=" * 60)
    print("GSS (Group Score Search) 实验 — 论文 Algorithm 2")
    print("=" * 60)

    print(f"\n实验参数:")
    print(f"  MASTER_SEED: {args.master_seed}")
    print(f"  S_noise 数量: {args.n_noise_seeds}")
    print(f"  DAPS sigma: {args.sigma}")
    print(f"  聚类 k 范围: [{args.k_min}, {args.k_max}]")
    print(f"  RSS top-k: {args.top_k}")
    print(f"  model_tag: {args.model_tag}")
    print(f"  cluster_max_samples: {args.cluster_max_samples}")
    print(f"  skip_cluster_vis: {args.skip_cluster_vis}")

    # 1. 创建配置
    cfg = FWIConfig()
    cfg.daps.batch_size = 1
    cfg.daps.sigma = args.sigma
    device = cfg.device
    checkpoint_path = _resolve_checkpoint_path(cfg, args)
    output_dir = _get_asset_dir(cfg, args)

    print(f"\n配置信息:")
    print(f"  设备: {device}")
    print(f"  图像尺寸: {cfg.image_size}")
    print(f"  模型标签: {args.model_tag}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  资产目录: {output_dir}")

    # 2. 加载训练域原始数据（聚类在原始 200x200 空间进行）
    dataset_tag, v_torch_train = _load_training_velocity_patches(cfg, args)

    # =================================================================
    # Phase 1: 聚类
    # =================================================================
    cluster_limit_tag = 'all' if args.cluster_max_samples <= 0 else str(args.cluster_max_samples)
    clustering_path = os.path.join(
        output_dir, f'clustering_dataset-{dataset_tag}_max{cluster_limit_tag}.pt'
    )
    clustering_vis_path = os.path.join(
        output_dir, f'clustering_dataset-{dataset_tag}_max{cluster_limit_tag}.pdf'
    )

    if not args.skip_clustering:
        print(f"\n" + "=" * 60)
        print(f"Phase 1: 对 {dataset_tag.upper()} patch 进行 SSIM + K-medoids 聚类")
        print("=" * 60)

        best_k, labels, info = cluster_velocity_patches(
            v_torch_train, k_range=(args.k_min, args.k_max),
            forced_k=args.k
        )
        centroid_indices, group_members = get_centroids(
            v_torch_train, labels, best_k, info=info
        )

        # 保存聚类结果
        save_clustering_results(
            clustering_path, best_k, labels,
            centroid_indices, group_members, info
        )

        # 可视化
        if args.skip_cluster_vis:
            print(f"跳过聚类可视化 (--skip_cluster_vis): {clustering_vis_path}")
        else:
            visualize_clustering(
                v_torch_train, labels, centroid_indices, info,
                save_path=clustering_vis_path
            )
    else:
        print(f"\n跳过聚类，加载已有结果: {clustering_path}")
        if not os.path.exists(clustering_path):
            print(f"错误: 聚类文件不存在 - {clustering_path}")
            return
        cl_results = load_clustering_results(clustering_path)
        best_k = cl_results['k']
        labels = cl_results['labels']
        centroid_indices = cl_results['centroid_indices']
        group_members = cl_results['group_members']
        info = cl_results['info']

    print(f"\n聚类结果: k={best_k}, centroid patch 索引={centroid_indices}")

    # 创建训练域数据集（后续 direct match / Phase2 使用）
    data = create_velocity_dataset(v_torch_train, image_size=cfg.image_size)
    print(f"✓ 训练域数据集已创建: {len(data)} 个样本")

    if args.clustering_only:
        print("\n--clustering_only 模式，跳过后续步骤。")
        return

    # =================================================================
    # Direct Match 模式: 在数据域直接匹配 d_obs 与 F(centroid)
    # 跳过 score model 加载和相似度矩阵构建
    # =================================================================
    if args.direct_match:
        print(f"\n" + "=" * 60)
        print("Direct Match 模式: d_obs vs F(centroid_i)")
        print("=" * 60)

        # 只需 SDE config（用于 operator 初始化），不加载 score model
        code_dir = cfg.paths.code_dir
        setup_score_sde_path(code_dir)
        config, sde = create_sde_config(code_dir, batch_size=1)
        operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
        print(f"✓ 正演算子已初始化（未加载 score model）")

        # 构造 d_obs
        if args.d_obs_index is not None:
            obs_patch = data[args.d_obs_index].unsqueeze(0).to(device)
            print(f"  d_obs 来源: patch[{args.d_obs_index}]")
        else:
            # 默认使用第一个 centroid 作为测试
            obs_patch = data[centroid_indices[0]].unsqueeze(0).to(device)
            print(f"  d_obs 来源: centroid patch[{centroid_indices[0]}]（默认）")

        with torch.no_grad():
            d_obs = operator(obs_patch)

        # 对每个 centroid 正演并计算距离
        distances = torch.zeros(best_k, device=device)
        for i in range(best_k):
            centroid_patch = data[centroid_indices[i]].unsqueeze(0).to(device)
            with torch.no_grad():
                d_centroid = operator(centroid_patch)
                distances[i] = torch.norm(d_obs - d_centroid).item()

        # 排序输出
        sorted_indices = torch.argsort(distances)
        print(f"\n--- Direct Match 结果（按 L2 距离升序）---")
        for rank, gi in enumerate(sorted_indices):
            print(f"  Rank {rank}: Group {gi.item()}, "
                  f"centroid=patch[{centroid_indices[gi.item()]}], "
                  f"distance={distances[gi].item():.6f}")

        best_group = sorted_indices[0].item()
        print(f"\n✓ 最优 group: {best_group} "
              f"(centroid=patch[{centroid_indices[best_group]}], "
              f"distance={distances[sorted_indices[0]].item():.6f})")

        # 保存结果
        dm_filename = (
            f'direct_match_dataset-{dataset_tag}_model-{args.model_tag}_k{best_k}.pt'
        )
        dm_path = os.path.join(output_dir, dm_filename)
        torch.save({
            'distances': distances.cpu(),
            'sorted_indices': sorted_indices.cpu(),
            'best_group': best_group,
            'centroid_indices': centroid_indices,
            'd_obs_index': args.d_obs_index,
            'model_tag': args.model_tag,
            'checkpoint_path': checkpoint_path,
        }, dm_path)
        print(f"✓ 结果已保存: {dm_path}")

        print(f"\n" + "=" * 60)
        print("Direct Match 完成")
        print("=" * 60)
        return

    # =================================================================
    # 3. 设置 Score SDE 路径和配置 + 创建模型
    # =================================================================
    code_dir = cfg.paths.code_dir
    setup_score_sde_path(code_dir)
    config, sde = create_sde_config(code_dir, batch_size=cfg.daps.batch_size)
    base_config, lgvd_config = build_daps_configs(cfg)
    print(f"✓ SDE / DAPS 配置已创建")

    print(f"\n创建模型...")
    model = NCSNpp_DAPS(
        model_config=config,
        base_config=base_config,
        lgvd_config=lgvd_config,
        checkpoint_path=checkpoint_path
    )
    model.set_device(device)
    print(f"✓ 模型已创建并加载到 {device}")

    # 4. 初始化算子和评估器
    print(f"\n初始化算子和评估器...")
    operator = DAPSSeismicOperator(config, image_size=200, sigma=cfg.daps.sigma)
    eval_fn = Wasserstein(operator)
    eval_us_fn = Wasserstein_us(operator)
    evaluator = Evaluator((eval_fn,))
    evaluator_us = Evaluator((eval_us_fn,))
    print(f"✓ 算子和评估器已初始化")

    # =================================================================
    # Phase 2: 构建相似度矩阵
    # =================================================================
    sm_filename = (
        f'sm_dataset-{dataset_tag}_model-{args.model_tag}'
        f'_k{best_k}_j{args.n_noise_seeds}_seed{args.master_seed}.pt'
    )
    sm_path = os.path.join(output_dir, sm_filename)

    if not args.skip_gss:
        print(f"\n" + "=" * 60)
        print("Phase 2: 构建相似度矩阵 M[i,j] = dist(F(centroid_i), F(sample_j))")
        print("=" * 60)
        print(f"  矩阵维度: [{best_k} groups] x [{args.n_noise_seeds} samples]")
        print(f"  MASTER_SEED: {args.master_seed}")
        print(f"  策略: 一次性采样 {args.n_noise_seeds} 个无条件样本，复用于所有 centroid")

        # --- Step 1: 一次性无条件采样 n_noise_seeds 个速度模型 ---
        print(f"\nStep 1: 无条件采样 {args.n_noise_seeds} 个速度模型...")
        torch.manual_seed(args.master_seed)
        torch.cuda.manual_seed_all(args.master_seed)

        ref_input = data[0].unsqueeze(0).to(device)  # [1, 1, 32, 32] 仅用于获取 shape
        from score_sde_pytorch import sampling as sde_sampling
        _, C, H, W = ref_input.shape
        sampling_fn = sde_sampling.get_pc_sampler(
            sde=model.daps.sde,
            shape=(args.n_noise_seeds, C, H, W),
            predictor=sde_sampling.ReverseDiffusionPredictor,
            corrector=sde_sampling.LangevinCorrector,
            inverse_scaler=model.daps.inverse_scaler,
            snr=0.16,
            n_steps=1,
            probability_flow=False,
            continuous=True,
            eps=model.daps.sampling_eps,
            device=device,
            seed=None,
        )
        x0hat_batch, _ = sampling_fn(model)
        print(f"✓ 已生成 {x0hat_batch.shape[0]} 个无条件样本: {x0hat_batch.shape}")

        # --- Step 2: 对所有样本做一次正演 ---
        print(f"\nStep 2: 对 {args.n_noise_seeds} 个样本做正演...")
        x0hat_high = F.interpolate(
            x0hat_batch, size=(128, 128),
            mode='bilinear', align_corners=True
        )
        with torch.no_grad():
            d_samples = operator(x0hat_high)  # [n_noise_seeds, 1, 128, 128]
        d_samples_2d = d_samples.reshape(args.n_noise_seeds, -1)  # [N, D]
        print(f"✓ 样本正演完成: {d_samples.shape}")

        # --- Step 3: 对 k 个 centroid 做正演 ---
        print(f"\nStep 3: 对 {best_k} 个 centroid 做正演...")
        d_centroids_list = []
        for i in range(best_k):
            centroid_patch = data[centroid_indices[i]].unsqueeze(0).to(device)
            with torch.no_grad():
                d_centroids_list.append(operator(centroid_patch))
        d_centroids = torch.cat(d_centroids_list, dim=0)  # [k, 1, 128, 128]
        d_centroids_2d = d_centroids.reshape(best_k, -1)  # [k, D]
        print(f"✓ Centroid 正演完成: {d_centroids.shape}")

        # --- Step 4: 计算距离矩阵 M[i,j] = ||F(centroid_i) - F(sample_j)|| ---
        print(f"\nStep 4: 计算距离矩阵...")
        similarity_matrix = torch.cdist(d_centroids_2d, d_samples_2d, p=2)  # [k, N]
        print(f"✓ 距离矩阵计算完成: {similarity_matrix.shape}")

        # 保存相似度矩阵及元信息
        sm_data = {
            'similarity_matrix': similarity_matrix.cpu(),
            'k': best_k,
            'centroid_indices': centroid_indices,
            'n_noise_seeds': args.n_noise_seeds,
            'master_seed': args.master_seed,
            'x0hat_batch': x0hat_batch.cpu(),
            'd_samples_2d': d_samples_2d.cpu(),
            'd_centroids_2d': d_centroids_2d.cpu(),
            'dataset_tag': dataset_tag,
            'model_tag': args.model_tag,
            'checkpoint_path': checkpoint_path,
            'sigma': cfg.daps.sigma,
            'image_size': cfg.image_size,
        }
        torch.save(sm_data, sm_path)
        print(f"\n✓ 相似度矩阵已保存: {sm_path}")
        print(f"  矩阵形状: {similarity_matrix.shape}")
        torch.set_printoptions(precision=1, sci_mode=False)
        print(f"  相似度矩阵:\n{similarity_matrix}")
    else:
        print(f"\n跳过 GSS 矩阵构建，加载已有矩阵: {sm_filename}")
        if not os.path.exists(sm_path):
            print(f"错误: 文件不存在 - {sm_path}")
            return
        sm_data = torch.load(sm_path, weights_only=False)
        similarity_matrix = sm_data['similarity_matrix'].to(device)
        x0hat_batch = sm_data['x0hat_batch'].to(device)
        print(f"✓ 已加载矩阵: {similarity_matrix.shape}")
        if 'model_tag' in sm_data:
            print(f"  资产模型标签: {sm_data['model_tag']}")
        if 'checkpoint_path' in sm_data:
            print(f"  资产checkpoint: {sm_data['checkpoint_path']}")

    # =================================================================
    # RSS: 从相似度矩阵中选取最优 seed
    # =================================================================
    if not args.skip_rss:
        print(f"\n" + "=" * 60)
        print("RSS: 从相似度矩阵中选取最优 noise seed")
        print("=" * 60)

        SM = similarity_matrix
        torch.set_printoptions(precision=4)
        print(f"Similarity Matrix shape: {SM.shape}")
        print(f"Similarity Matrix:\n{SM}")

        # 取每行 top-k 最小距离及对应 sample 索引
        candidate = min(args.top_k, SM.shape[1])
        W_4_RandomSeed, seed_4_RandomSeed = torch.topk(
            SM, candidate, dim=1, largest=False
        )
        print(f"\nW_4_RandomSeed:\n{W_4_RandomSeed}")
        print(f"seed_4_RandomSeed:\n{seed_4_RandomSeed}")

        # =============================================================
        # MSE 排序可视化: 直接使用已生成的 x0hat_batch
        # =============================================================
        TARGET = args.target_group_index
        if TARGET >= best_k:
            print(f"\n警告: target_group_index ({TARGET}) >= k ({best_k})，"
                  f"跳过可视化")
        else:
            print(f"\n" + "-" * 40)
            print(f"Group {TARGET} 候选分析, "
                  f"centroid=patch[{centroid_indices[TARGET]}]")
            print("-" * 40)

            centroid_patch = data[centroid_indices[TARGET]]
            centroid_input = centroid_patch.unsqueeze(0).to(device)

            top_indices = seed_4_RandomSeed[TARGET]
            x0hat_of_group = x0hat_batch[top_indices]

            mse_results = []
            gt_repr = centroid_input.detach().cpu()
            for top_i in range(top_indices.shape[0]):
                tensor_a = x0hat_of_group[top_i][None,].detach().cpu()
                mse_val = F.mse_loss(tensor_a, gt_repr).item()
                mse_results.append((mse_val, top_i))

            mse_results.sort(key=lambda x: x[0])

            print(f"\n--- 按 MSE 升序排列的候选结果 ---")
            for mse_val, orig_idx in mse_results[:10]:
                print(f"  候选 {orig_idx}: MSE={mse_val:.6f}")

            # 保存最优候选的可视化
            if len(mse_results) > 0:
                best_mse, best_idx = mse_results[0]
                best_sample = x0hat_of_group[best_idx][None,].detach().cpu()

                sample_filename = (f'{dataset_tag}_xhat_group{TARGET}'
                                   f'_seed{top_indices[best_idx]}'
                                   f'_master_seed_{args.master_seed}'
                                   f'_top0.pdf')
                gt_filename = (f'{dataset_tag}_centroid_group{TARGET}'
                               f'_patch{centroid_indices[TARGET]}'
                               f'_master_seed_{args.master_seed}.pdf')

                sample_path = os.path.join(output_dir, sample_filename)
                gt_path = os.path.join(output_dir, gt_filename)

                visualize_data(best_sample, save_path=sample_path)
                visualize_data(gt_repr, save_path=gt_path)
                print(f"\n✓ 最优候选已保存: {sample_filename}")
                print(f"✓ Centroid 已保存: {gt_filename}")

    print(f"\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
