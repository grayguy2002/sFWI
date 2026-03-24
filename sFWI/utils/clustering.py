"""
速度模型聚类工具 (SSIM + K-medoids)

用于 GSS (Group Score Search) Phase 1: 对 SEAM 速度模型 patch 进行聚类，
将大量 patch 压缩为少量代表性 group。

算法:
  1. 计算 224 个 patch 的两两 SSIM 距离矩阵 D[i,j] = 1 - SSIM(patch_i, patch_j)
  2. 基于距离矩阵做 K-medoids 聚类（medoid 即为真实 patch centroid）
  3. 层次聚类 dendrogram 辅助确定 k
  4. Silhouette score 自动选择最优 k

依赖: scipy, sklearn (核心); piq (可选, SSIM 加速)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ================================================================
#  SSIM 距离矩阵计算
# ================================================================

def _ssim_single(img1, img2, C1=1e-4, C2=9e-4):
    """计算两个 2D numpy 数组的 SSIM。

    使用全局统计量（非滑动窗口），适用于 patch 级别的结构相似性比较。
    img1, img2: np.ndarray, shape (H, W), 值域已归一化到 [0, 1]。
    """
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


def compute_ssim_distance_matrix(v_torch):
    """计算所有 patch 对的 SSIM 距离矩阵。

    参数:
        v_torch: torch.Tensor, shape (N, H, W), 原始速度 patch

    返回:
        dist_matrix: np.ndarray, shape (N, N), D[i,j] = 1 - SSIM(i, j)
    """
    if isinstance(v_torch, torch.Tensor):
        v_torch = v_torch.detach().to(device='cpu', dtype=torch.float32).contiguous()
        patches = v_torch.numpy()
    else:
        patches = np.asarray(v_torch, dtype=np.float32)

    N = patches.shape[0]

    # 归一化到 [0, 1] 以使 SSIM 的 C1, C2 常数有意义
    global_min = patches.min()
    global_max = patches.max()
    if global_max - global_min > 1e-8:
        patches_norm = (patches - global_min) / (global_max - global_min)
    else:
        patches_norm = np.zeros_like(patches)

    n_pairs = N * (N - 1) // 2
    print(f"计算 SSIM 距离矩阵: {N} 个 patch, {n_pairs} 对...")

    dist_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            ssim_val = _ssim_single(patches_norm[i], patches_norm[j])
            d = 1.0 - ssim_val
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    print(f"✓ SSIM 距离矩阵计算完成。"
          f"距离范围: [{dist_matrix[dist_matrix > 0].min():.4f}, "
          f"{dist_matrix.max():.4f}]")

    return dist_matrix


# ================================================================
#  K-medoids 聚类
# ================================================================

def _kmedoids(dist_matrix, k, max_iter=300, random_state=42):
    """基于预计算距离矩阵的 K-medoids (PAM 算法简化版)。

    参数:
        dist_matrix: np.ndarray, shape (N, N), 对称距离矩阵
        k: int, 聚类数
        max_iter: int, 最大迭代次数
        random_state: int, 随机种子

    返回:
        medoid_indices: np.ndarray, shape (k,), medoid 在原数据中的索引
        labels: np.ndarray, shape (N,), 每个样本的聚类标签
        inertia: float, 所有样本到其 medoid 的距离之和
    """
    rng = np.random.RandomState(random_state)
    N = dist_matrix.shape[0]
    if k > N:
        raise ValueError(f"k={k} 大于样本数 N={N}，无法无放回初始化 medoids。")

    # 初始化: 随机选 k 个 medoid
    medoid_indices = rng.choice(N, size=k, replace=False)

    for iteration in range(max_iter):
        # Assign: 每个点分配到最近的 medoid
        dists_to_medoids = dist_matrix[:, medoid_indices]  # (N, k)
        labels = np.argmin(dists_to_medoids, axis=1)

        # 强制每个 medoid 归属自己的 cluster，避免 argmin tie-breaking 导致空簇
        for c in range(k):
            labels[medoid_indices[c]] = c

        # Update: 对每个 cluster，选使组内距离和最小的点为新 medoid
        new_medoids = medoid_indices.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            # 组内距离子矩阵
            sub_dist = dist_matrix[np.ix_(members, members)]
            total_dists = sub_dist.sum(axis=1)
            best_local = np.argmin(total_dists)
            new_medoids[c] = members[best_local]

        # 收敛检查
        if np.array_equal(np.sort(new_medoids), np.sort(medoid_indices)):
            break
        medoid_indices = new_medoids

    # 最终分配
    dists_to_medoids = dist_matrix[:, medoid_indices]
    labels = np.argmin(dists_to_medoids, axis=1)
    # 强制每个 medoid 归属自己的 cluster，避免 argmin tie-breaking 导致空簇
    for c in range(k):
        labels[medoid_indices[c]] = c
    inertia = sum(dist_matrix[i, medoid_indices[labels[i]]] for i in range(N))

    return medoid_indices, labels, inertia


# ================================================================
#  主聚类函数
# ================================================================

def cluster_velocity_patches(v_torch, k_range=(5, 20), random_state=42,
                             forced_k=None):
    """对速度模型 patch 基于 SSIM 距离做 K-medoids 聚类，自动选择最优 k。

    参数:
        v_torch: torch.Tensor, shape (N, H, W), 原始速度 patch
        k_range: tuple (k_min, k_max), 搜索范围（含两端）
        random_state: int, 随机种子
        forced_k: int (可选), 直接指定 k 值，跳过自动选择

    返回:
        best_k: int, 最优聚类数
        labels: np.ndarray, shape (N,), 每个 patch 的聚类标签
        info: dict, 包含 dist_matrix, silhouettes, inertias, k_values,
              linkage_matrix 用于可视化
    """
    # Step 1: 计算 SSIM 距离矩阵
    dist_matrix = compute_ssim_distance_matrix(v_torch)

    # Step 2: 层次聚类 (用于 dendrogram 可视化)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='ward')

    # Step 3: K-medoids 搜索
    k_min, k_max = k_range
    N = v_torch.shape[0]
    if N < 2:
        raise ValueError(f"样本数过小 (N={N})，至少需要2个样本才能聚类。")

    # 限制搜索范围，避免 k > N 触发无放回采样错误
    k_min = max(2, int(k_min))
    k_max = min(int(k_max), int(N))

    if k_min > k_max:
        k_min, k_max = 2, min(20, int(N))

    # forced_k 模式：只评估指定k，不做范围搜索
    if forced_k is not None:
        forced_k = int(forced_k)
        if forced_k < 2:
            print(f"[WARN] forced_k={forced_k} 过小，自动调整为2")
            forced_k = 2
        if forced_k > N:
            print(f"[WARN] forced_k={forced_k} 大于样本数 N={N}，自动调整为 N={N}")
            forced_k = N
        k_values = [forced_k]
        print(f"\nK-medoids 指定模式: 仅评估 k={forced_k}...")
    else:
        k_values = list(range(k_min, k_max + 1))
        print(f"\nK-medoids 搜索 k={k_min}~{k_max}...")
    silhouettes = []
    inertias = []
    all_labels = {}
    all_medoids = {}

    for k in k_values:
        medoids, lab, inertia = _kmedoids(
            dist_matrix, k, random_state=random_state
        )
        # silhouette_score 要求标签数在 [2, N-1]，超出范围时跳过
        if 2 <= k < N:
            sil = silhouette_score(dist_matrix, lab, metric='precomputed')
        else:
            sil = 0.0
        silhouettes.append(sil)
        inertias.append(inertia)
        all_labels[k] = lab
        all_medoids[k] = medoids
        print(f"  k={k:2d}: silhouette={sil:.4f}, inertia={inertia:.2f}")

    # 选择 k
    if forced_k is not None:
        best_k = forced_k
        best_sil = silhouettes[k_values.index(forced_k)]
        print(f"\n指定 k={best_k} (silhouette={best_sil:.4f})")
    else:
        best_idx = int(np.argmax(silhouettes))
        best_k = k_values[best_idx]
        print(f"\n最优 k={best_k} (silhouette={silhouettes[best_idx]:.4f})")

    labels = all_labels[best_k]

    info = {
        'k_values': k_values,
        'silhouettes': silhouettes,
        'inertias': inertias,
        'dist_matrix': dist_matrix,
        'linkage_matrix': Z,
    }
    return best_k, labels, info


def get_centroids(v_torch, labels, k, info=None):
    """获取每个 cluster 的 centroid (medoid) 索引和 group 成员列表。

    K-medoids 的 medoid 已经是真实 patch，此函数从 labels 中提取。
    如果 info 中包含 dist_matrix，则用距离矩阵精确选取 medoid；
    否则退化为选取组内到其他成员平均距离最小的点。

    参数:
        v_torch: torch.Tensor, shape (N, H, W)
        labels: np.ndarray, shape (N,), 聚类标签
        k: int, 聚类数
        info: dict (可选), 包含 dist_matrix

    返回:
        centroid_indices: list[int], 长度 k
        group_members: dict[int, list[int]]
    """
    dist_matrix = info.get('dist_matrix') if info else None

    centroid_indices = []
    group_members = {}

    for g in range(k):
        member_indices = np.where(labels == g)[0].tolist()
        if len(member_indices) == 0:
            print(f"  Group {g}: 空簇 (跳过)")
            continue
        group_members[g] = member_indices

        if dist_matrix is not None and len(member_indices) > 1:
            members = np.array(member_indices)
            sub_dist = dist_matrix[np.ix_(members, members)]
            total_dists = sub_dist.sum(axis=1)
            best_local = int(np.argmin(total_dists))
            centroid = member_indices[best_local]
        else:
            centroid = member_indices[0]

        centroid_indices.append(centroid)
        print(f"  Group {g}: {len(member_indices)} 个成员, "
              f"centroid=patch[{centroid}]")

    return centroid_indices, group_members


# ================================================================
#  保存 / 加载
# ================================================================

def save_clustering_results(save_path, k, labels, centroid_indices,
                            group_members, info):
    """保存聚类结果为 .pt 文件。

    注意: dist_matrix 和 linkage_matrix 也会保存，方便后续分析。
    """
    results = {
        'k': k,
        'labels': labels,
        'centroid_indices': centroid_indices,
        'group_members': group_members,
        'info': info,
    }
    torch.save(results, save_path)
    print(f"聚类结果已保存: {save_path}")


def load_clustering_results(load_path):
    """加载聚类结果。

    返回:
        dict, 包含 k, labels, centroid_indices, group_members, info
    """
    results = torch.load(load_path, weights_only=False)
    print(f"已加载聚类结果: k={results['k']}, "
          f"centroids={results['centroid_indices']}")
    return results


# ================================================================
#  可视化
# ================================================================

def visualize_clustering(v_torch, labels, centroid_indices, info, save_path):
    """可视化聚类结果。

    Page 1: Dendrogram + Silhouette 曲线
    Page 2: 每个 group 的 centroid patch + SSIM 距离热力图

    参数:
        v_torch: torch.Tensor, shape (N, H, W)
        labels: np.ndarray
        centroid_indices: list[int]
        info: dict
        save_path: str, PDF 保存路径
    """
    from matplotlib.backends.backend_pdf import PdfPages

    k = len(centroid_indices)
    k_values = info['k_values']
    silhouettes = info['silhouettes']
    inertias = info['inertias']
    Z = info.get('linkage_matrix')
    dist_matrix = info.get('dist_matrix')

    with PdfPages(save_path) as pdf:
        # ---- Page 1: Dendrogram + Silhouette + Inertia ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Dendrogram
        ax = axes[0]
        if Z is not None:
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
                       leaf_rotation=90, leaf_font_size=8,
                       color_threshold=Z[-(k-1), 2] if k > 1 else 0)
            ax.axhline(y=Z[-(k-1), 2] if k > 1 else 0,
                       color='r', linestyle='--', label=f'k={k} cut')
            ax.legend(fontsize=9)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=11)
        ax.set_xlabel('Patch (truncated)')
        ax.set_ylabel('SSIM Distance')

        # Silhouette 曲线
        ax = axes[1]
        ax.plot(k_values, silhouettes, 'go-', linewidth=2)
        ax.axvline(x=k, color='r', linestyle='--', label=f'best k={k}')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Silhouette Score', fontsize=10)
        ax.set_title('Silhouette Score vs k', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Inertia 曲线
        ax = axes[2]
        ax.plot(k_values, inertias, 'bo-', linewidth=2)
        ax.axvline(x=k, color='r', linestyle='--', label=f'best k={k}')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Inertia (sum of distances)', fontsize=10)
        ax.set_title('Inertia vs k', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ---- Page 2: Centroid patches ----
        cols = min(5, max(2, k))
        rows = int(np.ceil(k / cols))
        fig, axes_p2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes_p2 = axes_p2[np.newaxis, :] if k > 1 else np.array([[axes_p2]])
        if cols == 1:
            axes_p2 = axes_p2[:, np.newaxis]

        for i, idx in enumerate(centroid_indices):
            r, c = i // cols, i % cols
            ax = axes_p2[r, c]
            patch = v_torch[idx].numpy()
            im = ax.imshow(patch, cmap='viridis', aspect='auto')
            n_members = int(np.sum(labels == i))
            ax.set_title(f'Group {i}\npatch[{idx}] ({n_members} members)',
                         fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 隐藏多余子图
        for i in range(k, rows * cols):
            r, c = i // cols, i % cols
            axes_p2[r, c].axis('off')

        fig.suptitle(f'K-medoids Centroids (k={k}, SSIM distance)',
                     fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ---- Page 3: SSIM 距离热力图 (按 group 排序) ----
        if dist_matrix is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # 按 group 标签排序
            sorted_order = np.argsort(labels)
            sorted_dist = dist_matrix[np.ix_(sorted_order, sorted_order)]

            im = ax.imshow(sorted_dist, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='SSIM Distance (1 - SSIM)')
            ax.set_title(f'SSIM Distance Matrix (sorted by group, k={k})',
                         fontsize=11)
            ax.set_xlabel('Patch index (sorted)')
            ax.set_ylabel('Patch index (sorted)')

            # 画 group 分界线
            boundaries = []
            for g in range(k):
                count = int(np.sum(labels == g))
                boundaries.append(count)
            cum = np.cumsum(boundaries)
            for b in cum[:-1]:
                ax.axhline(y=b - 0.5, color='red', linewidth=0.8)
                ax.axvline(x=b - 0.5, color='red', linewidth=0.8)

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"聚类可视化已保存: {save_path}")
