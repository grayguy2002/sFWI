"""
GSS 示意图生成脚本

用于生成论文中 Group Score Search (GSS) 算法三个阶段的示意图组件。
支持导出为 PDF（适用于 Illustrator）和 PNG（方便预览）。

输出文件:
- gss_p1_ssimmatrix.pdf/png           - Phase 1: SSIM 距离矩阵
- gss_p1_clustering.pdf/png           - Phase 1: 聚类结果示意图
- gss_p1_centroids.pdf/png            - Phase 1: Centroid Patch 示例
- gss_p2_sampling.pdf/png             - Phase 2: 无条件采样（速度域）
- gss_p2_forwardop.pdf/png            - Phase 2: 正演算子流程图
- gss_p2_batchforward.pdf/png         - Phase 2: 批量正演结果
- gss_p2_similaritymatrix.pdf/png     - Phase 2: 相似度矩阵（数据域）
- gss_p2_formulation.pdf/png          - Phase 2: 数学公式说明
- gss_p3_matching.pdf/png             - Phase 3: 观测引导的组匹配
- gss_p3_datamatching.pdf/png         - Phase 3: 数据域匹配
- gss_p3_localrss.pdf/png             - Phase 3: 局部 RSS 结果
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置字体，避免 Times New Roman 不存在的问题
# 使用 DejaVu Serif 作为替代（matplotlib 自带）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'mathtext.fontset': 'dejavuserif',
})


# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))


def set_font():
    """设置 LaTeX 风格字体"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'mathtext.fontset': 'dejavuserif',
    })


def save_figure(filename_base):
    """保存为 PDF 和 PNG，适用于 Illustrator 和预览"""
    filepath_pdf = os.path.join(current_dir, filename_base + '.pdf')
    filepath_png = os.path.join(current_dir, filename_base + '.png')

    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(filepath_png, dpi=150, bbox_inches='tight', format='png')

    print(f"已保存 PDF: {filepath_pdf}")
    print(f"已保存 PNG: {filepath_png}")
    plt.close()


def generate_p1_ssimmatrix():
    """Phase 1: SSIM 距离矩阵"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 4))

    k = 8
    n_patches = 100
    np.random.seed(42)
    dist_matrix = np.zeros((n_patches, n_patches))
    for i in range(n_patches):
        for j in range(i+1, n_patches):
            d = 0.3 + 0.4 * np.random.random()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title('SSIM Distance Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Patch Index')
    plt.colorbar(im, ax=ax, label='SSIM Distance (1 - SSIM)')

    centroids = [5, 18, 32, 45, 61, 77, 89, 94]
    for i, c in enumerate(centroids):
        ax.plot(c, c, 'r*', markersize=10, markeredgecolor='white', markeredgewidth=1)
        ax.text(c + 3, c + 3, f'G{i}', fontsize=8, color='red')

    save_figure('gss_p1_ssimmatrix')


def generate_p1_clustering():
    """Phase 1: 聚类结果示意图"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3))

    np.random.seed(123)
    n_patches = 100
    for i in range(n_patches):
        group = i // 12
        x = np.random.uniform(group * 2, (group + 1) * 2)
        y = np.random.uniform(0, 1)
        color = plt.cm.tab10(group % 10)
        ax.scatter(x, y, c=[color], s=30, alpha=0.7)

    centroids = [5, 18, 32, 45, 61, 77, 89, 94]
    for i, c in enumerate(centroids):
        group = c // 12
        x = group * 2 + 1
        y = 0.5
        ax.plot(x, y, 'r*', markersize=12, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_title('Clustered Groups (k=8)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Group')
    ax.set_yticks([])
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_xticklabels([f'G{i}' for i in range(8)])

    save_figure('gss_p1_clustering')


def generate_p1_centroids():
    """Phase 1: Centroid Patch 示例"""
    set_font()
    fig, axes = plt.subplots(2, 2, figsize=(5, 4))

    titles = ['G0 Centroid', 'G1 Centroid', 'G2 Centroid', 'G3 Centroid']
    patches = []

    # G0: 线性梯度
    patch0 = np.zeros((32, 32))
    for z in range(32):
        patch0[:, z] = 1500 + 50 * z
    patches.append(patch0)

    # G1: 两层结构
    patch1 = np.zeros((32, 32))
    patch1[:, :16] = 2000
    patch1[:, 16:] = 3500
    patches.append(patch1)

    # G2: 正弦扰动
    patch2 = np.zeros((32, 32))
    patch2[:, :] = 2500 + 1000 * np.sin(np.linspace(0, np.pi, 32))[:, None]
    patches.append(patch2)

    # G3: 圆形异常体
    patch3 = np.zeros((32, 32))
    center_x, center_z = 16, 10
    for x in range(32):
        for z in range(32):
            dist = np.sqrt((x - center_x)**2 + (z - center_z)**2)
            patch3[x, z] = 4500 if dist < 8 else 2500
    patches.append(patch3)

    for i, (ax, patch, title) in enumerate(zip(axes.flat, patches, titles)):
        im = ax.imshow(patch, cmap='seismic', aspect='auto', vmin=1500, vmax=5500)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_figure('gss_p1_centroids')


def generate_p2_sampling():
    """Phase 2: 无条件采样（速度域）"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.set_title('Velocity Domain: Batch Sampling', fontsize=12, fontweight='bold')
    np.random.seed(456)
    n_samples = 50
    n_centroids = 8
    for i in range(n_samples):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        ax.scatter(x, y, c='gray', s=20, alpha=0.6, marker='o')

    for i in range(n_centroids):
        x = 0.1 + i * 0.11
        y = 0.8
        ax.plot(x, y, 'r*', markersize=10, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('Latent Space (sample)')
    ax.set_yticks([])

    save_figure('gss_p2_sampling')


def generate_p2_forwardop():
    """Phase 2: 正演算子流程图"""
    set_font()
    fig, ax = plt.subplots(figsize=(5, 2))

    ax.set_title('Forward Operator: Acoustic Wave Equations', fontsize=12, fontweight='bold')
    ax.text(0.15, 0.7, 'Velocity\nModel', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.arrow(0.35, 0.7, 0.15, 0, head_width=0.03, head_length=0.05, fc='black', ec='black')
    ax.text(0.6, 0.7, 'DeepWave\nForward', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.arrow(0.8, 0.7, 0.15, 0, head_width=0.03, head_length=0.05, fc='black', ec='black')
    ax.text(0.95, 0.7, 'Seismic\nRecord', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    save_figure('gss_p2_forwardop')


def generate_p2_batchforward():
    """Phase 2: 批量正演结果"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.set_title('Batched Forward Modeling', fontsize=12, fontweight='bold')
    nt, nr = 100, 50
    n_samples = 50
    n_centroids = 8

    for i in range(min(5, n_samples)):
        seismic = np.random.randn(nt, nr) * 0.5
        ax.imshow(seismic.T, cmap='gray', aspect='auto', vmin=-2, vmax=2,
                extent=[0, n_samples, nt, 0], alpha=0.3)

    for i in range(n_centroids):
        ax.plot(i * 2 + 1, 50, 'r*', markersize=6)

    ax.set_xlabel('Sample/Index')
    ax.set_ylabel('Time')
    ax.set_ylim(nt, 0)

    save_figure('gss_p2_batchforward')


def generate_p2_similaritymatrix():
    """Phase 2: 相似度矩阵（数据域）"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.set_title('Data Domain: Similarity Matrix M', fontsize=12, fontweight='bold')
    k, n_s = 8, 50
    np.random.seed(789)
    M = np.zeros((k, n_s))
    for i in range(k):
        for j in range(n_s):
            d_base = 0.5 + 0.3 * np.random.random()
            if j % 10 == i:
                d_base *= 0.5
            M[i, j] = d_base

    im = ax.imshow(M, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index j')
    ax.set_ylabel('Group Index i')
    ax.set_xticks([0, 25, 49])
    ax.set_xticklabels([0, 25, 49])
    plt.colorbar(im, ax=ax, label='ell_2 Distance')

    save_figure('gss_p2_similaritymatrix')


def generate_p2_formulation():
    """Phase 2: 数学公式说明"""
    set_font()
    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.set_title('Mathematical Formulation', fontsize=12, fontweight='bold')
    ax.text(0.1, 0.85, r'M[i,j] = ||F(centroid_i) - F(sample_j)||_2',
            fontsize=14)
    ax.text(0.1, 0.65, 'where:', fontsize=11)
    ax.text(0.1, 0.5, 'F: Acoustic wave equation solver', fontsize=10)
    ax.text(0.1, 0.35, 'centroid_i: Centroid of group i', fontsize=10)
    ax.text(0.1, 0.2, 'sample_j: Unconditional sample j', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    save_figure('gss_p2_formulation')


def generate_p3_matching():
    """Phase 3: 观测引导的组匹配"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3))

    ax.set_title('Observation-Guided Matching', fontsize=12, fontweight='bold')
    np.random.seed(999)
    for i in range(20):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        ax.scatter(x, y, c='gray', s=20, alpha=0.5, marker='o')

    ax.scatter(0.6, 0.6, c='red', s=80, marker='*', label=r'$d_{\text{obs}}$ sample',
                edgecolors='darkred', linewidths=2, zorder=10)
    ax.text(0.6, 0.65, r'$d_{\text{obs}}$', fontsize=10, ha='center', color='darkred')

    centroids_x = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    centroids_y = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for i, (x, y, c) in enumerate(zip(centroids_x, centroids_y, colors)):
        ax.plot(x, y, 's', markersize=10, markerfacecolor=c,
                markeredgecolor='black', markeredgewidth=1.5, label=f'Group {i}')

    ax.set_xlabel('Velocity Domain')
    ax.set_ylabel('Velocity Domain')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='upper left', fontsize=8, ncol=2)

    save_figure('gss_p3_matching')


def generate_p3_datamatching():
    """Phase 3: 数据域匹配"""
    set_font()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    k, n_s = 6, 50
    np.random.seed(111)
    M = np.zeros((k, n_s))
    for i in range(k):
        for j in range(n_s):
            d_base = 0.3 + 0.4 * np.random.random()
            M[i, j] = d_base

    d_obs = 0.45
    d_obs_distribution = np.abs(M - d_obs).mean(axis=1)
    im = ax.imshow(M, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index j')
    ax.set_ylabel('Group Index i')
    ax.set_xticks([0, 25, 49])

    best_group = int(np.argmin(d_obs_distribution))
    ax.axhline(y=best_group, xmin=0, xmax=1, color='red', linestyle='--', linewidth=2)
    plt.colorbar(im, ax=ax, label='ell_2 Distance')

    save_figure('gss_p3_datamatching')


def generate_p3_localrss():
    """Phase 3: 局部 RSS 结果"""
    set_font()
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.set_title('Localized RSS in Best Group', fontsize=12, fontweight='bold')
    group_members_x = [0.65 + np.random.random() * 0.15 for _ in range(8)]
    group_members_y = [0.45 + np.random.random() * 0.15 for _ in range(8)]

    for i, (x, y) in enumerate(zip(group_members_x, group_members_y)):
        ax.scatter(x, y, c='gray', s=25, alpha=0.6)

    best_idx = 3
    ax.scatter(group_members_x[best_idx], group_members_y[best_idx],
                c='gold', s=100, marker='*', label='Best candidate',
                edgecolors='orange', linewidths=2, zorder=10)
    ax.text(group_members_x[best_idx], group_members_y[best_idx] + 0.05,
            'Selected', fontsize=9, ha='center', color='orange', fontweight='bold')

    ax.set_xlabel('Group Space')
    ax.set_ylabel('Group Space')
    ax.set_xlim(0.5, 0.9)
    ax.set_ylim(0.3, 0.7)
    ax.legend(loc='upper left', fontsize=8)

    save_figure('gss_p3_localrss')


def main():
    print('=' * 60)
    print('GSS Schematic Diagram Generator')
    print('=' * 60)
    print(f'输出目录: {current_dir}')

    # Phase 1: 聚类相关（3张图）
    print('生成 Phase 1: 聚类...')
    generate_p1_ssimmatrix()
    generate_p1_clustering()
    generate_p1_centroids()

    # Phase 2: 相似度矩阵（5张图）
    print('生成 Phase 2: 相似度矩阵...')
    generate_p2_sampling()
    generate_p2_forwardop()
    generate_p2_batchforward()
    generate_p2_similaritymatrix()
    generate_p2_formulation()

    # Phase 3: 组匹配（3张图）
    print('生成 Phase 3: 组匹配...')
    generate_p3_matching()
    generate_p3_datamatching()
    generate_p3_localrss()

    print('\n所有示意图已生成完成！')
    print('生成的文件 (PDF + PNG):')
    print('  Phase 1 (聚类):')
    print(f'    {current_dir}/gss_p1_ssimmatrix.{{pdf,png}}')
    print(f'    {current_dir}/gss_p1_clustering.{{pdf,png}}')
    print(f'    {current_dir}/gss_p1_centroids.{{pdf,png}}')
    print('  Phase 2 (相似度矩阵):')
    print(f'    {current_dir}/gss_p2_sampling.{{pdf,png}}')
    print(f'    {current_dir}/gss_p2_forwardop.{{pdf,png}}')
    print(f'    {current_dir}/gss_p2_batchforward.{{pdf,png}}')
    print(f'    {current_dir}/gss_p2_similaritymatrix.{{pdf,png}}')
    print(f'    {current_dir}/gss_p2_formulation.{{pdf,png}}')
    print('  Phase 3 (组匹配):')
    print(f'    {current_dir}/gss_p3_matching.{{pdf,png}}')
    print(f'    {current_dir}/gss_p3_datamatching.{{pdf,png}}')
    print(f'    {current_dir}/gss_p3_localrss.{{pdf,png}}')


if __name__ == '__main__':
    main()
