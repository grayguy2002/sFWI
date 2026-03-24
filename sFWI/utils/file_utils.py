"""
时间戳文件命名工具函数

合并自 exp_daps_sampling.py, exp_rss_dms.py, exp_evaluation.py 中的重复定义。
"""

import os
from datetime import datetime


def generate_timestamped_filename(base_name, extension='', timestamp_format='%Y%m%d_%H%M%S'):
    """
    生成带时间戳的文件名，避免文件覆盖

    参数:
        base_name: str, 基础文件名（不含扩展名）
        extension: str, 文件扩展名（如'.pt', '.pdf', '.png'），可选
        timestamp_format: str, 时间戳格式，默认为'YYYYMMDD_HHMMSS'

    返回:
        str, 带时间戳的完整文件名

    示例:
        >>> generate_timestamped_filename('similarity_matrix', '.pt')
        'similarity_matrix_20260205_143022.pt'
    """
    timestamp = datetime.now().strftime(timestamp_format)

    if extension and not extension.startswith('.'):
        extension = '.' + extension

    return f"{base_name}_{timestamp}{extension}"


def generate_timestamped_path(directory, base_name, extension='', timestamp_format='%Y%m%d_%H%M%S'):
    """
    生成带时间戳的完整文件路径

    参数:
        directory: str, 目标目录路径
        base_name: str, 基础文件名（不含扩展名）
        extension: str, 文件扩展名，可选
        timestamp_format: str, 时间戳格式

    返回:
        str, 带时间戳的完整文件路径
    """
    filename = generate_timestamped_filename(base_name, extension, timestamp_format)
    return os.path.join(directory, filename)
