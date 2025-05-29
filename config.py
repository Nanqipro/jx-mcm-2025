#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度预测系统配置文件

该文件包含了系统的各种配置参数，包括模型参数、数据处理参数等。

Author: Data Analysis Team
Date: 2024
"""

import os
from typing import Dict, List, Any


class AnalysisConfig:
    """
    分析配置类
    
    包含了能见度预测分析系统的所有配置参数。
    """
    
    # ===========================================
    # 数据相关配置
    # ===========================================
    
    # 默认数据文件路径
    DEFAULT_DATA_FILE: str = os.getenv('VISIBILITY_DATA_PATH', 'complete_synced_data.csv')
    
    # 目标变量列名
    TARGET_COLUMN: str = 'visibility_mor_raw'
    
    # 数据划分参数
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # ===========================================
    # 特征相关配置
    # ===========================================
    
    # 手动选择的核心特征
    MANUAL_FEATURES: List[str] = [
        'laplacian_var',
        'high_freq_ratio', 
        'edge_density',
        'contrast_std',
        'weather_humidity_pct',
        'weather_temperature_c',
        'wind_wind_speed_10m'
    ]
    
    # 图像特征关键词
    IMAGE_FEATURE_KEYWORDS: List[str] = [
        'laplacian', 'sobel', 'contrast', 'edge', 'freq', 'var',
        'gradient', 'texture', 'entropy', 'local'
    ]
    
    # 气象特征关键词
    WEATHER_FEATURE_KEYWORDS: List[str] = [
        'weather', 'temperature', 'humidity', 'pressure'
    ]
    
    # 风速特征关键词
    WIND_FEATURE_KEYWORDS: List[str] = ['wind']
    
    # 能见度特征关键词
    VISIBILITY_FEATURE_KEYWORDS: List[str] = ['visibility']
    
    # VIF阈值（方差膨胀因子）
    VIF_THRESHOLD: float = 10.0
    
    # ===========================================
    # 模型相关配置  
    # ===========================================
    
    # 回归模型配置
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        '普通线性回归': {},
        '岭回归(Ridge)': {'alpha': 1.0},
        'Lasso回归': {'alpha': 0.1},
        '弹性网络': {'alpha': 0.1, 'l1_ratio': 0.5}
    }
    
    # 模型评估指标
    EVALUATION_METRICS: List[str] = ['r2', 'rmse', 'mae', 'mape']
    
    # 是否进行目标变量变换
    TARGET_TRANSFORM: bool = True  # 平方根变换
    
    # ===========================================
    # 可视化相关配置
    # ===========================================
    
    # 图表尺寸配置
    FIGURE_SIZES: Dict[str, tuple] = {
        'correlation_heatmap': (15, 12),
        'residual_analysis': (15, 10),
        'feature_distribution': (12, 8),
        'model_comparison': (10, 6)
    }
    
    # 颜色配置
    COLOR_PALETTE: str = 'coolwarm'
    
    # 字体配置
    CHINESE_FONTS: List[str] = ['SimHei', 'DejaVu Sans']
    
    # DPI设置
    FIGURE_DPI: int = 100
    
    # ===========================================
    # 输出相关配置
    # ===========================================
    
    # 输出目录
    OUTPUT_DIR: str = os.getenv('VISIBILITY_OUTPUT_DIR', 'output')
    
    # 结果文件名
    RESULT_FILES: Dict[str, str] = {
        'model_summary': 'model_summary.json',
        'feature_importance': 'feature_importance.csv',
        'predictions': 'predictions.csv',
        'residual_plot': 'residual_analysis.png',
        'correlation_plot': 'correlation_matrix.png'
    }
    
    # 是否保存中间结果
    SAVE_INTERMEDIATE: bool = True
    
    # ===========================================
    # 性能相关配置
    # ===========================================
    
    # 内存使用限制（MB）
    MEMORY_LIMIT: int = int(os.getenv('MEMORY_LIMIT', '2048'))
    
    # 并行处理参数
    N_JOBS: int = int(os.getenv('N_JOBS', '-1'))  # -1表示使用所有可用CPU
    
    # 是否启用详细输出
    VERBOSE: bool = os.getenv('VERBOSE', 'True').lower() == 'true'
    
    # ===========================================
    # 质量控制配置
    # ===========================================
    
    # 数据质量检查
    MIN_SAMPLES: int = 50  # 最少样本数
    MAX_MISSING_RATIO: float = 0.1  # 最大缺失比例
    
    # 相关性阈值
    MIN_CORRELATION: float = 0.1  # 最小相关性
    MAX_CORRELATION: float = 0.95  # 避免完全相关
    
    # 模型性能要求
    MIN_R2_SCORE: float = 0.5  # 最小R²要求
    
    # ===========================================
    # 日志配置
    # ===========================================
    
    # 日志级别
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # 日志文件
    LOG_FILE: str = os.path.join(OUTPUT_DIR, 'analysis.log')
    
    # 是否在控制台显示日志
    LOG_TO_CONSOLE: bool = True


class ProductionConfig(AnalysisConfig):
    """
    生产环境配置
    
    继承基础配置并覆盖生产环境特定的参数。
    """
    
    # 生产环境更严格的质量要求
    MIN_R2_SCORE: float = 0.7
    MIN_SAMPLES: int = 100
    MAX_MISSING_RATIO: float = 0.05
    
    # 关闭详细输出以提升性能
    VERBOSE: bool = False
    
    # 使用更保守的模型参数
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        '岭回归(Ridge)': {'alpha': 1.0},
        'Lasso回归': {'alpha': 0.01},
    }


class DevelopmentConfig(AnalysisConfig):
    """
    开发环境配置
    
    继承基础配置并覆盖开发环境特定的参数。
    """
    
    # 开发环境允许更宽松的质量要求
    MIN_R2_SCORE: float = 0.3
    MIN_SAMPLES: int = 20
    
    # 启用所有详细输出
    VERBOSE: bool = True
    SAVE_INTERMEDIATE: bool = True
    
    # 使用更多模型进行实验
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        '普通线性回归': {},
        '岭回归(Ridge)': {'alpha': 1.0},
        'Lasso回归': {'alpha': 0.1},
        '弹性网络': {'alpha': 0.1, 'l1_ratio': 0.5}
    }


def get_config(env: str = None) -> AnalysisConfig:
    """
    根据环境获取配置
    
    Parameters
    ----------
    env : str, optional
        环境名称，可选 'development', 'production'
        如果未指定，从环境变量 ENVIRONMENT 获取
    
    Returns
    -------
    AnalysisConfig
        配置对象
    """
    if env is None:
        env = os.getenv('ENVIRONMENT', 'development')
    
    if env.lower() == 'production':
        return ProductionConfig()
    elif env.lower() == 'development':
        return DevelopmentConfig()
    else:
        return AnalysisConfig()


# 默认配置实例
config = get_config()


if __name__ == "__main__":
    # 测试配置
    print("当前配置:")
    print(f"环境: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"数据文件: {config.DEFAULT_DATA_FILE}")
    print(f"目标列: {config.TARGET_COLUMN}")
    print(f"手动特征: {config.MANUAL_FEATURES}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"最小R²要求: {config.MIN_R2_SCORE}") 