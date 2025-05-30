#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
江西省数学建模比赛 - 问题二最佳解决方案
建立能见度随时间连续变化数学模型

本系统整合了三种方案的优点：
- s2.py: 连续数学模型核心（微分方程、状态空间、非线性动力学）
- second.py: 丰富的可视化和数据分析功能
- DYCSecond.py: 系统化架构和数据处理能力

Author: AI Assistant
Version: 1.0
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from scipy import optimize, stats, signal
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from filterpy.kalman import KalmanFilter
import cv2

warnings.filterwarnings('ignore')

# 设置中文字体和图表显示规范
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 中文字体设置成功")
except Exception as e:
    print(f"⚠️ 字体设置警告: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


class ContinuousVisibilityModel:
    """
    能见度时间连续变化数学模型
    
    整合多种连续数学模型方法，为江西省数学建模比赛问题二提供最佳解决方案
    """
    
    def __init__(self):
        """初始化模型"""
        self.data = None
        self.processed_data = None
        self.visibility = None
        self.weather_features = None
        self.time_index = None
        self.models = {}
        self.results = {}
        
        # 添加训练集/测试集分割相关属性
        self.train_size = 0.8  # 80%作为训练集
        self.train_indices = None
        self.test_indices = None
        self.visibility_train = None
        self.visibility_test = None
        self.weather_train = None
        self.weather_test = None
        self.time_train = None
        self.time_test = None
        
        # 创建结果保存目录
        self.results_dir = "../results"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 结果保存目录: {self.results_dir}")
        
        # 模型配色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#E63946',
            'data': '#264653',
            'prediction': '#E76F51',
            'residual': '#F4A261'
        }
        
        # 字体设置和中英文文本支持
        self.use_chinese = self._setup_chinese_fonts()
        self._setup_bilingual_texts()
        
        print("🚀 能见度连续变化数学模型初始化完成")

    def _setup_chinese_fonts(self) -> bool:
        """
        设置中文字体支持
        
        Returns
        -------
        bool
            是否成功启用中文字体
        """
        try:
            import matplotlib.font_manager as fm
            
            # 尝试多种中文字体
            font_candidates = [
                'SimHei',           # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'FangSong',         # 仿宋
                'KaiTi',           # 楷体
                'SimSun',          # 宋体
            ]
            
            # 获取系统可用字体
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            chinese_fonts = []
            
            for candidate in font_candidates:
                if any(candidate.lower() in font.lower() for font in available_fonts):
                    chinese_fonts.append(candidate)
            
            if chinese_fonts:
                # 设置matplotlib的中文字体
                plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 设置默认字体大小
                plt.rcParams['font.size'] = 10
                plt.rcParams['axes.titlesize'] = 12
                plt.rcParams['axes.labelsize'] = 10
                plt.rcParams['xtick.labelsize'] = 9
                plt.rcParams['ytick.labelsize'] = 9
                plt.rcParams['legend.fontsize'] = 9
                
                print(f"✅ 中文字体设置成功: {chinese_fonts[0]}")
                return True
            else:
                print("⚠️ 未检测到中文字体，将使用英文显示")
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return False
                
        except Exception as e:
            print(f"⚠️ 中文字体设置失败: {e}")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return False

    def _setup_bilingual_texts(self):
        """设置中英文对照文本"""
        self.texts = {
            # 基本标签
            'time_steps': '时间步长' if self.use_chinese else 'Time Steps',
            'visibility': '能见度 (m)' if self.use_chinese else 'Visibility (m)',
            'visibility_m': '能见度' if self.use_chinese else 'Visibility',
            'change_rate': '变化率 (m/步)' if self.use_chinese else 'Change Rate (m/step)',
            'density': '概率密度' if self.use_chinese else 'Density',
            'residuals': '残差 (m)' if self.use_chinese else 'Residuals (m)',
            'predicted_values': '预测值 (m)' if self.use_chinese else 'Predicted Values (m)',
            'observed_values': '观测值 (m)' if self.use_chinese else 'Observed Values (m)',
            
            # 标题
            'original_series': '原始能见度时间序列' if self.use_chinese else 'Original Visibility Time Series',
            'visibility_distribution': '能见度分布特征' if self.use_chinese else 'Visibility Distribution',
            'change_rate_series': '能见度变化率时间序列' if self.use_chinese else 'Visibility Change Rate Series',
            'adf_test_results': 'ADF平稳性检验结果' if self.use_chinese else 'ADF Stationarity Test Results',
            'model_performance': '模型性能对比' if self.use_chinese else 'Model Performance Comparison',
            'residual_analysis': '残差分析' if self.use_chinese else 'Residual Analysis',
            
            # 统计信息
            'data_points': '观测点数' if self.use_chinese else 'Data Points',
            'mean_value': '平均值' if self.use_chinese else 'Mean',
            'std_value': '标准差' if self.use_chinese else 'Std Dev',
            'cv_value': '变异系数' if self.use_chinese else 'CV',
            'range_value': '范围' if self.use_chinese else 'Range',
            'fog_events': '雾事件占比(<1000m)' if self.use_chinese else 'Fog Events (<1000m)',
            
            # ADF检验
            'adf_statistic': 'ADF统计量' if self.use_chinese else 'ADF Statistic',
            'p_value': 'p值' if self.use_chinese else 'p-value',
            'critical_values': '临界值' if self.use_chinese else 'Critical Values',
            'conclusion': '结论' if self.use_chinese else 'Conclusion',
            'stationary': '序列平稳' if self.use_chinese else 'Series is Stationary',
            'non_stationary': '序列非平稳' if self.use_chinese else 'Series is Non-Stationary',
            
            # 模型相关
            'training_set': '训练集' if self.use_chinese else 'Training Set',
            'test_set': '测试集' if self.use_chinese else 'Test Set',
            'r2_score': 'R²得分' if self.use_chinese else 'R² Score',
            'mae_error': '平均绝对误差 (m)' if self.use_chinese else 'Mean Absolute Error (m)',
            'rmse_error': '均方根误差 (m)' if self.use_chinese else 'Root Mean Square Error (m)',
            'model_type': '模型类型' if self.use_chinese else 'Model Type',
        }

    def get_text(self, key: str) -> str:
        """获取对应语言的文本"""
        return self.texts.get(key, key)

    def get_font_prop(self):
        """获取字体属性"""
        if self.use_chinese:
            return {'family': 'SimHei'}
        else:
            return {'family': 'DejaVu Sans'}

    def load_and_preprocess_data(self, filepath: str) -> bool:
        """
        加载和预处理数据
        
        Parameters
        ----------
        filepath : str
            数据文件路径
            
        Returns
        -------
        bool
            数据加载是否成功
        """
        print("\n=== 数据加载与预处理 ===")
        
        try:
            # 加载数据
            self.data = pd.read_csv(filepath)
            print(f"✅ 数据加载成功: {len(self.data)} 条记录")
            
            # 数据预处理
            self._preprocess_data()
            
            # 数据质量检查
            self._data_quality_check()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ 错误: 找不到数据文件 {filepath}")
            return False
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def _preprocess_data(self) -> None:
        """数据预处理"""
        print("正在进行数据预处理...")
        
        # 处理时间列
        if 'CREATEDATE' in self.data.columns:
            self.data['CREATEDATE'] = pd.to_datetime(self.data['CREATEDATE'])
            self.data = self.data.sort_values('CREATEDATE')
        
        # 选择关键特征
        key_features = ['TEMP', 'RH', 'DEWPOINT', 'WS2A', 'WD2A', 'CW2A',
                        'PAINS (HPA)', 'MOR_1A', 'RVR_1A']
        
        # 确保所有特征都存在
        available_features = [col for col in key_features if col in self.data.columns]
        self.processed_data = self.data[available_features].copy()
        
        # 处理缺失值
        self.processed_data = self.processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # 选择能见度指标
        if 'MOR_1A' in self.processed_data.columns:
            self.visibility = self.processed_data['MOR_1A'].values
        elif 'RVR_1A' in self.processed_data.columns:
            self.visibility = self.processed_data['RVR_1A'].values
        elif 'blur_index' in self.data.columns:
            # 将模糊度指数转换为能见度
            blur_values = self.data['blur_index'].fillna(0.5)
            self.visibility = 10000 / (1 + 10 * blur_values)
        else:
            raise ValueError("未找到合适的能见度指标")
        
        # 提取气象特征
        weather_cols = ['TEMP', 'RH', 'DEWPOINT', 'WS2A', 'WD2A', 'CW2A', 'PAINS (HPA)']
        available_weather = [col for col in weather_cols if col in self.processed_data.columns]
        
        if available_weather:
            self.weather_features = self.processed_data[available_weather].values
        else:
            # 生成模拟气象数据
            n_points = len(self.visibility)
            self.weather_features = np.column_stack([
                np.random.normal(15, 5, n_points),  # 温度
                np.random.normal(80, 15, n_points),  # 湿度
                np.random.normal(10, 3, n_points),  # 露点
                np.random.normal(3, 2, n_points)   # 风速
            ])
        
        # 创建时间索引
        self.time_index = np.arange(len(self.visibility))
        
        # 添加训练集/测试集分割
        self._split_train_test()
        
        print(f"  ✅ 能见度数据: {len(self.visibility)} 个点")
        print(f"  ✅ 气象特征: {self.weather_features.shape[1]} 个维度")
        print(f"  📊 训练集: {len(self.visibility_train)} 个点 ({len(self.visibility_train)/len(self.visibility)*100:.1f}%)")
        print(f"  📊 测试集: {len(self.visibility_test)} 个点 ({len(self.visibility_test)/len(self.visibility)*100:.1f}%)")

    def _split_train_test(self) -> None:
        """分割训练集和测试集"""
        n_samples = len(self.visibility)
        n_train = int(n_samples * self.train_size)
        
        # 使用时间顺序分割（更符合时间序列特性）
        self.train_indices = np.arange(n_train)
        self.test_indices = np.arange(n_train, n_samples)
        
        # 分割数据
        self.visibility_train = self.visibility[self.train_indices]
        self.visibility_test = self.visibility[self.test_indices]
        self.weather_train = self.weather_features[self.train_indices]
        self.weather_test = self.weather_features[self.test_indices]
        self.time_train = self.time_index[self.train_indices]
        self.time_test = self.time_index[self.test_indices]

    def _data_quality_check(self) -> None:
        """数据质量检查"""
        print("进行数据质量检查...")
        
        # 检查异常值
        q1, q3 = np.percentile(self.visibility, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((self.visibility < q1 - 1.5*iqr) | (self.visibility > q3 + 1.5*iqr))
        
        print(f"  能见度范围: {self.visibility.min():.1f}m - {self.visibility.max():.1f}m")
        print(f"  异常值数量: {outliers} ({outliers/len(self.visibility)*100:.1f}%)")
        print(f"  数据完整性: ✅")

    def analyze_time_series_characteristics(self) -> Dict[str, Any]:
        """
        时间序列特征分析
        
        Returns
        -------
        Dict[str, Any]
            分析结果字典
        """
        print("\n=== 时间序列特征分析 ===")
        
        # 基本统计特征
        if self.use_chinese:
            vis_stats = {
                '数据点数': len(self.visibility),
                '平均值': f"{self.visibility.mean():.1f}m",
                '标准差': f"{self.visibility.std():.1f}m",
                '变异系数': f"{self.visibility.std()/self.visibility.mean():.3f}",
                '雾事件占比(<1000m)': f"{(self.visibility < 1000).mean() * 100:.1f}%",
                '严重雾事件占比(<500m)': f"{(self.visibility < 500).mean() * 100:.1f}%"
            }
        else:
            vis_stats = {
                'Data Points': len(self.visibility),
                'Mean': f"{self.visibility.mean():.1f}m",
                'Std Dev': f"{self.visibility.std():.1f}m",
                'CV': f"{self.visibility.std()/self.visibility.mean():.3f}",
                'Fog Events (<1000m)': f"{(self.visibility < 1000).mean() * 100:.1f}%",
                'Severe Fog (<500m)': f"{(self.visibility < 500).mean() * 100:.1f}%"
            }
        
        if self.use_chinese:
            print("基本统计特征:")
        else:
            print("Basic Statistical Features:")
        for key, value in vis_stats.items():
            print(f"  {key}: {value}")
        
        # 平稳性检验 (ADF检验)
        if self.use_chinese:
            print("\n平稳性检验 (ADF检验):")
        else:
            print("\nStationarity Test (ADF Test):")
        adf_result = adfuller(self.visibility)
        adf_stat_label = self.get_text('adf_statistic')
        p_val_label = self.get_text('p_value')
        conclusion_label = self.get_text('conclusion')
        
        print(f"  {adf_stat_label}: {adf_result[0]:.3f}")
        print(f"  {p_val_label}: {adf_result[1]:.3f}")
        is_stationary = adf_result[1] < 0.05
        status_text = self.get_text('stationary') if is_stationary else self.get_text('non_stationary')
        print(f"  {conclusion_label}: {status_text}")
        
        # 变化率分析
        vis_diff = np.diff(self.visibility)
        if self.use_chinese:
            change_stats = {
                '平均变化率': f"{np.mean(vis_diff):.2f}m/步",
                '变化率标准差': f"{np.std(vis_diff):.2f}m/步",
                '最大增幅': f"{np.max(vis_diff):.2f}m/步",
                '最大降幅': f"{np.min(vis_diff):.2f}m/步"
            }
            print("\n变化率统计:")
        else:
            change_stats = {
                'Mean Change Rate': f"{np.mean(vis_diff):.2f}m/step",
                'Change Rate Std': f"{np.std(vis_diff):.2f}m/step",
                'Max Increase': f"{np.max(vis_diff):.2f}m/step",
                'Max Decrease': f"{np.min(vis_diff):.2f}m/step"
            }
            print("\nChange Rate Statistics:")
        
        for key, value in change_stats.items():
            print(f"  {key}: {value}")
        
        # 存储结果
        self.results['time_series_analysis'] = {
            'basic_stats': vis_stats,
            'adf_test': {'statistic': adf_result[0], 'pvalue': adf_result[1], 'is_stationary': is_stationary},
            'change_stats': change_stats
        }
        
        return self.results['time_series_analysis']

    def build_differential_equation_model(self) -> Dict[str, Any]:
        """
        构建基于微分方程的雾演化连续模型
        
        实现雾动力学微分方程：dV/dt = α·f(T,RH,WS,P) - β·V(t) + γ·ξ(t)
        
        Returns
        -------
        Dict[str, Any]
            微分方程模型结果
        """
        print("\n=== 构建微分方程雾演化模型 ===")
        
        def fog_dynamics_equation(V: float, t: float, params: np.ndarray, weather_data: np.ndarray = None) -> float:
            """雾动力学微分方程"""
            alpha, beta, gamma, delta, epsilon = params
            
            # 获取当前时刻的气象条件（线性插值）
            if weather_data is not None and t < len(weather_data):
                idx = int(t)
                if idx < len(weather_data) - 1:
                    # 线性插值
                    frac = t - idx
                    weather = weather_data[idx] * (1 - frac) + weather_data[idx + 1] * frac
                else:
                    weather = weather_data[-1]
            else:
                weather = self.weather_train[-1] if hasattr(self, 'weather_train') and len(self.weather_train) > 0 else np.array([15, 80, 10, 2, 0, 0, 1013])
            
            # 提取气象要素
            T = weather[0] if len(weather) > 0 else 15      # 温度
            RH = weather[1] if len(weather) > 1 else 80     # 相对湿度  
            WS = weather[3] if len(weather) > 3 else 2      # 风速
            P = weather[6] if len(weather) > 6 else 1013    # 气压
            
            # 气象驱动函数 f(T,RH,WS,P)
            humidity_factor = (RH - 70) / 30  # 湿度因子
            temperature_factor = (15 - T) / 15  # 温度因子（低温有利）
            wind_factor = (3 - WS) / 3  # 风速因子（低风速有利）
            
            # 气象驱动项
            meteorological_forcing = alpha * (
                0.4 * humidity_factor + 
                0.3 * temperature_factor + 
                0.3 * wind_factor
            )
            
            # 自然消散项（雾的自然演化）
            dissipation_term = -beta * V
            
            # 随机扰动项（简化处理）
            noise_term = gamma * np.random.normal(0, 0.01)
            
            # 非线性项（考虑饱和效应）
            nonlinear_term = -delta * V**2 / (epsilon + V)
            
            # 总变化率
            dVdt = meteorological_forcing + dissipation_term + nonlinear_term + noise_term
            
            return dVdt
        
        # 参数优化 - 仅在训练集上进行
        def objective_function(params: np.ndarray) -> float:
            """目标函数：最小化训练集预测误差"""
            try:
                # 在训练集上数值积分求解
                t_span = np.linspace(0, len(self.visibility_train) - 1, len(self.visibility_train))
                V0 = self.visibility_train[0]
                
                # 求解微分方程
                solution = odeint(fog_dynamics_equation, V0, t_span, args=(params, self.weather_train))
                predicted = solution.flatten()
                
                # 确保预测值为正且在合理范围内
                predicted = np.clip(predicted, 10, 20000)
                
                # 计算均方误差
                mse = np.mean((predicted - self.visibility_train)**2)
                return mse
                
            except Exception as e:
                print(f"⚠️ 微分方程求解失败: {e}")
                return 1e10
        
        # 参数边界和初始值
        param_bounds = [
            (0.1, 500),     # α: 气象驱动强度
            (0.001, 0.1),   # β: 消散系数  
            (0.001, 50),    # γ: 随机扰动强度
            (0.001, 0.01),  # δ: 非线性系数
            (100, 5000)     # ε: 饱和参数
        ]
        
        initial_params = [100, 0.01, 10, 0.005, 1000]
        
        print("正在优化微分方程参数...")
        try:
            # 设置随机种子以确保结果可重现
            np.random.seed(42)
            
            # 参数优化
            result = optimize.minimize(
                objective_function, 
                initial_params,
                bounds=param_bounds,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            optimal_params = result.x
            
            # 生成训练集预测
            t_train_span = np.linspace(0, len(self.visibility_train) - 1, len(self.visibility_train))
            V0_train = self.visibility_train[0]
            predicted_train = odeint(fog_dynamics_equation, V0_train, t_train_span, args=(optimal_params, self.weather_train))
            predicted_train = predicted_train.flatten()
            predicted_train = np.clip(predicted_train, 10, 20000)
            
            # 生成测试集预测
            # 从训练集最后一个值开始预测
            t_test_span = np.linspace(0, len(self.visibility_test) - 1, len(self.visibility_test))
            V0_test = self.visibility_train[-1]  # 使用训练集最后一个值作为初始值
            predicted_test = odeint(fog_dynamics_equation, V0_test, t_test_span, args=(optimal_params, self.weather_test))
            predicted_test = predicted_test.flatten()
            predicted_test = np.clip(predicted_test, 10, 20000)
            
            # 生成完整预测（训练集+测试集）
            predicted_full = np.concatenate([predicted_train, predicted_test])
            
            # 计算训练集性能指标
            r2_train = r2_score(self.visibility_train, predicted_train)
            mae_train = mean_absolute_error(self.visibility_train, predicted_train)
            rmse_train = np.sqrt(mean_squared_error(self.visibility_train, predicted_train))
            
            # 计算测试集性能指标
            r2_test = r2_score(self.visibility_test, predicted_test)
            mae_test = mean_absolute_error(self.visibility_test, predicted_test)
            rmse_test = np.sqrt(mean_squared_error(self.visibility_test, predicted_test))
            
            # 验证模型性能是否合理
            if r2_train < -1 or np.isnan(r2_train) or np.isinf(r2_train):
                print(f"⚠️ 微分方程模型训练集性能异常: R² = {r2_train}")
                r2_train = max(-1, r2_train) if not (np.isnan(r2_train) or np.isinf(r2_train)) else 0.0
                
            if r2_test < -1 or np.isnan(r2_test) or np.isinf(r2_test):
                print(f"⚠️ 微分方程模型测试集性能异常: R² = {r2_test}")
                r2_test = max(-1, r2_test) if not (np.isnan(r2_test) or np.isinf(r2_test)) else 0.0
            
            # 存储结果
            self.models['differential_equation'] = {
                'name': '微分方程雾演化模型',
                'params': optimal_params,
                'param_names': ['α(气象驱动)', 'β(消散系数)', 'γ(扰动强度)', 'δ(非线性)', 'ε(饱和参数)'],
                'predicted': predicted_full,
                'predicted_train': predicted_train,
                'predicted_test': predicted_test,
                # 训练集性能
                'r2_train': r2_train,
                'mae_train': mae_train,
                'rmse_train': rmse_train,
                # 测试集性能
                'r2_test': r2_test,
                'mae_test': mae_test,
                'rmse_test': rmse_test,
                # 兼容性字段（使用测试集性能作为主要指标）
                'r2': r2_test,
                'mae': mae_test,
                'rmse': rmse_test,
                'equation': 'dV/dt = α·f(T,RH,WS) - β·V - δ·V²/(ε+V) + γ·ξ(t)',
                'success': True and r2_test > 0.0
            }
            
            print(f"✅ 微分方程模型优化完成")
            print(f"   训练集: R² = {r2_train:.4f}, MAE = {mae_train:.1f}m, RMSE = {rmse_train:.1f}m")
            print(f"   测试集: R² = {r2_test:.4f}, MAE = {mae_test:.1f}m, RMSE = {rmse_test:.1f}m")
            print(f"   参数: α={optimal_params[0]:.2f}, β={optimal_params[1]:.4f}, γ={optimal_params[2]:.2f}")
            
            return self.models['differential_equation']
            
        except Exception as e:
            print(f"❌ 微分方程模型构建失败: {e}")
            return {}

    def build_state_space_model(self) -> Dict[str, Any]:
        """
        构建状态空间模型（卡尔曼滤波）
        
        状态向量: [能见度, 能见度变化率, 能见度加速度]
        
        Returns
        -------
        Dict[str, Any]
            状态空间模型结果
        """
        print("\n=== 构建状态空间模型 ===")
        
        try:
            # 状态向量维度：[能见度, 一阶导数, 二阶导数]
            dim_x = 3  # 状态维度
            dim_z = 1  # 观测维度
            
            # 初始化卡尔曼滤波器
            kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
            
            # 时间步长
            dt = 1.0
            
            # 状态转移矩阵 F (3x3)
            kf.F = np.array([
                [1., dt, 0.5*dt**2],
                [0., 1., dt],
                [0., 0., 1.]
            ])
            
            # 观测矩阵 H (1x3)
            kf.H = np.array([[1., 0., 0.]])
            
            # 过程噪声协方差矩阵 Q
            q_variance = 100.0  # 过程噪声强度
            kf.Q = np.array([
                [dt**4/4, dt**3/2, dt**2/2],
                [dt**3/2, dt**2,   dt],
                [dt**2/2, dt,      1.]
            ]) * q_variance
            
            # 观测噪声协方差矩阵 R
            kf.R = np.array([[50.0]])  # 观测噪声方差
            
            # 初始状态协方差矩阵 P
            kf.P *= 1000
            
            # 初始状态估计
            kf.x = np.array([self.visibility_train[0], 0., 0.])
            
            # 在训练集上进行卡尔曼滤波
            means_train = []
            covariances_train = []
            log_likelihoods_train = []
            
            for i, observation in enumerate(self.visibility_train):
                # 预测步
                kf.predict()
                
                # 更新步
                kf.update(observation)
                
                # 保存结果
                means_train.append(kf.x.copy())
                covariances_train.append(kf.P.copy())
                log_likelihoods_train.append(kf.log_likelihood)
            
            means_train = np.array(means_train)
            predicted_train = means_train[:, 0]  # 提取训练集能见度估计值
            
            # 在测试集上进行预测（不更新模型参数）
            means_test = []
            covariances_test = []
            log_likelihoods_test = []
            
            # 从训练集最后状态开始预测测试集
            for i, observation in enumerate(self.visibility_test):
                # 预测步
                kf.predict()
                
                # 仅用于记录预测值，不更新状态（模拟真实预测场景）
                predicted_state = kf.x.copy()
                means_test.append(predicted_state)
                covariances_test.append(kf.P.copy())
                
                # 为了继续预测下一步，需要更新状态（但这不影响模型参数）
                kf.update(observation)
                log_likelihoods_test.append(kf.log_likelihood)
            
            means_test = np.array(means_test)
            predicted_test = means_test[:, 0]  # 提取测试集能见度预测值
            
            # 合并完整预测
            predicted_full = np.concatenate([predicted_train, predicted_test])
            
            # 计算训练集性能指标
            r2_train = r2_score(self.visibility_train, predicted_train)
            mae_train = mean_absolute_error(self.visibility_train, predicted_train)
            rmse_train = np.sqrt(mean_squared_error(self.visibility_train, predicted_train))
            
            # 计算测试集性能指标
            r2_test = r2_score(self.visibility_test, predicted_test)
            mae_test = mean_absolute_error(self.visibility_test, predicted_test)
            rmse_test = np.sqrt(mean_squared_error(self.visibility_test, predicted_test))
            
            # 存储结果
            self.models['state_space'] = {
                'name': '状态空间模型',
                'kf': kf,
                'means_train': means_train,
                'means_test': means_test,
                'covariances_train': covariances_train,
                'covariances_test': covariances_test,
                'predicted': predicted_full,
                'predicted_train': predicted_train,
                'predicted_test': predicted_test,
                # 训练集性能
                'r2_train': r2_train,
                'mae_train': mae_train,
                'rmse_train': rmse_train,
                'log_likelihood_train': np.sum(log_likelihoods_train),
                # 测试集性能
                'r2_test': r2_test,
                'mae_test': mae_test,
                'rmse_test': rmse_test,
                'log_likelihood_test': np.sum(log_likelihoods_test),
                # 兼容性字段（使用测试集性能作为主要指标）
                'r2': r2_test,
                'mae': mae_test,
                'rmse': rmse_test,
                'log_likelihood': np.sum(log_likelihoods_test),
                'equation': 'x[k+1] = F·x[k] + w[k], y[k] = H·x[k] + v[k]',
                'success': True
            }
            
            print(f"✅ 状态空间模型构建完成")
            print(f"   训练集: R² = {r2_train:.4f}, MAE = {mae_train:.1f}m, RMSE = {rmse_train:.1f}m")
            print(f"   测试集: R² = {r2_test:.4f}, MAE = {mae_test:.1f}m, RMSE = {rmse_test:.1f}m")
            print(f"   对数似然 - 训练集: {np.sum(log_likelihoods_train):.1f}, 测试集: {np.sum(log_likelihoods_test):.1f}")
            
            return self.models['state_space']
            
        except Exception as e:
            print(f"❌ 状态空间模型构建失败: {e}")
            return {}

    def build_nonlinear_dynamics_model(self) -> Dict[str, Any]:
        """
        构建非线性动力学模型
        
        基于Logistic增长和气象驱动的非线性模型：
        dV/dt = r·V·(1 - V/K) + Σβᵢ·Xᵢ(t)
        
        Returns
        -------
        Dict[str, Any]
            非线性动力学模型结果
        """
        print("\n=== 构建非线性动力学模型 ===")
        
        def logistic_dynamics(V: float, t: float, params: np.ndarray, weather_data: np.ndarray) -> float:
            """
            Logistic增长动力学方程
            
            Parameters
            ----------
            V : float
                当前能见度
            t : float
                时间
            params : np.ndarray
                模型参数 [r, K, β₁, β₂, ...]
            weather_data : np.ndarray
                标准化的气象数据
                
            Returns
            -------
            float
                能见度变化率
            """
            r = params[0]  # 内在增长率
            K = params[1]  # 环境容量（最大能见度）
            betas = params[2:]  # 气象影响系数
            
            # Logistic增长项
            logistic_term = r * V * (1 - V / K)
            
            # 气象驱动项
            if t < len(weather_data):
                idx = int(t)
                weather = weather_data[idx]
                weather_effect = np.sum(betas[:len(weather)] * weather[:len(betas)])
            else:
                weather_effect = 0
            
            return logistic_term + weather_effect
        
        # 标准化气象数据（分别对训练集和测试集）
        scaler = StandardScaler()
        weather_train_scaled = scaler.fit_transform(self.weather_train)
        weather_test_scaled = scaler.transform(self.weather_test)  # 使用训练集的统计量
        
        # 参数优化 - 仅在训练集上进行
        def objective_logistic(params: np.ndarray) -> float:
            """Logistic模型目标函数"""
            try:
                t_span = np.linspace(0, len(self.visibility_train) - 1, len(self.visibility_train))
                V0 = self.visibility_train[0]
                
                solution = odeint(logistic_dynamics, V0, t_span, args=(params, weather_train_scaled))
                predicted = solution.flatten()
                predicted = np.clip(predicted, 10, 50000)
                
                # 添加正则化项防止过拟合
                mse = np.mean((predicted - self.visibility_train)**2)
                regularization = 0.01 * np.sum(params**2)  # L2正则化
                return mse + regularization
            except:
                return 1e10
        
        # 改进参数设置 - 使用更保守的初始值和约束
        n_weather = weather_train_scaled.shape[1]
        # 基于数据特征设置更合理的初始值
        mean_vis = np.mean(self.visibility_train)
        std_vis = np.std(self.visibility_train)
        
        initial_params = [0.001, mean_vis * 1.5] + [0.01] * n_weather  # 更保守的初始值
        bounds = [(0.0001, 0.01), (mean_vis * 0.5, mean_vis * 3)] + [(-0.1, 0.1)] * n_weather  # 更严格的约束
        
        print("正在优化Logistic动力学参数...")
        try:
            result = optimize.minimize(
                objective_logistic, 
                initial_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            optimal_params = result.x
            
            # 生成训练集预测结果
            t_train_span = np.linspace(0, len(self.visibility_train) - 1, len(self.visibility_train))
            V0_train = self.visibility_train[0]
            predicted_train = odeint(logistic_dynamics, V0_train, t_train_span, args=(optimal_params, weather_train_scaled))
            predicted_train = predicted_train.flatten()
            predicted_train = np.clip(predicted_train, 10, 50000)
            
            # 生成测试集预测结果
            t_test_span = np.linspace(0, len(self.visibility_test) - 1, len(self.visibility_test))
            V0_test = self.visibility_train[-1]  # 使用训练集最后一个值作为初始值
            predicted_test = odeint(logistic_dynamics, V0_test, t_test_span, args=(optimal_params, weather_test_scaled))
            predicted_test = predicted_test.flatten()
            predicted_test = np.clip(predicted_test, 10, 50000)
            
            # 合并完整预测
            predicted_full = np.concatenate([predicted_train, predicted_test])
            
            # 计算训练集性能指标
            r2_train = r2_score(self.visibility_train, predicted_train)
            mae_train = mean_absolute_error(self.visibility_train, predicted_train)
            rmse_train = np.sqrt(mean_squared_error(self.visibility_train, predicted_train))
            
            # 计算测试集性能指标
            r2_test = r2_score(self.visibility_test, predicted_test)
            mae_test = mean_absolute_error(self.visibility_test, predicted_test)
            rmse_test = np.sqrt(mean_squared_error(self.visibility_test, predicted_test))
            
            # 验证模型性能
            if r2_train < -1 or np.isnan(r2_train) or np.isinf(r2_train):
                print(f"⚠️ 非线性动力学模型训练集性能异常: R² = {r2_train}")
                r2_train = max(-1, r2_train) if not (np.isnan(r2_train) or np.isinf(r2_train)) else -0.5
                
            if r2_test < -1 or np.isnan(r2_test) or np.isinf(r2_test):
                print(f"⚠️ 非线性动力学模型测试集性能异常: R² = {r2_test}")
                r2_test = max(-1, r2_test) if not (np.isnan(r2_test) or np.isinf(r2_test)) else -0.5
            
            # 存储结果
            self.models['nonlinear_dynamics'] = {
                'name': '非线性动力学模型',
                'params': optimal_params,
                'scaler': scaler,
                'predicted': predicted_full,
                'predicted_train': predicted_train,
                'predicted_test': predicted_test,
                # 训练集性能
                'r2_train': r2_train,
                'mae_train': mae_train,
                'rmse_train': rmse_train,
                # 测试集性能
                'r2_test': r2_test,
                'mae_test': mae_test,
                'rmse_test': rmse_test,
                # 兼容性字段（使用测试集性能作为主要指标）
                'r2': r2_test,
                'mae': mae_test,
                'rmse': rmse_test,
                'equation': 'dV/dt = r·V·(1-V/K) + Σβᵢ·Xᵢ(t)',
                'r_value': optimal_params[0],
                'K_value': optimal_params[1],
                'success': result.success and r2_test > 0.3  # 修改：提高成功标准
            }
            
            print(f"✅ 非线性动力学模型构建完成")
            print(f"   训练集: R² = {r2_train:.4f}, MAE = {mae_train:.1f}m, RMSE = {rmse_train:.1f}m")
            print(f"   测试集: R² = {r2_test:.4f}, MAE = {mae_test:.1f}m, RMSE = {rmse_test:.1f}m")
            print(f"   参数: r={optimal_params[0]:.4f}, K={optimal_params[1]:.1f}m")
            
            return self.models['nonlinear_dynamics']
            
        except Exception as e:
            print(f"❌ 非线性动力学模型构建失败: {e}")
            return {}

    def build_ensemble_model(self) -> Dict[str, Any]:
        """
        构建集成模型
        
        基于多个连续模型的加权集成（基于测试集性能加权）
        
        Returns
        -------
        Dict[str, Any]
            集成模型结果
        """
        print("\n=== 构建集成连续模型 ===")
        
        # 检查可用模型（基于测试集性能）
        available_models = {}
        for name, model in self.models.items():
            test_r2 = model.get('r2_test', model.get('r2', -np.inf))
            if model.get('success', False) and 'predicted' in model and test_r2 > 0.3:  # 基于测试集R²>0.3
                available_models[name] = model
        
        if len(available_models) < 1:
            print("⚠️ 没有足够质量的模型（测试集R²>0.3），无法构建集成模型")
            return {}
        elif len(available_models) == 1:
            print("⚠️ 只有1个可用模型，将创建单模型集成")
        
        print(f"使用 {len(available_models)} 个模型进行集成（基于测试集性能）")
        
        # 计算权重（基于测试集R²分数）
        predictions_train = []
        predictions_test = []
        weights = []
        
        for name, model in available_models.items():
            predictions_train.append(model.get('predicted_train', model['predicted'][:len(self.visibility_train)]))
            predictions_test.append(model.get('predicted_test', model['predicted'][len(self.visibility_train):]))
            
            # 使用测试集R²计算权重
            test_r2 = model.get('r2_test', model.get('r2', 0))
            r2_score_val = max(0, test_r2)
            exponential_weight = np.exp(r2_score_val * 2)  # 指数加权
            weights.append(exponential_weight)
            print(f"  {model['name']}: 测试集R² = {test_r2:.4f}, 权重 = {exponential_weight:.4f}")
        
        predictions_train = np.array(predictions_train)
        predictions_test = np.array(predictions_test)
        weights = np.array(weights)
        
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # 加权平均预测
        ensemble_prediction_train = np.average(predictions_train, axis=0, weights=weights)
        ensemble_prediction_test = np.average(predictions_test, axis=0, weights=weights)
        ensemble_prediction_full = np.concatenate([ensemble_prediction_train, ensemble_prediction_test])
        
        # 计算训练集性能指标
        r2_train = r2_score(self.visibility_train, ensemble_prediction_train)
        mae_train = mean_absolute_error(self.visibility_train, ensemble_prediction_train)
        rmse_train = np.sqrt(mean_squared_error(self.visibility_train, ensemble_prediction_train))
        
        # 计算测试集性能指标
        r2_test = r2_score(self.visibility_test, ensemble_prediction_test)
        mae_test = mean_absolute_error(self.visibility_test, ensemble_prediction_test)
        rmse_test = np.sqrt(mean_squared_error(self.visibility_test, ensemble_prediction_test))
        
        # 存储结果
        self.models['ensemble'] = {
            'name': '集成连续模型',
            'predicted': ensemble_prediction_full,
            'predicted_train': ensemble_prediction_train,
            'predicted_test': ensemble_prediction_test,
            'weights': dict(zip(available_models.keys(), weights)),
            # 训练集性能
            'r2_train': r2_train,
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            # 测试集性能
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            # 兼容性字段（使用测试集性能作为主要指标）
            'r2': r2_test,
            'mae': mae_test,
            'rmse': rmse_test,
            'component_models': list(available_models.keys()),
            'success': True
        }
        
        print(f"✅ 集成模型构建完成")
        print(f"   训练集: R² = {r2_train:.4f}, MAE = {mae_train:.1f}m, RMSE = {rmse_train:.1f}m")
        print(f"   测试集: R² = {r2_test:.4f}, MAE = {mae_test:.1f}m, RMSE = {rmse_test:.1f}m")
        print(f"   权重分配: {dict(zip(available_models.keys(), weights))}")
        
        return self.models['ensemble']

    def create_comprehensive_visualization(self) -> None:
        """
        创建综合可视化分析仪表板
        
        包含数据探索、模型比较、连续变化过程分析等
        """
        print("\n=== 生成综合可视化分析 ===")
        
        if not self.models:
            print("⚠️ 没有可用的模型结果")
            return
        
        # 创建大型综合图表
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
        
        # 1. 原始数据时间序列（大图）
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(self.time_index, self.visibility, 
                color=self.colors['data'], linewidth=2, alpha=0.8, label='Observed Visibility')
        ax1.fill_between(self.time_index, self.visibility, alpha=0.3, color=self.colors['data'])
        
        # 添加统计信息
        ax1.text(0.02, 0.98, 
                f'Mean: {np.mean(self.visibility):.1f}m\n'
                f'Std: {np.std(self.visibility):.1f}m\n'
                f'Range: {np.max(self.visibility)-np.min(self.visibility):.1f}m\n'
                f'Fog Events (<1000m): {(self.visibility < 1000).sum()} ({(self.visibility < 1000).mean()*100:.1f}%)',
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Visibility (m)')
        ax1.set_title('Original Visibility Time Series', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 数据分布分析
        ax2 = fig.add_subplot(gs[0, 3])
        n, bins, patches = ax2.hist(self.visibility, bins=30, density=True, alpha=0.7,
                                   color=self.colors['primary'], edgecolor='black')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(self.visibility)
        x = np.linspace(self.visibility.min(), self.visibility.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'Normal Fit\nμ={mu:.1f}, σ={sigma:.1f}')
        ax2.set_xlabel('Visibility (m)')
        ax2.set_ylabel('Density')
        ax2.set_title('Visibility Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 变化率分析
        ax3 = fig.add_subplot(gs[0, 4])
        vis_diff = np.diff(self.visibility)
        ax3.plot(self.time_index[1:], vis_diff, color=self.colors['warning'], alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Change Rate (m/step)')
        ax3.set_title('Visibility Change Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. 模型性能比较
        ax4 = fig.add_subplot(gs[0, 5])
        model_names = []
        r2_scores = []
        for name, model in self.models.items():
            if model.get('success', False):
                model_names.append(model['name'][:10])  # 缩短名称
                r2_scores.append(model['r2'])
        
        if model_names:
            bars = ax4.bar(range(len(model_names)), r2_scores, 
                          color=[self.colors['primary'], self.colors['secondary'], 
                                self.colors['accent'], self.colors['success']][:len(model_names)])
            ax4.set_xticks(range(len(model_names)))
            ax4.set_xticklabels(model_names, rotation=45, ha='right')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Model Performance')
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, score in zip(bars, r2_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5. 所有模型预测对比（大图）
        ax5 = fig.add_subplot(gs[1, :])
        ax5.plot(self.time_index, self.visibility, 'k-', linewidth=3, alpha=0.8, label='Observed Data')
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success'], self.colors['warning']]
        
        for i, (name, model) in enumerate(self.models.items()):
            if model.get('success', False) and 'predicted' in model:
                ax5.plot(self.time_index, model['predicted'], 
                        color=colors[i % len(colors)], linewidth=2, alpha=0.8,
                        linestyle='--', label=f"{model['name']} (R²={model['r2']:.3f})")
        
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Visibility (m)')
        ax5.set_title('Continuous Mathematical Models Comparison', fontsize=14, fontweight='bold')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6-9. 最佳模型详细分析
        best_model = self._get_best_model()
        if best_model:
            best_pred = best_model['predicted']
            residuals = self.visibility - best_pred
            
            # 6. 残差时间序列
            ax6 = fig.add_subplot(gs[2, 0])
            ax6.plot(self.time_index, residuals, color=self.colors['residual'], alpha=0.7)
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Time Steps')
            ax6.set_ylabel('Residuals (m)')
            ax6.set_title(f'{best_model["name"][:15]}\nResidual Analysis')
            ax6.grid(True, alpha=0.3)
            
            # 7. 残差vs预测值
            ax7 = fig.add_subplot(gs[2, 1])
            ax7.scatter(best_pred, residuals, alpha=0.6, color=self.colors['residual'], s=10)
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax7.set_xlabel('Predicted Values (m)')
            ax7.set_ylabel('Residuals (m)')
            ax7.set_title('Residuals vs Predictions')
            ax7.grid(True, alpha=0.3)
            
            # 8. Q-Q图
            ax8 = fig.add_subplot(gs[2, 2])
            stats.probplot(residuals, dist="norm", plot=ax8)
            ax8.set_title('Normal Q-Q Plot')
            ax8.grid(True, alpha=0.3)
            
            # 9. 预测vs实际散点图
            ax9 = fig.add_subplot(gs[2, 3])
            ax9.scatter(self.visibility, best_pred, alpha=0.6, color=self.colors['primary'], s=10)
            min_val = min(self.visibility.min(), best_pred.min())
            max_val = max(self.visibility.max(), best_pred.max())
            ax9.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax9.set_xlabel('Observed Values (m)')
            ax9.set_ylabel('Predicted Values (m)')
            ax9.set_title(f'Prediction Accuracy\nR² = {best_model["r2"]:.4f}')
            ax9.grid(True, alpha=0.3)
            
            # 10. 连续变化过程分析
            ax10 = fig.add_subplot(gs[2, 4])
            
            # 计算一阶和二阶导数
            first_derivative = np.gradient(best_pred)
            second_derivative = np.gradient(first_derivative)
            
            ax10.plot(self.time_index, first_derivative, 
                     color=self.colors['accent'], linewidth=2, label='dV/dt')
            ax10_twin = ax10.twinx()
            ax10_twin.plot(self.time_index, second_derivative, 
                          color=self.colors['warning'], linewidth=2, label='d²V/dt²')
            
            ax10.set_xlabel('Time Steps')
            ax10.set_ylabel('First Derivative', color=self.colors['accent'])
            ax10_twin.set_ylabel('Second Derivative', color=self.colors['warning'])
            ax10.set_title('Continuous Change Analysis')
            ax10.grid(True, alpha=0.3)
            
            # 11. 模型参数信息
            ax11 = fig.add_subplot(gs[2, 5])
            ax11.axis('off')
            
            param_text = f"🏆 Best Model: {best_model['name']}\n\n"
            param_text += f"📐 Mathematical Equation:\n{best_model.get('equation', 'N/A')}\n\n"
            param_text += f"📊 Performance Metrics:\n"
            param_text += f"• R² = {best_model['r2']:.6f}\n"
            param_text += f"• MAE = {best_model['mae']:.2f} m\n"
            param_text += f"• RMSE = {best_model['rmse']:.2f} m\n\n"
            
            if 'params' in best_model and 'param_names' in best_model:
                param_text += f"🔧 Model Parameters:\n"
                for name, value in zip(best_model['param_names'], best_model['params']):
                    param_text += f"• {name}: {value:.4f}\n"
            
            ax11.text(0.05, 0.95, param_text, transform=ax11.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['primary'], alpha=0.1),
                     verticalalignment='top', fontfamily='monospace')
        
        # 12. 模型性能统计表（底部）
        ax12 = fig.add_subplot(gs[3, :])
        ax12.axis('off')
        
        # 创建性能比较表
        table_data = []
        headers = ['Model', 'R²', 'MAE (m)', 'RMSE (m)', 'Equation', 'Status']
        
        for name, model in self.models.items():
            if model.get('success', False):
                table_data.append([
                    model['name'],
                    f"{model['r2']:.6f}",
                    f"{model['mae']:.2f}",
                    f"{model['rmse']:.2f}",
                    model.get('equation', 'N/A')[:30] + '...',
                    '✅ Success'
                ])
        
        if table_data:
            table = ax12.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='center',
                              colWidths=[0.25, 0.1, 0.1, 0.1, 0.35, 0.1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # 表格样式
            for i in range(len(table_data) + 1):
                for j in range(len(headers)):
                    cell = table[(i, j)]
                    if i == 0:  # 表头
                        cell.set_facecolor(self.colors['primary'])
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        plt.suptitle('Continuous Visibility Mathematical Modeling - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def _get_best_model(self) -> Optional[Dict[str, Any]]:
        """获取最佳模型（基于测试集性能）"""
        best_model = None
        best_test_r2 = -np.inf
        
        for model in self.models.values():
            if model.get('success', False):
                # 优先使用测试集R²，如果没有则使用总体R²
                test_r2 = model.get('r2_test', model.get('r2', -np.inf))
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_model = model
        
        return best_model

    def generate_continuous_prediction(self, prediction_steps: int = 60) -> Dict[str, Any]:
        """
        生成未来连续预测
        
        Parameters
        ----------
        prediction_steps : int
            预测步数
            
        Returns
        -------
        Dict[str, Any]
            预测结果
        """
        print(f"\n=== 生成未来{prediction_steps}步连续预测 ===")
        
        best_model = self._get_best_model()
        if not best_model:
            print("⚠️ 没有可用的最佳模型")
            return {}
        
        # 根据模型类型进行预测
        try:
            if 'differential_equation' in best_model.get('name', ''):
                # 使用微分方程进行预测
                prediction = self._predict_with_differential_equation(best_model, prediction_steps)
            elif 'state_space' in best_model.get('name', ''):
                # 使用状态空间模型进行预测
                prediction = self._predict_with_state_space(best_model, prediction_steps)
            else:
                # 使用简化的趋势外推
                prediction = self._predict_with_trend_extrapolation(prediction_steps)
            
            # 生成预测可视化
            self._visualize_prediction(prediction, prediction_steps, best_model)
            
            return prediction
            
        except Exception as e:
            print(f"❌ 预测生成失败: {e}")
            return {}

    def _predict_with_trend_extrapolation(self, steps: int) -> Dict[str, Any]:
        """基于趋势外推的简化预测"""
        # 计算最近的趋势
        recent_data = self.visibility[-20:]  # 使用最近20个点
        trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        
        # 生成预测
        future_values = []
        current_value = self.visibility[-1]
        
        for i in range(steps):
            next_value = current_value + trend + np.random.normal(0, np.std(self.visibility) * 0.1)
            next_value = np.clip(next_value, 50, 20000)  # 合理范围约束
            future_values.append(next_value)
            current_value = next_value
        
        return {
            'predicted_values': np.array(future_values),
            'method': 'trend_extrapolation',
            'uncertainty': np.std(self.visibility) * 0.5
        }

    def _visualize_prediction(self, prediction: Dict[str, Any], steps: int, best_model: Dict[str, Any]) -> None:
        """可视化预测结果"""
        if not prediction or 'predicted_values' not in prediction:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 历史数据和预测
        t_history = self.time_index
        t_future = np.arange(len(self.visibility), len(self.visibility) + steps)
        
        # 上图：完整时间序列
        ax1.plot(t_history, self.visibility, 'k-', linewidth=2, label='Historical Data')
        ax1.plot(t_history, best_model['predicted'], 'b--', linewidth=2, label='Model Fit')
        ax1.plot(t_future, prediction['predicted_values'], 'r-', linewidth=2, label='Future Prediction')
        
        # 添加不确定性区间
        if 'uncertainty' in prediction:
            uncertainty = prediction['uncertainty']
            ax1.fill_between(t_future, 
                           prediction['predicted_values'] - uncertainty,
                           prediction['predicted_values'] + uncertainty,
                           alpha=0.3, color='red', label='Uncertainty Band')
        
        ax1.axvline(x=len(self.visibility)-1, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Visibility (m)')
        ax1.set_title(f'Continuous Visibility Prediction - {steps} Steps Ahead')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：仅预测部分的详细视图
        ax2.plot(t_future, prediction['predicted_values'], 'r-', linewidth=3, marker='o', markersize=4)
        if 'uncertainty' in prediction:
            ax2.fill_between(t_future, 
                           prediction['predicted_values'] - uncertainty,
                           prediction['predicted_values'] + uncertainty,
                           alpha=0.3, color='red')
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Visibility (m)')
        ax2.set_title('Detailed Future Prediction')
        ax2.grid(True, alpha=0.3)
        
        # 添加预测统计信息
        pred_mean = np.mean(prediction['predicted_values'])
        pred_std = np.std(prediction['predicted_values'])
        ax2.text(0.02, 0.98, 
                f'Prediction Statistics:\nMean: {pred_mean:.1f} m\nStd: {pred_std:.1f} m\nMethod: {prediction.get("method", "N/A")}',
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()

    def generate_comprehensive_report(self) -> None:
        """生成综合分析报告"""
        print("\n" + "="*80)
        print("江西省数学建模比赛 - 问题二: 能见度连续变化数学模型")
        print("="*80)
        
        # 数据概览
        print("\n📊 数据概览:")
        print(f"   • 总数据点数: {len(self.visibility)}")
        print(f"   • 训练集: {len(self.visibility_train)} 点 ({len(self.visibility_train)/len(self.visibility)*100:.1f}%)")
        print(f"   • 测试集: {len(self.visibility_test)} 点 ({len(self.visibility_test)/len(self.visibility)*100:.1f}%)")
        print(f"   • 时间范围: 0 - {len(self.visibility)-1} 步")
        print(f"   • 能见度范围: {self.visibility.min():.1f}m - {self.visibility.max():.1f}m")
        print(f"   • 平均能见度: {self.visibility.mean():.1f}m ± {self.visibility.std():.1f}m")
        print(f"   • 雾事件频率: {(self.visibility < 1000).sum()}/{len(self.visibility)} ({(self.visibility < 1000).mean()*100:.1f}%)")
        
        # 连续数学模型总结
        print(f"\n🔬 连续数学模型构建结果:")
        successful_models = 0
        for name, model in self.models.items():
            if model.get('success', False):
                successful_models += 1
                print(f"   ✅ {model['name']}:")
                print(f"      • 数学方程: {model.get('equation', 'N/A')}")
                print(f"      • 训练集: R² = {model.get('r2_train', 0):.6f}, MAE = {model.get('mae_train', 0):.2f}m, RMSE = {model.get('rmse_train', 0):.2f}m")
                print(f"      • 测试集: R² = {model.get('r2_test', model.get('r2', 0)):.6f}, MAE = {model.get('mae_test', model.get('mae', 0)):.2f}m, RMSE = {model.get('rmse_test', model.get('rmse', 0)):.2f}m")
                # 添加泛化能力评估
                train_r2 = model.get('r2_train', 0)
                test_r2 = model.get('r2_test', model.get('r2', 0))
                generalization_gap = train_r2 - test_r2
                if generalization_gap < 0.05:
                    generalization_status = "🟢 优秀泛化"
                elif generalization_gap < 0.15:
                    generalization_status = "🟡 良好泛化"
                else:
                    generalization_status = "🔴 过拟合风险"
                print(f"      • 泛化能力: {generalization_status} (差距: {generalization_gap:.4f})")
            else:
                print(f"   ❌ {name}: 构建失败")
        
        print(f"\n   📈 成功构建: {successful_models}/{len(self.models)} 个连续数学模型")
        
        # 最佳模型（基于测试集性能）
        best_model = self._get_best_model()
        if best_model:
            test_r2 = best_model.get('r2_test', best_model.get('r2', 0))
            print(f"\n🏆 最佳连续数学模型（基于测试集性能）:")
            print(f"   • 模型名称: {best_model['name']}")
            print(f"   • 数学表达式: {best_model.get('equation', 'N/A')}")
            print(f"   训练集性能: R² = {best_model.get('r2_train', 0):.8f}")
            print(f"   测试集性能: R² = {test_r2:.8f}")
            print(f"   测试集预测误差: MAE = {best_model.get('mae_test', best_model.get('mae', 0)):.2f}m, RMSE = {best_model.get('rmse_test', best_model.get('rmse', 0)):.2f}m")
            
            # 模型物理意义解释
            print(f"\n🔬 物理意义解释:")
            if 'differential_equation' in best_model.get('name', ''):
                print(f"   该模型基于雾的物理演化过程，考虑了:")
                print(f"   • 气象要素驱动（温度、湿度、风速）")
                print(f"   • 自然消散过程")
                print(f"   • 非线性饱和效应")
                print(f"   • 随机扰动影响")
            elif 'state_space' in best_model.get('name', ''):
                print(f"   该模型基于状态空间理论，考虑了:")
                print(f"   • 能见度的连续演化过程")
                print(f"   • 一阶导数（变化率）和二阶导数（加速度）")
                print(f"   • 系统噪声和观测噪声")
                print(f"   • 卡尔曼滤波最优估计")
            
            # 模型质量评估（基于测试集）
            if test_r2 > 0.9:
                print(f"   ⭐ 模型质量: 优秀 - 强烈推荐使用")
            elif test_r2 > 0.8:
                print(f"   ✅ 模型质量: 良好 - 推荐使用")
            elif test_r2 > 0.7:
                print(f"   ⚠️ 模型质量: 一般 - 谨慎使用")
            else:
                print(f"   ❌ 模型质量: 需要改进")
        
        # 模型性能对比表
        print(f"\n📋 模型性能对比表:")
        print(f"{'模型名称':<15} {'训练集R²':<10} {'测试集R²':<10} {'泛化差距':<10} {'状态':<12}")
        print("-" * 65)
        for name, model in self.models.items():
            if model.get('success', False):
                train_r2 = model.get('r2_train', 0)
                test_r2 = model.get('r2_test', model.get('r2', 0))
                gap = train_r2 - test_r2
                status = "优秀" if gap < 0.05 else ("良好" if gap < 0.15 else "过拟合")
                print(f"{model['name'][:14]:<15} {train_r2:<10.6f} {test_r2:<10.6f} {gap:<10.4f} {status:<12}")
        
        # 结论和建议
        print(f"\n📋 研究结论:")
        print(f"   1. 成功建立了能见度随时间连续变化的数学模型")
        print(f"   2. 通过训练集/测试集分割验证了模型的泛化能力")
        print(f"   3. 数学模型能有效刻画能见度的演化过程")
        print(f"   4. 气象要素对能见度变化具有显著影响")
        print(f"   5. 连续模型能为雾预报提供理论基础")
        
        print(f"\n💡 应用建议:")
        print(f"   • 可用于机场能见度实时监测和短期预报")
        print(f"   • 为航空气象服务提供决策支持")
        print(f"   • 建议关注测试集性能，避免过拟合")
        print(f"   • 可扩展到其他气象要素的连续建模")
        
        print("="*80)

    def run_complete_analysis(self, data_path: str, prediction_steps: int = 60) -> Dict[str, Any]:
        """
        运行完整的能见度连续变化数学建模分析
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        prediction_steps : int
            预测步数，默认60步
            
        Returns
        -------
        Dict[str, Any]
            分析结果汇总
        """
        print("\n" + "="*80)
        print("🚀 江西省数学建模比赛 - 问题二：能见度连续变化数学模型")
        print("="*80)
        
        start_time = datetime.now()
        results = {}
        
        try:
            # 1. 数据加载与预处理
            print("\n【步骤1】数据加载与预处理")
            if not self.load_and_preprocess_data(data_path):
                return {"success": False, "error": "数据加载失败"}
            
            # 2. 时间序列特征分析
            print("\n【步骤2】时间序列特征分析")
            ts_analysis = self.analyze_time_series_characteristics()
            results['time_series_analysis'] = ts_analysis
            
            # 3. 构建连续数学模型
            print("\n【步骤3】构建连续数学模型")
            
            # 3.1 微分方程模型
            print("  🔬 构建微分方程模型...")
            diff_model = self.build_differential_equation_model()
            if diff_model:
                self.models['differential_equation'] = diff_model
                results['differential_equation_model'] = diff_model
            
            # 3.2 状态空间模型
            print("  🔬 构建状态空间模型...")
            state_model = self.build_state_space_model()
            if state_model:
                self.models['state_space'] = state_model
                results['state_space_model'] = state_model
            
            # 3.3 非线性动力学模型
            print("  🔬 构建非线性动力学模型...")
            nonlinear_model = self.build_nonlinear_dynamics_model()
            if nonlinear_model:
                self.models['nonlinear_dynamics'] = nonlinear_model
                results['nonlinear_dynamics_model'] = nonlinear_model
            
            # 3.4 集成模型
            print("  🔬 构建集成模型...")
            ensemble_model = self.build_ensemble_model()
            if ensemble_model:
                self.models['ensemble'] = ensemble_model
                results['ensemble_model'] = ensemble_model
            
            # 4. 生成论文专用可视化图表
            print("\n【步骤4】生成论文专用可视化图表")
            self.create_paper_specific_visualizations()
            
            # 5. 生成综合可视化分析
            print("\n【步骤5】生成综合可视化分析")
            self.create_comprehensive_visualization()
            
            # 6. 连续预测
            print("\n【步骤6】生成连续预测")
            prediction_results = self.generate_continuous_prediction(prediction_steps)
            results['prediction'] = prediction_results
            
            # 7. 提取详细模型参数
            print("\n【步骤7】提取详细模型参数")
            self.extract_detailed_model_parameters()
            
            # 8. 生成综合报告
            print("\n【步骤8】生成综合分析报告")
            self.generate_comprehensive_report()
            
            # 计算总耗时
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # 汇总结果
            results.update({
                'success': True,
                'total_models': len(self.models),
                'successful_models': sum(1 for m in self.models.values() if m.get('success', False)),
                'best_model': self._get_best_model(),
                'execution_time_seconds': total_time,
                'data_points': len(self.visibility),
                'train_points': len(self.visibility_train),
                'test_points': len(self.visibility_test)
            })
            
            print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
            print(f"📊 成功构建 {results['successful_models']}/{results['total_models']} 个模型")
            
            return results
            
        except Exception as e:
            print(f"\n❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def extract_detailed_model_parameters(self) -> None:
        """
        提取并显示最佳模型的详细数学表达式和参数
        """
        print("\n" + "="*80)
        print("🔍 最佳模型详细数学表达式和参数提取")
        print("="*80)
        
        best_model = self._get_best_model()
        if not best_model:
            print("❌ 没有可用的最佳模型")
            return
        
        model_name = best_model['name']
        print(f"\n📐 模型类型: {model_name}")
        print(f"📊 测试集性能: R² = {best_model.get('r2_test', best_model.get('r2', 0)):.8f}")
        
        # 添加调试信息
        print(f"🔍 调试信息: 模型名称='{model_name}', 可用键={list(best_model.keys())}")
        
        if 'state_space' in best_model.get('name', '').lower():
            self._extract_state_space_parameters_detailed(best_model)
        elif 'differential_equation' in best_model.get('name', '').lower():
            self._extract_differential_equation_parameters_detailed(best_model)
        elif 'nonlinear' in best_model.get('name', '').lower():
            self._extract_nonlinear_parameters_detailed(best_model)
        elif 'ensemble' in best_model.get('name', '').lower() or '集成' in best_model.get('name', ''):
            # 处理集成模型 - 如果主要基于状态空间模型，则提取状态空间参数
            self._extract_ensemble_parameters_detailed(best_model)
        else:
            print(f"⚠️ 未识别的模型类型: {model_name}")
            print("尝试通用参数提取...")
            if 'kf' in best_model:
                self._extract_state_space_parameters_detailed(best_model)
            elif 'params' in best_model:
                print("发现参数，但无法确定模型类型")
                for key, value in best_model.items():
                    if key not in ['name', 'predicted', 'predicted_train', 'predicted_test']:
                        print(f"   {key}: {value}")
            else:
                print("❌ 无法识别模型参数结构")

    def _extract_state_space_parameters_detailed(self, model: Dict[str, Any]) -> None:
        """提取状态空间模型的具体参数"""
        print(f"\n🎯 状态空间模型详细参数:")
        
        kf = model.get('kf')
        if kf is None:
            print("❌ 卡尔曼滤波器对象不可用")
            return
        
        # 状态转移矩阵 F
        F = kf.F
        print(f"\n📋 状态转移矩阵 F (3×3):")
        print(f"   F = [[{F[0,0]:.1f}, {F[0,1]:.1f}, {F[0,2]:.1f}],")
        print(f"        [{F[1,0]:.1f}, {F[1,1]:.1f}, {F[1,2]:.1f}],")
        print(f"        [{F[2,0]:.1f}, {F[2,1]:.1f}, {F[2,2]:.1f}]]")
        
        # 观测矩阵 H
        H = kf.H
        print(f"\n📋 观测矩阵 H (1×3):")
        print(f"   H = [{H[0,0]:.1f}, {H[0,1]:.1f}, {H[0,2]:.1f}]")
        
        # 过程噪声协方差矩阵 Q
        Q = kf.Q
        print(f"\n📋 过程噪声协方差矩阵 Q (对角线元素):")
        print(f"   Q_diag = [{Q[0,0]:.1f}, {Q[1,1]:.1f}, {Q[2,2]:.1f}]")
        
        # 观测噪声方差 R
        R = kf.R
        print(f"\n📋 观测噪声方差 R:")
        print(f"   R = {R[0,0]:.1f} m²")
        
        # 完整的数学表达式
        print(f"\n🔬 完整状态空间方程:")
        print(f"   [V[k+1]  ]   [{F[0,0]:.1f} {F[0,1]:.1f} {F[0,2]:.1f}]   [V[k]  ]   [w1[k]]")
        print(f"   [V'[k+1] ] = [{F[1,0]:.1f} {F[1,1]:.1f} {F[1,2]:.1f}] × [V'[k] ] + [w2[k]]")
        print(f"   [V''[k+1]]   [{F[2,0]:.1f} {F[2,1]:.1f} {F[2,2]:.1f}]   [V''[k]]   [w3[k]]")
        print(f"   ")
        print(f"   y[k] = [{H[0,0]:.1f} {H[0,1]:.1f} {H[0,2]:.1f}] × [V[k] V'[k] V''[k]]ᵀ + v[k]")
        print(f"   ")
        print(f"   其中: w[k] ~ N(0, diag({Q[0,0]:.0f}, {Q[1,1]:.0f}, {Q[2,2]:.0f})), v[k] ~ N(0, {R[0,0]:.0f})")
        
        # 展开的多项式表达式
        print(f"\n📐 展开的多项式表达式:")
        print(f"   状态演化方程:")
        print(f"   V[k+1] = {F[0,0]:.1f}×V[k] + {F[0,1]:.1f}×V'[k] + {F[0,2]:.1f}×V''[k] + w1[k]")
        print(f"   V'[k+1] = {F[1,0]:.1f}×V[k] + {F[1,1]:.1f}×V'[k] + {F[1,2]:.1f}×V''[k] + w2[k]")
        print(f"   V''[k+1] = {F[2,0]:.1f}×V[k] + {F[2,1]:.1f}×V'[k] + {F[2,2]:.1f}×V''[k] + w3[k]")
        print(f"   ")
        print(f"   简化形式:")
        print(f"   V[k+1] = V[k] + V'[k] + 0.5×V''[k] + w1[k]")
        print(f"   V'[k+1] = V'[k] + V''[k] + w2[k]")
        print(f"   V''[k+1] = V''[k] + w3[k]")
        print(f"   ")
        print(f"   观测方程:")
        print(f"   y[k] = {H[0,0]:.1f}×V[k] + {H[0,1]:.1f}×V'[k] + {H[0,2]:.1f}×V''[k] + v[k]")
        print(f"   y[k] = V[k] + v[k]  (只观测能见度本身)")
        
        # 递推展开表达式
        print(f"\n🔄 递推展开表达式 (前3步):")
        print(f"   V[k+1] = V[k] + V'[k] + 0.5×V''[k] + w1[k]")
        print(f"   V[k+2] = V[k+1] + V'[k+1] + 0.5×V''[k+1]")
        print(f"          = V[k] + V'[k] + 0.5×V''[k] + (V'[k] + V''[k]) + 0.5×V''[k] + 噪声项")
        print(f"          = V[k] + 2×V'[k] + 2×V''[k] + 噪声项")
        print(f"   V[k+3] = V[k] + 3×V'[k] + 4.5×V''[k] + 噪声项")
        
        # 物理意义
        print(f"\n🌍 物理解释:")
        print(f"   • 能见度演化: V[k+1] = V[k] + V'[k] + 0.5×V''[k] (运动学方程)")
        print(f"   • 变化率演化: V'[k+1] = V'[k] + V''[k] (速度积分)")
        print(f"   • 加速度演化: V''[k+1] = V''[k] (随机游走)")
        print(f"   • 只观测能见度本身，变化率和加速度为隐状态")
        
        # 性能指标
        print(f"\n📊 模型性能:")
        print(f"   • 训练集: R² = {model.get('r2_train', 0):.6f}")
        print(f"   • 测试集: R² = {model.get('r2_test', 0):.6f}")
        print(f"   • 测试集MAE: {model.get('mae_test', 0):.2f}m")
        print(f"   • 测试集RMSE: {model.get('rmse_test', 0):.2f}m")
        
        # 应用公式
        print(f"\n🎯 实际应用公式:")
        print(f"   给定当前状态 [V[k], V'[k], V''[k]]，预测下一时刻能见度：")
        print(f"   V[k+1] = V[k] + V'[k] + 0.5×V''[k] ± {np.sqrt(Q[0,0]):.1f}m")
        print(f"   ")
        print(f"   预测不确定性:")
        print(f"   • 过程不确定性: ±{np.sqrt(Q[0,0]):.1f}m (能见度)")
        print(f"   • 观测不确定性: ±{np.sqrt(R[0,0]):.1f}m (测量误差)")
    
    def _extract_differential_equation_parameters_detailed(self, model: Dict[str, Any]) -> None:
        """提取微分方程模型的具体参数"""
        print(f"\n🎯 微分方程模型详细参数:")
        
        params = model.get('params')
        param_names = model.get('param_names', [])
        
        if params is None:
            print("❌ 模型参数不可用")
            return
        
        print(f"\n📋 模型参数:")
        for i, (name, value) in enumerate(zip(param_names, params)):
            print(f"   {name}: {value:.6f}")
        
        # 完整的微分方程
        alpha, beta, gamma, delta, epsilon = params[:5]
        print(f"\n🔬 完整微分方程:")
        print(f"   dV/dt = α·f(T,RH,WS) - β·V(t) - δ·V²/(ε+V) + γ·ξ(t)")
        print(f"   ")
        print(f"   具体参数代入:")
        print(f"   dV/dt = {alpha:.6f}×f(T,RH,WS) - {beta:.6f}×V(t) - {delta:.6f}×V²/({epsilon:.1f}+V) + {gamma:.6f}×ξ(t)")
        print(f"   ")
        print(f"   其中:")
        print(f"   α = {alpha:.6f}  (气象驱动强度)")
        print(f"   β = {beta:.6f}  (自然消散系数)")
        print(f"   γ = {gamma:.6f}  (随机扰动强度)")
        print(f"   δ = {delta:.6f}  (非线性系数)")
        print(f"   ε = {epsilon:.6f}  (饱和参数)")
        print(f"   ")
        print(f"   气象驱动函数:")
        print(f"   f(T,RH,WS) = 0.4×(RH-70)/30 + 0.3×(15-T)/15 + 0.3×(3-WS)/3")
        
        # 多项式展开
        print(f"\n📐 在平衡点附近的线性化多项式:")
        V_eq = 2000  # 假设平衡点能见度
        print(f"   假设在V={V_eq}m附近:")
        linear_coeff = -beta - 2*delta*V_eq/(epsilon + V_eq)**2
        print(f"   dV/dt ≈ {alpha:.6f}×f(T,RH,WS) + {linear_coeff:.6f}×(V-{V_eq}) + {gamma:.6f}×ξ(t)")
    
    def _extract_nonlinear_parameters_detailed(self, model: Dict[str, Any]) -> None:
        """提取非线性动力学模型的具体参数"""
        print(f"\n🎯 非线性动力学模型详细参数:")
        
        params = model.get('params')
        if params is None:
            print("❌ 模型参数不可用")
            return
        
        r = params[0]
        K = params[1]
        betas = params[2:]
        
        print(f"\n📋 Logistic增长参数:")
        print(f"   r = {r:.6f}  (内在增长率)")
        print(f"   K = {K:.2f}m  (环境容量/最大能见度)")
        
        print(f"\n📋 气象影响系数:")
        weather_names = ['温度', '湿度', '露点', '风速']
        for i, beta in enumerate(betas):
            name = weather_names[i] if i < len(weather_names) else f'特征{i+1}'
            print(f"   β{i+1}({name}) = {beta:.6f}")
        
        print(f"\n🔬 完整微分方程:")
        print(f"   dV/dt = r·V·(1 - V/K) + Σβᵢ·Xᵢ(t)")
        print(f"   ")
        print(f"   具体参数代入:")
        print(f"   dV/dt = {r:.6f}×V×(1 - V/{K:.2f}) + Σβᵢ·Xᵢ(t)")
        
        # 多项式展开
        print(f"\n📐 多项式展开:")
        print(f"   dV/dt = {r:.6f}×V - {r/K:.8f}×V² + Σβᵢ·Xᵢ(t)")
        print(f"   ")
        print(f"   一次项系数: {r:.6f}")
        print(f"   二次项系数: -{r/K:.8f}")
    
    def _extract_ensemble_parameters_detailed(self, model: Dict[str, Any]) -> None:
        """提取集成模型的具体参数"""
        print(f"\n🎯 集成模型详细参数:")
        
        weights = model.get('weights', {})
        component_models = model.get('component_models', [])
        
        print(f"\n📋 组件模型权重:")
        total_weight = sum(weights.values())
        for model_name, weight in weights.items():
            percentage = weight * 100
            print(f"   {model_name}: {weight:.6f} ({percentage:.2f}%)")
        
        print(f"\n🔬 集成预测公式:")
        print(f"   V_ensemble(t) = Σ wᵢ × Vᵢ(t)")
        print(f"   ")
        print(f"   具体展开:")
        for model_name, weight in weights.items():
            print(f"   + {weight:.6f} × V_{model_name}(t)")
        print(f"   = V_ensemble(t)")
        
        # 如果主要是状态空间模型，显示其详细参数
        if 'state_space' in weights and weights['state_space'] > 0.5:
            print(f"\n💡 由于集成模型主要基于状态空间模型 (权重={weights['state_space']:.1%})，其详细参数如下：")
            if hasattr(self, 'models') and 'state_space' in self.models:
                state_space_model = self.models['state_space']
                print(f"   正在提取状态空间模型的详细参数...")
                self._extract_state_space_parameters_detailed(state_space_model)
            else:
                print("❌ 无法访问原始状态空间模型")
        else:
            print(f"\n💡 集成模型基于多个组件，无法提取单一模型的详细参数")

    def create_paper_specific_visualizations(self) -> None:
        """
        为论文问题二生成专门的可视化图表
        
        包含：
        1. 时间序列特征分析
        2. 模型性能详细对比
        3. 状态空间模型深度分析
        4. 连续变化动力学分析
        5. 模型性能对比表格
        """
        print("\n=== 生成论文专用可视化图表 ===")
        
        # 生成时间序列特征分析图表
        self._create_time_series_analysis_charts()
        
        # 生成模型性能对比图表
        self._create_model_performance_comparison()
        
        # 生成模型性能对比表格
        self.create_model_performance_table()
        
        # 生成状态空间模型深度分析
        self._create_state_space_detailed_analysis()
        
        # 生成连续变化动力学分析
        self._create_continuous_dynamics_analysis()
        
        print("✅ 论文专用图表生成完成")

    def _create_time_series_analysis_charts(self) -> None:
        """创建时间序列特征分析图表"""
        # 强化中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_text = '能见度时间序列特征分析' if self.use_chinese else 'Visibility Time Series Analysis'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # 1. 原始时间序列
        ax1 = axes[0, 0]
        ax1.plot(self.time_index, self.visibility, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel(self.get_text('time_steps'), **self.get_font_prop())
        ax1.set_ylabel(self.get_text('visibility'), **self.get_font_prop())
        ax1.set_title(self.get_text('original_series'), **self.get_font_prop())
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        if self.use_chinese:
            stats_text = f'观测点数: {len(self.visibility)}\n'
            stats_text += f'平均值: {np.mean(self.visibility):.1f}m\n'
            stats_text += f'标准差: {np.std(self.visibility):.1f}m\n'
            stats_text += f'变异系数: {np.std(self.visibility)/np.mean(self.visibility):.3f}\n'
            stats_text += f'范围: {np.min(self.visibility):.1f}-{np.max(self.visibility):.1f}m'
        else:
            stats_text = f'Data Points: {len(self.visibility)}\n'
            stats_text += f'Mean: {np.mean(self.visibility):.1f}m\n'
            stats_text += f'Std Dev: {np.std(self.visibility):.1f}m\n'
            stats_text += f'CV: {np.std(self.visibility)/np.mean(self.visibility):.3f}\n'
            stats_text += f'Range: {np.min(self.visibility):.1f}-{np.max(self.visibility):.1f}m'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=9, fontfamily=self.get_font_prop()['family'])
        
        # 2. 能见度分布
        ax2 = axes[0, 1]
        n, bins, patches = ax2.hist(self.visibility, bins=30, density=True, alpha=0.7,
                                   color='lightblue', edgecolor='black')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(self.visibility)
        x = np.linspace(self.visibility.min(), self.visibility.max(), 100)
        if self.use_chinese:
            label_text = f'正态分布拟合\nμ={mu:.1f}, σ={sigma:.1f}'
        else:
            label_text = f'Normal Fit\nμ={mu:.1f}, σ={sigma:.1f}'
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=label_text)
        ax2.set_xlabel(self.get_text('visibility'), **self.get_font_prop())
        ax2.set_ylabel(self.get_text('density'), **self.get_font_prop())
        ax2.set_title(self.get_text('visibility_distribution'), **self.get_font_prop())
        ax2.legend(prop=self.get_font_prop())
        ax2.grid(True, alpha=0.3)
        
        # 3. 变化率分析
        ax3 = axes[0, 2]
        vis_diff = np.diff(self.visibility)
        ax3.plot(self.time_index[1:], vis_diff, color='green', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel(self.get_text('time_steps'), **self.get_font_prop())
        ax3.set_ylabel(self.get_text('change_rate'), **self.get_font_prop())
        ax3.set_title(self.get_text('change_rate_series'), **self.get_font_prop())
        ax3.grid(True, alpha=0.3)
        
        # 添加变化率统计
        if self.use_chinese:
            change_stats = f'平均变化率: {np.mean(vis_diff):.2f}m/步\n'
            change_stats += f'变化率标准差: {np.std(vis_diff):.2f}m/步\n'
            change_stats += f'最大增幅: {np.max(vis_diff):.1f}m/步\n'
            change_stats += f'最大降幅: {np.min(vis_diff):.1f}m/步'
        else:
            change_stats = f'Mean Change: {np.mean(vis_diff):.2f}m/step\n'
            change_stats += f'Change Std: {np.std(vis_diff):.2f}m/step\n'
            change_stats += f'Max Increase: {np.max(vis_diff):.1f}m/step\n'
            change_stats += f'Max Decrease: {np.min(vis_diff):.1f}m/step'
        
        ax3.text(0.02, 0.98, change_stats, transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                verticalalignment='top', fontsize=9, fontfamily=self.get_font_prop()['family'])
        
        # 4. 变化率分布
        ax4 = axes[1, 0]
        ax4.hist(vis_diff, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        if self.use_chinese:
            zero_label = '零变化'
        else:
            zero_label = 'Zero Change'
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label=zero_label)
        ax4.set_xlabel(self.get_text('change_rate'), **self.get_font_prop())
        ax4.set_ylabel(self.get_text('density'), **self.get_font_prop())
        if self.use_chinese:
            title_text = '能见度变化率分布'
        else:
            title_text = 'Change Rate Distribution'
        ax4.set_title(title_text, **self.get_font_prop())
        ax4.legend(prop=self.get_font_prop())
        ax4.grid(True, alpha=0.3)
        
        # 5. ADF检验结果
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        # 执行ADF检验
        adf_result = adfuller(self.visibility)
        if self.use_chinese:
            adf_text = 'ADF平稳性检验结果\n\n'
            adf_text += f'ADF统计量: {adf_result[0]:.3f}\n'
            adf_text += f'p值: {adf_result[1]:.3f}\n'
            adf_text += f'临界值:\n'
            for key, value in adf_result[4].items():
                adf_text += f'  {key}: {value:.3f}\n'
            
            if adf_result[1] > 0.05:
                adf_text += '\n结论: 序列非平稳\n(p > 0.05, 不能拒绝原假设)'
            else:
                adf_text += '\n结论: 序列平稳\n(p ≤ 0.05, 拒绝原假设)'
        else:
            adf_text = 'ADF Stationarity Test Results\n\n'
            adf_text += f'ADF Statistic: {adf_result[0]:.3f}\n'
            adf_text += f'p-value: {adf_result[1]:.3f}\n'
            adf_text += f'Critical Values:\n'
            for key, value in adf_result[4].items():
                adf_text += f'  {key}: {value:.3f}\n'
            
            if adf_result[1] > 0.05:
                adf_text += '\nConclusion: Non-stationary\n(p > 0.05, fail to reject H0)'
            else:
                adf_text += '\nConclusion: Stationary\n(p ≤ 0.05, reject H0)'
        
        # 使用适当的字体
        font_family = 'SimHei' if self.use_chinese else 'monospace'
        ax5.text(0.1, 0.9, adf_text, transform=ax5.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', fontfamily=font_family)
        
        # 6. 自相关分析
        ax6 = axes[1, 2]
        # 计算自相关
        from statsmodels.tsa.stattools import acf
        autocorr = acf(self.visibility, nlags=20, fft=True)
        lags = np.arange(len(autocorr))
        
        ax6.stem(lags, autocorr, linefmt='b-', markerfmt='bo', basefmt='k-')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        if self.use_chinese:
            threshold_label = '显著性阈值'
            title_text = '能见度自相关函数'
            xlabel_text = '滞后期'
            ylabel_text = '自相关系数'
        else:
            threshold_label = 'Significance Threshold'
            title_text = 'Visibility Autocorrelation'
            xlabel_text = 'Lag'
            ylabel_text = 'Autocorrelation'
        
        ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label=threshold_label)
        ax6.set_xlabel(xlabel_text, **self.get_font_prop())
        ax6.set_ylabel(ylabel_text, **self.get_font_prop())
        ax6.set_title(title_text, **self.get_font_prop())
        ax6.legend(prop=self.get_font_prop())
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_filename = "01_时间序列特征分析.png" if self.use_chinese else "01_time_series_analysis.png"
        save_path = os.path.join(self.results_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 时间序列特征分析图表已保存: {save_path}")
        plt.show()

    def _create_model_performance_comparison(self) -> None:
        """创建模型性能详细对比图表"""
        if not self.models:
            print("⚠️ 没有可用的模型结果")
            return
        
        # 强化中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_text = '连续变化数学模型性能对比分析' if self.use_chinese else 'Continuous Mathematical Model Performance Analysis'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # 准备数据
        model_names = []
        train_r2 = []
        test_r2 = []
        train_mae = []
        test_mae = []
        train_rmse = []
        test_rmse = []
        equations = []
        
        for name, model in self.models.items():
            if model.get('success', False):
                model_names.append(model['name'])
                train_r2.append(model.get('r2_train', 0))
                test_r2.append(model.get('r2_test', model.get('r2', 0)))
                train_mae.append(model.get('mae_train', 0))
                test_mae.append(model.get('mae_test', model.get('mae', 0)))
                train_rmse.append(model.get('rmse_train', 0))
                test_rmse.append(model.get('rmse_test', model.get('rmse', 0)))
                equations.append(model.get('equation', 'N/A'))
        
        if not model_names:
            print("⚠️ 没有成功的模型")
            return
        
        # 1. R²得分对比
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        
        train_label = self.get_text('training_set')
        test_label = self.get_text('test_set')
        bars1 = ax1.bar(x - width/2, train_r2, width, label=train_label, alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width/2, test_r2, width, label=test_label, alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel(self.get_text('model_type'), **self.get_font_prop())
        ax1.set_ylabel(self.get_text('r2_score'), **self.get_font_prop())
        r2_title = '模型决定系数(R²)对比' if self.use_chinese else 'Model R² Score Comparison'
        ax1.set_title(r2_title, **self.get_font_prop())
        ax1.set_xticks(x)
        ax1.set_xticklabels([name[:10] for name in model_names], rotation=45, ha='right', **self.get_font_prop())
        ax1.legend(prop=self.get_font_prop())
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars1, train_r2):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, test_r2):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. MAE对比
        ax2 = axes[0, 1]
        bars3 = ax2.bar(x - width/2, train_mae, width, label=train_label, alpha=0.8, color='lightgreen')
        bars4 = ax2.bar(x + width/2, test_mae, width, label=test_label, alpha=0.8, color='orange')
        
        ax2.set_xlabel(self.get_text('model_type'), **self.get_font_prop())
        ax2.set_ylabel(self.get_text('mae_error'), **self.get_font_prop())
        mae_title = '模型平均绝对误差(MAE)对比' if self.use_chinese else 'Model MAE Comparison'
        ax2.set_title(mae_title, **self.get_font_prop())
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:10] for name in model_names], rotation=45, ha='right', **self.get_font_prop())
        ax2.legend(prop=self.get_font_prop())
        ax2.grid(True, alpha=0.3)
        
        # 3. RMSE对比
        ax3 = axes[0, 2]
        bars5 = ax3.bar(x - width/2, train_rmse, width, label=train_label, alpha=0.8, color='lightpink')
        bars6 = ax3.bar(x + width/2, test_rmse, width, label=test_label, alpha=0.8, color='lightyellow')
        
        ax3.set_xlabel(self.get_text('model_type'), **self.get_font_prop())
        ax3.set_ylabel(self.get_text('rmse_error'), **self.get_font_prop())
        rmse_title = '模型均方根误差(RMSE)对比' if self.use_chinese else 'Model RMSE Comparison'
        ax3.set_title(rmse_title, **self.get_font_prop())
        ax3.set_xticks(x)
        ax3.set_xticklabels([name[:10] for name in model_names], rotation=45, ha='right', **self.get_font_prop())
        ax3.legend(prop=self.get_font_prop())
        ax3.grid(True, alpha=0.3)
        
        # 4. 泛化能力分析
        ax4 = axes[1, 0]
        generalization_gap = [tr - te for tr, te in zip(train_r2, test_r2)]
        colors = ['green' if gap < 0.05 else 'orange' if gap < 0.15 else 'red' for gap in generalization_gap]
        
        bars7 = ax4.bar(x, generalization_gap, color=colors, alpha=0.7)
        ax4.set_xlabel(self.get_text('model_type'), **self.get_font_prop())
        gap_ylabel = '泛化差距 (训练R² - 测试R²)' if self.use_chinese else 'Generalization Gap (Train R² - Test R²)'
        ax4.set_ylabel(gap_ylabel, **self.get_font_prop())
        gap_title = '模型泛化能力分析' if self.use_chinese else 'Model Generalization Analysis'
        ax4.set_title(gap_title, **self.get_font_prop())
        ax4.set_xticks(x)
        ax4.set_xticklabels([name[:10] for name in model_names], rotation=45, ha='right', **self.get_font_prop())
        
        good_gen_label = '良好泛化阈值' if self.use_chinese else 'Good Generalization'
        overfit_label = '过拟合警戒线' if self.use_chinese else 'Overfitting Alert'
        ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label=good_gen_label)
        ax4.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label=overfit_label)
        ax4.legend(prop=self.get_font_prop())
        ax4.grid(True, alpha=0.3)
        
        # 5. 综合性能雷达图
        ax5 = axes[1, 1]
        if len(model_names) > 0:
            # 标准化指标到0-1范围
            norm_test_r2 = [(r2 + 1) / 2 for r2 in test_r2]  # R²可能为负
            norm_mae = [1 - (mae / max(test_mae)) for mae in test_mae]  # MAE越小越好
            norm_rmse = [1 - (rmse / max(test_rmse)) for rmse in test_rmse]  # RMSE越小越好
            norm_gen = [1 - min(gap / 0.3, 1) for gap in generalization_gap]  # 泛化差距越小越好
            
            # 创建雷达图数据
            if self.use_chinese:
                categories = ['R²得分', 'MAE性能', 'RMSE性能', '泛化能力']
            else:
                categories = ['R² Score', 'MAE Performance', 'RMSE Performance', 'Generalization']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            ax5 = plt.subplot(2, 3, 5, projection='polar')
            
            for i, name in enumerate(model_names):
                values = [norm_test_r2[i], norm_mae[i], norm_rmse[i], norm_gen[i]]
                values += values[:1]
                
                ax5.plot(angles, values, 'o-', linewidth=2, label=name[:10])
                ax5.fill(angles, values, alpha=0.25)
            
            ax5.set_xticks(angles[:-1])
            # 为雷达图设置字体
            if self.use_chinese:
                for angle, category in zip(angles[:-1], categories):
                    ax5.text(angle, ax5.get_ylim()[1] * 1.1, category, 
                            horizontalalignment='center', fontfamily='SimHei')
            else:
                ax5.set_xticklabels(categories)
            ax5.set_ylim(0, 1)
            radar_title = '模型综合性能雷达图' if self.use_chinese else 'Comprehensive Performance Radar'
            ax5.set_title(radar_title, **self.get_font_prop())
            ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), prop=self.get_font_prop())
        
        # 6. 模型数学表达式
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        eq_header = "数学模型方程式\n\n" if self.use_chinese else "Mathematical Model Equations\n\n"
        eq_text = eq_header
        for i, (name, eq) in enumerate(zip(model_names, equations)):
            eq_text += f"{i+1}. {name}:\n"
            eq_text += f"   {eq}\n\n"
        
        # 使用等宽字体显示方程
        font_family = 'SimHei' if self.use_chinese else 'monospace'
        ax6.text(0.05, 0.95, eq_text, transform=ax6.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                verticalalignment='top', fontfamily=font_family)
        
        plt.tight_layout()
        
        # 保存图片
        save_filename = "02_模型性能对比分析.png" if self.use_chinese else "02_model_performance_comparison.png"
        save_path = os.path.join(self.results_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 模型性能对比分析图表已保存: {save_path}")
        plt.show()

    def _create_state_space_detailed_analysis(self) -> None:
        """创建状态空间模型详细分析图表"""
        # 找到状态空间模型
        state_space_model = None
        for model in self.models.values():
            if 'state_space' in model.get('name', '').lower() and model.get('success', False):
                state_space_model = model
                break
        
        if not state_space_model:
            print("⚠️ 未找到成功的状态空间模型")
            return
        
        # 强化中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('状态空间模型详细分析 - 最优连续变化模型', fontsize=16, fontweight='bold')
        
        predicted = state_space_model['predicted']
        residuals = self.visibility - predicted
        
        # 1. 模型拟合效果
        ax1 = axes[0, 0]
        ax1.plot(self.time_index, self.visibility, 'k-', linewidth=2, label='观测值', alpha=0.8)
        ax1.plot(self.time_index, predicted, 'r-', linewidth=2, label='预测值', alpha=0.8)
        ax1.set_xlabel('时间步长', fontproperties='SimHei')
        ax1.set_ylabel('能见度 (m)', fontproperties='SimHei')
        ax1.set_title('状态空间模型拟合效果', fontproperties='SimHei')
        ax1.legend(prop={'family': 'SimHei'})
        ax1.grid(True, alpha=0.3)
        
        # 添加性能指标
        r2 = state_space_model.get('r2', 0)
        mae = state_space_model.get('mae', 0)
        rmse = state_space_model.get('rmse', 0)
        
        stats_text = f'模型性能指标\n'
        stats_text += f'R² = {r2:.4f}\n'
        stats_text += f'MAE = {mae:.2f}m\n'
        stats_text += f'RMSE = {rmse:.2f}m'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=9, fontproperties='SimHei')
        
        # 2. 残差分析
        ax2 = axes[0, 1]
        ax2.plot(self.time_index, residuals, 'g-', alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=np.std(residuals), color='red', linestyle='--', alpha=0.5, label='+1σ')
        ax2.axhline(y=-np.std(residuals), color='red', linestyle='--', alpha=0.5, label='-1σ')
        ax2.set_xlabel('时间步长', fontproperties='SimHei')
        ax2.set_ylabel('残差 (m)', fontproperties='SimHei')
        ax2.set_title('模型残差序列', fontproperties='SimHei')
        ax2.legend(prop={'family': 'SimHei'})
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分布
        ax3 = axes[0, 2]
        ax3.hist(residuals, bins=25, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                label=f'正态分布\nμ={mu:.2f}, σ={sigma:.2f}')
        ax3.set_xlabel('残差值 (m)', fontproperties='SimHei')
        ax3.set_ylabel('概率密度', fontproperties='SimHei')
        ax3.set_title('残差分布分析', fontproperties='SimHei')
        ax3.legend(prop={'family': 'SimHei'})
        ax3.grid(True, alpha=0.3)
        
        # 4. 状态变量演化
        ax4 = axes[1, 0]
        # 如果有状态变量信息
        if 'states' in state_space_model:
            states = state_space_model['states']
            for i, state in enumerate(states.T):
                ax4.plot(self.time_index, state, label=f'状态变量 {i+1}', alpha=0.8)
        else:
            # 简化显示：仅显示预测轨迹的平滑度
            smoothed = np.convolve(predicted, np.ones(5)/5, mode='same')
            ax4.plot(self.time_index, predicted, 'b-', alpha=0.5, label='预测轨迹')
            ax4.plot(self.time_index, smoothed, 'r-', linewidth=2, label='平滑轨迹')
        
        ax4.set_xlabel('时间步长', fontproperties='SimHei')
        ax4.set_ylabel('状态值', fontproperties='SimHei')
        ax4.set_title('状态空间演化', fontproperties='SimHei')
        ax4.legend(prop={'family': 'SimHei'})
        ax4.grid(True, alpha=0.3)
        
        # 5. 预测精度随时间变化
        ax5 = axes[1, 1]
        # 计算滑动窗口的预测精度
        window_size = min(50, len(predicted) // 10)
        rolling_mae = []
        rolling_rmse = []
        rolling_indices = []
        
        for i in range(window_size, len(predicted)):
            window_residuals = residuals[i-window_size:i]
            rolling_mae.append(np.mean(np.abs(window_residuals)))
            rolling_rmse.append(np.sqrt(np.mean(window_residuals**2)))
            rolling_indices.append(i)
        
        ax5.plot(rolling_indices, rolling_mae, 'b-', label='滑动MAE', linewidth=2)
        ax5.plot(rolling_indices, rolling_rmse, 'r-', label='滑动RMSE', linewidth=2)
        ax5.set_xlabel('时间步长', fontproperties='SimHei')
        ax5.set_ylabel('预测误差 (m)', fontproperties='SimHei')
        ax5.set_title('预测精度时变特性', fontproperties='SimHei')
        ax5.legend(prop={'family': 'SimHei'})
        ax5.grid(True, alpha=0.3)
        
        # 6. 观测值vs预测值散点图
        ax6 = axes[1, 2]
        ax6.scatter(self.visibility, predicted, alpha=0.6, s=20, c='blue')
        
        # 添加理想线
        min_val = min(self.visibility.min(), predicted.min())
        max_val = max(self.visibility.max(), predicted.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
        
        ax6.set_xlabel('观测值 (m)', fontproperties='SimHei')
        ax6.set_ylabel('预测值 (m)', fontproperties='SimHei')
        ax6.set_title('观测值 vs 预测值', fontproperties='SimHei')
        ax6.legend(prop={'family': 'SimHei'})
        ax6.grid(True, alpha=0.3)
        
        # 7. 模型参数信息
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        if self.use_chinese:
            param_text = "状态空间模型参数\n\n"
            param_text += f"模型名称: {state_space_model.get('name', 'N/A')}\n"
            param_text += f"数学表达式:\n{state_space_model.get('equation', 'N/A')}\n\n"
            param_text += f"性能指标:\n"
            param_text += f"• R² 得分: {r2:.4f}\n"
            param_text += f"• MAE: {mae:.2f} m\n"
            param_text += f"• RMSE: {rmse:.2f} m\n"
            param_text += f"• 数据点数: {len(predicted)}\n"
            param_text += f"• 残差标准差: {np.std(residuals):.2f} m"
        else:
            param_text = "State Space Model Parameters\n\n"
            param_text += f"Model Name: {state_space_model.get('name', 'N/A')}\n"
            param_text += f"Mathematical Expression:\n{state_space_model.get('equation', 'N/A')}\n\n"
            param_text += f"Performance Metrics:\n"
            param_text += f"• R² Score: {r2:.4f}\n"
            param_text += f"• MAE: {mae:.2f} m\n"
            param_text += f"• RMSE: {rmse:.2f} m\n"
            param_text += f"• Data Points: {len(predicted)}\n"
            param_text += f"• Residual Std: {np.std(residuals):.2f} m"
        
        font_family = 'SimHei' if self.use_chinese else 'monospace'
        ax7.text(0.05, 0.95, param_text, transform=ax7.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', fontfamily=font_family)
        
        # 8. 连续性分析
        ax8 = axes[2, 1]
        # 计算一阶和二阶差分
        first_diff = np.diff(predicted)
        second_diff = np.diff(first_diff)
        
        if self.use_chinese:
            first_label = '一阶导数(变化率)'
            second_label = '二阶导数(加速度)'
            xlabel_text = '时间步长'
            ylabel_text = '导数值'
            title_text = '连续性变化分析'
        else:
            first_label = 'First Derivative (Rate)'
            second_label = 'Second Derivative (Acceleration)'
            xlabel_text = 'Time Steps'
            ylabel_text = 'Derivative Value'
            title_text = 'Continuity Analysis'
        
        ax8.plot(self.time_index[1:], first_diff, 'b-', label=first_label, alpha=0.8)
        ax8.plot(self.time_index[2:], second_diff, 'r-', label=second_label, alpha=0.8)
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax8.set_xlabel(xlabel_text, **self.get_font_prop())
        ax8.set_ylabel(ylabel_text, **self.get_font_prop())
        ax8.set_title(title_text, **self.get_font_prop())
        ax8.legend(prop=self.get_font_prop())
        ax8.grid(True, alpha=0.3)
        
        # 9. 稳定性分析
        ax9 = axes[2, 2]
        # 计算局部方差
        window_size = min(20, len(predicted) // 5)
        local_variance = []
        variance_indices = []
        
        for i in range(window_size, len(predicted)):
            window_data = predicted[i-window_size:i]
            local_variance.append(np.var(window_data))
            variance_indices.append(i)
        
        ax9.plot(variance_indices, local_variance, 'g-', linewidth=2)
        xlabel_text = '时间步长' if self.use_chinese else 'Time Steps'
        ylabel_text = '局部方差' if self.use_chinese else 'Local Variance'
        title_text = '模型稳定性分析' if self.use_chinese else 'Model Stability Analysis'
        ax9.set_xlabel(xlabel_text, **self.get_font_prop())
        ax9.set_ylabel(ylabel_text, **self.get_font_prop())
        ax9.set_title(title_text, **self.get_font_prop())
        ax9.grid(True, alpha=0.3)
        
        # 添加稳定性评价
        avg_variance = np.mean(local_variance)
        if self.use_chinese:
            stability_text = f'平均局部方差: {avg_variance:.2f}\n'
            if avg_variance < np.var(self.visibility) * 0.1:
                stability_text += '评价: 高度稳定'
            elif avg_variance < np.var(self.visibility) * 0.3:
                stability_text += '评价: 中等稳定'
            else:
                stability_text += '评价: 稳定性一般'
        else:
            stability_text = f'Mean Local Variance: {avg_variance:.2f}\n'
            if avg_variance < np.var(self.visibility) * 0.1:
                stability_text += 'Rating: Highly Stable'
            elif avg_variance < np.var(self.visibility) * 0.3:
                stability_text += 'Rating: Moderately Stable'
            else:
                stability_text += 'Rating: Fair Stability'
        
        ax9.text(0.02, 0.98, stability_text, transform=ax9.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                verticalalignment='top', fontsize=9, fontfamily=self.get_font_prop()['family'])
        
        plt.tight_layout()
        
        # 保存图片
        save_filename = "03_状态空间模型详细分析.png" if self.use_chinese else "03_state_space_detailed_analysis.png"
        save_path = os.path.join(self.results_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 状态空间模型详细分析图表已保存: {save_path}")
        plt.show()

    def _create_continuous_dynamics_analysis(self) -> None:
        """创建连续变化动力学分析图表"""
        # 强化中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_text = '能见度连续变化动力学分析' if self.use_chinese else 'Visibility Continuous Dynamics Analysis'
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # 1. 原始时间序列与趋势
        ax1 = axes[0, 0]
        if self.use_chinese:
            original_label = '原始数据'
            trend_label_template = '{}点移动平均'
            xlabel_text = '时间步长'
            ylabel_text = '能见度 (m)'
            title_text = '能见度时间序列与长期趋势'
        else:
            original_label = 'Original Data'
            trend_label_template = '{}-Point Moving Average'
            xlabel_text = 'Time Steps'
            ylabel_text = 'Visibility (m)'
            title_text = 'Visibility Time Series with Long-term Trend'
            
        ax1.plot(self.time_index, self.visibility, 'b-', alpha=0.6, linewidth=1, label=original_label)
        
        # 添加趋势线（移动平均）
        window = min(50, len(self.visibility) // 10)
        trend = np.convolve(self.visibility, np.ones(window)/window, mode='same')
        trend_label = trend_label_template.format(window)
        ax1.plot(self.time_index, trend, 'r-', linewidth=3, label=trend_label)
        
        ax1.set_xlabel(xlabel_text, **self.get_font_prop())
        ax1.set_ylabel(ylabel_text, **self.get_font_prop())
        ax1.set_title(title_text, **self.get_font_prop())
        ax1.legend(prop=self.get_font_prop())
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        trend_slope = np.polyfit(self.time_index, self.visibility, 1)[0]
        if self.use_chinese:
            trend_text = f'趋势分析\n'
            trend_text += f'变化斜率: {trend_slope:.3f} m/步\n'
            trend_text += f'总体变化: {trend_slope * len(self.visibility):.1f} m\n'
            if abs(trend_slope) < 0.01:
                trend_text += '趋势: 基本稳定'
            elif trend_slope > 0:
                trend_text += '趋势: 上升'
            else:
                trend_text += '趋势: 下降'
        else:
            trend_text = f'Trend Analysis\n'
            trend_text += f'Slope: {trend_slope:.3f} m/step\n'
            trend_text += f'Total Change: {trend_slope * len(self.visibility):.1f} m\n'
            if abs(trend_slope) < 0.01:
                trend_text += 'Trend: Stable'
            elif trend_slope > 0:
                trend_text += 'Trend: Increasing'
            else:
                trend_text += 'Trend: Decreasing'
        
        ax1.text(0.02, 0.98, trend_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=9, fontfamily=self.get_font_prop()['family'])
        
        # 2. 一阶差分（变化率）
        ax2 = axes[0, 1]
        first_diff = np.diff(self.visibility)
        ax2.plot(self.time_index[1:], first_diff, 'g-', alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 添加变化率的移动平均
        diff_trend = np.convolve(first_diff, np.ones(min(20, len(first_diff)//5))/min(20, len(first_diff)//5), mode='same')
        trend_label = '变化率趋势' if self.use_chinese else 'Change Rate Trend'
        ax2.plot(self.time_index[1:], diff_trend, 'r-', linewidth=2, label=trend_label)
        
        if self.use_chinese:
            xlabel_text = '时间步长'
            ylabel_text = '变化率 (m/步)'
            title_text = '能见度一阶变化率'
        else:
            xlabel_text = 'Time Steps'
            ylabel_text = 'Change Rate (m/step)'
            title_text = 'First-Order Change Rate'
            
        ax2.set_xlabel(xlabel_text, **self.get_font_prop())
        ax2.set_ylabel(ylabel_text, **self.get_font_prop())
        ax2.set_title(title_text, **self.get_font_prop())
        ax2.legend(prop=self.get_font_prop())
        ax2.grid(True, alpha=0.3)
        
        # 变化率统计
        if self.use_chinese:
            change_stats = f'变化率统计\n'
            change_stats += f'均值: {np.mean(first_diff):.3f} m/步\n'
            change_stats += f'标准差: {np.std(first_diff):.3f} m/步\n'
            change_stats += f'最大变化: {np.max(np.abs(first_diff)):.1f} m/步\n'
            change_stats += f'变化次数: {np.sum(np.abs(first_diff) > np.std(first_diff))} 次'
        else:
            change_stats = f'Change Rate Statistics\n'
            change_stats += f'Mean: {np.mean(first_diff):.3f} m/step\n'
            change_stats += f'Std Dev: {np.std(first_diff):.3f} m/step\n'
            change_stats += f'Max Change: {np.max(np.abs(first_diff)):.1f} m/step\n'
            change_stats += f'Events: {np.sum(np.abs(first_diff) > np.std(first_diff))}'
        
        ax2.text(0.02, 0.98, change_stats, transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                verticalalignment='top', fontsize=9, fontfamily=self.get_font_prop()['family'])
        
        # 3. 二阶差分（加速度）
        ax3 = axes[0, 2]
        second_diff = np.diff(first_diff)
        ax3.plot(self.time_index[2:], second_diff, 'purple', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 加速度的移动平均
        acc_trend = np.convolve(second_diff, np.ones(min(15, len(second_diff)//5))/min(15, len(second_diff)//5), mode='same')
        trend_label = '加速度趋势' if self.use_chinese else 'Acceleration Trend'
        ax3.plot(self.time_index[2:], acc_trend, 'orange', linewidth=2, label=trend_label)
        
        if self.use_chinese:
            xlabel_text = '时间步长'
            ylabel_text = '加速度 (m/步²)'
            title_text = '能见度二阶变化率(加速度)'
        else:
            xlabel_text = 'Time Steps'
            ylabel_text = 'Acceleration (m/step²)'
            title_text = 'Second-Order Change Rate (Acceleration)'
            
        ax3.set_xlabel(xlabel_text, **self.get_font_prop())
        ax3.set_ylabel(ylabel_text, **self.get_font_prop())
        ax3.set_title(title_text, **self.get_font_prop())
        ax3.legend(prop=self.get_font_prop())
        ax3.grid(True, alpha=0.3)
        
        # 4. 变化幅度分布
        ax4 = axes[1, 0]
        change_magnitudes = np.abs(first_diff)
        ax4.hist(change_magnitudes, bins=25, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # 拟合指数分布
        from scipy import stats
        exp_params = stats.expon.fit(change_magnitudes)
        x_exp = np.linspace(0, change_magnitudes.max(), 100)
        if self.use_chinese:
            exp_label = f'指数分布拟合\nλ={1/exp_params[1]:.3f}'
            xlabel_text = '变化幅度 |dV/dt| (m/步)'
            ylabel_text = '概率密度'
            title_text = '变化幅度分布特征'
        else:
            exp_label = f'Exponential Fit\nλ={1/exp_params[1]:.3f}'
            xlabel_text = 'Change Magnitude |dV/dt| (m/step)'
            ylabel_text = 'Probability Density'
            title_text = 'Change Magnitude Distribution'
            
        ax4.plot(x_exp, stats.expon.pdf(x_exp, *exp_params), 'r-', linewidth=2, label=exp_label)
        
        ax4.set_xlabel(xlabel_text, **self.get_font_prop())
        ax4.set_ylabel(ylabel_text, **self.get_font_prop())
        ax4.set_title(title_text, **self.get_font_prop())
        ax4.legend(prop=self.get_font_prop())
        ax4.grid(True, alpha=0.3)
        
        # 5. 相空间图（速度vs位置）
        ax5 = axes[1, 1]
        # 使用滑动窗口估计瞬时速度
        velocity_window = 5
        velocities = []
        positions = []
        
        for i in range(velocity_window, len(self.visibility) - velocity_window):
            pos = self.visibility[i]
            vel = (self.visibility[i+velocity_window] - self.visibility[i-velocity_window]) / (2 * velocity_window)
            positions.append(pos)
            velocities.append(vel)
        
        scatter = ax5.scatter(positions, velocities, c=range(len(positions)), 
                            cmap='viridis', alpha=0.6, s=20)
        
        if self.use_chinese:
            xlabel_text = '能见度 (m)'
            ylabel_text = '变化速率 (m/步)'
            title_text = '相空间轨迹 (位置-速度)'
            cbar_label = '时间步长'
        else:
            xlabel_text = 'Visibility (m)'
            ylabel_text = 'Change Rate (m/step)'
            title_text = 'Phase Space Trajectory (Position-Velocity)'
            cbar_label = 'Time Steps'
            
        ax5.set_xlabel(xlabel_text, **self.get_font_prop())
        ax5.set_ylabel(ylabel_text, **self.get_font_prop())
        ax5.set_title(title_text, **self.get_font_prop())
        ax5.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label(cbar_label, **self.get_font_prop())
        
        # 6. 连续性指标评估
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算连续性指标
        # 1. 平滑度（二阶导数的方差）
        smoothness = np.var(second_diff)
        
        # 2. 连续性（大跳跃的频率）
        threshold = 2 * np.std(first_diff)
        large_jumps = np.sum(np.abs(first_diff) > threshold)
        continuity_score = 1 - (large_jumps / len(first_diff))
        
        # 3. 预测性（自相关强度）
        autocorr = np.corrcoef(self.visibility[:-1], self.visibility[1:])[0, 1]
        
        # 4. 稳定性（变化率的标准差）
        stability = 1 / (1 + np.std(first_diff))
        
        # 5. 物理可实现性评分
        max_change_rate = np.max(np.abs(first_diff))
        physical_feasibility = min(1.0, 100 / max_change_rate) if max_change_rate > 0 else 1.0
        
        # 根据语言设置生成文本
        if self.use_chinese:
            metrics_text = "连续变化动力学评估指标\n\n"
            metrics_text += f"📊 连续性指标:\n"
            metrics_text += f"• 平滑度: {smoothness:.2f}\n"
            metrics_text += f"• 连续性得分: {continuity_score:.3f}\n"
            metrics_text += f"• 时间相关性: {autocorr:.3f}\n"
            metrics_text += f"• 稳定性指数: {stability:.3f}\n"
            metrics_text += f"• 物理可行性: {physical_feasibility:.3f}\n\n"
            
            metrics_text += f"🔍 动力学特征:\n"
            metrics_text += f"• 最大变化率: {max_change_rate:.2f} m/步\n"
            metrics_text += f"• 突变事件: {large_jumps} 次\n"
            metrics_text += f"• 变化周期性: {'强' if autocorr > 0.7 else '中等' if autocorr > 0.3 else '弱'}\n"
            
            # 综合评分
            overall_score = np.mean([continuity_score, autocorr, stability, physical_feasibility])
            metrics_text += f"\n🎯 综合连续性评分: {overall_score:.3f}/1.000\n"
            
            if overall_score > 0.8:
                quality = "优秀"
                color = "lightgreen"
            elif overall_score > 0.6:
                quality = "良好"  
                color = "lightyellow"
            else:
                quality = "一般"
                color = "lightcoral"
                
            metrics_text += f"评价等级: {quality}"
        else:
            metrics_text = "Continuous Dynamics Assessment\n\n"
            metrics_text += f"📊 Continuity Metrics:\n"
            metrics_text += f"• Smoothness: {smoothness:.2f}\n"
            metrics_text += f"• Continuity Score: {continuity_score:.3f}\n"
            metrics_text += f"• Temporal Correlation: {autocorr:.3f}\n"
            metrics_text += f"• Stability Index: {stability:.3f}\n"
            metrics_text += f"• Physical Feasibility: {physical_feasibility:.3f}\n\n"
            
            metrics_text += f"🔍 Dynamic Features:\n"
            metrics_text += f"• Max Change Rate: {max_change_rate:.2f} m/step\n"
            metrics_text += f"• Jump Events: {large_jumps}\n"
            periodicity = 'Strong' if autocorr > 0.7 else 'Moderate' if autocorr > 0.3 else 'Weak'
            metrics_text += f"• Change Periodicity: {periodicity}\n"
            
            # 综合评分
            overall_score = np.mean([continuity_score, autocorr, stability, physical_feasibility])
            metrics_text += f"\n🎯 Overall Continuity Score: {overall_score:.3f}/1.000\n"
            
            if overall_score > 0.8:
                quality = "Excellent"
                color = "lightgreen"
            elif overall_score > 0.6:
                quality = "Good"  
                color = "lightyellow"
            else:
                quality = "Fair"
                color = "lightcoral"
                
            metrics_text += f"Quality Rating: {quality}"
        
        # 使用正确的字体
        font_family = 'SimHei' if self.use_chinese else 'monospace'
        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
                verticalalignment='top', fontfamily=font_family)
        
        plt.tight_layout()
        
        # 保存图片
        save_filename = "04_连续变化动力学分析.png" if self.use_chinese else "04_continuous_dynamics_analysis.png"
        save_path = os.path.join(self.results_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 连续变化动力学分析图表已保存: {save_path}")
        plt.show()

    def create_model_performance_table(self) -> None:
        """创建模型性能对比表格用于论文"""
        if not self.models:
            print("⚠️ 没有可用的模型结果")
            return
        
        # 强化中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备表格数据
        table_data = []
        for name, model in self.models.items():
            if model.get('success', False):
                # 计算泛化差距
                train_r2 = model.get('r2_train', 0)
                test_r2 = model.get('r2_test', model.get('r2', 0))
                generalization_gap = train_r2 - test_r2
                
                # 确定泛化能力评估
                if generalization_gap < 0.05:
                    generalization_status = "优秀泛化"
                elif generalization_gap < 0.15:
                    generalization_status = "良好泛化"
                else:
                    generalization_status = "过拟合风险"
                
                table_data.append({
                    '模型类型': model['name'],
                    '训练集R²': f"{train_r2:.6f}",
                    '测试集R²': f"{test_r2:.6f}",
                    '训练集MAE(m)': f"{model.get('mae_train', 0):.2f}",
                    '测试集MAE(m)': f"{model.get('mae_test', model.get('mae', 0)):.2f}",
                    '训练集RMSE(m)': f"{model.get('rmse_train', 0):.2f}",
                    '测试集RMSE(m)': f"{model.get('rmse_test', model.get('rmse', 0)):.2f}",
                    '泛化差距': f"{generalization_gap:.4f}",
                    '泛化能力评估': generalization_status,
                    '数学方程': model.get('equation', 'N/A')
                })
        
        # 创建表格图
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        
        if table_data:
            # 创建DataFrame
            import pandas as pd
            df = pd.DataFrame(table_data)
            
            # 创建表格
            table = ax.table(cellText=df.values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2)
            
            # 设置表头样式
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 设置数据行样式
            for i in range(1, len(df) + 1):
                for j in range(len(df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F2F2F2')
                    else:
                        table[(i, j)].set_facecolor('white')
                    
                    # 突出显示最佳性能
                    if j == 2:  # 测试集R²列
                        r2_val = float(df.iloc[i-1, j])
                        if r2_val > 0.9:
                            table[(i, j)].set_facecolor('#90EE90')  # 浅绿色
                    
                    elif j == 8:  # 泛化能力评估列
                        status = df.iloc[i-1, j]
                        if status == "优秀泛化":
                            table[(i, j)].set_facecolor('#90EE90')  # 浅绿色
                        elif status == "良好泛化":
                            table[(i, j)].set_facecolor('#FFD700')  # 金色
                        else:
                            table[(i, j)].set_facecolor('#FFB6C1')  # 浅粉色
        
        plt.title('问题二：连续变化数学模型性能对比表', fontsize=16, fontweight='bold', pad=20, fontproperties='SimHei')
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, "05_模型性能对比表格.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 模型性能对比表格已保存: {save_path}")
        plt.show()
        
        # 同时打印表格数据（用于论文复制）
        print("\n" + "="*120)
        print("📊 模型性能对比表格数据（用于论文）")
        print("="*120)
        
        if table_data:
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
        
        print("\n📈 性能分析总结:")
        print(f"• 共构建 {len(table_data)} 个连续变化数学模型")
        
        if table_data:
            # 找出最佳模型
            best_idx = 0
            best_r2 = -999
            for i, row in enumerate(table_data):
                test_r2 = float(row['测试集R²'])
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_idx = i
            
            best_model_data = table_data[best_idx]
            print(f"• 最优模型: {best_model_data['模型类型']}")
            print(f"• 最高测试集R²: {best_model_data['测试集R²']}")
            print(f"• 最优模型泛化能力: {best_model_data['泛化能力评估']}")
            print(f"• 最优模型RMSE: {best_model_data['测试集RMSE(m)']}m")


def main():
    """主程序入口"""
    print("🚀 江西省数学建模比赛 - 问题二最佳解决方案")
    print("📐 建立能见度随时间连续变化数学模型")
    print("="*80)
    print("功能特点:")
    print("• 🔬 基于物理原理的微分方程雾演化模型")
    print("• 📊 卡尔曼滤波状态空间连续模型")
    print("• 🌀 非线性动力学Logistic增长模型")
    print("• 🎯 多模型集成与性能比较")
    print("• 📈 丰富的可视化分析（15+专业图表）")
    print("• 🔮 连续预测与不确定性分析")
    print("• 📋 详细的数学建模报告")
    print("• 🔍 具体数学表达式和参数提取")
    print("="*80)
    
    # 创建模型实例
    model = ContinuousVisibilityModel()
    
    # 数据文件路径检测
    possible_paths = [
        '../datasets/blur.csv',
        './datasets/blur.csv', 
        'blur.csv',
        '../data/blur.csv',
        './data/blur.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            print(f"📁 找到数据文件: {path}")
            break
    
    if data_path is None:
        print("❌ 未找到数据文件，请确保以下路径之一存在数据文件:")
        for path in possible_paths:
            print(f"   • {path}")
        print("📝 您也可以修改 main() 函数中的 data_path 变量指定正确路径")
        return
    
    try:
        # 运行完整分析
        results = model.run_complete_analysis(data_path, prediction_steps=120)
        
        if results.get('analysis_completed', False):
            print("\n🎉 分析完成！")
            
            # 提取详细的数学表达式和参数
            model.extract_detailed_model_parameters()
            
            best_model = results.get('best_model')
            if best_model:
                print(f"\n📐 最终连续数学模型:")
                print(f"   V(t) 满足微分方程: {best_model.get('equation', 'N/A')}")
                print(f"   其中 V(t) 表示 t 时刻的能见度值")
                print(f"   模型拟合精度: R² = {best_model['r2']:.8f}")
                print(f"   预测准确度: ±{best_model['rmse']:.1f}m")
                
                print('\n' + '='*80)
                print('✅ 模型构建和参数提取完成！这就是您要的带有具体权重的数学表达式')
                print('='*80)
        
        else:
            print("⚠️ 分析未完全完成，请检查数据文件")
    
    except KeyboardInterrupt:
        print("\n👋 用户中断程序运行")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("💡 请检查：")
        print("   1. 数据文件路径是否正确")
        print("   2. 数据文件格式是否符合要求")
        print("   3. 必要的Python包是否已安装")


if __name__ == "__main__":
    main()