#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机场能见度数据探索性分析
========================

基于AMOS气象观测数据和机场视频数据进行全面的探索性数据分析，
为后续能见度回归建模提供数据基础。

作者: 分析团队
日期: 2024
"""

import os
import sys
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import seaborn as sns
import cv2
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import glob
import matplotlib
from matplotlib.font_manager import FontProperties
import platform

# 设置中文字体支持
def setup_chinese_fonts():
    """
    设置中文字体支持
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows系统中文字体
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun', 'FangSong']
    elif system == "Darwin":  # macOS
        # macOS系统中文字体
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    else:  # Linux
        # Linux系统中文字体
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei']
    
    # 添加默认字体作为备选
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS', 'sans-serif'])
    
    # 设置matplotlib中文字体
    matplotlib.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 验证字体设置
    try:
        # 测试中文字体渲染
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文字体', fontsize=12, ha='center')
        plt.close(fig)
        print("✓ 中文字体配置成功")
    except Exception as e:
        print(f"⚠️  中文字体配置可能有问题: {e}")
        # 备用方案：使用系统默认字体
        matplotlib.rcParams['font.family'] = 'sans-serif'

# 调用字体设置函数
setup_chinese_fonts()

def list_available_chinese_fonts():
    """
    列出系统中可用的中文字体
    """
    from matplotlib.font_manager import fontManager
    
    chinese_fonts = []
    for font in fontManager.ttflist:
        # 检查字体名称是否包含中文相关字符
        font_name = font.name
        if any(keyword in font_name for keyword in [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
            'PingFang', 'Hiragino', 'STHeiti', 'WenQuanYi', 'Noto Sans CJK'
        ]):
            chinese_fonts.append(font_name)
    
    if chinese_fonts:
        print("系统中检测到的中文字体:")
        for font in set(chinese_fonts):
            print(f"  - {font}")
    else:
        print("⚠️  未检测到常见的中文字体")
        print("建议安装以下字体之一:")
        print("  - Windows: Microsoft YaHei, SimHei")
        print("  - macOS: PingFang SC")  
        print("  - Linux: WenQuanYi Micro Hei")
    
    return list(set(chinese_fonts))

# 检测可用中文字体
available_fonts = list_available_chinese_fonts()

def set_chinese_text(ax, title=None, xlabel=None, ylabel=None):
    """
    为图表设置中文标题和标签，确保正确显示
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        图表轴对象
    title : str, optional
        图表标题
    xlabel : str, optional  
        x轴标签
    ylabel : str, optional
        y轴标签
    """
    # 设置字体属性
    font_prop = FontProperties()
    if available_fonts:
        font_prop.set_family(available_fonts[0])
    
    if title:
        ax.set_title(title, fontproperties=font_prop, fontsize=12, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=font_prop, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=10)
    
    # 设置图例字体
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

class WeatherDataEDA:
    """
    气象数据探索性分析类
    
    主要功能包括:
    - AMOS数据加载和预处理
    - 数据质量评估  
    - 时间序列分析
    - 相关性分析
    - 异常值检测
    """
    
    def __init__(self, data_dir: str = "../AMOS20200313"):
        """
        初始化分析类
        
        Parameters
        ----------
        data_dir : str
            AMOS数据文件目录路径
        """
        self.data_dir = data_dir
        self.vis_data: Optional[pd.DataFrame] = None
        self.ptu_data: Optional[pd.DataFrame] = None  
        self.wind_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        
    def load_his_file(self, file_path: str) -> pd.DataFrame:
        """
        加载.his格式的AMOS数据文件
        
        Parameters
        ----------
        file_path : str
            文件路径
            
        Returns
        -------
        pd.DataFrame
            加载的数据框
        """
        try:
            # 读取文件，跳过第一行注释
            df = pd.read_csv(file_path, sep='\t', skiprows=1, encoding='utf-8')
            
            # 转换时间列
            df['LOCALDATE (BEIJING)'] = pd.to_datetime(df['LOCALDATE (BEIJING)'])
            df.set_index('LOCALDATE (BEIJING)', inplace=True)
            
            print(f"成功加载 {os.path.basename(file_path)}: {df.shape[0]} 条记录")
            return df
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return pd.DataFrame()
    
    def load_all_data(self) -> None:
        """
        加载所有AMOS数据文件
        """
        print("=" * 60)
        print("加载AMOS气象数据文件...")
        print("=" * 60)
        
        # 加载能见度数据
        vis_file = os.path.join(self.data_dir, "VIS_R06_12.his")
        if os.path.exists(vis_file):
            self.vis_data = self.load_his_file(vis_file)
            
        # 加载温湿压数据  
        ptu_file = os.path.join(self.data_dir, "PTU_R06_12.his")
        if os.path.exists(ptu_file):
            self.ptu_data = self.load_his_file(ptu_file)
            
        # 加载风速风向数据
        wind_file = os.path.join(self.data_dir, "WIND_R06_12.his")
        if os.path.exists(wind_file):
            self.wind_data = self.load_his_file(wind_file)
    
    def analyze_data_quality(self) -> None:
        """
        分析数据质量，包括缺失值、异常值等
        """
        print("\n" + "=" * 60)
        print("数据质量分析")
        print("=" * 60)
        
        datasets = {
            'VIS (能见度)': self.vis_data,
            'PTU (气压/温度/湿度)': self.ptu_data, 
            'WIND (风速风向)': self.wind_data
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 设置整体标题的中文字体
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            fig.suptitle('数据质量分析概览', fontproperties=font_prop, fontsize=16, fontweight='bold')
        else:
            fig.suptitle('数据质量分析概览', fontsize=16, fontweight='bold')
        
        for i, (name, data) in enumerate(datasets.items()):
            if data is not None and not data.empty:
                print(f"\n{name} 数据质量报告:")
                print(f"  时间范围: {data.index.min()} 至 {data.index.max()}")
                print(f"  总记录数: {len(data):,}")
                print(f"  数据列数: {data.shape[1]}")
                
                # 缺失值统计
                missing_stats = data.isnull().sum()
                missing_pct = (missing_stats / len(data) * 100).round(2)
                
                if missing_stats.sum() > 0:
                    print(f"  缺失值情况:")
                    for col, miss_count in missing_stats.items():
                        if miss_count > 0:
                            print(f"    {col}: {miss_count} ({missing_pct[col]:.1f}%)")
                else:
                    print("  ✓ 无缺失值")
                
                # 绘制缺失值热图
                row, col = divmod(i, 2)
                if row < 2:  # 只绘制前3个数据集
                    ax = axes[row, col]
                    
                    # 选择部分数值列进行可视化
                    numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]
                    if len(numeric_cols) > 0:
                        missing_matrix = data[numeric_cols].isnull()
                        sns.heatmap(missing_matrix.T, ax=ax, cbar=True, 
                                  cmap='YlOrRd', yticklabels=True)
                        
                        # 使用中文字体设置
                        if available_fonts:
                            font_prop = FontProperties()
                            font_prop.set_family(available_fonts[0])
                            ax.set_title(f'{name}\n缺失值热图', fontproperties=font_prop, fontsize=12)
                            ax.set_xlabel('时间 (秒级精度)', fontproperties=font_prop, fontsize=10)
                            ax.set_ylabel('参数', fontproperties=font_prop, fontsize=10)
                        else:
                            ax.set_title(f'{name}\n缺失值热图', fontsize=12)
                            ax.set_xlabel('时间 (秒级精度)', fontsize=10)
                            ax.set_ylabel('参数', fontsize=10)
                        
                        # 优化时间横坐标显示 - 只显示部分标签，格式为时:分:秒
                        n_ticks = min(6, len(data))  # 最多显示6个时间点
                        if n_ticks > 1:
                            tick_indices = np.linspace(0, len(data)-1, n_ticks, dtype=int)
                            tick_labels = [data.index[idx].strftime('%H:%M:%S') for idx in tick_indices]
                            ax.set_xticks(tick_indices)
                            ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        
        # 删除空白子图
        if len(datasets) < 4:
            fig.delaxes(axes[1, 1])
            
        plt.tight_layout()
        
        # 确保结果目录存在
        os.makedirs('../results', exist_ok=True)
        
        plt.savefig('../results/data_quality_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_visibility_trends(self) -> None:
        """
        分析能见度随时间的变化趋势
        """
        print("\n" + "=" * 60)
        print("能见度趋势分析")
        print("=" * 60)
        
        if self.vis_data is None or self.vis_data.empty:
            print("❌ 能见度数据不可用")
            return
            
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 设置整体标题的中文字体
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            fig.suptitle('能见度时间序列分析', fontproperties=font_prop, fontsize=16, fontweight='bold')
        else:
            fig.suptitle('能见度时间序列分析', fontsize=16, fontweight='bold')
        
        # 获取能见度相关列
        vis_cols = [col for col in self.vis_data.columns if any(keyword in col.upper() 
                   for keyword in ['RVR', 'MOR', 'VIS'])]
        
        if vis_cols:
            # 第一个子图：RVR数据
            rvr_cols = [col for col in vis_cols if 'RVR' in col.upper()]
            if rvr_cols:
                ax1 = axes[0]
                for col in rvr_cols[:3]:  # 最多显示3个RVR参数
                    if col in self.vis_data.columns:
                        data_to_plot = pd.to_numeric(self.vis_data[col], errors='coerce')
                        ax1.plot(self.vis_data.index, data_to_plot, 
                               label=col, linewidth=1.5, alpha=0.8)
                
                # 使用中文字体设置RVR图表
                if available_fonts:
                    font_prop = FontProperties()
                    font_prop.set_family(available_fonts[0])
                    ax1.set_title('跑道视程距离 (RVR) 趋势', fontproperties=font_prop, fontsize=14)
                    ax1.set_ylabel('RVR距离 (米)', fontproperties=font_prop, fontsize=12)
                    # 设置图例字体
                    legend = ax1.legend(fontsize=10)
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(font_prop)
                else:
                    ax1.set_title('跑道视程距离 (RVR) 趋势', fontsize=14)
                    ax1.set_ylabel('RVR距离 (米)', fontsize=12)
                    ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # 优化时间横坐标显示
                n_ticks = 8
                tick_indices = np.linspace(0, len(self.vis_data)-1, n_ticks, dtype=int)
                tick_labels = [self.vis_data.index[idx].strftime('%H:%M:%S') 
                             for idx in tick_indices]
                ax1.set_xticks([self.vis_data.index[idx] for idx in tick_indices])
                ax1.set_xticklabels(tick_labels, rotation=45, fontsize=10)
            
            # 第二个子图：MOR数据
            mor_cols = [col for col in vis_cols if 'MOR' in col.upper()]
            if mor_cols:
                ax2 = axes[1]
                for col in mor_cols[:3]:  # 最多显示3个MOR参数
                    if col in self.vis_data.columns:
                        data_to_plot = pd.to_numeric(self.vis_data[col], errors='coerce')
                        ax2.plot(self.vis_data.index, data_to_plot, 
                               label=col, linewidth=1.5, alpha=0.8)
                
                # 使用中文字体设置MOR图表
                if available_fonts:
                    font_prop = FontProperties()
                    font_prop.set_family(available_fonts[0])
                    ax2.set_title('气象光学距离 (MOR) 趋势', fontproperties=font_prop, fontsize=14)
                    ax2.set_ylabel('MOR距离 (米)', fontproperties=font_prop, fontsize=12)
                    ax2.set_xlabel('时间 (时:分:秒)', fontproperties=font_prop, fontsize=12)
                    # 设置图例字体
                    legend = ax2.legend(fontsize=10)
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(font_prop)
                else:
                    ax2.set_title('气象光学距离 (MOR) 趋势', fontsize=14)
                    ax2.set_ylabel('MOR距离 (米)', fontsize=12)
                    ax2.set_xlabel('时间 (时:分:秒)', fontsize=12)
                    ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # 优化时间横坐标显示
                n_ticks = 8
                tick_indices = np.linspace(0, len(self.vis_data)-1, n_ticks, dtype=int)
                tick_labels = [self.vis_data.index[idx].strftime('%H:%M:%S') 
                             for idx in tick_indices]
                ax2.set_xticks([self.vis_data.index[idx] for idx in tick_indices])
                ax2.set_xticklabels(tick_labels, rotation=45, fontsize=10)
            
            # 统计信息
            print(f"能见度参数数量: {len(vis_cols)}")
            for col in vis_cols:
                if col in self.vis_data.columns:
                    data_series = pd.to_numeric(self.vis_data[col], errors='coerce')
                    valid_data = data_series.dropna()
                    if len(valid_data) > 0:
                        print(f"  {col}:")
                        print(f"    平均值: {valid_data.mean():.1f}")
                        print(f"    最小值: {valid_data.min():.1f}")  
                        print(f"    最大值: {valid_data.max():.1f}")
                        print(f"    标准差: {valid_data.std():.1f}")
        
        plt.tight_layout()
        
        # 确保结果目录存在
        os.makedirs('../results', exist_ok=True)
        
        plt.savefig('../results/visibility_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 调用能见度分布分析
        self._plot_visibility_distributions(vis_cols)
    
    def _plot_visibility_distributions(self, vis_cols: List[str]) -> None:
        """
        绘制能见度数据的分布图
        
        Parameters
        ----------
        vis_cols : List[str]
            能见度列名列表
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 设置整体标题的中文字体
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            fig.suptitle('能见度数据分布分析', fontproperties=font_prop, fontsize=16, fontweight='bold')
        else:
            fig.suptitle('能见度数据分布分析', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(vis_cols[:6]):
            row, col_idx = divmod(i, 3)
            ax = axes[row, col_idx]
            
            if col in self.vis_data.columns:
                data_series = pd.to_numeric(self.vis_data[col], errors='coerce').dropna()
                
                # 直方图
                n, bins, patches = ax.hist(data_series, bins=50, alpha=0.7, 
                                         color='skyblue', edgecolor='black')
                
                # 添加统计线
                ax.axvline(data_series.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'均值: {data_series.mean():.0f}')
                ax.axvline(data_series.median(), color='green', linestyle='-', 
                          linewidth=2, label=f'中位数: {data_series.median():.0f}')
                
                # 使用中文字体设置
                if available_fonts:
                    font_prop = FontProperties()
                    font_prop.set_family(available_fonts[0])
                    ax.set_title(f'{col} 分布', fontproperties=font_prop)
                    ax.set_xlabel('能见度 (米)', fontproperties=font_prop)
                    ax.set_ylabel('频次', fontproperties=font_prop)
                    # 设置图例字体
                    legend = ax.legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(font_prop)
                else:
                    ax.set_title(f'{col} 分布')
                    ax.set_xlabel('能见度 (米)')
                    ax.set_ylabel('频次')
                    ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 删除空白子图
        for i in range(len(vis_cols), 6):
            row, col_idx = divmod(i, 3)
            if row < 2:
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig('../results/visibility_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_weather_correlations(self) -> None:
        """
        分析气象参数之间的相关性
        """
        print("\n" + "=" * 60)
        print("气象参数相关性分析")
        print("=" * 60)
        
        # 合并气象数据
        self._merge_weather_data()
        
        if self.merged_data is None or self.merged_data.empty:
            print("❌ 合并后的气象数据不可用")
            return
        
        # 选择数值型列进行相关性分析
        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns
        correlation_data = self.merged_data[numeric_cols]
        
        # 计算相关性矩阵
        corr_matrix = correlation_data.corr()
        
        # 创建相关性热图
        plt.figure(figsize=(14, 10))
        
        # 使用mask隐藏上三角（避免重复显示）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8}, fmt='.2f')
        
        # 使用中文字体设置标题和标签
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            plt.title('气象参数相关性矩阵', fontproperties=font_prop, fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('气象参数', fontproperties=font_prop, fontsize=12)
            plt.ylabel('气象参数', fontproperties=font_prop, fontsize=12)
        else:
            plt.title('气象参数相关性矩阵', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('气象参数', fontsize=12)
            plt.ylabel('气象参数', fontsize=12)
        
        # 旋转x轴标签以提高可读性
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        # 确保结果目录存在
        os.makedirs('../results', exist_ok=True)
        
        plt.savefig('../results/weather_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出强相关性信息
        self._print_strong_correlations(corr_matrix)
        
        print(f"\n✅ 相关性分析完成！")
        print(f"   分析参数数量: {len(numeric_cols)}")
        print(f"   数据记录数: {len(correlation_data):,}")
        print(f"   时间范围: {self.merged_data.index.min().strftime('%H:%M:%S')} - {self.merged_data.index.max().strftime('%H:%M:%S')}")
    
    def _merge_weather_data(self) -> None:
        """
        合并不同来源的气象数据
        """
        merged_list = []
        
        # 处理能见度数据 (15秒间隔)
        if self.vis_data is not None and not self.vis_data.empty:
            vis_subset = self.vis_data[['RVR_1A', 'MOR_1A', 'MOR_10A']].copy()
            vis_subset.columns = ['RVR', 'MOR_1min', 'MOR_10min']
            merged_list.append(vis_subset)
        
        # 处理温湿压数据 (1分钟间隔，需要插值到15秒)
        if self.ptu_data is not None and not self.ptu_data.empty:
            ptu_subset = self.ptu_data[['TEMP (°C)', 'RH (%)', 'DEWPOINT (°C)', 'PAINS (HPA)']].copy()
            ptu_subset.columns = ['Temperature', 'Humidity', 'Dewpoint', 'Pressure']
            
            # 重采样到15秒间隔（前向填充）
            ptu_resampled = ptu_subset.resample('15S').ffill()
            merged_list.append(ptu_resampled)
        
        # 处理风速风向数据 (15秒间隔)
        if self.wind_data is not None and not self.wind_data.empty:
            wind_cols = [col for col in ['WS2M', 'WD2M', 'WSV2M'] if col in self.wind_data.columns]
            if wind_cols:
                wind_subset = self.wind_data[wind_cols].copy()
                merged_list.append(wind_subset)
        
        # 合并所有数据
        if merged_list:
            self.merged_data = pd.concat(merged_list, axis=1, join='inner')
            # 确保所有列都是数值型
            for col in self.merged_data.columns:
                self.merged_data[col] = pd.to_numeric(self.merged_data[col], errors='coerce')
            
            print(f"成功合并气象数据: {self.merged_data.shape[0]} 条记录，{self.merged_data.shape[1]} 个参数")
        else:
            print("⚠️  没有可合并的数据")
    
    def _print_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> None:
        """
        打印强相关性参数对
        
        Parameters  
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
        threshold : float
            相关性阈值
        """
        print(f"\n强相关性参数对 (|r| > {threshold}):")
        print("-" * 50)
        
        strong_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # 避免重复
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > threshold and not pd.isna(corr_val):
                        strong_corrs.append((col1, col2, corr_val))
        
        if strong_corrs:
            for col1, col2, corr_val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {col1} ↔ {col2}: r = {corr_val:.3f}")
        else:
            print(f"  未发现超过阈值 {threshold} 的强相关性")
    
    def detect_anomalies(self) -> None:
        """
        检测气象数据中的异常值
        """
        print("\n" + "=" * 60)
        print("异常值检测")
        print("=" * 60)
        
        if self.merged_data is None or self.merged_data.empty:
            print("⚠️  无合并数据可供异常值检测")
            return
        
        # 使用Z-score方法检测异常值
        z_threshold = 3
        anomalies_summary = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 设置整体标题的中文字体
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            fig.suptitle('基于Z-Score方法的异常值检测', fontproperties=font_prop, fontsize=16, fontweight='bold')
        else:
            fig.suptitle('基于Z-Score方法的异常值检测', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(self.merged_data.columns[:6]):
            row, col_idx = divmod(i, 3)
            ax = axes[row, col_idx]
            
            data_series = self.merged_data[col].dropna()
            if len(data_series) == 0:
                continue
                
            # 计算Z-score
            z_scores = np.abs(stats.zscore(data_series))
            anomalies = z_scores > z_threshold
            anomaly_count = anomalies.sum()
            anomaly_pct = (anomaly_count / len(data_series)) * 100
            
            anomalies_summary[col] = {
                'count': anomaly_count,
                'percentage': anomaly_pct,
                'indices': data_series.index[anomalies].tolist()
            }
            
            # 绘制时间序列和异常点
            ax.plot(data_series.index, data_series, 'b-', alpha=0.7, linewidth=0.8, label='正常值')
            if anomaly_count > 0:
                anomaly_data = data_series[anomalies]
                ax.scatter(anomaly_data.index, anomaly_data, color='red', s=20, 
                          label=f'异常值 ({anomaly_count})', zorder=5)
            
            # 使用中文字体设置
            if available_fonts:
                font_prop = FontProperties()
                font_prop.set_family(available_fonts[0])
                ax.set_title(f'{col}\n异常值: {anomaly_count} 个 ({anomaly_pct:.1f}%)', fontproperties=font_prop)
                ax.set_ylabel(col, fontproperties=font_prop)
                # 设置图例字体
                legend = ax.legend()
                if legend:
                    for text in legend.get_texts():
                        text.set_fontproperties(font_prop)
            else:
                ax.set_title(f'{col}\n异常值: {anomaly_count} 个 ({anomaly_pct:.1f}%)')
                ax.set_ylabel(col)
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化时间轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 删除空白子图
        for i in range(len(self.merged_data.columns), 6):
            row, col_idx = divmod(i, 3)
            if row < 2:
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig('../results/anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印异常值摘要
        print("\n异常值检测摘要 (Z-score > 3):")
        print("-" * 40)
        for param, info in anomalies_summary.items():
            print(f"{param}: {info['count']} 个异常值 ({info['percentage']:.1f}%)")
    
    def generate_summary_report(self) -> None:
        """
        生成数据分析摘要报告
        """
        print("\n" + "=" * 60)
        print("数据分析摘要报告")
        print("=" * 60)
        
        report = {}
        
        # 数据集基本信息
        datasets = {
            'Visibility': self.vis_data,
            'PTU': self.ptu_data,
            'Wind': self.wind_data
        }
        
        for name, data in datasets.items():
            if data is not None and not data.empty:
                report[name] = {
                    'records': len(data),
                    'columns': data.shape[1], 
                    'time_range': f"{data.index.min()} 至 {data.index.max()}",
                    'missing_values': data.isnull().sum().sum(),
                    'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                }
        
        # 打印报告
        for dataset, info in report.items():
            print(f"\n{dataset} 数据集:")
            print(f"  ├─ 记录数量: {info['records']:,}")
            print(f"  ├─ 字段数量: {info['columns']}")
            print(f"  ├─ 时间范围: {info['time_range']}")
            print(f"  ├─ 缺失值: {info['missing_values']} ({info['missing_percentage']:.2f}%)")
            print(f"  └─ 数据质量: {'良好' if info['missing_percentage'] < 5 else '需要关注'}")
        
        # 保存报告到文件
        self._save_summary_report(report)
    
    def _save_summary_report(self, report: Dict[str, Any]) -> None:
        """
        保存摘要报告到文件
        
        Parameters
        ----------
        report : Dict[str, Any]
            报告数据
        """
        os.makedirs('../results', exist_ok=True)
        
        with open('../results/eda_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("AMOS气象数据探索性分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for dataset, info in report.items():
                f.write(f"{dataset} 数据集分析结果:\n")
                f.write(f"  记录数量: {info['records']:,}\n")
                f.write(f"  字段数量: {info['columns']}\n")
                f.write(f"  时间范围: {info['time_range']}\n")
                f.write(f"  缺失值: {info['missing_values']} ({info['missing_percentage']:.2f}%)\n")
                f.write(f"  数据质量: {'良好' if info['missing_percentage'] < 5 else '需要关注'}\n\n")


class VideoDataEDA:
    """
    视频数据探索性分析类
    
    主要功能包括:
    - 视频基本信息提取
    - 关键帧分析
    - 图像质量评估
    """
    
    def __init__(self, video_path: str = "../机场视频/a.mp4"):
        """
        初始化视频分析类
        
        Parameters
        ----------
        video_path : str
            视频文件路径
        """
        self.video_path = video_path
        self.video_info = {}
        
    def analyze_video_properties(self) -> None:
        """
        分析视频基本属性
        """
        print("\n" + "=" * 60)
        print("机场视频数据分析")
        print("=" * 60)
        
        if not os.path.exists(self.video_path):
            print(f"⚠️  视频文件不存在: {self.video_path}")
            return
        
        # 使用OpenCV读取视频信息
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("⚠️  无法打开视频文件")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps
        
        self.video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'file_size': os.path.getsize(self.video_path) / (1024**2)  # MB
        }
        
        print(f"视频基本信息:")
        print(f"  ├─ 分辨率: {width} × {height}")
        print(f"  ├─ 帧率: {fps:.2f} FPS")
        print(f"  ├─ 总帧数: {frame_count:,}")
        print(f"  ├─ 时长: {duration/3600:.2f} 小时")
        print(f"  ├─ 文件大小: {self.video_info['file_size']:.1f} MB")
        print(f"  └─ 平均码率: {(self.video_info['file_size'] * 8) / duration:.1f} Mbps")
        
        cap.release()
        
        # 分析关键帧提取方案
        self._analyze_keyframe_strategy()
    
    def _analyze_keyframe_strategy(self) -> None:
        """
        分析关键帧提取策略
        """
        print(f"\n关键帧提取策略分析:")
        print("-" * 30)
        
        # 根据论文描述：每5秒提取一帧
        extract_interval = 5  # 秒
        frames_per_interval = int(self.video_info['fps'] * extract_interval)
        total_keyframes = int(self.video_info['duration'] / extract_interval)
        
        print(f"  提取间隔: 每 {extract_interval} 秒")
        print(f"  每个间隔帧数: {frames_per_interval}")
        print(f"  预计关键帧数: {total_keyframes}")
        print(f"  数据压缩比: {self.video_info['frame_count']/total_keyframes:.1f}:1")
        
        # 计算与AMOS数据的时间对齐
        amos_interval = 15  # AMOS数据15秒间隔
        alignment_ratio = amos_interval / extract_interval
        print(f"  与AMOS数据对齐: 每 {alignment_ratio:.1f} 个关键帧对应1个AMOS记录")
    
    def sample_frame_analysis(self, sample_count: int = 10) -> None:
        """
        抽样帧图像质量分析
        
        Parameters
        ----------
        sample_count : int
            抽样帧数量
        """
        print(f"\n抽样帧图像质量分析 (样本数: {sample_count}):")
        print("-" * 40)
        
        if not os.path.exists(self.video_path):
            print("⚠️  视频文件不存在")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("⚠️  无法打开视频文件")
            return
        
        # 均匀抽样帧位置
        frame_indices = np.linspace(0, self.video_info['frame_count']-1, 
                                   sample_count, dtype=int)
        
        brightness_values = []
        contrast_values = []
        blur_values = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 计算图像质量指标
                brightness = np.mean(gray)
                contrast = np.std(gray)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                brightness_values.append(brightness)
                contrast_values.append(contrast)
                blur_values.append(blur)
        
        cap.release()
        
        # 计算统计量
        brightness_stats = {
            'mean': np.mean(brightness_values),
            'std': np.std(brightness_values),
            'min': np.min(brightness_values),
            'max': np.max(brightness_values)
        }
        
        contrast_stats = {
            'mean': np.mean(contrast_values),
            'std': np.std(contrast_values),
            'min': np.min(contrast_values),
            'max': np.max(contrast_values)
        }
        
        blur_stats = {
            'mean': np.mean(blur_values),
            'std': np.std(blur_values),
            'min': np.min(blur_values),
            'max': np.max(blur_values)
        }
        
        print(f"  亮度 (0-255): 均值={brightness_stats['mean']:.1f}, "
              f"标准差={brightness_stats['std']:.1f}")
        print(f"  对比度: 均值={contrast_stats['mean']:.1f}, "
              f"标准差={contrast_stats['std']:.1f}")
        print(f"  清晰度: 均值={blur_stats['mean']:.1f}, "
              f"标准差={blur_stats['std']:.1f}")
        
        # 评估图像质量
        self._evaluate_image_quality(brightness_stats, contrast_stats, blur_stats)
        
        # 可视化图像质量分布
        self._plot_image_quality_distribution(brightness_values, contrast_values, blur_values)
    
    def _evaluate_image_quality(self, brightness: Dict, contrast: Dict, blur: Dict) -> None:
        """
        评估图像质量
        
        Parameters
        ----------
        brightness : Dict
            亮度统计信息
        contrast : Dict
            对比度统计信息  
        blur : Dict
            清晰度统计信息
        """
        print(f"\n图像质量评估:")
        print("-" * 20)
        
        # 亮度评估 (理想范围: 50-200)
        if 50 <= brightness['mean'] <= 200:
            brightness_quality = "良好"
        elif brightness['mean'] < 50:
            brightness_quality = "偏暗"
        else:
            brightness_quality = "偏亮"
        print(f"  亮度质量: {brightness_quality}")
        
        # 对比度评估 (标准差 > 30 表示对比度较好)
        contrast_quality = "良好" if contrast['mean'] > 30 else "偏低"
        print(f"  对比度质量: {contrast_quality}")
        
        # 清晰度评估 (Laplacian方差 > 100 表示较清晰)
        blur_quality = "清晰" if blur['mean'] > 100 else "模糊"
        print(f"  清晰度质量: {blur_quality}")
        
        # 综合评估
        quality_score = 0
        if brightness_quality == "良好":
            quality_score += 1
        if contrast_quality == "良好":
            quality_score += 1
        if blur_quality == "清晰":
            quality_score += 1
        
        overall_quality = ["较差", "一般", "良好", "优秀"][quality_score]
        print(f"  ✓ 综合质量: {overall_quality} ({quality_score}/3)")
    
    def _plot_image_quality_distribution(self, brightness: List[float], 
                                       contrast: List[float], 
                                       blur: List[float]) -> None:
        """
        绘制图像质量指标分布图
        
        Parameters
        ----------
        brightness : List[float]
            亮度值列表
        contrast : List[float]
            对比度值列表
        blur : List[float]
            清晰度值列表
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 设置整体标题的中文字体
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            fig.suptitle('视频帧图像质量分布分析', fontproperties=font_prop, fontsize=14, fontweight='bold')
        else:
            fig.suptitle('视频帧图像质量分布分析', fontsize=14, fontweight='bold')
        
        # 亮度分布
        axes[0].hist(brightness, bins=10, alpha=0.7, color='gold', edgecolor='black')
        axes[0].axvline(np.mean(brightness), color='red', linestyle='--', 
                       label=f'均值: {np.mean(brightness):.1f}')
        
        # 使用中文字体设置亮度图
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            axes[0].set_title('亮度分布', fontproperties=font_prop)
            axes[0].set_xlabel('亮度 (0-255)', fontproperties=font_prop)
            axes[0].set_ylabel('频次', fontproperties=font_prop)
            legend = axes[0].legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_prop)
        else:
            axes[0].set_title('亮度分布')
            axes[0].set_xlabel('亮度 (0-255)')
            axes[0].set_ylabel('频次')
            axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 对比度分布
        axes[1].hist(contrast, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1].axvline(np.mean(contrast), color='red', linestyle='--',
                       label=f'均值: {np.mean(contrast):.1f}')
        
        # 使用中文字体设置对比度图
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            axes[1].set_title('对比度分布', fontproperties=font_prop)
            axes[1].set_xlabel('对比度 (标准差)', fontproperties=font_prop)
            axes[1].set_ylabel('频次', fontproperties=font_prop)
            legend = axes[1].legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_prop)
        else:
            axes[1].set_title('对比度分布')
            axes[1].set_xlabel('对比度 (标准差)')
            axes[1].set_ylabel('频次')
            axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 清晰度分布
        axes[2].hist(blur, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[2].axvline(np.mean(blur), color='red', linestyle='--',
                       label=f'均值: {np.mean(blur):.1f}')
        
        # 使用中文字体设置清晰度图
        if available_fonts:
            font_prop = FontProperties()
            font_prop.set_family(available_fonts[0])
            axes[2].set_title('清晰度分布', fontproperties=font_prop)
            axes[2].set_xlabel('拉普拉斯方差', fontproperties=font_prop)
            axes[2].set_ylabel('频次', fontproperties=font_prop)
            legend = axes[2].legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_prop)
        else:
            axes[2].set_title('清晰度分布')
            axes[2].set_xlabel('拉普拉斯方差')
            axes[2].set_ylabel('频次')
            axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/video_quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_eda() -> None:
    """
    运行综合探索性数据分析
    """
    print("🚀 开始机场能见度数据综合探索性分析")
    print("=" * 80)
    
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    
    # 1. 气象数据分析
    print("\n📊 第一部分: AMOS气象数据分析")
    weather_eda = WeatherDataEDA()
    
    try:
        weather_eda.load_all_data()
        weather_eda.analyze_data_quality()
        weather_eda.analyze_visibility_trends()
        weather_eda.analyze_weather_correlations()
        weather_eda.detect_anomalies()
        weather_eda.generate_summary_report()
    except Exception as e:
        print(f"⚠️  气象数据分析出错: {str(e)}")
    
    # 2. 视频数据分析
    print("\n🎥 第二部分: 机场视频数据分析")
    video_eda = VideoDataEDA()
    
    try:
        video_eda.analyze_video_properties()
        video_eda.sample_frame_analysis(sample_count=20)
    except Exception as e:
        print(f"⚠️  视频数据分析出错: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ 探索性数据分析完成！")
    print("📁 分析结果已保存到 '../results/' 目录")
    print("📋 主要输出文件:")
    print("   ├─ data_quality_overview.png - 数据质量概览")
    print("   ├─ visibility_time_series.png - 能见度时间序列")
    print("   ├─ visibility_distributions.png - 能见度分布")
    print("   ├─ weather_correlations.png - 气象参数相关性")
    print("   ├─ anomaly_detection.png - 异常值检测")
    print("   ├─ video_quality_distribution.png - 视频质量分布")
    print("   └─ eda_summary_report.txt - 分析摘要报告")


if __name__ == "__main__":
    """
    主程序入口
    """
    run_comprehensive_eda()
