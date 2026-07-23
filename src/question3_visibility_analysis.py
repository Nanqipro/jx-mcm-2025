#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析脚本：能见度与图像特征、天气因素关系分析

本脚本从Jupyter notebook转换而来，用于分析能见度数据与各种特征的关系。
包含数据探索、可视化、相关性分析、异常值检测等功能模块。

作者: 数据分析团队
版本: 1.0.0
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')

# 设置中文字体（支持中文注释显示）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class VisibilityAnalyzer:
    """
    能见度数据分析器
    
    用于分析能见度与各种特征（图像特征、天气因素等）之间的关系。
    提供数据加载、探索性分析、可视化、相关性分析等功能。
    """
    
    def __init__(self, data_path: str):
        """
        初始化分析器
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.image_features = [
            'laplacian_var', 'sobel_mean', 'sobel_std', 'contrast_std',
            'high_freq_ratio', 'edge_density', 'entropy', 'local_var_mean', 'brenner'
        ]
        self.visibility_indicators = [
            'visibility_vis_1a', 'visibility_vis_10a', 
            'visibility_mor_raw', 'visibility_vis_raw'
        ]
        self.weather_features = [
            'weather_temperature_c', 'weather_humidity_pct', 'weather_dewpoint_c',
            'weather_pressure_hpa', 'weather_qnh_hpa'
        ]
        self.wind_features = [
            'wind_wind_speed_2m', 'wind_wind_speed_10m', 
            'wind_wind_dir_2m', 'wind_wind_dir_10m', 'wind_gust_speed'
        ]
    
    def load_data(self) -> pd.DataFrame:
        """
        加载数据并进行基本检查
        
        Returns
        -------
        pd.DataFrame
            加载的数据框
        """
        try:
            self.df = pd.read_csv(self.data_path)
            print("=" * 50)
            print("数据基本信息")
            print("=" * 50)
            print(f"数据形状: {self.df.shape}")
            print(f"\n数据类型:")
            print(self.df.dtypes)
            print(f"\n前5行数据:")
            print(self.df.head())
            print(f"\n缺失值情况:")
            print(self.df.isnull().sum())
            print(f"\n数值型特征描述性统计:")
            print(self.df.describe())
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件 {self.data_path} 未找到")
    
    def analyze_visibility_correlations(self) -> None:
        """
        分析能见度指标之间的相关性
        
        生成可视化相关性矩阵图表
        """
        print("\n" + "=" * 50)
        print("能见度指标相关性分析")
        print("=" * 50)
        
        # 计算相关性矩阵
        vis_corr = self.df[self.visibility_indicators].corr()
        print("能见度指标相关性矩阵:")
        print(vis_corr)
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(vis_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Visibility Indicators Correlation Matrix')  # 英文标题
        plt.xlabel('Visibility Indicators')  # 英文X轴标签
        plt.ylabel('Visibility Indicators')  # 英文Y轴标签
        plt.tight_layout()
        plt.show()
        
        # 分析高相关性对
        print("\n高相关性指标对 (|r| > 0.8):")
        for i in range(len(vis_corr.columns)):
            for j in range(i+1, len(vis_corr.columns)):
                corr_val = vis_corr.iloc[i, j]
                if abs(corr_val) > 0.8:
                    print(f"{vis_corr.columns[i]} - {vis_corr.columns[j]}: {corr_val:.3f}")
    
    def plot_distribution_histograms(self) -> None:
        """
        绘制主要特征的分布直方图
        
        展示数据分布特征和统计信息
        """
        print("\n" + "=" * 50)
        print("特征分布分析")
        print("=" * 50)
        
        # 选择主要特征进行分布分析
        main_features = self.visibility_indicators + ['weather_temperature_c', 
                                                     'weather_humidity_pct', 'laplacian_var']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(main_features):
            if i < len(axes):
                # 绘制直方图
                axes[i].hist(self.df[feature], bins=50, alpha=0.7, 
                           color='skyblue', edgecolor='black')
                
                # 添加统计信息
                mean_val = self.df[feature].mean()
                std_val = self.df[feature].std()
                median_val = self.df[feature].median()
                
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', 
                              label=f'Median: {median_val:.2f}')
                
                axes[i].set_title(f'Distribution of {feature}')  # 英文标题
                axes[i].set_xlabel('Value')  # 英文X轴标签
                axes[i].set_ylabel('Frequency')  # 英文Y轴标签
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Feature Distribution Analysis', y=1.02, fontsize=16)  # 英文总标题
        plt.show()
    
    def analyze_image_features_correlation(self) -> None:
        """
        分析图像特征与能见度的相关性
        
        计算并可视化图像特征与visibility_mor_raw的相关性
        """
        print("\n" + "=" * 50)
        print("图像特征与能见度相关性分析")
        print("=" * 50)
        
        # 计算相关性
        correlations = {}
        target = 'visibility_mor_raw'
        
        for feature in self.image_features:
            corr = self.df[feature].corr(self.df[target])
            correlations[feature] = corr
            print(f"{feature} - {target}: {corr:.3f}")
        
        # 按相关性绝对值排序
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n图像特征与{target}相关性排序 (按绝对值):")
        for feature, corr in sorted_corr:
            print(f"{feature}: {corr:.3f}")
        
        # 绘制相关性条形图
        features, corr_values = zip(*sorted_corr)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        bars = plt.bar(range(len(features)), corr_values, color=colors, alpha=0.7)
        
        plt.title(f'Image Features Correlation with {target}')  # 英文标题
        plt.xlabel('Image Features')  # 英文X轴标签
        plt.ylabel('Correlation Coefficient')  # 英文Y轴标签
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, corr_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01 if value > 0 else value - 0.01,
                    f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_weather_correlation(self) -> None:
        """
        分析天气因素与能见度的相关性
        
        计算并可视化天气特征与visibility_mor_raw的相关性
        """
        print("\n" + "=" * 50)
        print("天气因素与能见度相关性分析")
        print("=" * 50)
        
        # 计算相关性
        correlations = {}
        target = 'visibility_mor_raw'
        
        all_weather_wind = self.weather_features + self.wind_features
        
        for feature in all_weather_wind:
            corr = self.df[feature].corr(self.df[target])
            correlations[feature] = corr
            print(f"{feature} - {target}: {corr:.3f}")
        
        # 按相关性绝对值排序
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n天气因素与{target}相关性排序 (按绝对值):")
        for feature, corr in sorted_corr:
            print(f"{feature}: {corr:.3f}")
        
        # 绘制相关性条形图
        features, corr_values = zip(*sorted_corr)
        
        plt.figure(figsize=(14, 8))
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        bars = plt.bar(range(len(features)), corr_values, color=colors, alpha=0.7)
        
        plt.title(f'Weather Factors Correlation with {target}')  # 英文标题
        plt.xlabel('Weather and Wind Features')  # 英文X轴标签
        plt.ylabel('Correlation Coefficient')  # 英文Y轴标签
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, corr_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.02 if value > 0 else value - 0.02,
                    f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top')
        
        plt.tight_layout()
        plt.show()
    
    def detect_outliers(self) -> Dict[str, Any]:
        """
        检测数据中的异常值
        
        使用IQR方法和Z-score方法检测异常值
        
        Returns
        -------
        Dict[str, Any]
            异常值检测结果
        """
        print("\n" + "=" * 50)
        print("异常值检测分析")
        print("=" * 50)
        
        outlier_results = {}
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # IQR方法检测异常值
        for col in numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100
            
            outlier_results[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_percentage > 5:  # 报告异常值比例超过5%的特征
                print(f"{col}: {outlier_count} 个异常值 ({outlier_percentage:.2f}%)")
        
        # 绘制主要特征的箱型图
        main_features = ['visibility_mor_raw', 'weather_humidity_pct', 
                        'weather_temperature_c', 'laplacian_var']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(main_features):
            axes[i].boxplot(self.df[feature])
            axes[i].set_title(f'Box Plot of {feature}')  # 英文标题
            axes[i].set_ylabel('Value')  # 英文Y轴标签
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Outlier Detection - Box Plots', fontsize=16)  # 英文总标题
        plt.tight_layout()
        plt.show()
        
        return outlier_results
    
    def feature_ranking_analysis(self) -> None:
        """
        特征重要性排名分析
        
        基于与target变量的相关性对所有特征进行排名
        """
        print("\n" + "=" * 50)
        print("特征重要性排名分析")
        print("=" * 50)
        
        target = 'visibility_mor_raw'
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        feature_features = [col for col in numeric_features if col != target and 'time' not in col.lower()]
        
        # 计算所有特征与目标变量的相关性
        correlations = {}
        for feature in feature_features:
            corr = abs(self.df[feature].corr(self.df[target]))
            correlations[feature] = corr
        
        # 按相关性排序
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"特征与{target}的相关性排名 (按绝对值):")
        print("-" * 60)
        for i, (feature, corr) in enumerate(sorted_features[:15], 1):
            print(f"{i:2d}. {feature:<25}: {corr:.3f}")
        
        # 绘制top 15特征相关性图
        top_features, top_corrs = zip(*sorted_features[:15])
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_corrs, color='steelblue', alpha=0.7)
        plt.title(f'Top 15 Features Correlation with {target}')  # 英文标题
        plt.xlabel('Absolute Correlation Coefficient')  # 英文X轴标签
        plt.ylabel('Features')  # 英文Y轴标签
        plt.yticks(range(len(top_features)), top_features)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, value in zip(bars, top_corrs):
            plt.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def extreme_values_analysis(self) -> None:
        """
        极值分析
        
        分析数据中的极端值及其分布特征
        """
        print("\n" + "=" * 50)
        print("极值分析")
        print("=" * 50)
        
        target = 'visibility_mor_raw'
        
        # 分析能见度的极值
        visibility_data = self.df[target]
        
        print(f"{target} 极值统计:")
        print(f"最小值: {visibility_data.min():.2f}")
        print(f"最大值: {visibility_data.max():.2f}")
        print(f"1%分位数: {visibility_data.quantile(0.01):.2f}")
        print(f"99%分位数: {visibility_data.quantile(0.99):.2f}")
        
        # 极值条件分析
        low_visibility = self.df[visibility_data <= visibility_data.quantile(0.1)]
        high_visibility = self.df[visibility_data >= visibility_data.quantile(0.9)]
        
        print(f"\n低能见度条件 (≤10%分位数) 统计:")
        print(f"样本数: {len(low_visibility)}")
        print(f"平均湿度: {low_visibility['weather_humidity_pct'].mean():.1f}%")
        print(f"平均温度: {low_visibility['weather_temperature_c'].mean():.1f}°C")
        
        print(f"\n高能见度条件 (≥90%分位数) 统计:")
        print(f"样本数: {len(high_visibility)}")
        print(f"平均湿度: {high_visibility['weather_humidity_pct'].mean():.1f}%")
        print(f"平均温度: {high_visibility['weather_temperature_c'].mean():.1f}°C")
    
    def run_complete_analysis(self) -> None:
        """
        运行完整的数据分析流程
        
        按序执行所有分析模块
        """
        print("开始完整数据分析流程...")
        print("=" * 80)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 能见度指标相关性分析
        self.analyze_visibility_correlations()
        
        # 3. 分布直方图分析
        self.plot_distribution_histograms()
        
        # 4. 图像特征相关性分析
        self.analyze_image_features_correlation()
        
        # 5. 天气因素相关性分析
        self.analyze_weather_correlation()
        
        # 6. 异常值检测
        outlier_results = self.detect_outliers()
        
        # 7. 特征排名分析
        self.feature_ranking_analysis()
        
        # 8. 极值分析
        self.extreme_values_analysis()
        
        print("\n" + "=" * 80)
        print("数据分析完成！")
        print("=" * 80)


def main() -> None:
    """
    主函数：执行完整的数据分析流程
    
    创建VisibilityAnalyzer实例并运行完整分析
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        parser = argparse.ArgumentParser(description="分析图像、气象要素与能见度的关系")
        parser.add_argument(
            "--data",
            type=Path,
            default=repo_root / "data" / "private" / "processed" / "complete_synced_data.csv",
            help="完成时间同步的图像与 AMOS 特征 CSV",
        )
        args = parser.parse_args()
        data_file = args.data.expanduser().resolve()
        if not data_file.is_file():
            raise SystemExit(
                f"找不到数据文件: {data_file}\n"
                "可先运行 scripts/generate_demo_data.py 生成演示数据。"
            )
        
        # 创建分析器实例并运行分析
        analyzer = VisibilityAnalyzer(str(data_file))
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
