#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雾模糊度分析系统 - 快速演示

展示核心功能和使用方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_feature_extraction():
    """演示特征提取功能"""
    print("="*60)
    print("演示：分层特征提取")
    print("="*60)
    
    # 创建模拟雾图像
    img_clear = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 清晰图像
    img_foggy = img_clear * 0.3 + np.ones_like(img_clear) * 180  # 雾图像
    img_foggy = img_foggy.astype(np.uint8)
    
    print("1. 物理特征提取演示：")
    print("-"*30)
    
    # 暗通道计算
    def calculate_dark_channel(image, patch_size=15):
        b, g, r = cv2.split(image)
        dark = np.minimum(np.minimum(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        return cv2.erode(dark, kernel)
    
    # 计算清晰和雾图像的暗通道
    dark_clear = calculate_dark_channel(img_clear)
    dark_foggy = calculate_dark_channel(img_foggy)
    
    print(f"清晰图像暗通道均值: {np.mean(dark_clear):.2f}")
    print(f"雾图像暗通道均值: {np.mean(dark_foggy):.2f}")
    print(f"差异: {np.mean(dark_foggy) - np.mean(dark_clear):.2f}")
    
    print("\n2. 感知特征提取演示：")
    print("-"*30)
    
    # 拉普拉斯方差计算
    def calculate_laplacian_variance(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    lap_clear = calculate_laplacian_variance(img_clear)
    lap_foggy = calculate_laplacian_variance(img_foggy)
    
    print(f"清晰图像拉普拉斯方差: {lap_clear:.2f}")
    print(f"雾图像拉普拉斯方差: {lap_foggy:.2f}")
    print(f"清晰度损失: {(lap_clear - lap_foggy)/lap_clear*100:.1f}%")
    
    print("\n3. 雾浓度估算演示：")
    print("-"*30)
    
    # 简化的雾浓度计算
    def estimate_fog_density(image):
        dark_channel = calculate_dark_channel(image)
        atmospheric_light = np.percentile(image, 99)
        transmission = 1 - 0.95 * (dark_channel / atmospheric_light)
        return 1 - np.mean(transmission)
    
    fog_density_clear = estimate_fog_density(img_clear)
    fog_density_foggy = estimate_fog_density(img_foggy)
    
    print(f"清晰图像雾浓度: {fog_density_clear:.3f}")
    print(f"雾图像雾浓度: {fog_density_foggy:.3f}")
    print(f"雾浓度增加: {fog_density_foggy - fog_density_clear:.3f}")


def demo_mathematical_model():
    """演示数学模型"""
    print("\n" + "="*60)
    print("演示：数学模型构建")
    print("="*60)
    
    print("1. 物理散射模型：")
    print("-"*30)
    print("大气散射模型: I(x) = J(x)·t(x) + A·(1-t(x))")
    print("透射率模型: t(x) = exp(-β·d)")
    print("散射系数: β = 3.912 / 能见度")
    
    # 模拟不同能见度下的透射率
    visibility_range = np.arange(100, 2000, 100)  # 100m到2000m
    beta_values = 3.912 / visibility_range
    transmission_values = np.exp(-beta_values * 0.5)  # 假设距离0.5km
    
    print(f"\n能见度范围: {visibility_range[0]}m - {visibility_range[-1]}m")
    print(f"对应透射率: {transmission_values[0]:.3f} - {transmission_values[-1]:.3f}")
    
    print("\n2. 综合模糊度模型：")
    print("-"*30)
    print("B(t) = α·B_physical(t) + β·B_perceptual(t) + γ·B_statistical(t)")
    print("其中：")
    print("  B_physical = 1 - transmission  (物理模糊度)")
    print("  B_perceptual = 1/(1 + laplacian_var)  (感知模糊度)")  
    print("  B_statistical = 1 - entropy/entropy_max  (统计模糊度)")
    
    # 权重示例
    alpha, beta, gamma = 0.5, 0.3, 0.2
    print(f"\n权重分配示例: α={alpha}, β={beta}, γ={gamma}")
    
    # 计算示例
    physical_blur = 1 - transmission_values[10]  # 选择一个透射率值
    perceptual_blur = 0.3  # 示例值
    statistical_blur = 0.2  # 示例值
    
    comprehensive_blur = (alpha * physical_blur + 
                         beta * perceptual_blur + 
                         gamma * statistical_blur)
    
    print(f"综合模糊度计算示例:")
    print(f"  物理模糊度: {physical_blur:.3f}")
    print(f"  感知模糊度: {perceptual_blur:.3f}")
    print(f"  统计模糊度: {statistical_blur:.3f}")
    print(f"  综合模糊度: {comprehensive_blur:.3f}")


def demo_visualization():
    """演示可视化功能"""
    print("\n" + "="*60)
    print("演示：数据可视化")
    print("="*60)
    
    # 生成模拟数据
    time_points = np.arange(0, 100, 1)
    visibility = 1000 + 500 * np.sin(time_points * 0.1) + np.random.normal(0, 50, len(time_points))
    visibility = np.clip(visibility, 200, 2000)
    
    # 计算衍生特征
    beta_coeff = 3.912 / visibility
    transmission = np.exp(-beta_coeff * 0.5)
    fog_density = 1 - transmission
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Fog Blur Analysis Visualization Demo', fontsize=14, fontweight='bold')
    
    # 1. 能见度时间序列
    axes[0, 0].plot(time_points, visibility, 'b-', linewidth=2)
    axes[0, 0].set_title('Visibility Time Series')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Visibility (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 雾浓度时间序列
    axes[0, 1].plot(time_points, fog_density, 'r-', linewidth=2)
    axes[0, 1].set_title('Fog Density Time Series')
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Fog Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 能见度vs雾浓度关系
    axes[1, 0].scatter(visibility, fog_density, alpha=0.6, s=20)
    axes[1, 0].set_title('Visibility vs Fog Density')
    axes[1, 0].set_xlabel('Visibility (m)')
    axes[1, 0].set_ylabel('Fog Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 雾浓度分布直方图
    axes[1, 1].hist(fog_density, bins=20, alpha=0.7, color='green')
    axes[1, 1].set_title('Fog Density Distribution')
    axes[1, 1].set_xlabel('Fog Density')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if not os.path.exists("../results"):
        os.makedirs("../results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../results/demo_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {filename}")
    
    plt.show()
    
    # 输出统计信息
    print(f"\n数据统计:")
    print(f"能见度范围: {visibility.min():.0f}m - {visibility.max():.0f}m")
    print(f"平均能见度: {visibility.mean():.0f}m")
    print(f"雾浓度范围: {fog_density.min():.3f} - {fog_density.max():.3f}")
    print(f"平均雾浓度: {fog_density.mean():.3f}")
    
    # 雾事件统计
    fog_events = np.sum(visibility < 1000)
    severe_fog = np.sum(visibility < 500)
    print(f"雾事件 (能见度<1000m): {fog_events} 次 ({fog_events/len(visibility)*100:.1f}%)")
    print(f"严重雾霾 (能见度<500m): {severe_fog} 次 ({severe_fog/len(visibility)*100:.1f}%)")


def demo_model_comparison():
    """演示模型对比"""
    print("\n" + "="*60)
    print("演示：模型性能对比")
    print("="*60)
    
    # 模拟不同模型的性能
    models = {
        'Physical Model': {'R2': 0.72, 'RMSE': 0.158, 'MAE': 0.124},
        'Ensemble Model (RF)': {'R2': 0.89, 'RMSE': 0.098, 'MAE': 0.076},
        'Ensemble Model (GB)': {'R2': 0.87, 'RMSE': 0.105, 'MAE': 0.082},
        'Integrated Model': {'R2': 0.92, 'RMSE': 0.085, 'MAE': 0.067}
    }
    
    print("模型性能对比:")
    print("-"*50)
    print(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
    print("-"*50)
    
    for model_name, metrics in models.items():
        print(f"{model_name:<20} {metrics['R2']:<8.3f} {metrics['RMSE']:<8.3f} {metrics['MAE']:<8.3f}")
    
    # 找出最佳模型
    best_model = max(models.items(), key=lambda x: x[1]['R2'])
    print(f"\n最佳模型: {best_model[0]} (R² = {best_model[1]['R2']:.3f})")
    
    print("\n特征重要性排序 (模拟):")
    print("-"*30)
    feature_importance = {
        'fog_density': 0.25,
        'transmission_mean': 0.18,
        'dark_channel_mean': 0.15,
        'laplacian_variance': 0.12,
        'atmospheric_light': 0.10,
        'entropy': 0.08,
        'rms_contrast': 0.07,
        'edge_density': 0.05
    }
    
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"{i:2d}. {feature:<20}: {importance:.3f}")


def main():
    """主演示函数"""
    print("="*80)
    print("雾模糊度分析系统 - 核心功能演示")
    print("="*80)
    print("基于江西省数学建模比赛题目的整合分析方案")
    print("整合了物理理论、机器学习和图像处理的多维度分析方法")
    
    try:
        # 运行各个演示模块
        demo_feature_extraction()
        demo_mathematical_model()
        demo_visualization()
        demo_model_comparison()
        
        print("\n" + "="*80)
        print("演示完成！主要展示内容:")
        print("="*80)
        print("✅ 1. 分层特征提取（物理+感知+统计）")
        print("✅ 2. 数学模型构建（散射理论+综合模糊度）")
        print("✅ 3. 数据可视化（时序分析+关系图表）")
        print("✅ 4. 模型性能对比（多算法集成评估）")
        
        print("\n📝 使用建议:")
        print("1. 对于江西省数学建模比赛第一问，重点关注物理理论模型")
        print("2. 结合暗通道先验和大气散射理论建立数学模型")
        print("3. 使用分层特征体系增强模型的准确性和解释性")
        print("4. 通过可视化分析验证模型的合理性")
        
        print(f"\n📁 相关文件:")
        print("- 完整实现: src/integrated_fog_blur_analysis.py")
        print("- 分析报告: src/analysis_comparison_report.md")
        print("- 结果保存: ../results/ 目录")
        
    except Exception as e:
        print(f"演示过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 