#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度预测系统使用示例

该脚本展示了如何使用VisibilityAnalyzer类进行完整的能见度预测分析。

Author: Data Analysis Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加src目录到Python路径
sys.path.append('src')

from three import VisibilityAnalyzer


def create_sample_data() -> pd.DataFrame:
    """
    创建示例数据（用于演示）
    
    Returns
    -------
    pd.DataFrame
        包含示例特征的数据框
    """
    np.random.seed(42)
    n_samples = 100
    
    # 创建相关的特征数据
    data = {
        # 图像特征
        'laplacian_var': np.random.normal(100, 30, n_samples),
        'sobel_mean': np.random.normal(50, 15, n_samples),
        'high_freq_ratio': np.random.normal(0.15, 0.05, n_samples),
        'edge_density': np.random.normal(0.25, 0.1, n_samples),
        'contrast_std': np.random.normal(15, 5, n_samples),
        'local_var_mean': np.random.normal(80, 20, n_samples),
        'gradient_magnitude': np.random.normal(40, 12, n_samples),
        'texture_energy': np.random.normal(0.3, 0.1, n_samples),
        'entropy_value': np.random.normal(7, 1, n_samples),
        
        # 气象特征
        'weather_humidity_pct': np.random.normal(70, 20, n_samples),
        'weather_temperature_c': np.random.normal(15, 10, n_samples),
        'weather_pressure_hpa': np.random.normal(1013, 20, n_samples),
        
        # 风速特征
        'wind_wind_speed_10m': np.random.normal(5, 3, n_samples),
        
        # 能见度指标
        'visibility_vis_1a': np.random.normal(3000, 1500, n_samples),
        'visibility_vis_10a': np.random.normal(3200, 1600, n_samples),
        'visibility_vis_raw': np.random.normal(3100, 1550, n_samples),
    }
    
    # 创建目标变量（基于特征的简单模拟）
    visibility_mor_raw = (
        5000 - 
        data['weather_humidity_pct'] * 40 + 
        data['weather_temperature_c'] * 80 +
        data['laplacian_var'] * -15 +
        np.random.normal(0, 500, n_samples)
    )
    
    # 确保能见度为正值
    visibility_mor_raw = np.maximum(visibility_mor_raw, 100)
    data['visibility_mor_raw'] = visibility_mor_raw
    
    return pd.DataFrame(data)


def run_complete_analysis() -> VisibilityAnalyzer:
    """
    运行完整的能见度预测分析流程
    
    Returns
    -------
    VisibilityAnalyzer
        训练完成的分析器
    """
    print("=" * 80)
    print("能见度预测分析系统演示")
    print("=" * 80)
    
    # 检查是否存在真实数据文件
    data_file = 'complete_synced_data.csv'
    
    if os.path.exists(data_file):
        print(f"发现数据文件 '{data_file}'，使用真实数据进行分析...")
        analyzer = VisibilityAnalyzer(data_file)
    else:
        print(f"未找到数据文件 '{data_file}'，使用模拟数据进行演示...")
        # 创建和保存示例数据
        sample_data = create_sample_data()
        sample_data.to_csv('sample_data.csv', index=False)
        analyzer = VisibilityAnalyzer('sample_data.csv')
    
    try:
        # 1. 数据探索
        print("\n1. 执行数据探索分析...")
        basic_info = analyzer.explore_data()
        
        # 2. 特征分析
        print("\n2. 执行特征相关性分析...")
        correlation_results = analyzer.analyze_features()
        
        # 3. 可视化相关性
        print("\n3. 绘制相关性热力图...")
        analyzer.visualize_correlations()
        
        # 4. 特征选择
        print("\n4. 执行特征选择...")
        selected_features = analyzer.select_features(method='manual')
        
        # 5. 多重共线性检测
        print("\n5. 检测多重共线性...")
        vif_results = analyzer.detect_multicollinearity(selected_features)
        
        # 6. 准备数据
        print("\n6. 准备训练和测试数据...")
        X_train, X_test, y_train, y_test = analyzer.prepare_data()
        
        # 7. 训练模型
        print("\n7. 训练多种回归模型...")
        model_results = analyzer.train_models(X_train, X_test, y_train, y_test)
        
        # 8. 残差分析
        print("\n8. 执行残差分析...")
        analyzer.analyze_residuals(X_test, y_test)
        
        # 9. 获取模型公式
        print("\n9. 获取模型数学公式...")
        equation = analyzer.get_model_equation()
        
        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        
        return analyzer
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        raise


def demonstrate_prediction(analyzer: VisibilityAnalyzer) -> None:
    """
    演示如何使用训练好的模型进行预测
    
    Parameters
    ----------
    analyzer : VisibilityAnalyzer
        训练完成的分析器
    """
    print("\n" + "=" * 80)
    print("预测演示")
    print("=" * 80)
    
    # 创建新的预测数据
    new_data = pd.DataFrame({
        'laplacian_var': [120.5, 85.2, 150.3],
        'high_freq_ratio': [0.18, 0.12, 0.22],
        'edge_density': [0.28, 0.20, 0.35],
        'contrast_std': [18.2, 12.8, 22.1],
        'weather_humidity_pct': [80.0, 60.0, 90.0],
        'weather_temperature_c': [25.5, 15.2, 5.8],
        'wind_wind_speed_10m': [3.2, 7.5, 1.8]
    })
    
    print("新数据样本:")
    print(new_data)
    
    try:
        # 进行预测
        predictions = analyzer.predict(new_data)
        
        print(f"\n预测结果:")
        for i, pred in enumerate(predictions):
            print(f"样本 {i+1}: 预测能见度 = {pred:.1f} 米")
        
        # 分析预测条件
        print(f"\n预测条件分析:")
        for i in range(len(new_data)):
            humidity = new_data.iloc[i]['weather_humidity_pct']
            temp = new_data.iloc[i]['weather_temperature_c']
            wind = new_data.iloc[i]['wind_wind_speed_10m']
            visibility = predictions[i]
            
            print(f"样本 {i+1}: 湿度{humidity:.1f}%, 温度{temp:.1f}°C, 风速{wind:.1f}m/s → 能见度{visibility:.0f}m")
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {e}")


def analyze_feature_importance(analyzer: VisibilityAnalyzer) -> None:
    """
    分析特征重要性
    
    Parameters
    ----------
    analyzer : VisibilityAnalyzer
        训练完成的分析器
    """
    print("\n" + "=" * 80)
    print("特征重要性分析")
    print("=" * 80)
    
    if analyzer.best_model is None:
        print("❌ 模型尚未训练，无法分析特征重要性")
        return
    
    # 获取特征系数
    coefficients = analyzer.best_model.coef_
    features = analyzer.selected_features
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        '特征': features,
        '系数': coefficients,
        '重要性': np.abs(coefficients)
    }).sort_values('重要性', ascending=False)
    
    print("特征重要性排序（按系数绝对值）:")
    print(importance_df.to_string(index=False))
    
    print(f"\n系数解释:")
    for _, row in importance_df.iterrows():
        feature = row['特征']
        coef = row['系数']
        direction = "正相关" if coef > 0 else "负相关"
        print(f"- {feature}: {direction} (系数 = {coef:.6f})")


def main():
    """主函数"""
    try:
        # 执行完整分析
        analyzer = run_complete_analysis()
        
        # 特征重要性分析
        analyze_feature_importance(analyzer)
        
        # 预测演示
        demonstrate_prediction(analyzer)
        
        print(f"\n🎉 示例演示完成！")
        print(f"📊 查看生成的图表了解模型性能")
        print(f"📄 查看控制台输出了解分析结果")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 