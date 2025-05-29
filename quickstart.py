#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度预测系统快速开始脚本

该脚本提供了最简单的方式来快速体验能见度预测系统的功能。

Author: Data Analysis Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np

# 确保能够导入我们的模块
sys.path.append('src')

def print_header(title: str) -> None:
    """打印标题"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_step(step_num: int, description: str) -> None:
    """打印步骤"""
    print(f"\n📋 步骤 {step_num}: {description}")
    print("-" * 50)

def create_demo_data() -> None:
    """创建演示数据"""
    print_step(1, "创建演示数据")
    
    np.random.seed(42)
    n_samples = 200
    
    print(f"正在生成 {n_samples} 个样本的演示数据...")
    
    # 生成相关的特征数据
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
        
        # 其他能见度指标
        'visibility_vis_1a': np.random.normal(3000, 1500, n_samples),
        'visibility_vis_10a': np.random.normal(3200, 1600, n_samples),
        'visibility_vis_raw': np.random.normal(3100, 1550, n_samples),
    }
    
    # 创建目标变量（基于特征的真实关系模拟）
    visibility_mor_raw = (
        5000 +                                        # 基础能见度
        data['weather_temperature_c'] * 80 +          # 温度正相关
        data['weather_humidity_pct'] * -40 +          # 湿度负相关  
        data['laplacian_var'] * -15 +                 # 图像清晰度负相关
        data['wind_wind_speed_10m'] * 50 +            # 风速正相关
        np.random.normal(0, 500, n_samples)           # 随机噪声
    )
    
    # 确保能见度为正值
    visibility_mor_raw = np.maximum(visibility_mor_raw, 100)
    data['visibility_mor_raw'] = visibility_mor_raw
    
    # 保存数据
    df = pd.DataFrame(data)
    df.to_csv('demo_data.csv', index=False)
    
    print(f"✅ 演示数据已保存为 'demo_data.csv'")
    print(f"📊 数据集包含 {n_samples} 个样本，{len(data)} 个特征")
    
    return df

def run_quick_analysis() -> None:
    """运行快速分析"""
    print_step(2, "执行快速分析")
    
    try:
        from three import VisibilityAnalyzer
        
        # 初始化分析器
        print("正在初始化能见度预测分析器...")
        analyzer = VisibilityAnalyzer('demo_data.csv')
        
        # 快速数据探索
        print("\n🔍 数据探索分析...")
        basic_info = analyzer.explore_data()
        
        # 特征选择
        print("\n🎯 特征选择...")
        selected_features = analyzer.select_features(method='manual')
        print(f"选择了 {len(selected_features)} 个核心特征")
        
        # 准备数据
        print("\n📊 准备训练数据...")
        X_train, X_test, y_train, y_test = analyzer.prepare_data()
        
        # 训练模型
        print("\n🤖 训练预测模型...")
        model_results = analyzer.train_models(X_train, X_test, y_train, y_test)
        
        # 获取最佳模型性能
        best_model_name = max(model_results.keys(), 
                             key=lambda x: model_results[x]['test_r2'])
        best_performance = model_results[best_model_name]
        
        print(f"\n✅ 模型训练完成！")
        print(f"🏆 最佳模型: {best_model_name}")
        print(f"📈 测试集 R² = {best_performance['test_r2']:.4f}")
        print(f"📉 测试集 RMSE = {best_performance['test_rmse']:.4f}")
        
        return analyzer
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保已正确安装所有依赖包：pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        return None

def demonstrate_prediction(analyzer) -> None:
    """演示预测功能"""
    if analyzer is None:
        return
    
    print_step(3, "预测演示")
    
    # 创建几个典型的预测场景
    scenarios = [
        {
            'name': '高温低湿清晰天气',
            'data': {
                'laplacian_var': 150.0,
                'high_freq_ratio': 0.20,
                'edge_density': 0.35,
                'contrast_std': 20.0,
                'weather_humidity_pct': 40.0,
                'weather_temperature_c': 25.0,
                'wind_wind_speed_10m': 8.0
            }
        },
        {
            'name': '低温高湿雾霾天气',
            'data': {
                'laplacian_var': 60.0,
                'high_freq_ratio': 0.08,
                'edge_density': 0.15,
                'contrast_std': 8.0,
                'weather_humidity_pct': 95.0,
                'weather_temperature_c': 5.0,
                'wind_wind_speed_10m': 2.0
            }
        },
        {
            'name': '中等条件',
            'data': {
                'laplacian_var': 100.0,
                'high_freq_ratio': 0.15,
                'edge_density': 0.25,
                'contrast_std': 15.0,
                'weather_humidity_pct': 70.0,
                'weather_temperature_c': 15.0,
                'wind_wind_speed_10m': 5.0
            }
        }
    ]
    
    print("🔮 预测不同天气条件下的能见度...")
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            # 准备预测数据
            pred_data = pd.DataFrame([scenario['data']])
            
            # 进行预测
            prediction = analyzer.predict(pred_data)[0]
            
            print(f"\n场景 {i}: {scenario['name']}")
            print(f"  💨 湿度: {scenario['data']['weather_humidity_pct']:.1f}%")
            print(f"  🌡️  温度: {scenario['data']['weather_temperature_c']:.1f}°C")
            print(f"  💨 风速: {scenario['data']['wind_wind_speed_10m']:.1f}m/s")
            print(f"  🖼️  图像清晰度: {scenario['data']['laplacian_var']:.0f}")
            print(f"  👁️  预测能见度: {prediction:.0f} 米")
            
            # 简单的能见度等级评估
            if prediction >= 10000:
                level = "极佳 🌟"
            elif prediction >= 5000:
                level = "良好 ✅"
            elif prediction >= 1000:
                level = "一般 ⚠️"
            else:
                level = "较差 ❌"
            
            print(f"  📊 能见度等级: {level}")
            
        except Exception as e:
            print(f"❌ 场景 {i} 预测失败: {e}")

def show_feature_importance(analyzer) -> None:
    """显示特征重要性"""
    if analyzer is None or analyzer.best_model is None:
        return
    
    print_step(4, "特征重要性分析")
    
    try:
        # 获取特征系数
        coefficients = analyzer.best_model.coef_
        features = analyzer.selected_features
        
        # 创建重要性排序
        importance_data = list(zip(features, coefficients, np.abs(coefficients)))
        importance_data.sort(key=lambda x: x[2], reverse=True)
        
        print("📈 特征对能见度影响的重要性排序：")
        print()
        
        for i, (feature, coef, abs_coef) in enumerate(importance_data, 1):
            direction = "↗️ 正影响" if coef > 0 else "↘️ 负影响"
            
            # 特征名称映射
            feature_names = {
                'weather_temperature_c': '温度',
                'weather_humidity_pct': '湿度',
                'wind_wind_speed_10m': '风速',
                'laplacian_var': '图像清晰度',
                'high_freq_ratio': '高频比例',
                'edge_density': '边缘密度',
                'contrast_std': '对比度'
            }
            
            chinese_name = feature_names.get(feature, feature)
            
            print(f"  {i}. {chinese_name:<10} {direction:<8} (系数: {coef:8.4f})")
        
        print(f"\n💡 解释：")
        print(f"  • 正影响：该因素增加时，能见度倾向于提高")
        print(f"  • 负影响：该因素增加时，能见度倾向于降低")
        print(f"  • 系数绝对值越大，影响越显著")
        
    except Exception as e:
        print(f"❌ 特征重要性分析失败: {e}")

def cleanup_demo_files() -> None:
    """清理演示文件"""
    demo_files = ['demo_data.csv', 'sample_data.csv']
    
    for file in demo_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"🗑️  已清理临时文件: {file}")
            except:
                pass

def show_next_steps() -> None:
    """显示后续步骤建议"""
    print_step(5, "下一步操作建议")
    
    print("🎯 恭喜！您已完成能见度预测系统的快速体验。")
    print("\n📚 进一步探索：")
    print("  1. 📄 运行 'python example.py' - 查看详细的使用示例")
    print("  2. 🧪 运行 'python test_system.py' - 执行完整的系统测试")
    print("  3. 📊 将您的数据文件命名为 'complete_synced_data.csv' 并放在项目根目录")
    print("  4. 🔧 编辑 'config.py' 来自定义分析参数")
    print("  5. 📖 阅读 'README.md' 了解详细功能和使用方法")
    
    print("\n🛠️  自定义分析：")
    print("  • 修改特征选择策略")
    print("  • 调整模型参数")
    print("  • 添加新的特征工程")
    print("  • 集成到您的应用系统")
    
    print("\n🆘 需要帮助？")
    print("  • 查看代码注释和文档字符串")
    print("  • 检查 requirements.txt 确保依赖安装完整")
    print("  • 运行测试脚本诊断问题")

def main():
    """主函数"""
    print_header("🌟 能见度预测系统 - 快速开始")
    print("欢迎使用能见度预测分析系统！")
    print("本脚本将引导您快速体验系统的核心功能。")
    
    try:
        # 步骤1：创建演示数据
        demo_data = create_demo_data()
        
        # 步骤2：运行分析
        analyzer = run_quick_analysis()
        
        # 步骤3：预测演示
        demonstrate_prediction(analyzer)
        
        # 步骤4：特征重要性
        show_feature_importance(analyzer)
        
        # 步骤5：下一步建议
        show_next_steps()
        
        print_header("✨ 快速开始完成")
        print("🎉 您已成功体验了能见度预测系统的主要功能！")
        
        # 询问是否清理演示文件
        while True:
            try:
                choice = input("\n❓ 是否清理临时演示文件？(y/n): ").lower().strip()
                if choice in ['y', 'yes', '是']:
                    cleanup_demo_files()
                    break
                elif choice in ['n', 'no', '否']:
                    print("📁 临时文件已保留，您可以手动删除 demo_data.csv")
                    break
                else:
                    print("请输入 y 或 n")
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用！")
                break
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，感谢使用！")
    except Exception as e:
        print(f"\n❌ 快速开始过程中出现错误: {e}")
        print("请检查依赖安装或运行测试脚本诊断问题。")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main() 