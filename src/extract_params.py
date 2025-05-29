#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取最佳模型的具体数学表达式和参数
"""

from optimal_visibility_model import ContinuousVisibilityModel
import os

def main():
    print('🔍 提取最佳模型的具体数学表达式和参数...')
    
    # 创建模型实例
    model = ContinuousVisibilityModel()
    
    # 加载数据
    data_path = '../datasets/blur.csv'
    if not os.path.exists(data_path):
        print('❌ 数据文件不存在')
        return
    
    # 加载和预处理数据
    model.load_and_preprocess_data(data_path)
    
    # 只构建状态空间模型（最佳模型）
    print('\n构建状态空间模型...')
    state_space_result = model.build_state_space_model()
    
    if state_space_result and state_space_result.get('success', False):
        print('\n' + '='*80)
        print('📐 最佳模型详细数学表达式和参数')
        print('='*80)
        
        # 手动提取状态空间参数
        kf = state_space_result.get('kf')
        if kf:
            print(f"\n🎯 状态空间模型详细参数:")
            
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
            
            # 物理意义
            print(f"\n🌍 物理解释:")
            print(f"   • 能见度演化: V[k+1] = V[k] + V'[k] + 0.5×V''[k]")
            print(f"   • 变化率演化: V'[k+1] = V'[k] + V''[k]") 
            print(f"   • 加速度演化: V''[k+1] = V''[k] (随机游走)")
            print(f"   • 只观测能见度本身，变化率和加速度为隐状态")
            
            # 性能指标
            print(f"\n📊 模型性能:")
            print(f"   • 训练集: R² = {state_space_result.get('r2_train', 0):.6f}")
            print(f"   • 测试集: R² = {state_space_result.get('r2_test', 0):.6f}")
            print(f"   • 测试集MAE: {state_space_result.get('mae_test', 0):.2f}m")
            print(f"   • 测试集RMSE: {state_space_result.get('rmse_test', 0):.2f}m")
        
        print('\n' + '='*80)
        print('✅ 参数提取完成！这就是您要的带有具体权重的数学表达式')
        print('='*80)
    else:
        print('❌ 状态空间模型构建失败')

if __name__ == "__main__":
    main() 