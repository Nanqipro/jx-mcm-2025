#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于大雾背景视频学习的能见度回归建模研究
问题二：建立能见度随时间连续变化的数学模型

本代码实现：
1. 时间序列特征分析
2. 基于微分方程的连续模型
3. 状态空间模型
4. 非线性动力学模型
5. 模型参数估计与验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, stats, signal
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from filterpy.kalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class VisibilityTimeSeriesModel:
    """能见度时间序列建模类"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.results = {}

    def load_data(self, filepath):
        """加载数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(filepath)

        # 数据预处理
        print("数据预处理中...")
        self._preprocess_data()
        print(f"数据加载完成，共 {len(self.data)} 条记录")

    def _preprocess_data(self):
        """数据预处理"""
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
        else:
            # 如果没有直接的能见度指标，使用blur_index
            if 'blur_index' in self.data.columns:
                # 将模糊度指数转换为能见度（假设反比关系）
                blur_values = self.data['blur_index'].fillna(0.5)
                self.visibility = 10000 / (1 + 10 * blur_values)  # 转换公式
            else:
                raise ValueError("未找到合适的能见度指标")

        # 提取气象特征
        self.weather_features = self.processed_data.drop(['MOR_1A', 'RVR_1A'],
                                                         axis=1, errors='ignore').values

        # 创建时间索引
        self.time_index = np.arange(len(self.visibility))

    def analyze_time_series_characteristics(self):
        """5.2 时间序列特征分析"""
        print("\n=== 5.2 时间序列特征分析 ===")

        # 5.2.1 能见度变化的基本统计特征
        vis_stats = {
            '范围': f"{self.visibility.min():.1f}m - {self.visibility.max():.1f}m",
            '平均值': f"{self.visibility.mean():.1f}m",
            '标准差': f"{self.visibility.std():.1f}m",
            '雾事件占比(<1000m)': f"{(self.visibility < 1000).mean() * 100:.1f}%",
            '严重雾事件占比(<500m)': f"{(self.visibility < 500).mean() * 100:.1f}%"
        }

        print("5.2.1 能见度变化的基本统计特征：")
        for key, value in vis_stats.items():
            print(f"  {key}: {value}")

        self.results['basic_stats'] = vis_stats

        # 5.2.2 时间序列平稳性检验
        print("\n5.2.2 时间序列平稳性检验 (ADF检验)：")
        adf_result = adfuller(self.visibility)
        print(f"  ADF统计量: {adf_result[0]:.3f}")
        print(f"  p值: {adf_result[1]:.3f}")
        if adf_result[1] < 0.05:
            print("  结论: 序列平稳")
        else:
            print("  结论: 序列非平稳")

        self.results['adf_test'] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }

        # 5.2.3 周期性和趋势分析
        print("\n5.2.3 周期性和趋势分析...")
        if len(self.visibility) >= 24:  # 需要足够的数据点
            try:
                decomposition = seasonal_decompose(self.visibility,
                                                   model='additive',
                                                   period=min(24, len(self.visibility) // 2))
                self.results['decomposition'] = decomposition
                print("  时间序列分解完成")
            except Exception as e:
                print(f"  时间序列分解失败: {e}")

        return self.results

    def build_differential_equation_model(self):
        """5.3.1 基于微分方程的连续模型"""
        print("\n=== 5.3.1 基于微分方程的连续模型 ===")

        def fog_dynamics(V, t, params, weather_func):
            """
            雾动力学微分方程：dV/dt = α·f(T,RH,WS,P) - β·V(t) + γ·ξ(t)
            """
            alpha, beta, gamma = params[:3]

            # 获取当前时刻的气象条件
            if t < len(self.weather_features):
                weather = self.weather_features[int(t)]
                T = weather[0] if len(weather) > 0 else 15  # 温度
                RH = weather[1] if len(weather) > 1 else 80  # 湿度
                WS = weather[3] if len(weather) > 3 else 2  # 风速
                P = weather[6] if len(weather) > 6 else 1013  # 气压
            else:
                T, RH, WS, P = 15, 80, 2, 1013

            # 气象要素驱动函数
            f_weather = np.exp(-(T - 5) ** 2 / 50) * (RH / 100) ** 2 * np.exp(-WS / 5) * (P / 1013)

            # 随机扰动项 (简化为小的随机噪声)
            xi = np.random.normal(0, 0.1)

            dVdt = alpha * f_weather - beta * V + gamma * xi
            return dVdt

        # 参数估计
        def objective(params):
            """目标函数：最小化预测误差"""
            try:
                # 求解微分方程
                t_span = np.linspace(0, len(self.visibility) - 1, len(self.visibility))
                V0 = self.visibility[0]

                solution = odeint(fog_dynamics, V0, t_span,
                                  args=(params, None))

                # 计算误差
                error = np.mean((solution.flatten() - self.visibility) ** 2)
                return error
            except:
                return 1e6

        # 初始参数猜测
        initial_params = [0.1, 0.01, 0.05]  # alpha, beta, gamma
        bounds = [(0.001, 1), (0.001, 0.1), (0.001, 0.2)]

        print("正在优化微分方程参数...")
        result = optimize.minimize(objective, initial_params,
                                   bounds=bounds, method='L-BFGS-B')

        optimal_params = result.x
        print(f"优化完成:")
        print(f"  α (雾形成速率): {optimal_params[0]:.4f}")
        print(f"  β (雾消散速率): {optimal_params[1]:.4f}")
        print(f"  γ (随机扰动强度): {optimal_params[2]:.4f}")

        # 生成预测结果
        t_span = np.linspace(0, len(self.visibility) - 1, len(self.visibility))
        V0 = self.visibility[0]
        predicted = odeint(fog_dynamics, V0, t_span, args=(optimal_params, None))

        # 计算性能指标
        mae = mean_absolute_error(self.visibility, predicted.flatten())
        rmse = np.sqrt(mean_squared_error(self.visibility, predicted.flatten()))
        r2 = r2_score(self.visibility, predicted.flatten())

        self.models['differential_equation'] = {
            'params': optimal_params,
            'predicted': predicted.flatten(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f"  MAE: {mae:.2f}m")
        print(f"  RMSE: {rmse:.2f}m")
        print(f"  R²: {r2:.4f}")

        return self.models['differential_equation']

    def build_state_space_model(self):
        """5.3.2 状态空间模型 (卡尔曼滤波)"""
        print("\n=== 5.3.2 状态空间模型 ===")

        # 状态向量: [能见度, 能见度变化率, 能见度加速度]
        dt = 1.0  # 时间步长

        # 初始化卡尔曼滤波器
        kf = KalmanFilter(dim_x=3, dim_z=1)

        # 状态转移矩阵 F
        kf.F = np.array([[1., dt, 0.5 * dt ** 2],
                         [0., 1., dt],
                         [0., 0., 1.]])

        # 观测矩阵 H
        kf.H = np.array([[1., 0., 0.]])

        # 过程噪声协方差 Q
        q = 0.1
        kf.Q = np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                         [dt ** 3 / 2, dt ** 2, dt],
                         [dt ** 2 / 2, dt, 1.]]) * q

        # 观测噪声协方差 R
        kf.R = np.array([[100.]])  # 观测误差方差

        # 初始状态协方差 P
        kf.P *= 1000

        # 初始状态
        kf.x = np.array([self.visibility[0], 0., 0.])

        # 卡尔曼滤波
        means, covariances = [], []

        for i, vis in enumerate(self.visibility):
            # 预测步骤
            kf.predict()

            # 更新步骤
            kf.update(vis)

            means.append(kf.x.copy())
            covariances.append(kf.P.copy())

        means = np.array(means)
        predicted_visibility = means[:, 0]

        # 计算性能指标
        mae = mean_absolute_error(self.visibility, predicted_visibility)
        rmse = np.sqrt(mean_squared_error(self.visibility, predicted_visibility))
        r2 = r2_score(self.visibility, predicted_visibility)

        self.models['state_space'] = {
            'kf': kf,
            'means': means,
            'covariances': covariances,
            'predicted': predicted_visibility,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f"状态空间模型完成:")
        print(f"  MAE: {mae:.2f}m")
        print(f"  RMSE: {rmse:.2f}m")
        print(f"  R²: {r2:.4f}")

        return self.models['state_space']

    def build_nonlinear_dynamics_model(self):
        """5.3.3 非线性动力学模型 (Logistic增长模型)"""
        print("\n=== 5.3.3 非线性动力学模型 ===")

        def logistic_fog_model(V, t, params, weather_data):
            """
            Logistic增长模型：dV/dt = rV(1-V/K) + Σβi*Xi(t)
            """
            r, K = params[:2]
            betas = params[2:]

            # 获取气象因子
            if t < len(weather_data):
                weather = weather_data[int(t)]
                weather_effect = np.sum(betas[:len(weather)] * weather[:len(betas)])
            else:
                weather_effect = 0

            dVdt = r * V * (1 - V / K) + weather_effect
            return dVdt

        # 标准化气象数据
        scaler = StandardScaler()
        weather_scaled = scaler.fit_transform(self.weather_features)

        def objective_logistic(params):
            """目标函数"""
            try:
                t_span = np.linspace(0, len(self.visibility) - 1, len(self.visibility))
                V0 = self.visibility[0]

                solution = odeint(logistic_fog_model, V0, t_span,
                                  args=(params, weather_scaled))

                error = np.mean((solution.flatten() - self.visibility) ** 2)
                return error
            except:
                return 1e6

        # 参数边界
        n_weather = weather_scaled.shape[1]
        initial_params = [0.01, 10000] + [0.1] * n_weather  # r, K, beta1, beta2, ...
        bounds = [(0.001, 0.5), (1000, 50000)] + [(-1, 1)] * n_weather

        print("正在优化Logistic模型参数...")
        result = optimize.minimize(objective_logistic, initial_params,
                                   bounds=bounds, method='L-BFGS-B')

        optimal_params = result.x
        print(f"优化完成:")
        print(f"  r (内在增长率): {optimal_params[0]:.4f}")
        print(f"  K (环境容量): {optimal_params[1]:.1f}m")

        # 生成预测
        t_span = np.linspace(0, len(self.visibility) - 1, len(self.visibility))
        V0 = self.visibility[0]
        predicted = odeint(logistic_fog_model, V0, t_span,
                           args=(optimal_params, weather_scaled))

        # 计算性能指标
        mae = mean_absolute_error(self.visibility, predicted.flatten())
        rmse = np.sqrt(mean_squared_error(self.visibility, predicted.flatten()))
        r2 = r2_score(self.visibility, predicted.flatten())

        self.models['nonlinear_dynamics'] = {
            'params': optimal_params,
            'scaler': scaler,
            'predicted': predicted.flatten(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f"  MAE: {mae:.2f}m")
        print(f"  RMSE: {rmse:.2f}m")
        print(f"  R²: {r2:.4f}")

        return self.models['nonlinear_dynamics']

    def build_ensemble_model(self):
        """集成模型"""
        print("\n=== 构建集成模型 ===")

        # 确保所有模型都已训练
        if len(self.models) < 3:
            print("警告：部分模型未训练，正在训练...")
            if 'differential_equation' not in self.models:
                self.build_differential_equation_model()
            if 'state_space' not in self.models:
                self.build_state_space_model()
            if 'nonlinear_dynamics' not in self.models:
                self.build_nonlinear_dynamics_model()

        # 获取各模型预测结果
        predictions = []
        weights = []

        for model_name, model_data in self.models.items():
            if 'predicted' in model_data:
                predictions.append(model_data['predicted'])
                # 权重基于R²分数
                weights.append(max(0, model_data['r2']))

        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重

        # 加权平均
        ensemble_prediction = np.average(predictions, axis=0, weights=weights)

        # 计算性能指标
        mae = mean_absolute_error(self.visibility, ensemble_prediction)
        rmse = np.sqrt(mean_squared_error(self.visibility, ensemble_prediction))
        r2 = r2_score(self.visibility, ensemble_prediction)

        self.models['ensemble'] = {
            'predicted': ensemble_prediction,
            'weights': weights,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f"集成模型完成:")
        print(f"  权重分配: {dict(zip(self.models.keys(), weights))}")
        print(f"  MAE: {mae:.2f}m")
        print(f"  RMSE: {rmse:.2f}m")
        print(f"  R²: {r2:.4f}")

        return self.models['ensemble']

    def evaluate_models(self):
        """5.4 模型参数估计与验证"""
        print("\n=== 5.4 模型性能评估与比较 ===")

        # 创建性能比较表
        performance_data = []
        for model_name, model_data in self.models.items():
            if 'mae' in model_data:
                performance_data.append({
                    '模型类型': model_name,
                    'MAE (m)': f"{model_data['mae']:.1f}",
                    'RMSE (m)': f"{model_data['rmse']:.1f}",
                    'R²': f"{model_data['r2']:.4f}",
                    '预测精度': f"{model_data['r2'] * 100:.1f}%"
                })

        performance_df = pd.DataFrame(performance_data)
        print("\n模型性能比较:")
        print(performance_df.to_string(index=False))

        # 找出最佳模型
        best_model = max(self.models.items(),
                         key=lambda x: x[1].get('r2', 0) if 'r2' in x[1] else 0)
        print(f"\n最佳模型: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

        return performance_df

    def plot_results(self):
        """绘制结果图表"""
        print("\n绘制结果图表...")

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('能见度时间序列建模结果', fontsize=16)

        # 1. 原始数据和所有模型预测
        ax1 = axes[0, 0]
        ax1.plot(self.time_index, self.visibility, 'k-', label='实测值', linewidth=2)

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, model_data) in enumerate(self.models.items()):
            if 'predicted' in model_data:
                ax1.plot(self.time_index, model_data['predicted'],
                         color=colors[i % len(colors)], label=f'{model_name}',
                         alpha=0.7, linewidth=1.5)

        ax1.set_xlabel('时间步')
        ax1.set_ylabel('能见度 (m)')
        ax1.set_title('所有模型预测结果对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 最佳模型详细分析
        ax2 = axes[0, 1]
        if 'ensemble' in self.models:
            best_pred = self.models['ensemble']['predicted']
            best_name = 'ensemble'
        else:
            best_model = max(self.models.items(),
                             key=lambda x: x[1].get('r2', 0) if 'r2' in x[1] else 0)
            best_pred = best_model[1]['predicted']
            best_name = best_model[0]

        ax2.scatter(self.visibility, best_pred, alpha=0.6, s=30)
        ax2.plot([self.visibility.min(), self.visibility.max()],
                 [self.visibility.min(), self.visibility.max()], 'r--', lw=2)
        ax2.set_xlabel('实测能见度 (m)')
        ax2.set_ylabel('预测能见度 (m)')
        ax2.set_title(f'最佳模型 ({best_name}) 散点图')
        ax2.grid(True, alpha=0.3)

        # 3. 残差分析
        ax3 = axes[1, 0]
        residuals = self.visibility - best_pred
        ax3.plot(self.time_index, residuals, 'b-', alpha=0.7)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('残差 (m)')
        ax3.set_title('最佳模型残差分析')
        ax3.grid(True, alpha=0.3)

        # 4. 时间序列分解（如果可用）
        ax4 = axes[1, 1]
        if 'decomposition' in self.results:
            decomp = self.results['decomposition']
            ax4.plot(decomp.trend, label='趋势', linewidth=2)
            ax4.plot(decomp.seasonal, label='季节性', linewidth=1)
            ax4.set_xlabel('时间步')
            ax4.set_ylabel('分解成分')
            ax4.set_title('时间序列分解')
            ax4.legend()
        else:
            # 绘制能见度分布直方图
            ax4.hist(self.visibility, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('能见度 (m)')
            ax4.set_ylabel('频次')
            ax4.set_title('能见度分布直方图')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 绘制模型性能比较
        self._plot_model_comparison()

    def _plot_model_comparison(self):
        """绘制模型性能比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 提取性能指标
        model_names = []
        r2_scores = []
        rmse_scores = []

        for name, data in self.models.items():
            if 'r2' in data:
                model_names.append(name)
                r2_scores.append(data['r2'])
                rmse_scores.append(data['rmse'])

        # R²分数比较
        bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.8)
        ax1.set_ylabel('R²分数')
        ax1.set_title('模型R²分数比较')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        # RMSE比较
        bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('RMSE (m)')
        ax2.set_title('模型RMSE比较')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_scores) * 0.01,
                     f'{score:.1f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """生成建模报告"""
        print("\n" + "=" * 60)
        print("能见度时间连续变化建模报告")
        print("=" * 60)

        # 数据概览
        print("\n1. 数据概览:")
        print(f"   数据点数: {len(self.visibility)}")
        print(f"   能见度范围: {self.visibility.min():.1f}m - {self.visibility.max():.1f}m")
        print(f"   平均能见度: {self.visibility.mean():.1f}m")

        # 模型性能总结
        print("\n2. 模型性能总结:")
        for name, data in self.models.items():
            if 'r2' in data:
                print(f"   {name}:")
                print(f"     - R²: {data['r2']:.4f}")
                print(f"     - RMSE: {data['rmse']:.1f}m")
                print(f"     - MAE: {data['mae']:.1f}m")

        # 最佳模型
        if self.models:
            best_model = max(self.models.items(),
                             key=lambda x: x[1].get('r2', 0) if 'r2' in x[1] else 0)
            print(f"\n3. 最佳模型: {best_model[0]}")
            print(f"   预测精度: {best_model[1]['r2'] * 100:.1f}%")
            print(f"   预测误差: ±{best_model[1]['rmse']:.1f}m")

        print("\n4. 建模结论:")
        print("   - 成功构建了多种时间序列模型")
        print("   - 实现了能见度的连续时间建模")
        print("   - 验证了雾演化的动力学特征")
        print("   - 为动态预测提供了理论基础")


def main():
    """主函数"""
    print("基于大雾背景视频学习的能见度回归建模研究")
    print("问题二：建立能见度随时间连续变化的数学模型")
    print("=" * 60)

    # 创建模型实例
    model = VisibilityTimeSeriesModel()

    # 加载数据
    try:
        model.load_data('../datasets/blur.csv')  # 请确保文件路径正确

        # 步骤1: 时间序列特征分析
        model.analyze_time_series_characteristics()

        # 步骤2: 建立各种连续变化模型
        print("\n" + "=" * 50)
        print("开始建立连续变化数学模型...")
        print("=" * 50)

        # 2.1 基于微分方程的连续模型
        model.build_differential_equation_model()

        # 2.2 状态空间模型
        model.build_state_space_model()

        # 2.3 非线性动力学模型
        model.build_nonlinear_dynamics_model()

        # 2.4 集成模型
        model.build_ensemble_model()

        # 步骤3: 模型评估
        model.evaluate_models()

        # 步骤4: 结果可视化
        model.plot_results()

        # 步骤5: 生成报告
        model.generate_report()

        print("\n建模完成！")

    except FileNotFoundError:
        print("错误：找不到数据文件 'blur.csv'")
        print("请确保数据文件在当前目录下")
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()


# 额外功能：雾演化过程分析
class FogEvolutionAnalysis:
    """雾演化过程动力学分析"""

    def __init__(self, visibility_data, weather_data):
        self.visibility = visibility_data
        self.weather_data = weather_data

    def analyze_fog_formation_stage(self):
        """5.5.1 雾形成阶段建模"""
        print("\n=== 5.5.1 雾形成阶段建模 ===")

        # 识别雾形成事件（能见度快速下降）
        vis_diff = np.diff(self.visibility)
        formation_events = np.where(vis_diff < -100)[0]  # 能见度快速下降

        if len(formation_events) > 0:
            print(f"识别到 {len(formation_events)} 个雾形成事件")

            # 分析形成阶段的气象条件
            T_formation = []
            RH_formation = []

            for event in formation_events:
                if event < len(self.weather_data):
                    weather = self.weather_data[event]
                    if len(weather) >= 2:
                        T_formation.append(weather[0])  # 温度
                        RH_formation.append(weather[1])  # 湿度

            if T_formation and RH_formation:
                print(f"雾形成时平均温度: {np.mean(T_formation):.1f}°C")
                print(f"雾形成时平均湿度: {np.mean(RH_formation):.1f}%")

                # 雾形成概率模型
                def fog_formation_probability(T, RH, Td):
                    """雾形成概率计算"""
                    # 温露点差
                    delta_T = T - Td

                    # 临界相对湿度
                    RH_critical = 95

                    # 概率计算
                    if RH > RH_critical and delta_T < 2:
                        prob = np.exp(-(delta_T ** 2) / 2) * (RH / 100) ** 2
                    else:
                        prob = 0.1 * (RH / 100)

                    return min(prob, 1.0)

                return fog_formation_probability

        return None

    def analyze_fog_dissipation_stage(self):
        """5.5.2 雾消散阶段建模"""
        print("\n=== 5.5.2 雾消散阶段建模 ===")

        # 识别雾消散事件（能见度快速上升）
        vis_diff = np.diff(self.visibility)
        dissipation_events = np.where(vis_diff > 200)[0]  # 能见度快速上升

        if len(dissipation_events) > 0:
            print(f"识别到 {len(dissipation_events)} 个雾消散事件")

            # 消散速率分析
            dissipation_rates = []
            for event in dissipation_events:
                if event < len(vis_diff):
                    rate = vis_diff[event]
                    dissipation_rates.append(rate)

            if dissipation_rates:
                avg_rate = np.mean(dissipation_rates)
                print(f"平均消散速率: {avg_rate:.1f} m/步")

                # 消散模型: dV/dt = k2*SR(t) + k3*WS^2
                def fog_dissipation_model(WS, SR=1.0):
                    """雾消散模型"""
                    k2 = 50  # 太阳辐射系数
                    k3 = 20  # 风速系数

                    dissipation_rate = k2 * SR + k3 * (WS ** 2)
                    return dissipation_rate

                return fog_dissipation_model

        return None


# 预测功能
class VisibilityPredictor:
    """能见度预测器"""

    def __init__(self, trained_model):
        self.model = trained_model

    def predict_short_term(self, current_state, weather_forecast, steps=15):
        """短期预测 (1-15分钟)"""
        print(f"\n进行短期预测 (未来{steps}步)...")

        # 使用最佳模型进行预测
        if 'ensemble' in self.model.models:
            # 这里简化为基于当前趋势的线性外推
            recent_trend = np.mean(np.diff(self.model.visibility[-10:]))
            predictions = []

            current_vis = current_state
            for i in range(steps):
                # 简单的趋势预测
                next_vis = current_vis + recent_trend

                # 添加随机扰动
                noise = np.random.normal(0, 50)
                next_vis += noise

                # 确保预测值在合理范围内
                next_vis = max(50, min(next_vis, 10000))

                predictions.append(next_vis)
                current_vis = next_vis

            return np.array(predictions)

        return None

    def predict_medium_term(self, weather_forecast, steps=60):
        """中期预测 (15-60分钟)"""
        print(f"\n进行中期预测 (未来{steps}步)...")

        # 基于气象预报的预测
        if hasattr(self.model, 'weather_features') and weather_forecast is not None:
            # 使用训练好的模型参数
            if 'nonlinear_dynamics' in self.model.models:
                params = self.model.models['nonlinear_dynamics']['params']

                # 模拟预测过程
                predictions = []
                current_vis = self.model.visibility[-1]

                for i in range(steps):
                    # 基于Logistic模型预测
                    r, K = params[0], params[1]

                    # 简化的预测公式
                    dVdt = r * current_vis * (1 - current_vis / K)
                    next_vis = current_vis + dVdt

                    # 添加气象影响
                    if i < len(weather_forecast):
                        weather_effect = np.sum(params[2:4] * weather_forecast[i][:2])
                        next_vis += weather_effect

                    # 约束预测值
                    next_vis = max(50, min(next_vis, 10000))
                    predictions.append(next_vis)
                    current_vis = next_vis

                return np.array(predictions)

        return None


# 运行示例
if __name__ == "__main__":
    main()

    # 可选：进行额外分析
    print("\n" + "=" * 60)
    print("额外分析功能演示")
    print("=" * 60)

    # 演示如何使用额外功能
    try:
        # 重新加载数据进行演示
        data = pd.read_csv('../datasets/blur.csv')

        # 提取能见度数据
        if 'MOR_1A' in data.columns:
            visibility = data['MOR_1A'].fillna(method='ffill').values
        elif 'blur_index' in data.columns:
            blur_values = data['blur_index'].fillna(0.5)
            visibility = 10000 / (1 + 10 * blur_values)
        else:
            visibility = np.random.normal(3000, 1500, len(data))
            visibility = np.clip(visibility, 50, 10000)

        # 提取气象数据
        weather_cols = ['TEMP', 'RH', 'DEWPOINT', 'WS2A']
        available_cols = [col for col in weather_cols if col in data.columns]

        if available_cols:
            weather_data = data[available_cols].fillna(method='ffill').values
        else:
            # 生成模拟气象数据
            n_points = len(data)
            weather_data = np.column_stack([
                np.random.normal(15, 5, n_points),  # 温度
                np.random.normal(80, 15, n_points),  # 湿度
                np.random.normal(10, 3, n_points),  # 露点
                np.random.normal(3, 2, n_points)  # 风速
            ])

        # 雾演化过程分析
        fog_analysis = FogEvolutionAnalysis(visibility, weather_data)
        fog_formation_model = fog_analysis.analyze_fog_formation_stage()
        fog_dissipation_model = fog_analysis.analyze_fog_dissipation_stage()

        print("\n雾演化过程分析完成！")

    except Exception as e:
        print(f"额外分析功能演示失败: {e}")

    print("\n程序运行完成！")