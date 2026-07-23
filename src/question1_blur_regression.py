import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SimpleBlurPredictor:
    """简化版blur_index预测器 - 专注于折线图对比和多项式输出"""

    def __init__(self, degree=2, alpha=1.0):
        self.degree = degree
        self.alpha = alpha
        self.target_name = 'blur_index'
        self.selected_features = None
        self.pipeline = None

    def load_and_prepare_data(self, data_source='sample'):
        """加载和准备数据"""
        if data_source == 'sample':
            # 生成示例数据
            np.random.seed(42)
            n_samples = 694

            self.data = pd.DataFrame({
                'laplacian_var': np.random.normal(500, 200, n_samples),
                'sobel_mean': np.random.normal(50, 20, n_samples),
                'brenner': np.random.normal(1000, 300, n_samples),
                'std_contrast': np.random.normal(0.3, 0.1, n_samples),
                'DEWPOINT': np.random.normal(15, 10, n_samples),
                'VIS1K': np.random.normal(8, 3, n_samples),
                'LIGHTS': np.random.normal(100, 50, n_samples),
                'RH': np.random.normal(60, 20, n_samples),
            })

            # 生成目标变量
            self.data['blur_index'] = (
                0.8 - 0.0001 * self.data['laplacian_var']
                - 0.002 * self.data['sobel_mean']
                - 0.00005 * self.data['brenner']
                - 0.1 * self.data['std_contrast']
                - 0.01 * self.data['DEWPOINT']
                - 0.02 * self.data['VIS1K']
                + np.random.normal(0, 0.05, n_samples)
            ).clip(0.042, 0.971)

        else:
            # 加载真实数据
            self.data = pd.read_csv(data_source)

        print(f"✅ 数据加载成功: {self.data.shape[0]}行, {self.data.shape[1]}列")
        return self.data

    def select_top_features(self, n_features=6):
        """选择最重要的特征"""
        # 计算相关性
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = self.data[numeric_cols].corr()[self.target_name].drop(self.target_name)
        correlations_abs = correlations.abs().sort_values(ascending=False)

        # 选择前n个特征
        self.selected_features = correlations_abs.head(n_features).index.tolist()

        print(f"\n📊 选择的{len(self.selected_features)}个最重要特征:")
        for i, feature in enumerate(self.selected_features, 1):
            corr = correlations[feature]
            print(f"  {i}. {feature:<20} | 相关系数: {corr:7.4f}")

        return self.selected_features

    def train_polynomial_model(self):
        """训练多项式回归模型"""
        # 准备数据
        X = self.data[self.selected_features]
        y = self.data[self.target_name]

        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 构建多项式回归管道
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('model', Ridge(alpha=self.alpha))
        ])

        # 训练模型
        self.pipeline.fit(self.X_train, self.y_train)

        # 预测
        self.y_train_pred = self.pipeline.predict(self.X_train)
        self.y_test_pred = self.pipeline.predict(self.X_test)

        # 计算评估指标
        self.train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.test_r2 = r2_score(self.y_test, self.y_test_pred)
        self.train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        self.test_mae = mean_absolute_error(self.y_test, self.y_test_pred)

        print(f"\n🎯 模型训练完成:")
        print(f"  训练集 R²: {self.train_r2:.6f}")
        print(f"  测试集 R²: {self.test_r2:.6f}")
        print(f"  训练集 MAE: {self.train_mae:.6f}")
        print(f"  测试集 MAE: {self.test_mae:.6f}")

        return self.pipeline

    def plot_line_comparison(self):
        """🔥 核心功能1: 绘制预测值vs实际值的折线图对比"""
        print("\n" + "="*60)
        print("📈 绘制预测值 vs 实际值折线图对比")
        print("="*60)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('🔍 预测值 vs 实际值 - 详细折线图对比分析', fontsize=18, fontweight='bold')

        # === 图1: 训练集对比 (前50个样本) ===
        n_display = min(50, len(self.y_train))
        train_indices = np.arange(n_display)
        train_actual = self.y_train.iloc[:n_display]
        train_pred = self.y_train_pred[:n_display]

        axes[0, 0].plot(train_indices, train_actual, 'o-',
                       label='实际值', color='blue', alpha=0.8, linewidth=2, markersize=5)
        axes[0, 0].plot(train_indices, train_pred, 's-',
                       label='预测值', color='red', alpha=0.8, linewidth=2, markersize=5)
        axes[0, 0].fill_between(train_indices, train_actual, train_pred,
                               alpha=0.3, color='gray', label='误差区域')

        # 添加误差线
        for i in range(n_display):
            axes[0, 0].plot([i, i], [train_actual.iloc[i], train_pred[i]],
                           'k-', alpha=0.4, linewidth=1)

        axes[0, 0].set_xlabel('样本索引')
        axes[0, 0].set_ylabel('blur_index值')
        axes[0, 0].set_title(f'训练集预测对比 (前{n_display}样本)\nR² = {self.train_r2:.4f}, MAE = {self.train_mae:.4f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # === 图2: 测试集对比 (前50个样本) ===
        n_display = min(50, len(self.y_test))
        test_indices = np.arange(n_display)
        test_actual = self.y_test.iloc[:n_display]
        test_pred = self.y_test_pred[:n_display]

        axes[0, 1].plot(test_indices, test_actual, 'o-',
                       label='实际值', color='green', alpha=0.8, linewidth=2, markersize=5)
        axes[0, 1].plot(test_indices, test_pred, 's-',
                       label='预测值', color='orange', alpha=0.8, linewidth=2, markersize=5)
        axes[0, 1].fill_between(test_indices, test_actual, test_pred,
                               alpha=0.3, color='gray', label='误差区域')

        # 添加误差线
        for i in range(n_display):
            axes[0, 1].plot([i, i], [test_actual.iloc[i], test_pred[i]],
                           'k-', alpha=0.4, linewidth=1)

        axes[0, 1].set_xlabel('样本索引')
        axes[0, 1].set_ylabel('blur_index值')
        axes[0, 1].set_title(f'测试集预测对比 (前{n_display}样本)\nR² = {self.test_r2:.4f}, MAE = {self.test_mae:.4f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # === 图3: 全数据集对比 (智能采样) ===
        total_samples = len(self.y_train) + len(self.y_test)
        sample_step = max(1, total_samples // 150)  # 智能采样

        # 合并所有数据
        all_actual = np.concatenate([self.y_train, self.y_test])
        all_pred = np.concatenate([self.y_train_pred, self.y_test_pred])
        all_indices = np.arange(len(all_actual))

        # 采样
        sampled_indices = all_indices[::sample_step]
        sampled_actual = all_actual[sampled_indices]
        sampled_pred = all_pred[sampled_indices]

        # 区分训练集和测试集
        train_boundary = len(self.y_train)
        train_mask = sampled_indices < train_boundary
        test_mask = sampled_indices >= train_boundary

        # 绘制训练集部分
        if np.any(train_mask):
            axes[1, 0].plot(sampled_indices[train_mask], sampled_actual[train_mask],
                           'o-', label='训练集实际', color='blue', alpha=0.7, markersize=3)
            axes[1, 0].plot(sampled_indices[train_mask], sampled_pred[train_mask],
                           's-', label='训练集预测', color='lightblue', alpha=0.7, markersize=3)

        # 绘制测试集部分
        if np.any(test_mask):
            axes[1, 0].plot(sampled_indices[test_mask], sampled_actual[test_mask],
                           'o-', label='测试集实际', color='green', alpha=0.7, markersize=3)
            axes[1, 0].plot(sampled_indices[test_mask], sampled_pred[test_mask],
                           's-', label='测试集预测', color='orange', alpha=0.7, markersize=3)

        # 添加分界线
        axes[1, 0].axvline(x=train_boundary, color='red', linestyle='--',
                          alpha=0.8, linewidth=2, label=f'训练/测试分界')

        axes[1, 0].set_xlabel('样本索引')
        axes[1, 0].set_ylabel('blur_index值')
        axes[1, 0].set_title(f'全数据集预测对比 (采样步长={sample_step})\n总样本: {total_samples}, 显示: {len(sampled_indices)}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # === 图4: 误差分析 ===
        train_errors = np.abs(self.y_train - self.y_train_pred)
        test_errors = np.abs(self.y_test - self.y_test_pred)

        # 误差分布
        bins = np.linspace(0, max(train_errors.max(), test_errors.max()), 30)
        axes[1, 1].hist(train_errors, bins=bins, alpha=0.6, label='训练集误差',
                       color='blue', density=True)
        axes[1, 1].hist(test_errors, bins=bins, alpha=0.6, label='测试集误差',
                       color='orange', density=True)

        # 添加均值线
        axes[1, 1].axvline(train_errors.mean(), color='blue', linestyle='--',
                          label=f'训练集平均误差: {train_errors.mean():.4f}')
        axes[1, 1].axvline(test_errors.mean(), color='orange', linestyle='--',
                          label=f'测试集平均误差: {test_errors.mean():.4f}')

        axes[1, 1].set_xlabel('绝对误差')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].set_title('预测误差分布分析')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def output_polynomial_formula(self):
        """🔥 核心功能2: 输出具体的多项式公式"""
        print("\n" + "="*80)
        print("🔍 具体多项式回归公式详细输出")
        print("="*80)

        # 获取多项式特征名称和系数
        poly_features = self.pipeline.named_steps['poly']
        feature_names = poly_features.get_feature_names_out(self.selected_features)
        coefficients = self.pipeline.named_steps['model'].coef_
        intercept = self.pipeline.named_steps['model'].intercept_

        # 分类项
        linear_terms = []
        quadratic_terms = []
        interaction_terms = []

        for name, coef in zip(feature_names, coefficients):
            if abs(coef) > 1e-8:  # 只保留有意义的系数
                if '^2' in name:
                    quadratic_terms.append((name, coef))
                elif ' ' in name:
                    interaction_terms.append((name, coef))
                else:
                    linear_terms.append((name, coef))

        # 输出模型结构
        print(f"📊 多项式模型结构分析:")
        print(f"   🔹 多项式次数: {self.degree}")
        print(f"   🔹 输入特征数: {len(self.selected_features)}")
        print(f"   🔹 截距项: 1个")
        print(f"   🔹 一次项: {len(linear_terms)}个")
        print(f"   🔹 二次项: {len(quadratic_terms)}个")
        print(f"   🔹 交互项: {len(interaction_terms)}个")
        print(f"   🔹 总参数数: {1 + len(linear_terms) + len(quadratic_terms) + len(interaction_terms)}个")

        # 构建完整公式
        print(f"\n📝 详细数学公式:")
        print("-" * 70)

        formula_parts = [f"{intercept:.8f}"]

        # 一次项
        if linear_terms:
            print(f"\n🔸 一次项 (线性效应):")
            for name, coef in sorted(linear_terms, key=lambda x: abs(x[1]), reverse=True):
                sign = "+" if coef >= 0 else "-"
                print(f"   {sign} {abs(coef):.8f} × {name}")
                if coef >= 0:
                    formula_parts.append(f"+ {coef:.8f}×{name}")
                else:
                    formula_parts.append(f"- {abs(coef):.8f}×{name}")

        # 二次项
        if quadratic_terms:
            print(f"\n🔸 二次项 (非线性效应):")
            for name, coef in sorted(quadratic_terms, key=lambda x: abs(x[1]), reverse=True):
                sign = "+" if coef >= 0 else "-"
                print(f"   {sign} {abs(coef):.8f} × {name}")
                if coef >= 0:
                    formula_parts.append(f"+ {coef:.8f}×{name}")
                else:
                    formula_parts.append(f"- {abs(coef):.8f}×{name}")

        # 交互项
        if interaction_terms:
            print(f"\n🔸 交互项 (特征间相互作用):")
            for name, coef in sorted(interaction_terms, key=lambda x: abs(x[1]), reverse=True):
                sign = "+" if coef >= 0 else "-"
                print(f"   {sign} {abs(coef):.8f} × {name}")
                if coef >= 0:
                    formula_parts.append(f"+ {coef:.8f}×{name}")
                else:
                    formula_parts.append(f"- {abs(coef):.8f}×{name}")

        # 完整公式
        full_formula = f"blur_index = {' '.join(formula_parts)}"
        print(f"\n📋 完整多项式公式:")
        print("="*80)
        print(full_formula)
        print("="*80)

        # 生成Python预测函数
        self.generate_prediction_function(intercept, coefficients, feature_names)

        # 输出特征重要性分析
        self.analyze_feature_importance(linear_terms, quadratic_terms, interaction_terms)

        return full_formula

    def generate_prediction_function(self, intercept, coefficients, feature_names):
        """生成可执行的Python预测函数"""
        print(f"\n💻 可执行的Python预测函数:")
        print("="*50)

        # 获取标准化参数
        scaler = self.pipeline.named_steps['scaler']

        print("import numpy as np")
        print()
        print("def predict_blur_index(", end="")
        print(", ".join(self.selected_features), end="")
        print("):")
        print('    """')
        print('    使用训练好的多项式回归模型预测blur_index')
        print('    ')
        print('    参数:')
        for feature in self.selected_features:
            print(f'    {feature}: float - {feature}特征值')
        print('    ')
        print('    返回:')
        print('    float - 预测的blur_index值')
        print('    """')

        # 标准化参数
        print("    # 标准化参数 (训练时计算)")
        print("    scaler_mean = np.array([", end="")
        print(", ".join([f"{mean:.8f}" for mean in scaler.mean_]), end="")
        print("])")
        print("    scaler_scale = np.array([", end="")
        print(", ".join([f"{scale:.8f}" for scale in scaler.scale_]), end="")
        print("])")

        # 输入特征数组
        print("    ")
        print("    # 组织输入特征")
        print("    features = np.array([", end="")
        print(", ".join(self.selected_features), end="")
        print("])")

        # 标准化
        print("    ")
        print("    # 特征标准化")
        print("    features_scaled = (features - scaler_mean) / scaler_scale")
        print("    ")

        # 计算预测值
        print("    # 多项式回归计算")
        print(f"    result = {intercept:.8f}  # 截距")

        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            if abs(coef) > 1e-8:
                if '^2' in name:  # 二次项
                    base_feature = name.replace('^2', '')
                    idx = self.selected_features.index(base_feature)
                    print(f"    result += {coef:.8f} * (features_scaled[{idx}] ** 2)  # {name}")
                elif ' ' in name:  # 交互项
                    features_pair = name.split(' ')
                    idx1 = self.selected_features.index(features_pair[0])
                    idx2 = self.selected_features.index(features_pair[1])
                    print(f"    result += {coef:.8f} * features_scaled[{idx1}] * features_scaled[{idx2}]  # {name}")
                else:  # 一次项
                    idx = self.selected_features.index(name)
                    print(f"    result += {coef:.8f} * features_scaled[{idx}]  # {name}")

        print("    ")
        print("    return result")

        # 使用示例
        print(f"\n# 使用示例:")
        sample_values = []
        for feature in self.selected_features:
            sample_val = self.data[feature].mean()
            sample_values.append(f"{sample_val:.2f}")

        print(f"# predicted_value = predict_blur_index({', '.join(sample_values)})")
        print(f"# print(f'预测的blur_index: {{predicted_value:.6f}}')")

    def analyze_feature_importance(self, linear_terms, quadratic_terms, interaction_terms):
        """分析特征重要性"""
        print(f"\n📈 特征重要性分析:")
        print("-" * 40)

        # 收集所有影响
        all_effects = {}

        # 线性效应
        for name, coef in linear_terms:
            if name not in all_effects:
                all_effects[name] = {'linear': 0, 'quadratic': 0, 'total': 0}
            all_effects[name]['linear'] = abs(coef)
            all_effects[name]['total'] += abs(coef)

        # 二次效应
        for name, coef in quadratic_terms:
            base_name = name.replace('^2', '')
            if base_name not in all_effects:
                all_effects[base_name] = {'linear': 0, 'quadratic': 0, 'total': 0}
            all_effects[base_name]['quadratic'] = abs(coef)
            all_effects[base_name]['total'] += abs(coef)

        # 按总影响排序
        sorted_features = sorted(all_effects.items(), key=lambda x: x[1]['total'], reverse=True)

        print(f"{'特征名称':<20} {'线性系数':<12} {'二次系数':<12} {'总影响':<12} {'主要效应':<10}")
        print("-" * 75)

        for feature, effects in sorted_features:
            main_effect = "线性" if effects['linear'] > effects['quadratic'] else "二次"
            if effects['linear'] == 0:
                main_effect = "仅二次"
            elif effects['quadratic'] == 0:
                main_effect = "仅线性"

            print(f"{feature:<20} {effects['linear']:<12.6f} {effects['quadratic']:<12.6f} "
                  f"{effects['total']:<12.6f} {main_effect:<10}")

def run_complete_analysis(data_path):
    """运行完整的分析流程"""
    print("🚀 启动blur_index多项式回归分析")
    print("=" * 60)

    # 创建预测器
    predictor = SimpleBlurPredictor(degree=2, alpha=1.0)

    # 第1步: 加载数据
    print("\n📂 第1步: 加载数据")
    predictor.load_and_prepare_data(data_path)

    # 第2步: 特征选择
    print("\n🎯 第2步: 特征选择")
    predictor.select_top_features(n_features=6)

    # 第3步: 训练模型
    print("\n🤖 第3步: 训练多项式回归模型")
    predictor.train_polynomial_model()

    # 第4步: 绘制折线图对比 (核心功能1)
    print("\n📊 第4步: 绘制预测vs实际值折线图对比")
    predictor.plot_line_comparison()

    # 第5步: 输出多项式公式 (核心功能2)
    print("\n📝 第5步: 输出具体多项式公式")
    formula = predictor.output_polynomial_formula()

    print(f"\n✅ 分析完成!")
    print("="*60)

    return predictor, formula

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="拟合问题一的综合模糊指数回归模型")
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "data" / "private" / "processed" / "blur.csv",
        help="包含 blur_index 的 CSV 文件",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"找不到数据文件: {args.data}\n"
            "可先运行 scripts/generate_demo_data.py 生成演示数据。"
        )

    # 运行完整分析
    predictor, formula = run_complete_analysis(args.data)

    print(f"\n🎯 快速访问结果:")
    print(f"   predictor.polynomial_formula  # 完整公式字符串")
    print(f"   predictor.train_r2           # 训练集R²: {predictor.train_r2:.6f}")
    print(f"   predictor.test_r2            # 测试集R²: {predictor.test_r2:.6f}")
