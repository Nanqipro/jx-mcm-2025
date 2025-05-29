import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和高质量图表
# 替换为:
try:
    # 尝试设置中文字体
    font_path = None

    # 检查常见的中文字体位置
    possible_fonts = [
        # Windows
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑

        # macOS
        '/System/Library/Fonts/PingFang.ttc',  # 苹方
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',

        # Linux
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]

    # 查找系统中可用的中文字体
    for font in possible_fonts:
        if os.path.exists(font):
            font_path = font
            break

    # 如果找到字体文件，使用它
    if font_path:
        chinese_font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = chinese_font.get_name()
        print(f"✅ 使用中文字体: {os.path.basename(font_path)}")
    else:
        # 回退到支持中文的字体名称
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                                           'Arial Unicode MS', 'DejaVu Sans']
        print("⚠️ 未找到字体文件，尝试使用名称匹配")

    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"❌ 字体设置失败: {e}")
    # 最后的回退方案
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['savefig.dpi'] = 300
# plt.style.use('seaborn-v0_8')


class EnhancedVisibilityAnalyzer:
    """增强版能见度时间序列分析系统"""

    def __init__(self):
        self.models = {}
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

    def load_data(self, csv_path):
        """加载数据"""
        try:
            self.data = pd.read_csv(csv_path)
            self.data = self.data.dropna(subset=['MOR_1A'])
            self.t = np.arange(len(self.data))
            self.V = self.data['MOR_1A'].values
            self.n = len(self.V)

            print(f"✅ 数据加载成功: {self.n} 个数据点")
            print(f"时间范围: 0 - {self.n - 1} 分钟")
            print(f"能见度范围: {self.V.min():.1f} - {self.V.max():.1f} 米")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def data_exploration_dashboard(self):
        """数据探索性分析仪表板"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 原始时间序列
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.t, self.V, color=self.colors['data'], linewidth=1.5, alpha=0.8)
        ax1.fill_between(self.t, self.V, alpha=0.3, color=self.colors['data'])
        ax1.set_title('能见度时间序列', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间 (分钟)')
        ax1.set_ylabel('能见度 (米)')
        ax1.grid(True, alpha=0.3)

        # 添加统计信息
        ax1.text(0.02, 0.98, f'均值: {np.mean(self.V):.1f}m\n'
                             f'标准差: {np.std(self.V):.1f}m\n'
                             f'最大值: {np.max(self.V):.1f}m\n'
                             f'最小值: {np.min(self.V):.1f}m',
                 transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top')

        # 2. 能见度分布直方图
        ax2 = fig.add_subplot(gs[0, 2])
        n, bins, patches = ax2.hist(self.V, bins=30, density=True, alpha=0.7,
                                    color=self.colors['primary'], edgecolor='black')
        # 拟合正态分布
        mu, sigma = stats.norm.fit(self.V)
        x = np.linspace(self.V.min(), self.V.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label=f'正态拟合\nμ={mu:.1f}, σ={sigma:.1f}')
        ax2.set_title('能见度分布')
        ax2.set_xlabel('能见度 (米)')
        ax2.set_ylabel('密度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 箱线图
        ax3 = fig.add_subplot(gs[0, 3])
        box_data = [self.V]
        bp = ax3.boxplot(box_data, patch_artist=True, labels=['能见度'])
        bp['boxes'][0].set_facecolor(self.colors['accent'])
        ax3.set_title('能见度箱线图')
        ax3.set_ylabel('能见度 (米)')
        ax3.grid(True, alpha=0.3)

        # 计算箱线图统计量
        q1, median, q3 = np.percentile(self.V, [25, 50, 75])
        iqr = q3 - q1
        ax3.text(1.2, median, f'中位数: {median:.1f}\nIQR: {iqr:.1f}',
                 fontsize=10, verticalalignment='center')

        # 4. 滑动平均
        ax4 = fig.add_subplot(gs[1, :2])
        windows = [10, 30, 60]
        ax4.plot(self.t, self.V, color='lightgray', alpha=0.5, label='原始数据')
        for window in windows:
            if window < len(self.V):
                moving_avg = pd.Series(self.V).rolling(window=window, center=True).mean()
                ax4.plot(self.t, moving_avg, linewidth=2, label=f'{window}分钟滑动平均')
        ax4.set_title('滑动平均分析')
        ax4.set_xlabel('时间 (分钟)')
        ax4.set_ylabel('能见度 (米)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 一阶差分
        ax5 = fig.add_subplot(gs[1, 2])
        diff1 = np.diff(self.V)
        ax5.plot(self.t[1:], diff1, color=self.colors['warning'], alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('一阶差分 (变化率)')
        ax5.set_xlabel('时间 (分钟)')
        ax5.set_ylabel('变化量 (米/分钟)')
        ax5.grid(True, alpha=0.3)

        # 6. 自相关分析
        ax6 = fig.add_subplot(gs[1, 3])
        max_lag = min(50, len(self.V) // 4)
        lags = range(max_lag)
        autocorr = [np.corrcoef(self.V[:-lag] if lag > 0 else self.V,
                                self.V[lag:] if lag > 0 else self.V)[0, 1]
                    if lag > 0 else 1.0 for lag in lags]
        ax6.plot(lags, autocorr, 'o-', color=self.colors['secondary'])
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_title('自相关函数')
        ax6.set_xlabel('滞后期')
        ax6.set_ylabel('自相关系数')
        ax6.grid(True, alpha=0.3)

        # 7. 时间序列分解
        ax7 = fig.add_subplot(gs[2, :])
        # 简单的趋势和季节性分解
        window = min(60, len(self.V) // 4)
        if window > 2:
            trend = pd.Series(self.V).rolling(window=window, center=True).mean()
            detrended = self.V - trend.fillna(method='bfill').fillna(method='ffill')

            ax7_1 = ax7
            ax7_2 = ax7.twinx()

            line1 = ax7_1.plot(self.t, self.V, color=self.colors['data'],
                               linewidth=1, alpha=0.7, label='原始数据')
            line2 = ax7_1.plot(self.t, trend, color=self.colors['primary'],
                               linewidth=2, label='趋势')
            line3 = ax7_2.plot(self.t, detrended, color=self.colors['accent'],
                               linewidth=1, alpha=0.8, label='去趋势残差')

            ax7_1.set_xlabel('时间 (分钟)')
            ax7_1.set_ylabel('能见度 (米)', color=self.colors['data'])
            ax7_2.set_ylabel('残差 (米)', color=self.colors['accent'])

            # 合并图例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax7_1.legend(lines, labels, loc='upper right')
            ax7_1.set_title('时间序列分解')
            ax7_1.grid(True, alpha=0.3)

        # 8. 统计摘要表
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')

        # 基本统计量
        stats_data = {
            '统计量': ['样本数', '均值', '标准差', '最小值', '25%分位数', '中位数',
                       '75%分位数', '最大值', '偏度', '峰度'],
            '数值': [f'{len(self.V)}',
                     f'{np.mean(self.V):.2f}',
                     f'{np.std(self.V):.2f}',
                     f'{np.min(self.V):.2f}',
                     f'{np.percentile(self.V, 25):.2f}',
                     f'{np.percentile(self.V, 50):.2f}',
                     f'{np.percentile(self.V, 75):.2f}',
                     f'{np.max(self.V):.2f}',
                     f'{stats.skew(self.V):.3f}',
                     f'{stats.kurtosis(self.V):.3f}']
        }

        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(self.V[:min(5000, len(self.V))])

        # 创建表格
        table_data = []
        for i in range(len(stats_data['统计量'])):
            table_data.append([stats_data['统计量'][i], stats_data['数值'][i]])

        table_data.append(['', ''])
        table_data.append(['正态性检验 (Shapiro)', ''])
        table_data.append(['统计量', f'{shapiro_stat:.4f}'])
        table_data.append(['p值', f'{shapiro_p:.4f}'])
        table_data.append(['结论', '正态分布' if shapiro_p > 0.05 else '非正态分布'])

        table = ax8.table(cellText=table_data,
                          colLabels=['项目', '值'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表格样式
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # 偶数行
                    cell.set_facecolor('#f8f9fa')
                else:  # 奇数行
                    cell.set_facecolor('white')

        ax8.set_title('数据统计摘要', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('能见度数据探索性分析仪表板', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def generate_polynomial_expression(self, coeffs):
        """生成多项式表达式字符串"""
        degree = len(coeffs) - 1
        terms = []

        for i, coeff in enumerate(coeffs):
            power = degree - i

            if abs(coeff) < 1e-12:
                continue

            # 格式化系数
            if abs(coeff) >= 1e-3:
                coeff_str = f"{coeff:.6f}"
            else:
                coeff_str = f"{coeff:.3e}"

            # 处理符号
            if coeff < 0:
                sign = " - "
                coeff_str = coeff_str[1:]  # 移除负号
            else:
                sign = " + " if terms else ""

            # 构建项
            if power == 0:
                term = f"{sign}{coeff_str}"
            elif power == 1:
                term = f"{sign}{coeff_str}×t"
            else:
                term = f"{sign}{coeff_str}×t^{power}"

            terms.append(term)

        return "".join(terms) if terms else "0"

    def fit_all_models(self, max_degree=8):
        """拟合所有模型"""
        print("\n正在拟合多种模型...")

        # 1. 线性模型
        linear_coeffs = np.polyfit(self.t, self.V, 1)
        self.models['linear'] = {
            'coeffs': linear_coeffs,
            'function': lambda t: linear_coeffs[0] * t + linear_coeffs[1],
            'name': '线性模型',
            'type': 'polynomial',
            'degree': 1
        }

        V_pred = self.models['linear']['function'](self.t)
        self.models['linear']['r2'] = r2_score(self.V, V_pred)
        self.models['linear']['mae'] = mean_absolute_error(self.V, V_pred)
        self.models['linear']['rmse'] = np.sqrt(np.mean((self.V - V_pred) ** 2))
        self.models['linear']['expression'] = f"{linear_coeffs[0]:.6f}×t + {linear_coeffs[1]:.2f}"

        # 2. 多项式模型 (自动选择最佳度数)
        best_score = -np.inf
        best_degree = 1
        best_coeffs = None

        for degree in range(2, min(max_degree + 1, len(self.t) // 20)):
            try:
                coeffs = np.polyfit(self.t, self.V, degree)
                V_pred = np.polyval(coeffs, self.t)
                r2 = r2_score(self.V, V_pred)

                # 使用调整后的R²避免过拟合
                n = len(self.V)
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - degree - 1)

                if adjusted_r2 > best_score:
                    best_score = adjusted_r2
                    best_degree = degree
                    best_coeffs = coeffs
            except:
                continue

        if best_coeffs is not None:
            self.models['polynomial'] = {
                'coeffs': best_coeffs,
                'degree': best_degree,
                'function': lambda t: np.polyval(best_coeffs, t),
                'name': f'多项式模型 (度数 {best_degree})',
                'type': 'polynomial'
            }

            V_pred = np.polyval(best_coeffs, self.t)
            self.models['polynomial']['r2'] = r2_score(self.V, V_pred)
            self.models['polynomial']['mae'] = mean_absolute_error(self.V, V_pred)
            self.models['polynomial']['rmse'] = np.sqrt(np.mean((self.V - V_pred) ** 2))
            self.models['polynomial']['expression'] = self.generate_polynomial_expression(best_coeffs)

        # 3. 三次样条模型
        try:
            n_knots = min(20, len(self.t) // 25)
            if n_knots >= 3:
                knot_indices = np.linspace(0, len(self.t) - 1, n_knots, dtype=int)
                t_knots = self.t[knot_indices]
                V_knots = self.V[knot_indices]

                cs = CubicSpline(t_knots, V_knots, bc_type='natural')

                self.models['spline'] = {
                    'spline': cs,
                    'function': lambda t: cs(t),
                    'name': '三次样条模型',
                    'type': 'spline',
                    'knots': len(t_knots)
                }

                V_pred = cs(self.t)
                self.models['spline']['r2'] = r2_score(self.V, V_pred)
                self.models['spline']['mae'] = mean_absolute_error(self.V, V_pred)
                self.models['spline']['rmse'] = np.sqrt(np.mean((self.V - V_pred) ** 2))
                self.models['spline']['expression'] = f"三次样条插值 (节点数: {len(t_knots)})"
        except:
            pass

    def model_comparison_dashboard(self):
        """模型比较仪表板"""
        if not self.models:
            print("请先拟合模型")
            return

        n_models = len(self.models)
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 模型性能比较表
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')

        table_data = []
        headers = ['模型', 'R²', 'MAE(米)', 'RMSE(米)', '复杂度']

        for name, model in self.models.items():
            complexity = str(model.get('degree', model.get('knots', 'N/A')))
            table_data.append([
                model['name'],
                f"{model['r2']:.6f}",
                f"{model['mae']:.2f}",
                f"{model['rmse']:.2f}",
                complexity
            ])

        table = ax1.table(cellText=table_data,
                          colLabels=headers,
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 表格样式
        for i in range(len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # 找到最佳R²并高亮
                    if j == 1:  # R²列
                        r2_values = [float(row[1]) for row in table_data]
                        if float(table_data[i - 1][1]) == max(r2_values):
                            cell.set_facecolor(self.colors['success'])
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
                    else:
                        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

        ax1.set_title('模型性能比较', fontsize=14, fontweight='bold')

        # 2. R²比较柱状图
        ax2 = fig.add_subplot(gs[0, 2])
        model_names = [model['name'].split('(')[0].strip() for model in self.models.values()]
        r2_values = [model['r2'] for model in self.models.values()]

        bars = ax2.bar(range(len(model_names)), r2_values,
                       color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']][
                             :len(model_names)])
        ax2.set_title('R² 比较')
        ax2.set_ylabel('R² 值')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for bar, value in zip(bars, r2_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=9)

        # 3. MAE比较柱状图
        ax3 = fig.add_subplot(gs[0, 3])
        mae_values = [model['mae'] for model in self.models.values()]

        bars = ax3.bar(range(len(model_names)), mae_values,
                       color=[self.colors['warning'], self.colors['accent'], self.colors['success']][:len(model_names)])
        ax3.set_title('MAE 比较')
        ax3.set_ylabel('MAE (米)')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.grid(True, alpha=0.3)

        for bar, value in zip(bars, mae_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_values) * 0.01,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=9)

        # 4. 所有模型拟合对比
        ax4 = fig.add_subplot(gs[1, :])

        # 原始数据
        ax4.plot(self.t, self.V, color='black', linewidth=2, alpha=0.8, label='观测数据')

        # 各模型拟合结果
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'],
                  self.colors['warning'], self.colors['success']]

        for i, (name, model) in enumerate(self.models.items()):
            V_pred = model['function'](self.t)
            ax4.plot(self.t, V_pred, color=colors[i % len(colors)],
                     linewidth=2, alpha=0.8, linestyle='--',
                     label=f"{model['name']} (R²={model['r2']:.4f})")

        ax4.set_title('所有模型拟合对比', fontsize=14, fontweight='bold')
        ax4.set_xlabel('时间 (分钟)')
        ax4.set_ylabel('能见度 (米)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 5-7. 最佳模型的残差分析
        best_name, best_model = self.get_best_model()
        if best_model:
            V_pred_best = best_model['function'](self.t)
            residuals = self.V - V_pred_best

            # 5. 残差时间序列
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(self.t, residuals, color=self.colors['residual'], alpha=0.7)
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax5.set_title(f'{best_model["name"]}\n残差时间序列')
            ax5.set_xlabel('时间 (分钟)')
            ax5.set_ylabel('残差 (米)')
            ax5.grid(True, alpha=0.3)

            # 6. 残差vs拟合值
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.scatter(V_pred_best, residuals, alpha=0.6, color=self.colors['residual'])
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax6.set_title('残差 vs 拟合值')
            ax6.set_xlabel('拟合值 (米)')
            ax6.set_ylabel('残差 (米)')
            ax6.grid(True, alpha=0.3)

            # 7. 残差正态Q-Q图
            ax7 = fig.add_subplot(gs[2, 2])
            stats.probplot(residuals, dist="norm", plot=ax7)
            ax7.set_title('残差正态Q-Q图')
            ax7.grid(True, alpha=0.3)

            # 8. 残差直方图
            ax8 = fig.add_subplot(gs[2, 3])
            ax8.hist(residuals, bins=20, density=True, alpha=0.7,
                     color=self.colors['residual'], edgecolor='black')

            # 拟合正态分布
            mu, sigma = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax8.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                     label=f'正态拟合\nμ={mu:.2f}, σ={sigma:.2f}')
            ax8.set_title('残差分布')
            ax8.set_xlabel('残差 (米)')
            ax8.set_ylabel('密度')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # 9. 预测对比 (底部跨列)
        ax9 = fig.add_subplot(gs[3, :])

        # 训练测试分割
        split_point = int(len(self.t) * 0.8)
        t_train, t_test = self.t[:split_point], self.t[split_point:]
        V_train, V_test = self.V[:split_point], self.V[split_point:]

        # 绘制数据
        ax9.plot(t_train, V_train, color='blue', linewidth=2, alpha=0.8, label='训练数据')
        ax9.plot(t_test, V_test, color='green', linewidth=2, alpha=0.8, label='测试数据')

        # 绘制最佳模型预测
        if best_model:
            V_pred_train = best_model['function'](t_train)
            V_pred_test = best_model['function'](t_test)

            ax9.plot(t_train, V_pred_train, color='red', linewidth=2,
                     alpha=0.8, linestyle='--', label=f'{best_model["name"]} 训练预测')
            ax9.plot(t_test, V_pred_test, color='orange', linewidth=2,
                     alpha=0.8, linestyle='--', label=f'{best_model["name"]} 测试预测')

            # 计算测试集性能
            r2_test = r2_score(V_test, V_pred_test)
            mae_test = mean_absolute_error(V_test, V_pred_test)

            ax9.text(0.02, 0.98, f'测试集性能:\nR² = {r2_test:.4f}\nMAE = {mae_test:.2f}m',
                     transform=ax9.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                     verticalalignment='top')

        ax9.axvline(x=split_point, color='gray', linestyle=':', linewidth=2,
                    alpha=0.7, label=f'训练/测试分界点')
        ax9.set_title('模型泛化能力验证', fontsize=14, fontweight='bold')
        ax9.set_xlabel('时间 (分钟)')
        ax9.set_ylabel('能见度 (米)')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax9.grid(True, alpha=0.3)

        plt.suptitle('模型比较分析仪表板', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    def best_model_detailed_analysis(self):
        """最佳模型详细分析"""
        best_name, best_model = self.get_best_model()
        if not best_model:
            print("未找到有效模型")
            return None, None

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        V_pred = best_model['function'](self.t)
        residuals = self.V - V_pred

        # 1. 主要拟合图 (大图)
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(self.t, self.V, color='black', linewidth=2, alpha=0.8, label='观测数据')
        ax1.plot(self.t, V_pred, color=self.colors['primary'], linewidth=2,
                 linestyle='--', alpha=0.9, label='模型拟合')

        # 添加置信区间
        residual_std = np.std(residuals)
        ax1.fill_between(self.t, V_pred - 1.96 * residual_std, V_pred + 1.96 * residual_std,
                         alpha=0.2, color=self.colors['primary'], label='95%置信区间')

        ax1.set_title(f'{best_model["name"]} - 最佳拟合模型\nR² = {best_model["r2"]:.6f}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间 (分钟)')
        ax1.set_ylabel('能见度 (米)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 模型信息面板
        ax2 = fig.add_subplot(gs[0, 3])
        ax2.axis('off')

        info_text = f"""
        🏆 最佳模型信息

        模型类型: {best_model['name']}

        📊 性能指标:
        • R² = {best_model['r2']:.8f}
        • MAE = {best_model['mae']:.4f} 米
        • RMSE = {best_model['rmse']:.4f} 米

        📐 模型表达式:
        V(t) = {best_model.get('expression', 'N/A')}

        ✨ 模型特点:
        """

        if best_model['r2'] > 0.95:
            info_text += "• 拟合度极佳\n• 预测精度很高\n• 推荐使用"
        elif best_model['r2'] > 0.85:
            info_text += "• 拟合度优秀\n• 预测较可靠\n• 适合应用"
        elif best_model['r2'] > 0.7:
            info_text += "• 拟合度良好\n• 可用于趋势分析\n• 谨慎预测"
        else:
            info_text += "• 拟合度有限\n• 仅供参考\n• 需要改进"

        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['primary'], alpha=0.1),
                 verticalalignment='top', fontfamily='monospace')

        # 3. 残差时间序列
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.t, residuals, color=self.colors['residual'], alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.fill_between(self.t, -1.96 * np.std(residuals), 1.96 * np.std(residuals),
                         alpha=0.2, color='gray', label='±1.96σ')
        ax3.set_title('残差时间序列')
        ax3.set_xlabel('时间 (分钟)')
        ax3.set_ylabel('残差 (米)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 残差散点图
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(V_pred, residuals, alpha=0.6, color=self.colors['residual'])
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 添加趋势线
        z = np.polyfit(V_pred, residuals, 1)
        p = np.poly1d(z)
        ax4.plot(V_pred, p(V_pred), "r--", alpha=0.8, linewidth=2)

        ax4.set_title('残差 vs 拟合值')
        ax4.set_xlabel('拟合值 (米)')
        ax4.set_ylabel('残差 (米)')
        ax4.grid(True, alpha=0.3)

        # 5. 残差分布
        ax5 = fig.add_subplot(gs[1, 2])
        n, bins, patches = ax5.hist(residuals, bins=25, density=True, alpha=0.7,
                                    color=self.colors['residual'], edgecolor='black')

        # 正态分布拟合
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label=f'正态分布\nμ={mu:.2f}, σ={sigma:.2f}')

        ax5.set_title('残差分布')
        ax5.set_xlabel('残差 (米)')
        ax5.set_ylabel('密度')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Q-Q图
        ax6 = fig.add_subplot(gs[1, 3])
        stats.probplot(residuals, dist="norm", plot=ax6)
        ax6.set_title('残差正态Q-Q图')
        ax6.grid(True, alpha=0.3)

        # 7. 预测能力验证
        ax7 = fig.add_subplot(gs[2, :2])

        # 交叉验证
        split_point = int(len(self.t) * 0.8)
        t_train, t_test = self.t[:split_point], self.t[split_point:]
        V_train, V_test = self.V[:split_point], self.V[split_point:]

        V_pred_train = best_model['function'](t_train)
        V_pred_test = best_model['function'](t_test)

        ax7.plot(t_train, V_train, color='blue', linewidth=2, alpha=0.8, label='训练数据')
        ax7.plot(t_test, V_test, color='green', linewidth=2, alpha=0.8, label='测试数据')
        ax7.plot(t_train, V_pred_train, color='red', linewidth=2, alpha=0.8,
                 linestyle='--', label='训练预测')
        ax7.plot(t_test, V_pred_test, color='orange', linewidth=2, alpha=0.8,
                 linestyle='--', label='测试预测')

        ax7.axvline(x=split_point, color='gray', linestyle=':', linewidth=2, alpha=0.7)

        # 计算测试性能
        r2_test = r2_score(V_test, V_pred_test)
        mae_test = mean_absolute_error(V_test, V_pred_test)

        ax7.text(0.02, 0.98,
                 f'训练集 R² = {best_model["r2"]:.4f}\n测试集 R² = {r2_test:.4f}\n泛化误差 = {abs(best_model["r2"] - r2_test):.4f}',
                 transform=ax7.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top')

        ax7.set_title('模型泛化能力验证')
        ax7.set_xlabel('时间 (分钟)')
        ax7.set_ylabel('能见度 (米)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 预测vs实际散点图
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.scatter(self.V, V_pred, alpha=0.6, color=self.colors['primary'])

        # 完美预测线
        min_val = min(self.V.min(), V_pred.min())
        max_val = max(self.V.max(), V_pred.max())
        ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)

        ax8.set_xlabel('实际值 (米)')
        ax8.set_ylabel('预测值 (米)')
        ax8.set_title('预测值 vs 实际值')
        ax8.grid(True, alpha=0.3)

        # 9. 残差统计
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.axis('off')

        # 残差统计分析
        residual_stats = {
            '均值': f'{np.mean(residuals):.4f}',
            '标准差': f'{np.std(residuals):.4f}',
            '最大值': f'{np.max(residuals):.4f}',
            '最小值': f'{np.min(residuals):.4f}',
            '偏度': f'{stats.skew(residuals):.4f}',
            '峰度': f'{stats.kurtosis(residuals):.4f}'
        }

        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])

        stats_text = "📊 残差统计分析\n\n"
        for key, value in residual_stats.items():
            stats_text += f"{key}: {value}\n"

        stats_text += f"\n🧪 正态性检验:\n"
        stats_text += f"Shapiro统计量: {shapiro_stat:.4f}\n"
        stats_text += f"p值: {shapiro_p:.4f}\n"
        stats_text += f"结论: {'正态分布' if shapiro_p > 0.05 else '非正态分布'}"

        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3),
                 verticalalignment='top', fontfamily='monospace')

        # 10. 系数分析 (如果是多项式)
        ax10 = fig.add_subplot(gs[3, :])

        if best_model.get('type') == 'polynomial' and 'coeffs' in best_model:
            coeffs = best_model['coeffs']
            degree = best_model.get('degree', len(coeffs) - 1)
            powers = np.arange(degree, -1, -1)

            # 系数柱状图
            bars = ax10.bar(powers, np.abs(coeffs), alpha=0.7, color=self.colors['accent'])

            # 添加值标签
            for bar, coeff, power in zip(bars, coeffs, powers):
                height = bar.get_height()
                ax10.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                          f'{coeff:.3e}', ha='center', va='bottom', fontsize=9, rotation=45)

            ax10.set_xlabel('t的幂次')
            ax10.set_ylabel('系数绝对值')
            ax10.set_title(f'多项式系数分析 (度数={degree})')
            ax10.set_xticks(powers)
            ax10.grid(True, alpha=0.3)

            # 添加系数表格
            table_data = []
            for power, coeff in zip(powers, coeffs):
                table_data.append([f't^{power}', f'{coeff:.6e}'])

            # 在右侧添加系数表
            ax10_table = ax10.twinx()
            ax10_table.axis('off')

        else:
            ax10.axis('off')
            ax10.text(0.5, 0.5, f'最佳模型: {best_model["name"]}\n\n'
                                f'这是非多项式模型，无系数分析\n\n'
                                f'模型表达式:\n{best_model.get("expression", "N/A")}',
                      ha='center', va='center', fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['primary'], alpha=0.1))

        plt.suptitle(f'最佳模型详细分析 - {best_model["name"]}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

        return best_name, best_model

    def future_prediction_dashboard(self, predict_steps=120):
        """未来预测仪表板"""
        best_name, best_model = self.get_best_model()
        if not best_model:
            print("未找到有效模型")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 生成预测数据
        t_future = np.arange(len(self.t), len(self.t) + predict_steps)
        V_future = best_model['function'](t_future)

        # 1. 主预测图 (大图)
        ax1 = fig.add_subplot(gs[0, :])

        # 历史数据
        ax1.plot(self.t, self.V, color='black', linewidth=2, alpha=0.8, label='历史观测数据')

        # 模型拟合
        V_fitted = best_model['function'](self.t)
        ax1.plot(self.t, V_fitted, color=self.colors['primary'], linewidth=2,
                 linestyle='--', alpha=0.9, label='模型拟合')

        # 未来预测
        ax1.plot(t_future, V_future, color=self.colors['prediction'], linewidth=3,
                 linestyle='-', alpha=0.9, label=f'未来{predict_steps}分钟预测')

        # 预测置信区间
        residual_std = np.std(self.V - V_fitted)
        ax1.fill_between(t_future, V_future - 1.96 * residual_std, V_future + 1.96 * residual_std,
                         alpha=0.3, color=self.colors['prediction'], label='95%预测区间')

        # 分界线
        ax1.axvline(x=len(self.t) - 1, color='red', linestyle=':', linewidth=2, alpha=0.7,
                    label='预测起点')

        ax1.set_title(f'能见度预测 - {best_model["name"]}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间 (分钟)')
        ax1.set_ylabel('能见度 (米)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 预测统计信息
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')

        pred_stats = {
            '预测时长': f'{predict_steps} 分钟',
            '预测起点': f'{len(self.t)} 分钟',
            '预测终点': f'{len(self.t) + predict_steps} 分钟',
            '当前能见度': f'{self.V[-1]:.2f} 米',
            '预测终值': f'{V_future[-1]:.2f} 米',
            '变化幅度': f'{V_future[-1] - self.V[-1]:.2f} 米',
            '平均预测值': f'{np.mean(V_future):.2f} 米',
            '预测标准差': f'{np.std(V_future):.2f} 米'
        }

        stats_text = "📊 预测统计信息\n\n"
        for key, value in pred_stats.items():
            stats_text += f"{key}: {value}\n"

        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
                 verticalalignment='top', fontfamily='monospace')

        # 3. 预测趋势分析
        ax3 = fig.add_subplot(gs[1, 1])

        # 计算预测趋势
        trend_change = np.diff(V_future)

        ax3.plot(t_future[1:], trend_change, color=self.colors['accent'], linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.fill_between(t_future[1:], trend_change, 0,
                         where=(trend_change > 0), color='green', alpha=0.3, label='上升趋势')
        ax3.fill_between(t_future[1:], trend_change, 0,
                         where=(trend_change < 0), color='red', alpha=0.3, label='下降趋势')

        ax3.set_title('预测变化趋势')
        ax3.set_xlabel('时间 (分钟)')
        ax3.set_ylabel('变化率 (米/分钟)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 关键时间点预测
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')

        # 选择关键时间点
        key_times = [15, 30, 60, 90, predict_steps]
        key_times = [t for t in key_times if t <= predict_steps]

        key_predictions = []
        for t_key in key_times:
            if t_key <= len(V_future):
                pred_value = V_future[t_key - 1]
                key_predictions.append([f'{t_key}分钟后', f'{pred_value:.2f}米'])

        if key_predictions:
            table = ax4.table(cellText=key_predictions,
                              colLabels=['时间点', '预测能见度'],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # 表格样式
            for i in range(len(key_predictions) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # 表头
                        cell.set_facecolor(self.colors['primary'])
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

        ax4.set_title('关键时间点预测', fontsize=12, fontweight='bold')

        # 5. 预测可靠性分析
        ax5 = fig.add_subplot(gs[1, 3])

        # 计算预测不确定性
        time_horizon = np.arange(1, predict_steps + 1)
        uncertainty = residual_std * np.sqrt(1 + time_horizon / len(self.t))  # 随时间增加的不确定性

        ax5.plot(time_horizon, uncertainty, color=self.colors['warning'], linewidth=2)
        ax5.fill_between(time_horizon, 0, uncertainty, alpha=0.3, color=self.colors['warning'])

        ax5.set_title('预测不确定性')
        ax5.set_xlabel('预测时长 (分钟)')
        ax5.set_ylabel('预测误差标准差 (米)')
        ax5.grid(True, alpha=0.3)

        # 6. 多个预测场景
        ax6 = fig.add_subplot(gs[2, :])

        # 绘制历史数据
        ax6.plot(self.t, self.V, color='black', linewidth=2, alpha=0.8, label='历史数据')

        # 基准预测
        ax6.plot(t_future, V_future, color=self.colors['prediction'], linewidth=3,
                 label=f'基准预测 ({best_model["name"]})')

        # 乐观预测 (+ 1个标准差)
        V_optimistic = V_future + residual_std
        ax6.plot(t_future, V_optimistic, color='green', linewidth=2,
                 linestyle=':', alpha=0.8, label='乐观预测 (+1σ)')

        # 悲观预测 (- 1个标准差)
        V_pessimistic = V_future - residual_std
        ax6.plot(t_future, V_pessimistic, color='red', linewidth=2,
                 linestyle=':', alpha=0.8, label='悲观预测 (-1σ)')

        # 填充预测区间
        ax6.fill_between(t_future, V_pessimistic, V_optimistic, alpha=0.2,
                         color=self.colors['prediction'], label='预测区间')

        ax6.axvline(x=len(self.t) - 1, color='gray', linestyle=':', linewidth=2, alpha=0.7)

        ax6.set_title('多场景预测分析', fontsize=14, fontweight='bold')
        ax6.set_xlabel('时间 (分钟)')
        ax6.set_ylabel('能见度 (米)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'能见度未来预测分析 - 预测{predict_steps}分钟', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

        # 打印预测摘要
        self.print_prediction_summary(V_future, predict_steps, best_model)

    def print_prediction_summary(self, V_future, predict_steps, best_model):
        """打印预测摘要"""
        print(f"\n{'=' * 80}")
        print(f"🔮 能见度预测摘要报告")
        print(f"{'=' * 80}")
        print(f"预测模型: {best_model['name']}")
        print(f"模型性能: R² = {best_model['r2']:.6f}, MAE = {best_model['mae']:.2f}米")
        print(f"预测时长: {predict_steps} 分钟")
        print(f"当前时刻: {len(self.t)} 分钟")
        print(f"当前能见度: {self.V[-1]:.2f} 米")

        print(f"\n📈 预测结果:")
        key_times = [15, 30, 60, 90, 120]
        for t in key_times:
            if t <= predict_steps:
                pred_value = V_future[t - 1]
                change = pred_value - self.V[-1]
                trend = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
                print(f"  {t:3d}分钟后: {pred_value:6.2f}米 ({change:+6.2f}米) {trend}")

        print(f"\n📊 预测统计:")
        print(f"  预测均值: {np.mean(V_future):.2f} 米")
        print(f"  预测标准差: {np.std(V_future):.2f} 米")
        print(f"  预测最大值: {np.max(V_future):.2f} 米")
        print(f"  预测最小值: {np.min(V_future):.2f} 米")
        print(f"  总体变化: {V_future[-1] - self.V[-1]:+.2f} 米")

        # 趋势分析
        trend_direction = "上升" if V_future[-1] > self.V[-1] else "下降" if V_future[-1] < self.V[-1] else "稳定"
        print(f"\n🎯 趋势分析:")
        print(f"  整体趋势: {trend_direction}")

        if abs(V_future[-1] - self.V[-1]) > 100:
            print(f"  ⚠️  预警: 能见度变化较大，请注意!")

        print(f"{'=' * 80}")

    def get_best_model(self):
        """获取最佳模型"""
        if not self.models:
            return None, None

        best_model_name = None
        best_score = -np.inf

        for name, model in self.models.items():
            if 'r2' in model and model['r2'] > best_score:
                best_score = model['r2']
                best_model_name = name

        return best_model_name, self.models[best_model_name] if best_model_name else None

    def run_complete_analysis(self, predict_steps=120):
        """运行完整分析"""
        print("🚀 启动增强版能见度分析系统")
        print("=" * 80)

        # 1. 数据探索
        print("\n📊 Step 1: 数据探索性分析...")
        self.data_exploration_dashboard()

        # 2. 模型拟合
        print("\n🔧 Step 2: 模型拟合...")
        self.fit_all_models()

        # 3. 模型比较
        print("\n📈 Step 3: 模型比较分析...")
        self.model_comparison_dashboard()

        # 4. 最佳模型详细分析
        print("\n🏆 Step 4: 最佳模型详细分析...")
        best_name, best_model = self.best_model_detailed_analysis()

        # 5. 未来预测
        print("\n🔮 Step 5: 未来预测分析...")
        self.future_prediction_dashboard(predict_steps)

        # 6. 输出最终结果
        print(f"\n{'=' * 80}")
        print(f"🎯 最终分析结果")
        print(f"{'=' * 80}")

        if best_model:
            print(f"✅ 最佳模型: {best_model['name']}")
            print(f"📐 数学公式: V(t) = {best_model.get('expression', 'N/A')}")
            print(f"📊 模型性能: R² = {best_model['r2']:.8f}")
            print(f"🎯 预测精度: MAE = {best_model['mae']:.4f} 米")

            if best_model['r2'] > 0.9:
                print("🌟 模型质量: 优秀 - 强烈推荐使用")
            elif best_model['r2'] > 0.8:
                print("✅ 模型质量: 良好 - 推荐使用")
            elif best_model['r2'] > 0.7:
                print("⚠️ 模型质量: 一般 - 谨慎使用")
            else:
                print("❌ 模型质量: 较差 - 需要改进")

        print(f"{'=' * 80}")
        print("📋 分析报告已生成，包含以下可视化:")
        print("  1. 数据探索性分析仪表板 (8个图表)")
        print("  2. 模型比较分析仪表板 (9个图表)")
        print("  3. 最佳模型详细分析 (10个图表)")
        print("  4. 未来预测分析仪表板 (6个图表)")
        print("📊 总计: 33个专业图表 + 详细统计分析")
        print(f"{'=' * 80}")

        return best_name, best_model


def main():
    """主函数"""
    print("🎯 增强版能见度时间序列分析系统")
    print("=" * 80)
    print("功能特点:")
    print("• 🔍 全面的数据探索性分析")
    print("• 🤖 多模型智能拟合与比较")
    print("• 📊 丰富的可视化图表 (33个)")
    print("• 🔮 智能预测与不确定性分析")
    print("• 📈 专业的统计分析报告")
    print("=" * 80)

    # 创建分析器
    analyzer = EnhancedVisibilityAnalyzer()

    # 数据文件路径
    csv_file = 'blur.csv'  # 请替换为你的文件路径

    if analyzer.load_data(csv_file):
        # 运行完整分析
        best_name, best_model = analyzer.run_complete_analysis(predict_steps=120)

        if best_model:
            print(f"\n🎉 分析完成!")
            print(f"📐 最终数学模型公式:")
            print(f"")
            print(f"    V(t) = {best_model.get('expression', 'N/A')}")
            print(f"")
            print(f"其中:")
            print(f"  • t: 时间变量 (分钟)")
            print(f"  • V(t): t时刻的能见度预测值 (米)")
            print(f"  • R²: {best_model['r2']:.8f} (拟合优度)")
            print(f"  • MAE: {best_model['mae']:.4f} 米 (平均绝对误差)")

    else:
        print("❌ 请确保CSV文件存在且包含'MOR_1A'列")
        print("💡 数据格式要求:")
        print("  • CSV格式文件")
        print("  • 包含'MOR_1A'列(能见度数据)")
        print("  • 数据为数值型，单位：米")


# 快速启动函数
def quick_analysis(csv_file='blur.csv', predict_minutes=60):
    """快速分析函数 - 适合初学者"""
    print("🚀 快速分析模式启动...")

    analyzer = EnhancedVisibilityAnalyzer()

    if analyzer.load_data(csv_file):
        # 只运行核心分析
        analyzer.fit_all_models()
        best_name, best_model = analyzer.best_model_detailed_analysis()

        if best_model:
            print(f"\n✅ 快速分析完成!")
            print(f"🏆 最佳模型: {best_model['name']}")
            print(f"📐 数学公式: V(t) = {best_model.get('expression', 'N/A')}")
            print(f"📊 拟合质量: R² = {best_model['r2']:.6f}")

            return best_model.get('expression', 'N/A')

    return None


# 仅提取公式的简化函数
def extract_formula_only(csv_file='blur.csv'):
    """仅提取最佳模型公式"""
    analyzer = EnhancedVisibilityAnalyzer()

    if analyzer.load_data(csv_file):
        analyzer.fit_all_models()
        best_name, best_model = analyzer.get_best_model()

        if best_model:
            print(f"\n🎯 最佳模型公式:")
            print(f"V(t) = {best_model.get('expression', 'N/A')}")
            print(f"R² = {best_model['r2']:.8f}")
            return best_model.get('expression', 'N/A')

    return None


if __name__ == "__main__":
    # 运行完整分析
    main()

    # 如果只需要快速分析，取消下面的注释:
    # quick_analysis()

    # 如果只需要提取公式，取消下面的注释:
    # extract_formula_only()