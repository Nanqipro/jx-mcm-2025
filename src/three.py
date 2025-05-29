#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度预测分析模块

该模块基于图像特征和气象数据构建能见度预测模型，包含数据分析、特征选择、
模型训练和验证等完整流程。

Author: Data Analysis Team
Date: 2024
"""

import os
from typing import Tuple, List, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


class VisibilityAnalyzer:
    """
    能见度预测分析器
    
    该类实现了基于图像特征和气象数据的能见度预测模型，包含数据处理、
    特征分析、模型训练和验证等核心功能。
    
    Attributes
    ----------
    data : pd.DataFrame
        原始数据集
    target_col : str
        目标变量列名
    scaler_X : StandardScaler
        特征标准化器
    scaler_y : StandardScaler
        目标变量标准化器（如果需要）
    best_model : object
        最佳训练模型
    selected_features : List[str]
        选择的特征列表
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化分析器
        
        Parameters
        ----------
        data_path : str, optional
            数据文件路径，默认为None
        """
        self.data = None
        self.target_col = 'visibility_mor_raw'
        self.scaler_X = StandardScaler()
        self.scaler_y = None
        self.best_model = None
        self.selected_features = []
        self.model_results = {}
        
        # 配置中文字体
        self._setup_chinese_font()
        
        if data_path:
            self.load_data(data_path)
    
    def _setup_chinese_font(self) -> None:
        """配置matplotlib中文字体显示"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self, data_path: str) -> None:
        """
        加载数据文件
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        """
        try:
            self.data = pd.read_csv(data_path)
            print(f"数据加载成功！形状: {self.data.shape}")
            print(f"列名: {list(self.data.columns)}")
        except Exception as e:
            raise FileNotFoundError(f"无法加载数据文件: {e}")
    
    def explore_data(self) -> Dict[str, Any]:
        """
        执行数据探索分析
        
        Returns
        -------
        Dict[str, Any]
            包含数据基本信息的字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("=" * 70)
        print("数据基本信息")
        print("=" * 70)
        
        # 基本信息
        print(f"数据集形状: {self.data.shape}")
        print(f"数据列数: {len(self.data.columns)}")
        print(f"数据行数: {len(self.data)}")
        
        # 缺失值检查
        missing_info = self.data.isnull().sum()
        print(f"\n缺失值情况:")
        print(missing_info[missing_info > 0])
        
        # 数据类型
        print(f"\n数据类型:")
        print(self.data.dtypes.value_counts())
        
        # 目标变量统计
        if self.target_col in self.data.columns:
            target_stats = self.data[self.target_col].describe()
            print(f"\n目标变量 '{self.target_col}' 统计:")
            print(target_stats)
        
        return {
            'shape': self.data.shape,
            'missing_values': missing_info,
            'data_types': self.data.dtypes,
            'target_stats': target_stats if self.target_col in self.data.columns else None
        }
    
    def analyze_features(self) -> Dict[str, pd.DataFrame]:
        """
        分析特征与目标变量的相关性
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            包含不同类型特征相关性分析结果的字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        print("=" * 70)
        print("特征相关性分析")
        print("=" * 70)
        
        # 识别不同类型的特征
        image_features = [col for col in self.data.columns if any(
            keyword in col.lower() for keyword in 
            ['laplacian', 'sobel', 'contrast', 'edge', 'freq', 'var']
        )]
        
        weather_features = [col for col in self.data.columns if any(
            keyword in col.lower() for keyword in 
            ['weather', 'temperature', 'humidity', 'pressure']
        )]
        
        wind_features = [col for col in self.data.columns if 'wind' in col.lower()]
        
        visibility_features = [col for col in self.data.columns if 'visibility' in col.lower()]
        
        print(f"图像特征 ({len(image_features)}个):")
        for feat in image_features:
            print(f"  - {feat}")
        
        print(f"\n气象特征 ({len(weather_features)}个):")
        for feat in weather_features:
            print(f"  - {feat}")
        
        print(f"\n风速特征 ({len(wind_features)}个):")
        for feat in wind_features:
            print(f"  - {feat}")
        
        print(f"\n能见度指标 ({len(visibility_features)}个):")
        for feat in visibility_features:
            print(f"  - {feat}")
        
        # 计算各类特征与目标变量的相关性
        correlation_results = {}
        
        if image_features:
            img_corr = self._calculate_correlations(image_features)
            correlation_results['image'] = img_corr
            print(f"\n图像特征与目标变量相关性:")
            print(img_corr.head(10))
        
        if weather_features:
            weather_corr = self._calculate_correlations(weather_features)
            correlation_results['weather'] = weather_corr
            print(f"\n气象特征与目标变量相关性:")
            print(weather_corr)
        
        if wind_features:
            wind_corr = self._calculate_correlations(wind_features)
            correlation_results['wind'] = wind_corr
            print(f"\n风速特征与目标变量相关性:")
            print(wind_corr)
        
        return correlation_results
    
    def _calculate_correlations(self, features: List[str]) -> pd.DataFrame:
        """
        计算特征与目标变量的相关性
        
        Parameters
        ----------
        features : List[str]
            要计算相关性的特征列表
        
        Returns
        -------
        pd.DataFrame
            相关性分析结果
        """
        correlations = []
        
        for feat in features:
            if feat in self.data.columns and feat != self.target_col:
                # 去除缺失值
                valid_data = self.data[[feat, self.target_col]].dropna()
                
                if len(valid_data) > 0:
                    pearson_corr, p_value = pearsonr(valid_data[feat], valid_data[self.target_col])
                    correlations.append({
                        '特征': feat,
                        'Pearson相关系数': pearson_corr,
                        'p值': p_value,
                        '相关性强度': self._interpret_correlation(abs(pearson_corr))
                    })
        
        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            corr_df = corr_df.sort_values('Pearson相关系数', key=abs, ascending=False)
        
        return corr_df
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """
        解释相关性强度
        
        Parameters
        ----------
        corr_value : float
            相关性系数绝对值
        
        Returns
        -------
        str
            相关性强度描述
        """
        if corr_value >= 0.8:
            return "极强"
        elif corr_value >= 0.6:
            return "强"
        elif corr_value >= 0.4:
            return "中等"
        elif corr_value >= 0.2:
            return "弱"
        else:
            return "极弱"
    
    def visualize_correlations(self) -> None:
        """绘制相关性热力图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 选择数值特征
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_features].corr()
        
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def detect_multicollinearity(self, features: List[str]) -> pd.DataFrame:
        """
        检测多重共线性
        
        Parameters
        ----------
        features : List[str]
            要检测的特征列表
        
        Returns
        -------
        pd.DataFrame
            VIF分析结果
        """
        print("=" * 70)
        print("多重共线性检测")
        print("=" * 70)
        
        X = self.data[features].dropna()
        
        vif_data = pd.DataFrame()
        vif_data["特征"] = features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(features))]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        print("方差膨胀因子(VIF)分析:")
        print(vif_data)
        print("\n注：VIF > 10 表示存在严重多重共线性")
        
        return vif_data
    
    def select_features(self, method: str = 'manual') -> List[str]:
        """
        特征选择
        
        Parameters
        ----------
        method : str, optional
            特征选择方法，可选 'manual', 'correlation', 'pca'
        
        Returns
        -------
        List[str]
            选择的特征列表
        """
        print("=" * 70)
        print("特征选择")
        print("=" * 70)
        
        if method == 'manual':
            # 基于专业知识手动选择特征
            selected_features = [
                'laplacian_var',
                'high_freq_ratio',
                'edge_density',
                'contrast_std',
                'weather_humidity_pct',
                'weather_temperature_c',
                'wind_wind_speed_10m'
            ]
            
            # 检查特征是否存在于数据中
            available_features = [feat for feat in selected_features 
                                if feat in self.data.columns]
            
            print(f"手动选择的特征 ({len(available_features)}个):")
            for i, feat in enumerate(available_features, 1):
                print(f"{i}. {feat}")
            
            self.selected_features = available_features
            
        return self.selected_features
    
    def prepare_data(self, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练和测试数据
        
        Parameters
        ----------
        test_size : float, optional
            测试集比例，默认0.2
        random_state : int, optional
            随机种子，默认42
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            训练和测试集的特征和目标变量
        """
        if not self.selected_features:
            self.select_features()
        
        # 准备特征和目标变量
        X = self.data[self.selected_features].dropna()
        y = self.data.loc[X.index, self.target_col]
        
        # 目标变量平方根变换（改善偏态分布）
        y_transformed = np.sqrt(y)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        print(f"数据准备完成:")
        print(f"训练集形状: {X_train_scaled.shape}")
        print(f"测试集形状: {X_test_scaled.shape}")
        print(f"目标变量已进行平方根变换")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        训练多种回归模型
        
        Parameters
        ----------
        X_train : np.ndarray
            训练集特征
        X_test : np.ndarray
            测试集特征
        y_train : np.ndarray
            训练集目标变量
        y_test : np.ndarray
            测试集目标变量
        
        Returns
        -------
        Dict[str, Dict]
            各模型的训练结果
        """
        print("=" * 70)
        print("模型训练与评估")
        print("=" * 70)
        
        models = {
            '普通线性回归': LinearRegression(),
            '岭回归(Ridge)': Ridge(alpha=1.0),
            'Lasso回归': Lasso(alpha=0.1),
            '弹性网络': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"\n{'='*20} {name} {'='*20}")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 评估指标
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            print(f"训练集 R²: {train_r2:.4f}")
            print(f"测试集 R²: {test_r2:.4f}")
            print(f"训练集 RMSE: {train_rmse:.4f}")
            print(f"测试集 RMSE: {test_rmse:.4f}")
            print(f"训练集 MAE: {train_mae:.4f}")
            print(f"测试集 MAE: {test_mae:.4f}")
            
            # 保存结果
            model_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'coefficients': getattr(model, 'coef_', None),
                'intercept': getattr(model, 'intercept_', None)
            }
        
        # 选择最佳模型
        best_model_name = max(model_results.keys(), 
                             key=lambda x: model_results[x]['test_r2'])
        self.best_model = model_results[best_model_name]['model']
        self.model_results = model_results
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"测试集 R²: {model_results[best_model_name]['test_r2']:.4f}")
        
        return model_results
    
    def analyze_residuals(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        残差分析和模型验证
        
        Parameters
        ----------
        X_test : np.ndarray
            测试集特征
        y_test : np.ndarray
            测试集目标变量
        """
        if self.best_model is None:
            raise ValueError("请先训练模型")
        
        print("=" * 70)
        print("残差分析和模型验证")
        print("=" * 70)
        
        # 预测
        y_pred = self.best_model.predict(X_test)
        
        # 残差计算
        residuals = y_test - y_pred
        
        print(f"残差统计:")
        print(f"均值: {residuals.mean():.6f}")
        print(f"标准差: {residuals.std():.6f}")
        
        # 绘制残差分析图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 预测值vs残差图
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Predicted vs Residuals')
        
        # 残差直方图
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        
        # Q-Q图
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Normality Test)')
        
        # 真实值vs预测值图 (sqrt尺度)
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'red', linestyle='--')
        axes[1, 0].set_xlabel('True Values (sqrt scale)')
        axes[1, 0].set_ylabel('Predicted Values (sqrt scale)')
        axes[1, 0].set_title('True vs Predicted (sqrt scale)')
        
        # 转换回原始尺度
        y_test_original = y_test**2
        y_pred_original = y_pred**2
        
        axes[1, 1].scatter(y_test_original, y_pred_original, alpha=0.6)
        axes[1, 1].plot([y_test_original.min(), y_test_original.max()], 
                       [y_test_original.min(), y_test_original.max()], 'red', linestyle='--')
        axes[1, 1].set_xlabel('True Visibility (meters)')
        axes[1, 1].set_ylabel('Predicted Visibility (meters)')
        axes[1, 1].set_title('True vs Predicted (original scale)')
        
        # 原始尺度精度统计
        original_r2 = r2_score(y_test_original, y_pred_original)
        original_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        original_mae = mean_absolute_error(y_test_original, y_pred_original)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        axes[1, 2].text(0.1, 0.8, 'Original Scale Accuracy:', fontsize=14, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, f'R² = {original_r2:.4f}', fontsize=12)
        axes[1, 2].text(0.1, 0.6, f'RMSE = {original_rmse:.1f} m', fontsize=12)
        axes[1, 2].text(0.1, 0.5, f'MAE = {original_mae:.1f} m', fontsize=12)
        axes[1, 2].text(0.1, 0.4, f'MAPE = {mape:.1f}%', fontsize=12)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n原始尺度（米）的模型精度:")
        print(f"R² = {original_r2:.4f}")
        print(f"RMSE = {original_rmse:.1f} 米")
        print(f"MAE = {original_mae:.1f} 米")
        print(f"MAPE = {mape:.1f}%")
    
    def get_model_equation(self) -> str:
        """
        获取模型数学公式
        
        Returns
        -------
        str
            模型的数学表达式
        """
        if self.best_model is None or not self.selected_features:
            raise ValueError("请先训练模型并选择特征")
        
        print("=" * 70)
        print("模型数学公式")
        print("=" * 70)
        
        # 获取模型系数
        coefficients = self.best_model.coef_
        intercept = self.best_model.intercept_
        
        print("标准化特征的回归方程:")
        print("sqrt(visibility_mor_raw) = β₀ + Σ(βᵢ × Xᵢ_standardized)")
        print()
        print("其中：")
        print(f"β₀ (截距) = {intercept:.6f}")
        
        formula_parts = [f"{intercept:.6f}"]
        for i, (feat, coef) in enumerate(zip(self.selected_features, coefficients), 1):
            print(f"β{i} ({feat}) = {coef:10.6f}")
            sign = "+" if coef >= 0 else ""
            formula_parts.append(f"{sign}{coef:.6f}×{feat}_std")
        
        formula = "sqrt(visibility_mor_raw) = " + " ".join(formula_parts)
        print(f"\n完整标准化公式:")
        print(formula)
        
        print(f"\n最终能见度预测公式:")
        print("visibility_mor_raw = [上述sqrt结果]²")
        
        return formula
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Parameters
        ----------
        new_data : pd.DataFrame
            新的输入数据
        
        Returns
        -------
        np.ndarray
            预测的能见度值（原始尺度）
        """
        if self.best_model is None:
            raise ValueError("请先训练模型")
        
        # 提取所需特征
        X_new = new_data[self.selected_features]
        
        # 标准化
        X_new_scaled = self.scaler_X.transform(X_new)
        
        # 预测（sqrt尺度）
        y_pred_sqrt = self.best_model.predict(X_new_scaled)
        
        # 转换回原始尺度
        y_pred = y_pred_sqrt ** 2
        
        return y_pred


def main():
    """主函数：执行完整的能见度预测分析流程"""
    print("能见度预测分析系统")
    print("=" * 70)
    
    # 初始化分析器
    analyzer = VisibilityAnalyzer()
    
    # 检查数据文件是否存在
    data_file = 'complete_synced_data.csv'
    if os.path.exists(data_file):
        # 加载数据
        analyzer.load_data(data_file)
        
        # 数据探索
        basic_info = analyzer.explore_data()
        
        # 特征分析
        correlation_results = analyzer.analyze_features()
        
        # 可视化相关性
        analyzer.visualize_correlations()
        
        # 特征选择
        selected_features = analyzer.select_features(method='manual')
        
        # 多重共线性检测
        vif_results = analyzer.detect_multicollinearity(selected_features)
        
        # 准备数据
        X_train, X_test, y_train, y_test = analyzer.prepare_data()
        
        # 训练模型
        model_results = analyzer.train_models(X_train, X_test, y_train, y_test)
        
        # 残差分析
        analyzer.analyze_residuals(X_test, y_test)
        
        # 获取模型公式
        equation = analyzer.get_model_equation()
        
        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)
        
    else:
        print(f"数据文件 '{data_file}' 不存在，请检查文件路径。")
        print("请将数据文件放在当前目录下或修改 data_file 变量的路径。")


if __name__ == "__main__":
    main()
