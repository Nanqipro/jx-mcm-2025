#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于大雾背景视频学习的图像模糊程度分析模型

基于2020E.md中提供的思路，实现江西省数学建模比赛第一题：
建立反映大雾导致的视频图像模糊程度的数学模型

作者：AI助手
日期：2024年
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict, Any
import os
from skimage import filters, feature, measure
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FogBlurAnalyzer:
    """
    大雾图像模糊程度分析器
    
    基于多种图像质量评价指标，量化分析大雾对视频图像造成的模糊程度
    """
    
    def __init__(self):
        """初始化分析器"""
        self.blur_metrics: Dict[str, List[float]] = {
            'laplacian_variance': [],
            'gradient_magnitude': [],
            'sobel_variance': [],
            'tenengrad': [],
            'brenner_gradient': [],
            'dark_channel_prior': [],
            'visibility_index': []
        }
        
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        计算拉普拉斯方差 - 经典的图像清晰度评价指标
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            拉普拉斯方差值，越大表示越清晰
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    def calculate_gradient_magnitude(self, image: np.ndarray) -> float:
        """
        计算梯度幅值 - 基于边缘信息的清晰度评价
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            平均梯度幅值
        """
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(magnitude)
    
    def calculate_sobel_variance(self, image: np.ndarray) -> float:
        """
        计算Sobel算子方差
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            Sobel方差值
        """
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.var(sobel)
    
    def calculate_tenengrad(self, image: np.ndarray) -> float:
        """
        计算Tenengrad梯度 - 基于Sobel算子的清晰度评价
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            Tenengrad值
        """
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = sobel_x**2 + sobel_y**2
        return np.sum(tenengrad[tenengrad > np.percentile(tenengrad, 95)])
    
    def calculate_brenner_gradient(self, image: np.ndarray) -> float:
        """
        计算Brenner梯度 - 基于相邻像素差值的清晰度评价
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            Brenner梯度值
        """
        h, w = image.shape
        brenner = 0
        for i in range(h):
            for j in range(w-2):
                brenner += (int(image[i, j+2]) - int(image[i, j]))**2
        return brenner / (h * (w-2))
    
    def calculate_dark_channel_prior(self, image: np.ndarray, patch_size: int = 15) -> float:
        """
        计算暗通道先验值 - 基于何凯明暗通道先验算法
        
        Parameters
        ----------
        image : np.ndarray
            输入彩色图像 (BGR格式)
        patch_size : int
            局部区域大小
            
        Returns
        -------
        float
            暗通道先验值，越大表示雾越浓
        """
        if len(image.shape) == 3:
            # 对于彩色图像，取RGB三个通道的最小值
            min_channel = np.min(image, axis=2)
        else:
            min_channel = image
            
        # 计算暗通道
        kernel = np.ones((patch_size, patch_size), np.uint8)
        dark_channel = cv2.erode(min_channel, kernel)
        
        # 返回暗通道的平均值
        return np.mean(dark_channel) / 255.0
    
    def calculate_visibility_index(self, image: np.ndarray) -> float:
        """
        计算能见度指数 - 综合多个指标的能见度评价
        
        基于图像对比度、亮度分布和边缘信息的综合评价
        
        Parameters
        ----------
        image : np.ndarray
            输入灰度图像
            
        Returns
        -------
        float
            能见度指数，越大表示能见度越好
        """
        # 计算对比度
        contrast = np.std(image)
        
        # 计算边缘密度
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # 计算亮度分布熵
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # 归一化
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 综合指数（权重可调）
        visibility_index = 0.4 * contrast + 0.3 * edge_density * 1000 + 0.3 * entropy
        
        return visibility_index
    
    def analyze_single_image(self, image_path: str) -> Dict[str, float]:
        """
        分析单张图像的模糊程度
        
        Parameters
        ----------
        image_path : str
            图像文件路径
            
        Returns
        -------
        Dict[str, float]
            包含各种模糊程度指标的字典
        """
        # 读取图像
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图像
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 计算各种模糊程度指标
        metrics = {
            'laplacian_variance': self.calculate_laplacian_variance(image_gray),
            'gradient_magnitude': self.calculate_gradient_magnitude(image_gray),
            'sobel_variance': self.calculate_sobel_variance(image_gray),
            'tenengrad': self.calculate_tenengrad(image_gray),
            'brenner_gradient': self.calculate_brenner_gradient(image_gray),
            'dark_channel_prior': self.calculate_dark_channel_prior(image_bgr),
            'visibility_index': self.calculate_visibility_index(image_gray)
        }
        
        return metrics
    
    def process_image_sequence(self, image_folder: str) -> pd.DataFrame:
        """
        处理图像序列，分析时间序列的模糊程度变化
        
        Parameters
        ----------
        image_folder : str
            包含图像序列的文件夹路径
            
        Returns
        -------
        pd.DataFrame
            包含所有图像模糊程度指标的数据框
        """
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort()  # 按文件名排序
        
        results = []
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            print(f"处理图像 {i+1}/{len(image_files)}: {image_file}")
            
            try:
                metrics = self.analyze_single_image(image_path)
                metrics['filename'] = image_file
                metrics['frame_index'] = i
                results.append(metrics)
                
                # 更新内部存储
                for key, value in metrics.items():
                    if key in self.blur_metrics:
                        self.blur_metrics[key].append(value)
                        
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def create_blur_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        创建图像模糊程度数学模型
        
        Parameters
        ----------
        df : pd.DataFrame
            包含图像模糊程度指标的数据框
            
        Returns
        -------
        Dict[str, Any]
            模型参数和统计信息
        """
        # 选择主要指标
        feature_columns = ['laplacian_variance', 'gradient_magnitude', 'sobel_variance', 
                          'tenengrad', 'dark_channel_prior', 'visibility_index']
        
        # 标准化处理
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_columns])
        
        # 计算权重（基于方差贡献度）
        variances = np.var(features_scaled, axis=0)
        weights = variances / np.sum(variances)
        
        # 创建综合模糊度指数
        blur_index = np.dot(features_scaled, weights)
        
        # 将暗通道先验值反向（因为暗通道值越大表示雾越浓）
        dark_channel_weight_idx = feature_columns.index('dark_channel_prior')
        weights[dark_channel_weight_idx] *= -1
        
        model_params = {
            'feature_names': feature_columns,
            'weights': weights,
            'scaler': scaler,
            'statistics': {
                'mean_blur_index': np.mean(blur_index),
                'std_blur_index': np.std(blur_index),
                'min_blur_index': np.min(blur_index),
                'max_blur_index': np.max(blur_index)
            }
        }
        
        return model_params
    
    def visualize_blur_analysis(self, df: pd.DataFrame, save_path: str = None):
        """
        可视化模糊程度分析结果
        
        Parameters
        ----------
        df : pd.DataFrame
            包含分析结果的数据框
        save_path : str, optional
            保存图片的路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Image Blur Analysis Results', fontsize=16, fontweight='bold')
        
        # 定义指标和对应的图表标题
        metrics = [
            ('laplacian_variance', 'Laplacian Variance'),
            ('gradient_magnitude', 'Gradient Magnitude'), 
            ('sobel_variance', 'Sobel Variance'),
            ('tenengrad', 'Tenengrad'),
            ('dark_channel_prior', 'Dark Channel Prior'),
            ('visibility_index', 'Visibility Index')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            axes[row, col].plot(df['frame_index'], df[metric], 'b-', linewidth=2, markersize=4)
            axes[row, col].set_title(title, fontweight='bold')
            axes[row, col].set_xlabel('Frame Index')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)
            
            # 添加趋势线
            z = np.polyfit(df['frame_index'], df[metric], 1)
            p = np.poly1d(z)
            axes[row, col].plot(df['frame_index'], p(df['frame_index']), 'r--', alpha=0.8, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分析结果图已保存至: {save_path}")
        
        plt.show()
    
    def export_results(self, df: pd.DataFrame, model_params: Dict[str, Any], 
                      output_file: str = "blur_analysis_results.xlsx"):
        """
        导出分析结果到Excel文件
        
        Parameters
        ----------
        df : pd.DataFrame
            分析结果数据框
        model_params : Dict[str, Any]
            模型参数
        output_file : str
            输出文件名
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 导出原始数据
            df.to_excel(writer, sheet_name='原始数据', index=False)
            
            # 导出统计摘要
            summary_stats = df.describe()
            summary_stats.to_excel(writer, sheet_name='统计摘要')
            
            # 导出模型参数
            model_df = pd.DataFrame({
                'Feature': model_params['feature_names'],
                'Weight': model_params['weights']
            })
            model_df.to_excel(writer, sheet_name='模型参数', index=False)
            
            # 导出相关性矩阵
            feature_columns = ['laplacian_variance', 'gradient_magnitude', 'sobel_variance', 
                              'tenengrad', 'dark_channel_prior', 'visibility_index']
            correlation_matrix = df[feature_columns].corr()
            correlation_matrix.to_excel(writer, sheet_name='相关性矩阵')
        
        print(f"分析结果已导出至: {output_file}")


def generate_synthetic_fog_images(output_dir: str, num_images: int = 50):
    """
    生成合成的雾化图像序列用于测试
    
    Parameters
    ----------
    output_dir : str
        输出目录
    num_images : int
        生成图像数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建基础图像（模拟机场跑道场景）
    base_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加跑道和地面纹理
    base_image[200:280, :] = [100, 100, 100]  # 跑道
    base_image[240:260, ::20] = [255, 255, 255]  # 跑道标线
    
    # 添加建筑物和远景
    base_image[100:200, 100:200] = [80, 80, 120]  # 建筑物
    base_image[50:100, :] = [150, 180, 200]  # 天空
    
    for i in range(num_images):
        # 逐渐增加雾的浓度
        fog_intensity = i / num_images
        
        # 创建雾化效果
        fog_image = base_image.copy().astype(np.float32)
        
        # 添加高斯模糊模拟雾的散射效果
        blur_kernel_size = int(1 + fog_intensity * 20)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        
        fog_image = cv2.GaussianBlur(fog_image, (blur_kernel_size, blur_kernel_size), 0)
        
        # 增加亮度模拟雾的散射
        fog_image = fog_image + fog_intensity * 100
        
        # 降低对比度
        fog_image = fog_image * (1 - fog_intensity * 0.5)
        
        # 限制像素值范围
        fog_image = np.clip(fog_image, 0, 255).astype(np.uint8)
        
        # 保存图像
        filename = f"frame_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), fog_image)
    
    print(f"已生成 {num_images} 张合成雾化图像到目录: {output_dir}")


def main():
    """
    主函数 - 演示图像模糊程度分析的完整流程
    """
    print("=" * 60)
    print("基于大雾背景视频学习的图像模糊程度分析系统")
    print("=" * 60)
    
    # 创建分析器实例
    analyzer = FogBlurAnalyzer()
    
    # 生成测试数据（如果没有真实数据）
    test_images_dir = "test_fog_images"
    if not os.path.exists(test_images_dir):
        print("生成测试用雾化图像序列...")
        generate_synthetic_fog_images(test_images_dir, num_images=30)
    
    # 处理图像序列
    print("分析图像序列中的模糊程度...")
    results_df = analyzer.process_image_sequence(test_images_dir)
    
    # 创建数学模型
    print("建立图像模糊程度数学模型...")
    model_params = analyzer.create_blur_model(results_df)
    
    # 打印模型参数
    print("\n模型参数:")
    print("-" * 40)
    for i, (feature, weight) in enumerate(zip(model_params['feature_names'], 
                                            model_params['weights'])):
        print(f"{feature:20s}: {weight:8.4f}")
    
    print("\n模型统计信息:")
    print("-" * 40)
    stats = model_params['statistics']
    for key, value in stats.items():
        print(f"{key:20s}: {value:8.4f}")
    
    # 可视化结果
    print("\n生成可视化结果...")
    analyzer.visualize_blur_analysis(results_df, save_path="blur_analysis_results.png")
    
    # 导出结果
    print("\n导出分析结果...")
    analyzer.export_results(results_df, model_params)
    
    print("\n分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main() 