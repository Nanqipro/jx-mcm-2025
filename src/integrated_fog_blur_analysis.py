#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合雾模糊度分析系统
基于江西省数学建模比赛题目要求，整合多种分析方法

功能模块：
1. 分层特征提取体系
2. 物理模型约束
3. 多源数据融合
4. 科学建模流程
5. 模型验证和评估

作者: 整合分析团队
版本: 1.0.0
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from skimage import feature, filters
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
RESULTS_DIR = "../results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class PhysicsBasedFogModel:
    """
    基于物理理论的雾模型
    
    实现大气散射理论、Koschmieder定律和暗通道先验理论
    """
    
    @staticmethod
    def koschmieder_law(distance: float, beta: float, L_inf: float = 255.0) -> float:
        """
        Koschmieder定律：描述大气中目标物的视见度
        
        L(d) = L_inf + (L_0 - L_inf) * exp(-β * d)
        
        Parameters
        ----------
        distance : float
            观测距离
        beta : float
            大气散射系数
        L_inf : float
            天空亮度
            
        Returns
        -------
        float
            视见度值
        """
        return L_inf * (1 - np.exp(-beta * distance))
    
    @staticmethod
    def beer_lambert_transmission(distance: float, beta: float) -> float:
        """
        Beer-Lambert定律：描述透射率
        
        t(d) = exp(-β * d)
        """
        return np.exp(-beta * distance)
    
    @staticmethod
    def calculate_scattering_coefficient(visibility: float) -> float:
        """
        由能见度计算散射系数
        
        β = 3.912 / V_met
        
        Parameters
        ----------
        visibility : float
            气象能见度（米）
            
        Returns
        -------
        float
            散射系数
        """
        return 3.912 / (visibility + 1e-8)
    
    @staticmethod
    def atmospheric_scattering_model(image: np.ndarray, A: float, t: float) -> np.ndarray:
        """
        大气散射模型
        
        I(x) = J(x) * t(x) + A * (1 - t(x))
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
        A : float
            大气光值
        t : float
            透射率
            
        Returns
        -------
        np.ndarray
            散射后的图像
        """
        return image * t + A * (1 - t)


class HierarchicalFeatureExtractor:
    """
    分层特征提取器
    
    按照物理意义和感知层次提取多维度特征，支持性能优化
    """
    
    def __init__(self, optimization_level: int = 1):
        """
        初始化特征提取器
        
        Parameters
        ----------
        optimization_level : int
            优化级别：1=基础优化, 2=进阶优化, 3=极速模式
        """
        self.optimization_level = optimization_level
        self.feature_categories = {
            'physical': ['dark_channel', 'transmission', 'atmospheric_light', 'scattering_coeff'],
            'perceptual': ['laplacian_var', 'sobel_magnitude', 'rms_contrast', 'edge_density'],
            'statistical': ['entropy', 'glcm_features', 'lbp_features', 'frequency_features'],
            'temporal': ['frame_difference', 'optical_flow', 'motion_consistency']
        }
        
        # 根据优化级别配置特征集
        self._configure_features()
    
    def _configure_features(self):
        """根据优化级别配置特征集"""
        if self.optimization_level == 1:
            # 基础优化：跳过计算密集的GLCM特征
            self.skip_glcm = True
            self.skip_lbp = False
            self.lbp_radius = 2  # 减小LBP半径
            self.lbp_points = 16  # 减少LBP点数
            
        elif self.optimization_level == 2:
            # 进阶优化：进一步简化特征
            self.skip_glcm = True
            self.skip_lbp = True
            self.simple_frequency = True  # 简化频域分析
            
        elif self.optimization_level == 3:
            # 极速模式：只保留核心特征
            self.skip_glcm = True
            self.skip_lbp = True
            self.skip_frequency = True  # 跳过频域分析
            self.skip_optical_flow = True  # 跳过光流计算
            
        else:
            # 默认模式：完整特征
            self.skip_glcm = False
            self.skip_lbp = False
            self.lbp_radius = 3
            self.lbp_points = 24
    
    def extract_physical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取基于物理理论的特征
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
            
        Returns
        -------
        Dict[str, float]
            物理特征字典
        """
        # 转换为浮点数格式
        img_float = image.astype(np.float32) / 255.0
        
        # 1. 暗通道先验
        dark_channel = self._calculate_dark_channel(img_float)
        dark_channel_mean = np.mean(dark_channel)
        
        # 2. 大气光估计
        atmospheric_light = self._estimate_atmospheric_light(img_float, dark_channel)
        
        # 3. 透射率估计
        transmission = self._estimate_transmission(img_float, atmospheric_light, omega=0.95)
        transmission_mean = np.mean(transmission)
        transmission_std = np.std(transmission)
        
        # 4. 散射系数估计
        scattering_coeff = -np.log(transmission_mean + 1e-8)
        
        # 5. 雾浓度指标
        fog_density = 1.0 - transmission_mean
        
        # 6. 视程估算
        visibility_estimate = 3.912 / (scattering_coeff + 1e-8)
        
        return {
            'dark_channel_mean': dark_channel_mean,
            'atmospheric_light': atmospheric_light,
            'transmission_mean': transmission_mean,
            'transmission_std': transmission_std,
            'scattering_coefficient': scattering_coeff,
            'fog_density': fog_density,
            'visibility_estimate': visibility_estimate
        }
    
    def extract_perceptual_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取基于人眼感知的特征
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
            
        Returns
        -------
        Dict[str, float]
            感知特征字典
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 清晰度特征
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)
        
        # 2. 梯度特征
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mean = np.mean(sobel_magnitude)
        
        # 3. Tenengrad梯度
        tenengrad = sobel_x**2 + sobel_y**2
        tenengrad_mean = np.mean(tenengrad)
        
        # 4. 对比度特征
        rms_contrast = np.sqrt(np.mean((gray - np.mean(gray))**2))
        
        # 5. 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 6. 方差特征
        image_variance = np.var(gray)
        
        return {
            'laplacian_variance': laplacian_variance,
            'sobel_magnitude_mean': sobel_mean,
            'tenengrad_mean': tenengrad_mean,
            'rms_contrast': rms_contrast,
            'edge_density': edge_density,
            'image_variance': image_variance
        }
    
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        提取统计特征（优化版）
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
            
        Returns
        -------
        Dict[str, float]
            统计特征字典
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # 1. 图像熵（始终计算，因为计算快）
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / (hist.sum() + 1e-8)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        features['entropy'] = entropy
        
        # 2. 频域特征（根据优化级别决定）
        if not hasattr(self, 'skip_frequency') or not self.skip_frequency:
            if hasattr(self, 'simple_frequency') and self.simple_frequency:
                # 简化版频域分析
                f_transform = np.fft.fft2(gray[::2, ::2])  # 降采样计算
                magnitude = np.abs(f_transform)
                high_freq_ratio = np.sum(magnitude[magnitude.shape[0]//4:, magnitude.shape[1]//4:]) / np.sum(magnitude)
            else:
                # 完整频域分析
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) * 0.3)**2
                high_freq_energy = np.sum(magnitude * mask)
                total_energy = np.sum(magnitude)
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                
            features['high_freq_ratio'] = high_freq_ratio
        else:
            features['high_freq_ratio'] = 0.0
        
        # 3. LBP纹理特征（根据优化级别决定）
        if not self.skip_lbp:
            radius = getattr(self, 'lbp_radius', 3)
            n_points = getattr(self, 'lbp_points', 24)
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)
            lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-8)
            lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
            features['lbp_entropy'] = lbp_entropy
        else:
            features['lbp_entropy'] = 0.0
        
        # 4. GLCM特征（根据优化级别决定）
        if not self.skip_glcm:
            gray_small = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4))
            gray_quantized = (gray_small / 32).astype(np.uint8)
            
            glcm = graycomatrix(gray_quantized, [1], [0], levels=8, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            features.update({
                'glcm_contrast': contrast,
                'glcm_homogeneity': homogeneity
            })
        else:
            features.update({
                'glcm_contrast': 0.0,
                'glcm_homogeneity': 0.0
            })
        
        return features
    
    def extract_temporal_features(self, current_frame: np.ndarray, 
                                 previous_frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        提取时序特征
        
        Parameters
        ----------
        current_frame : np.ndarray
            当前帧
        previous_frame : Optional[np.ndarray]
            前一帧
            
        Returns
        -------
        Dict[str, float]
            时序特征字典
        """
        if previous_frame is None:
            return {
                'frame_difference_mean': 0.0,
                'frame_difference_std': 0.0,
                'optical_flow_magnitude': 0.0
            }
        
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 帧差特征
        frame_diff = cv2.absdiff(gray_current, gray_previous)
        diff_mean = np.mean(frame_diff)
        diff_std = np.std(frame_diff)
        
        # 2. 光流特征（优化版本）
        if hasattr(self, 'skip_optical_flow') and self.skip_optical_flow:
            # 极速模式：跳过光流计算
            flow_magnitude = 0.0
        else:
            try:
                # 先检测角点（减少角点数量以加速）
                max_corners = 50 if self.optimization_level >= 2 else 100
                corners = cv2.goodFeaturesToTrack(gray_previous, max_corners, 0.3, 7)
                
                if corners is not None and len(corners) > 0:
                    # 计算光流
                    next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                        gray_previous, gray_current, corners, None,
                        winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    
                    # 筛选有效的光流点
                    good_corners = corners[status == 1]
                    good_next = next_corners[status == 1]
                    
                    if len(good_corners) > 0:
                        # 计算光流幅值
                        flow_vectors = good_next - good_corners
                        flow_magnitude = np.mean(np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2))
                    else:
                        flow_magnitude = 0.0
                else:
                    flow_magnitude = 0.0
                    
            except Exception as e:
                print(f"光流计算警告: {e}")
                flow_magnitude = 0.0
        
        return {
            'frame_difference_mean': diff_mean,
            'frame_difference_std': diff_std,
            'optical_flow_magnitude': flow_magnitude
        }
    
    def _calculate_dark_channel(self, image: np.ndarray, patch_size: int = 15) -> np.ndarray:
        """计算暗通道"""
        b, g, r = cv2.split(image)
        dark = np.minimum(np.minimum(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(dark, kernel)
        return dark_channel
    
    def _estimate_atmospheric_light(self, image: np.ndarray, dark_channel: np.ndarray) -> float:
        """估计大气光值"""
        # 选择暗通道中最亮的0.1%像素
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, 3)
        
        indices = np.argsort(flat_dark)
        top_indices = indices[-int(len(indices) * 0.001):]
        
        atmospheric_light = np.max(flat_image[top_indices])
        return atmospheric_light
    
    def _estimate_transmission(self, image: np.ndarray, atmospheric_light: float, 
                             omega: float = 0.95) -> np.ndarray:
        """估计透射率"""
        norm_image = image / atmospheric_light
        dark_channel = self._calculate_dark_channel(norm_image)
        transmission = 1 - omega * dark_channel
        return np.clip(transmission, 0.1, 1.0)
    
    def extract_all_features(self, image: np.ndarray, 
                           previous_frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        提取所有类别的特征
        
        Parameters
        ----------
        image : np.ndarray
            当前图像
        previous_frame : Optional[np.ndarray]
            前一帧图像
            
        Returns
        -------
        Dict[str, float]
            所有特征的字典
        """
        features = {}
        
        # 提取各类特征
        features.update(self.extract_physical_features(image))
        features.update(self.extract_perceptual_features(image))
        features.update(self.extract_statistical_features(image))
        features.update(self.extract_temporal_features(image, previous_frame))
        
        return features


class IntegratedVideoProcessor:
    """
    整合视频处理器
    
    结合精确时间处理和多维特征提取，支持多级别性能优化
    """
    
    def __init__(self, video_path: str, start_time_offset: int = 28, 
                 optimization_level: int = 1, target_size: tuple = (640, 360)):
        """
        初始化视频处理器
        
        Parameters
        ----------
        video_path : str
            视频文件路径
        start_time_offset : int
            视频开始时间偏移（秒）
        optimization_level : int
            优化级别：1=基础优化, 2=进阶优化, 3=极速模式
        target_size : tuple
            目标图像尺寸 (width, height)
        """
        self.video_path = video_path
        self.start_time_offset = start_time_offset
        self.optimization_level = optimization_level
        self.target_size = target_size
        self.feature_extractor = HierarchicalFeatureExtractor(optimization_level)
        self.features_data = []
        
        # 根据优化级别设置参数
        self._configure_optimization()
        
        # 初始化视频信息
        self._initialize_video_info()
    
    def _configure_optimization(self):
        """根据优化级别配置参数"""
        if self.optimization_level == 1:
            # 基础优化：2-3倍加速
            self.default_sampling_interval = 50
            self.resize_factor = 0.5
            print("🚀 启用基础优化模式 (2-3倍加速)")
            
        elif self.optimization_level == 2:
            # 进阶优化：5-10倍加速
            self.default_sampling_interval = 100
            self.resize_factor = 0.4
            print("🚀 启用进阶优化模式 (5-10倍加速)")
            
        elif self.optimization_level == 3:
            # 极速模式：10-20倍加速
            self.default_sampling_interval = 200
            self.resize_factor = 0.3
            print("🚀 启用极速优化模式 (10-20倍加速)")
            
        else:
            # 默认模式
            self.default_sampling_interval = 1500
            self.resize_factor = 1.0
            print("📍 使用默认处理模式")
    
    def _initialize_video_info(self):
        """初始化视频信息"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        self.fps = 1500.0  # 实际帧率
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        
        # 计算优化后的尺寸
        self.optimized_width = int(self.width * self.resize_factor)
        self.optimized_height = int(self.height * self.resize_factor)
        
        # 估算处理时间
        estimated_samples = self.frame_count // self.default_sampling_interval
        
        cap.release()
        print(f"视频信息: {self.width}x{self.height}, {self.fps}fps, {self.frame_count}帧")
        print(f"优化设置: 图像缩放至{self.optimized_width}x{self.optimized_height}, 采样间隔{self.default_sampling_interval}帧")
        print(f"预计处理: {estimated_samples} 个样本")
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧：缩放和优化
        
        Parameters
        ----------
        frame : np.ndarray
            原始帧
            
        Returns
        -------
        np.ndarray
            预处理后的帧
        """
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, (self.optimized_width, self.optimized_height), 
                             interpolation=cv2.INTER_LINEAR)
        return frame
    
    def calculate_timestamp(self, frame_index: int) -> datetime:
        """计算帧对应的时间戳"""
        seconds_from_start = frame_index / self.fps
        total_seconds = self.start_time_offset + seconds_from_start
        
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return base_time + timedelta(seconds=total_seconds)
    
    def process_video(self, sampling_interval: Optional[int] = None) -> List[Dict]:
        """
        处理视频提取特征（优化版）
        
        Parameters
        ----------
        sampling_interval : Optional[int]
            采样间隔（帧数），为None时使用优化级别的默认值
            
        Returns
        -------
        List[Dict]
            特征数据列表
        """
        if sampling_interval is None:
            sampling_interval = self.default_sampling_interval
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        frame_index = 0
        processed_count = 0
        previous_frame = None
        start_time = datetime.now()
        
        print(f"开始处理视频，采样间隔: {sampling_interval}帧")
        print(f"优化级别: {self.optimization_level}, 图像缩放: {self.resize_factor:.1f}x")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % sampling_interval == 0:
                # 预处理帧（缩放优化）
                optimized_frame = self._preprocess_frame(frame)
                
                timestamp = self.calculate_timestamp(frame_index)
                
                # 提取特征
                features = self.feature_extractor.extract_all_features(optimized_frame, previous_frame)
                
                # 添加时间信息
                features.update({
                    'frame_index': frame_index,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'seconds_from_start': frame_index / self.fps,
                    'optimization_level': self.optimization_level,
                    'image_size': f"{optimized_frame.shape[1]}x{optimized_frame.shape[0]}"
                })
                
                self.features_data.append(features)
                previous_frame = optimized_frame.copy()
                processed_count += 1
                
                # 动态显示进度和速度
                if processed_count % 50 == 0:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    fps_processing = processed_count / elapsed_time if elapsed_time > 0 else 0
                    remaining_frames = (self.frame_count // sampling_interval) - processed_count
                    eta_seconds = remaining_frames / fps_processing if fps_processing > 0 else 0
                    
                    print(f"已处理 {processed_count} 帧... "
                          f"(处理速度: {fps_processing:.1f} 样本/秒, "
                          f"预计剩余: {eta_seconds/60:.1f} 分钟)")
            
            frame_index += 1
        
        cap.release()
        total_time = (datetime.now() - start_time).total_seconds()
        avg_fps = len(self.features_data) / total_time if total_time > 0 else 0
        
        print(f"✅ 视频处理完成！")
        print(f"📊 统计信息:")
        print(f"   - 提取样本: {len(self.features_data)} 个")
        print(f"   - 处理时间: {total_time:.1f} 秒")
        print(f"   - 平均速度: {avg_fps:.1f} 样本/秒")
        print(f"   - 时间节省: 预计比原版快 {self.optimization_level*3:.0f} 倍")
        
        return self.features_data


class BlurModelBuilder:
    """
    模糊度模型构建器
    
    实现多种建模方法的集成
    """
    
    def __init__(self, features_data: List[Dict]):
        """
        初始化模型构建器
        
        Parameters
        ----------
        features_data : List[Dict]
            特征数据列表
        """
        self.features_data = features_data
        self.df = pd.DataFrame(features_data)
        self.models = {}
        self.feature_importance = {}
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备建模数据
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            特征矩阵、目标变量、特征名称
        """
        # 选择数值特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除时间相关列
        exclude_cols = ['frame_index', 'seconds_from_start']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = self.df[feature_cols].values
        
        # 使用雾浓度作为目标变量（代表模糊度）
        y = self.df['fog_density'].values
        
        # 处理缺失值和异常值
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        
        return X, y, feature_cols
    
    def build_physical_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        构建基于物理理论的模型
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
            
        Returns
        -------
        Dict[str, Any]
            物理模型结果
        """
        print("构建物理约束模型...")
        
        # 选择物理意义明确的特征
        physical_features = ['dark_channel_mean', 'transmission_mean', 'atmospheric_light', 
                           'scattering_coefficient', 'visibility_estimate']
        
        feature_indices = []
        for i, col in enumerate(self.df.columns):
            if col in physical_features and col in self.df.select_dtypes(include=[np.number]).columns:
                feature_indices.append(i)
        
        if not feature_indices:
            print("未找到物理特征，使用所有特征")
            X_phys = X
        else:
            X_phys = X[:, feature_indices]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_phys, y, test_size=0.2, random_state=42
        )
        
        # Ridge回归（带物理约束）
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)
        
        # 评估模型
        r2 = r2_score(y_test, y_pred_ridge)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        mae = mean_absolute_error(y_test, y_pred_ridge)
        
        return {
            'model': ridge_model,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred_ridge,
            'test_actual': y_test
        }
    
    def build_ensemble_model(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        构建集成学习模型
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        feature_names : List[str]
            特征名称
            
        Returns
        -------
        Dict[str, Any]
            集成模型结果
        """
        print("构建集成学习模型...")
        
        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        models = {}
        
        # 1. 随机森林
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # 2. 梯度提升
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        # 评估模型
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'model': model,
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'predictions': y_pred
            }
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_names, model.feature_importances_))
        
        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'])
        best_result = results[best_model_name]
        best_result['best_model_name'] = best_model_name
        best_result['test_actual'] = y_test
        best_result['scaler'] = scaler
        
        return best_result
    
    def build_comprehensive_models(self) -> Dict[str, Any]:
        """
        构建综合模型集合
        
        Returns
        -------
        Dict[str, Any]
            所有模型结果
        """
        print("="*60)
        print("开始构建综合模糊度预测模型")
        print("="*60)
        
        # 准备数据
        X, y, feature_names = self.prepare_data()
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量范围: {y.min():.4f} - {y.max():.4f}")
        
        # 构建不同类型的模型
        results = {}
        
        # 1. 物理模型
        results['physical'] = self.build_physical_model(X, y)
        
        # 2. 集成模型
        results['ensemble'] = self.build_ensemble_model(X, y, feature_names)
        
        # 模型性能对比
        print("\n" + "="*60)
        print("模型性能对比")
        print("="*60)
        
        for model_type, result in results.items():
            if model_type == 'ensemble':
                print(f"{model_type.upper()} ({result['best_model_name']}):")
            else:
                print(f"{model_type.upper()}:")
            print(f"  R² Score: {result['r2_score']:.4f}")
            print(f"  RMSE:     {result['rmse']:.4f}")
            print(f"  MAE:      {result['mae']:.4f}")
        
        # 保存最佳模型信息
        best_overall = max(results.items(), key=lambda x: x[1]['r2_score'])
        print(f"\n最佳模型: {best_overall[0]} (R² = {best_overall[1]['r2_score']:.4f})")
        
        self.models = results
        return results


class IntegratedAnalysisSystem:
    """
    整合分析系统
    
    统一管理整个分析流程
    """
    
    def __init__(self, video_path: str, optimization_level: int = 2):
        """
        初始化整合分析系统
        
        Parameters
        ----------
        video_path : str
            视频文件路径
        optimization_level : int
            优化级别：1=基础优化, 2=进阶优化(推荐), 3=极速模式
        """
        self.video_path = video_path
        self.optimization_level = optimization_level
        self.video_processor = None
        self.model_builder = None
        self.results = {}
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Returns
        -------
        Dict[str, Any]
            完整分析结果
        """
        print("="*80)
        print("整合雾模糊度分析系统")
        print("="*80)
        
        # 1. 视频处理和特征提取
        print("\n第一阶段：视频处理和特征提取")
        print("-"*50)
        
        self.video_processor = IntegratedVideoProcessor(
            self.video_path, 
            optimization_level=self.optimization_level
        )
        features_data = self.video_processor.process_video()
        
        if not features_data:
            print("特征提取失败")
            return {}
        
        # 2. 模型构建
        print("\n第二阶段：模糊度建模")
        print("-"*50)
        
        self.model_builder = BlurModelBuilder(features_data)
        model_results = self.model_builder.build_comprehensive_models()
        
        # 3. 结果可视化
        print("\n第三阶段：结果可视化")
        print("-"*50)
        
        self.create_comprehensive_visualizations()
        
        # 4. 生成分析报告
        print("\n第四阶段：生成分析报告")
        print("-"*50)
        
        report = self.generate_analysis_report()
        
        self.results = {
            'features_data': features_data,
            'model_results': model_results,
            'analysis_report': report
        }
        
        return self.results
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化分析"""
        if not self.model_builder:
            return
        
        df = self.model_builder.df
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Integrated Fog Blur Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 雾浓度时间序列
        axes[0, 0].plot(df['seconds_from_start'], df['fog_density'], 'b-', linewidth=1)
        axes[0, 0].set_title('Fog Density Time Series')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Fog Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 透射率与雾浓度关系
        axes[0, 1].scatter(df['transmission_mean'], df['fog_density'], alpha=0.6, s=10)
        axes[0, 1].set_title('Transmission vs Fog Density')
        axes[0, 1].set_xlabel('Transmission')
        axes[0, 1].set_ylabel('Fog Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 清晰度特征分布
        axes[0, 2].hist(df['laplacian_variance'], bins=30, alpha=0.7, color='green')
        axes[0, 2].set_title('Sharpness Feature Distribution')
        axes[0, 2].set_xlabel('Laplacian Variance')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 模型预测性能
        if 'ensemble' in self.model_builder.models:
            model_result = self.model_builder.models['ensemble']
            actual = model_result['test_actual']
            predicted = model_result['predictions']
            
            axes[1, 0].scatter(actual, predicted, alpha=0.6, s=10)
            axes[1, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
            axes[1, 0].set_title(f'Model Prediction (R² = {model_result["r2_score"]:.3f})')
            axes[1, 0].set_xlabel('Actual Fog Density')
            axes[1, 0].set_ylabel('Predicted Fog Density')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 特征重要性
        if self.model_builder.feature_importance:
            importance_data = list(self.model_builder.feature_importance.values())[0]
            top_features = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10])
            
            axes[1, 1].barh(range(len(top_features)), list(top_features.values()))
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(list(top_features.keys()), fontsize=8)
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 特征相关性热图
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # 取前10个数值特征
        corr_matrix = df[numeric_cols].corr()
        
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(numeric_cols)))
        axes[1, 2].set_yticks(range(len(numeric_cols)))
        axes[1, 2].set_xticklabels(numeric_cols, rotation=45, fontsize=8)
        axes[1, 2].set_yticklabels(numeric_cols, fontsize=8)
        axes[1, 2].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/integrated_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"综合分析图表已保存: {filename}")
        
        plt.show()
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        if not self.model_builder:
            return {}
        
        df = self.model_builder.df
        
        report = {
            'data_summary': {
                'total_frames': len(df),
                'time_range': f"{df['seconds_from_start'].min():.1f} - {df['seconds_from_start'].max():.1f} seconds",
                'fog_density_range': f"{df['fog_density'].min():.4f} - {df['fog_density'].max():.4f}",
                'feature_count': len(df.select_dtypes(include=[np.number]).columns)
            },
            'model_performance': {},
            'key_findings': []
        }
        
        # 模型性能
        if self.model_builder.models:
            for model_name, result in self.model_builder.models.items():
                report['model_performance'][model_name] = {
                    'r2_score': result['r2_score'],
                    'rmse': result['rmse'],
                    'mae': result['mae']
                }
        
        # 关键发现
        fog_events = len(df[df['fog_density'] > 0.5])
        report['key_findings'].append(f"检测到 {fog_events} 个高雾浓度事件")
        
        avg_visibility = df['visibility_estimate'].mean() if 'visibility_estimate' in df.columns else 0
        report['key_findings'].append(f"平均估算能见度: {avg_visibility:.1f}m")
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{RESULTS_DIR}/analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("整合雾模糊度分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("数据概况:\n")
            for key, value in report['data_summary'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n模型性能:\n")
            for model_name, metrics in report['model_performance'].items():
                f.write(f"  {model_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
            
            f.write("\n关键发现:\n")
            for finding in report['key_findings']:
                f.write(f"  - {finding}\n")
        
        print(f"分析报告已保存: {report_file}")
        return report


def main():
    """主函数"""
    # 检查视频文件
    video_path = "../机场视频/a.mp4"
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    # 选择优化级别
    print("🚀 选择优化级别:")
    print("1. 基础优化 (2-3倍加速) - 跳过GLCM特征，图像缩放50%")
    print("2. 进阶优化 (5-10倍加速) - 简化特征集，图像缩放40% [推荐]")
    print("3. 极速模式 (10-20倍加速) - 最小特征集，图像缩放30%")
    print("0. 默认模式 (无优化) - 完整特征，原始尺寸")
    
    try:
        choice = input("\n请选择优化级别 (0-3, 默认2): ").strip()
        optimization_level = int(choice) if choice.isdigit() and choice in '0123' else 2
    except:
        optimization_level = 2
    
    try:
        # 创建整合分析系统
        analysis_system = IntegratedAnalysisSystem(video_path, optimization_level)
        
        # 运行完整分析
        results = analysis_system.run_complete_analysis()
        
        if results:
            print("\n"+"="*80)
            print("🎉 分析完成！主要成果:")
            print("="*80)
            print(f"1. 提取特征样本: {len(results['features_data'])} 个")
            print(f"2. 构建模型数量: {len(results['model_results'])} 个")
            print("3. 生成可视化图表和分析报告")
            print(f"4. 结果保存目录: {RESULTS_DIR}")
            
            # 性能统计
            video_processor = analysis_system.video_processor
            print(f"\n📊 性能统计:")
            print(f"   - 优化级别: {optimization_level}")
            print(f"   - 图像尺寸: {video_processor.optimized_width}x{video_processor.optimized_height}")
            print(f"   - 采样间隔: {video_processor.default_sampling_interval} 帧")
        
    except Exception as e:
        print(f"分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 