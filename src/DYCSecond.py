#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机场能见度回归建模系统
基于大雾背景视频学习的能见度回归建模

本系统整合了气象数据分析、视频图像处理、机器学习建模等功能，
用于实现基于多源数据的机场能见度智能预测。

Author: DYC
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.optimize import curve_fit, minimize
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置matplotlib中文显示（注意：图表标签仍使用英文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AirportVisibilityAnalyzer:
    """
    机场能见度分析器
    
    整合多源数据进行机场能见度分析和预测的主要类
    """
    
    def __init__(self, data_dir: str = "../AMOS20200313/", video_file: str = "../机场视频/a.mp4"):
        """
        初始化分析器
        
        Parameters
        ----------
        data_dir : str
            气象数据文件目录路径
        video_file : str
            机场视频文件路径
        """
        self.data_dir = data_dir
        self.video_file = video_file
        self.vis_data = None
        self.ptu_data = None  
        self.wind_data = None
        self.merged_data = None
        self.features_data = None
        self.time_series_data = None
        self.models = {}
        
        print("机场能见度分析器初始化完成！")
    
    def check_files(self) -> Tuple[List[str], List[str]]:
        """
        检查所有数据文件是否存在
        
        Returns
        -------
        Tuple[List[str], List[str]]
            存在的文件列表和缺失的文件列表
        """
        files_info = {
            os.path.join(self.data_dir, "VIS_R06_12.his"): "能见度数据文件",
            os.path.join(self.data_dir, "PTU_R06_12.his"): "气象数据文件", 
            os.path.join(self.data_dir, "WIND_R06_12.his"): "风速数据文件",
            self.video_file: "机场视频文件(可选)"
        }
        
        print("=" * 50)
        print("数据文件检查结果:")
        print("=" * 50)
        
        existing_files = []
        missing_files = []
        
        for filename, description in files_info.items():
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"✓ {os.path.basename(filename):20s} - {description} ({file_size:,} bytes)")
                existing_files.append(filename)
            else:
                print(f"✗ {os.path.basename(filename):20s} - {description} (文件不存在)")
                missing_files.append(filename)
        
        print(f"\n存在的文件: {len(existing_files)} 个")
        print(f"缺失的文件: {len(missing_files)} 个")
        
        return existing_files, missing_files

    def load_vis_data(self, filename: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        加载能见度数据
        
        Parameters
        ----------
        filename : Optional[str]
            VIS文件路径，默认使用标准路径
            
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            解析后的能见度数据列表
        """
        if filename is None:
            filename = os.path.join(self.data_dir, "VIS_R06_12.his")
            
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在解析能见度数据: {filename}")
        
        try:
            vis_records = []
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"文件总行数: {len(lines)}")
            
            # 跳过前两行标题，从第三行开始解析数据
            processed_count = 0
            for line_no, line in enumerate(lines[2:], start=3):
                if line.strip():
                    try:
                        # VIS文件使用Tab分隔符
                        parts = line.strip().split('\t')
                        
                        if len(parts) >= 23:  # 确保有足够的列
                            # 解析时间戳 - 使用LOCALDATE (BEIJING)，即第2列（索引1）
                            timestamp_str = parts[1]  # LOCALDATE (BEIJING)
                            try:
                                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                continue
                            
                            # 提取能见度值 - 根据文件格式，使用VIS_RAW列（第23列，索引22）
                            visibility = None
                            
                            # 尝试从VIS_RAW列获取（第23列）
                            if len(parts) > 22 and parts[22] and parts[22].strip() != '':
                                try:
                                    vis_raw_str = parts[22].strip()
                                    # 去掉可能的空格
                                    if vis_raw_str.replace('.', '').replace('-', '').isdigit():
                                        visibility = float(vis_raw_str)
                                except (ValueError, IndexError):
                                    pass
                            
                            # 如果VIS_RAW没有数据，尝试从MOR_RAW列获取（第22列，索引21）
                            if visibility is None and len(parts) > 21 and parts[21] and parts[21].strip() != '':
                                try:
                                    mor_raw_str = parts[21].strip()
                                    if mor_raw_str.replace('.', '').replace('-', '').isdigit():
                                        visibility = float(mor_raw_str)
                                except (ValueError, IndexError):
                                    pass
                            
                            # 如果还是没有，尝试从其他可能的能见度列获取
                            if visibility is None:
                                # 尝试VIS1K列（第17列，索引16）
                                for col_idx in [16, 17, 18]:  # VIS1K, VIS1A, VIS10A
                                    if col_idx < len(parts) and parts[col_idx] and parts[col_idx].strip() != '':
                                        try:
                                            vis_str = parts[col_idx].strip()
                                            if vis_str.replace('.', '').replace('-', '').isdigit():
                                                visibility = float(vis_str)
                                                break
                                        except (ValueError, IndexError):
                                            continue
                            
                            if visibility is not None and visibility > 0:
                                vis_records.append({
                                    'timestamp': timestamp,
                                    'visibility': visibility,
                                    'raw_data': parts
                                })
                                processed_count += 1
                            
                    except (ValueError, IndexError) as e:
                        continue
            
            print(f"成功解析 {len(vis_records)} 条能见度记录")
            self.vis_data = vis_records
            return vis_records
            
        except Exception as e:
            print(f"解析能见度数据时出错: {e}")
            return None

    def load_ptu_data(self, filename: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        加载气象数据（压力、温度、湿度）
        
        Parameters
        ----------
        filename : Optional[str]
            PTU文件路径，默认使用标准路径
            
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            解析后的气象数据列表
        """
        if filename is None:
            filename = os.path.join(self.data_dir, "PTU_R06_12.his")
            
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在解析气象数据: {filename}")
        
        try:
            ptu_records = []
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 从第三行开始解析数据
            for line in lines[2:]:
                if line.strip():
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 16:
                            timestamp_str = parts[1]  # 北京时间
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            # 提取气象参数
                            pressure = float(parts[3]) if parts[3] != '' else 0.0
                            temperature = float(parts[13]) if parts[13] != '' else 0.0
                            humidity = float(parts[14]) if parts[14] != '' else 0.0
                            dewpoint = float(parts[15]) if parts[15] != '' else 0.0
                            
                            ptu_records.append({
                                'timestamp': timestamp,
                                'pressure': pressure,
                                'temperature': temperature,
                                'humidity': humidity,
                                'dewpoint': dewpoint
                            })
                    except (ValueError, IndexError):
                        continue
            
            print(f"成功解析 {len(ptu_records)} 条气象记录")
            self.ptu_data = ptu_records
            return ptu_records
            
        except Exception as e:
            print(f"解析气象数据时出错: {e}")
            return None

    def load_wind_data(self, filename: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        加载风速数据
        
        Parameters
        ----------
        filename : Optional[str]
            WIND文件路径，默认使用标准路径
            
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            解析后的风速数据列表
        """
        if filename is None:
            filename = os.path.join(self.data_dir, "WIND_R06_12.his")
            
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在解析风速数据: {filename}")
        
        try:
            wind_records = []
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 从第三行开始解析数据
            for line in lines[2:]:
                if line.strip():
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 20:
                            timestamp_str = parts[1]  # 北京时间
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            # 提取风速和风向数据
                            wind_speed = float(parts[5]) if parts[5] != '' else 0.0
                            wind_direction = float(parts[12]) if parts[12] != '' else 0.0
                            vertical_wind = float(parts[22]) if len(parts) > 22 and parts[22] != '' else 0.0
                            
                            wind_records.append({
                                'timestamp': timestamp,
                                'wind_speed': wind_speed,
                                'wind_direction': wind_direction,
                                'vertical_wind': vertical_wind
                            })
                    except (ValueError, IndexError):
                        continue
            
            print(f"成功解析 {len(wind_records)} 条风速记录")
            self.wind_data = wind_records
            return wind_records
            
        except Exception as e:
            print(f"解析风速数据时出错: {e}")
            return None

    def merge_weather_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        合并所有气象数据
        
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            合并后的数据列表
        """
        if not all([self.vis_data, self.ptu_data, self.wind_data]):
            print("错误: 缺少必要的数据文件")
            return None
        
        print("正在合并气象数据...")
        
        # 创建时间戳索引
        vis_dict = {record['timestamp']: record for record in self.vis_data}
        ptu_dict = {record['timestamp']: record for record in self.ptu_data}
        wind_dict = {record['timestamp']: record for record in self.wind_data}
        
        merged_records = []
        
        # 以能见度数据的时间戳为基准进行合并
        for vis_record in self.vis_data:
            timestamp = vis_record['timestamp']
            
            # 寻找最接近的气象数据
            ptu_record = self._find_closest_record(timestamp, self.ptu_data)
            wind_record = self._find_closest_record(timestamp, self.wind_data)
            
            if ptu_record and wind_record:
                merged_record = {
                    'timestamp': timestamp,
                    'visibility': vis_record['visibility'],
                    'temperature': ptu_record['temperature'],
                    'humidity': ptu_record['humidity'],
                    'pressure': ptu_record['pressure'],
                    'dewpoint': ptu_record['dewpoint'],
                    'wind_speed': wind_record['wind_speed'],
                    'wind_direction': wind_record['wind_direction'],
                    'vertical_wind': wind_record['vertical_wind']
                }
                merged_records.append(merged_record)
        
        print(f"成功合并 {len(merged_records)} 条记录")
        self.merged_data = merged_records
        return merged_records

    def _find_closest_record(self, target_time: datetime, records: List[Dict]) -> Optional[Dict]:
        """
        查找最接近目标时间的记录
        
        Parameters
        ----------
        target_time : datetime
            目标时间
        records : List[Dict]
            记录列表
            
        Returns
        -------
        Optional[Dict]
            最接近的记录
        """
        if not records:
            return None
        
        min_diff = float('inf')
        closest_record = None
        
        for record in records:
            diff = abs((record['timestamp'] - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_record = record
        
        return closest_record

    def calculate_fog_formation_probability(self, temperature: float, humidity: float, wind_speed: float) -> float:
        """
        计算雾形成概率
        
        Parameters
        ----------
        temperature : float
            温度（摄氏度）
        humidity : float
            相对湿度（%）
        wind_speed : float
            风速（m/s）
            
        Returns
        -------
        float
            雾形成概率（0-1）
        """
        humidity_factor = max(0, (humidity - 70) / 30)
        wind_factor = max(0, (3 - wind_speed) / 3)
        temp_factor = max(0, (15 - temperature) / 15)
        
        fog_prob = (humidity_factor * 0.5 + wind_factor * 0.3 + temp_factor * 0.2)
        return min(1.0, fog_prob)

    def extract_image_features_from_frame(self, image: np.ndarray) -> Dict[str, float]:
        """
        从单帧图像提取清晰度特征
        
        Parameters
        ----------
        image : np.ndarray
            输入的灰度图像
            
        Returns
        -------
        Dict[str, float]
            提取的图像特征字典
        """
        # 1. Tenengrad梯度方差
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobel_x**2 + sobel_y**2)
        
        # 2. 拉普拉斯方差
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # 3. 高频能量占比
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) * 0.3)**2
        
        high_freq_energy = np.sum(magnitude * mask)
        total_energy = np.sum(magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # 4. RMS对比度
        mean_intensity = np.mean(image)
        contrast_rms = np.sqrt(np.mean((image - mean_intensity)**2))
        
        # 5. 边缘密度
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # 6. 暗通道值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(image, kernel)
        dark_channel = np.mean(dark) / 255.0
        
        return {
            'tenengrad': tenengrad,
            'laplacian_var': laplacian_var,
            'high_freq_ratio': high_freq_ratio,
            'contrast_rms': contrast_rms,
            'edge_density': edge_density,
            'dark_channel': dark_channel
        }

    def process_video_with_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        处理视频并结合气象数据
        
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            包含图像特征和气象特征的综合数据列表
        """
        if not os.path.exists(self.video_file):
            print(f"视频文件不存在: {self.video_file}")
            return None
            
        if not self.merged_data:
            print("错误: 请先合并气象数据")
            return None
        
        print("开始处理视频文件...")
        
        # 打开视频
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            print("无法打开视频文件")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {fps} FPS, {total_frames:,} 总帧数")
        print(f"数据记录: {len(self.merged_data)} 条")
        
        # 计算采样策略 - 假设数据间隔约15秒
        frames_per_data_point = int(15 * fps)
        
        features_list = []
        processed_count = 0
        
        print(f"开始处理，每{frames_per_data_point}帧采样一次...")
        
        for i, data_record in enumerate(self.merged_data):
            # 计算对应的视频帧位置
            frame_position = i * frames_per_data_point
            
            # 如果超出视频长度，停止处理
            if frame_position >= total_frames:
                print(f"视频结束，处理了前{i}个数据点")
                break
            
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 提取图像特征
            image_features = self.extract_image_features_from_frame(gray)
            
            # 计算雾形成概率
            fog_prob = self.calculate_fog_formation_probability(
                data_record['temperature'], 
                data_record['humidity'], 
                data_record['wind_speed']
            )
            
            # 组合所有特征
            combined_features = {
                'timestamp': data_record['timestamp'],
                'visibility': data_record['visibility'],
                # 图像特征
                **image_features,
                # 气象特征
                'temperature': data_record['temperature'],
                'humidity': data_record['humidity'],
                'pressure': data_record['pressure'],
                'dewpoint': data_record['dewpoint'],
                'wind_speed': data_record['wind_speed'],
                'wind_direction': data_record['wind_direction'],
                'vertical_wind': data_record['vertical_wind'],
                # 派生特征
                'fog_formation_prob': fog_prob,
                'temp_dewpoint_diff': data_record['temperature'] - data_record['dewpoint']
            }
            
            features_list.append(combined_features)
            processed_count += 1
            
            # 显示进度
            if processed_count % 100 == 0:
                print(f"已处理: {processed_count}/{len(self.merged_data)} ({processed_count/len(self.merged_data)*100:.1f}%)")
        
        cap.release()
        
        print(f"视频处理完成！提取了 {len(features_list)} 个特征样本")
        self.features_data = features_list
        return features_list

    def build_regression_models(self) -> Dict[str, Any]:
        """
        构建多种回归模型
        
        Returns
        -------
        Dict[str, Any]
            包含所有模型结果的字典
        """
        if not self.features_data:
            print("错误: 没有特征数据")
            return {}
        
        print("开始构建回归模型...")
        
        # 准备数据
        df = pd.DataFrame(self.features_data)
        
        # 选择特征列
        feature_columns = [
            'tenengrad', 'laplacian_var', 'high_freq_ratio', 'contrast_rms', 
            'edge_density', 'dark_channel', 'temperature', 'humidity', 
            'pressure', 'dewpoint', 'wind_speed', 'wind_direction', 
            'vertical_wind', 'fog_formation_prob', 'temp_dewpoint_diff'
        ]
        
        X = df[feature_columns].values
        y = df['visibility'].values
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 特征选择
        selector = SelectKBest(score_func=f_regression, k=min(10, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        print(f"选择的特征: {selected_features}")
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # 定义模型
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0)
        }
        
        results = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'selected_features': selected_features,
            'scaler': scaler,
            'selector': selector,
            'models': {}
        }
        
        # 训练和评估模型
        for name, model in models.items():
            try:
                print(f"训练模型: {name}")
                
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
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                results['models'][name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'y_test_pred': y_test_pred,
                    'success': True
                }
                
                print(f"  训练R²: {train_r2:.4f}, 测试R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
                
            except Exception as e:
                print(f"  模型训练失败: {e}")
                results['models'][name] = {'success': False, 'error': str(e)}
        
        # 找到最佳模型
        best_model_name = max(
            [name for name, result in results['models'].items() if result['success']], 
            key=lambda x: results['models'][x]['test_r2']
        )
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"测试R²: {results['models'][best_model_name]['test_r2']:.4f}")
        
        self.models = results
        return results

    def create_visualization(self) -> None:
        """
        创建综合可视化图表
        """
        if not self.models:
            print("错误: 请先训练模型")
            return
        
        print("生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Airport Visibility Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 特征重要性
        ax1 = axes[0, 0]
        if self.features_data:
            img_features = ['tenengrad', 'laplacian_var', 'high_freq_ratio', 
                           'contrast_rms', 'edge_density', 'dark_channel']
            img_values = [np.mean([item[feat] for item in self.features_data]) for feat in img_features]
            img_labels = ['Tenengrad', 'Laplacian', 'HighFreq', 'Contrast', 'EdgeDens', 'DarkChan']
            
            bars = ax1.bar(range(len(img_features)), img_values, color='skyblue', alpha=0.7)
            ax1.set_xticks(range(len(img_features)))
            ax1.set_xticklabels(img_labels, rotation=45, ha='right')
            ax1.set_title('Image Features Average Values')
            ax1.set_ylabel('Feature Value')
            ax1.grid(True, alpha=0.3)
        
        # 2. 模型性能对比
        ax2 = axes[0, 1]
        model_names = []
        r2_scores = []
        for name, result in self.models['models'].items():
            if result['success']:
                model_names.append(name.upper())
                r2_scores.append(result['test_r2'])
        
        bars = ax2.bar(range(len(model_names)), r2_scores, color='lightcoral', alpha=0.7)
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Model Performance Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. 预测vs真实值
        ax3 = axes[0, 2]
        best_model_name = max(
            [name for name, result in self.models['models'].items() if result['success']], 
            key=lambda x: self.models['models'][x]['test_r2']
        )
        
        y_test = self.models['y_test']
        y_test_pred = self.models['models'][best_model_name]['y_test_pred']
        
        ax3.scatter(y_test, y_test_pred, alpha=0.6, s=20, color='blue')
        min_val = min(np.min(y_test), np.min(y_test_pred))
        max_val = max(np.max(y_test), np.max(y_test_pred))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('True Visibility (m)')
        ax3.set_ylabel('Predicted Visibility (m)')
        ax3.set_title(f'Prediction vs Truth\nR² = {self.models["models"][best_model_name]["test_r2"]:.4f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差分布
        ax4 = axes[1, 0]
        residuals = y_test - y_test_pred
        ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Residuals (m)')
        ax4.set_ylabel('Density')
        ax4.set_title('Residual Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. 时间序列预测效果
        ax5 = axes[1, 1]
        n_show = min(100, len(y_test))
        indices = np.arange(n_show)
        
        ax5.plot(indices, y_test[:n_show], 'b-', label='True Values', linewidth=2, alpha=0.8)
        ax5.plot(indices, y_test_pred[:n_show], 'r-', label='Predictions', linewidth=2, alpha=0.8)
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Visibility (m)')
        ax5.set_title(f'Time Series Prediction')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能指标总结
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 显示性能指标
        metrics_text = f"""
Best Model: {best_model_name.upper()}
Test R²: {self.models['models'][best_model_name]['test_r2']:.4f}
RMSE: {self.models['models'][best_model_name]['test_rmse']:.2f} m
MAE: {self.models['models'][best_model_name]['test_mae']:.2f} m

Features Used: {len(self.models['selected_features'])}
Training Samples: {len(self.models['X_train'])}
Test Samples: {len(self.models['X_test'])}
        """
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax6.set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        执行完整的分析流程
        
        Returns
        -------
        Dict[str, Any]
            完整分析结果
        """
        print("=" * 60)
        print("开始执行完整的机场能见度分析流程")
        print("=" * 60)
        
        results = {}
        
        # 1. 检查文件
        existing_files, missing_files = self.check_files()
        results['file_check'] = {'existing': existing_files, 'missing': missing_files}
        
        if missing_files:
            print(f"警告: 有 {len(missing_files)} 个文件缺失，分析可能受限")
        
        # 2. 加载数据
        print("\n步骤 1: 加载数据文件...")
        vis_data = self.load_vis_data()
        ptu_data = self.load_ptu_data()
        wind_data = self.load_wind_data()
        
        if not all([vis_data, ptu_data, wind_data]):
            print("错误: 无法加载必要的数据文件")
            return results
        
        # 3. 合并数据
        print("\n步骤 2: 合并气象数据...")
        merged_data = self.merge_weather_data()
        if not merged_data:
            print("错误: 数据合并失败")
            return results
        
        results['data_summary'] = {
            'vis_records': len(vis_data),
            'ptu_records': len(ptu_data),
            'wind_records': len(wind_data),
            'merged_records': len(merged_data)
        }
        
        # 4. 处理视频（如果存在）
        if os.path.exists(self.video_file):
            print("\n步骤 3: 处理视频数据...")
            features_data = self.process_video_with_data()
            if features_data:
                results['features_extracted'] = len(features_data)
            else:
                print("警告: 视频处理失败，仅使用气象数据")
                # 如果视频处理失败，使用气象数据构建特征
                self.features_data = merged_data
        else:
            print("警告: 视频文件不存在，仅使用气象数据")
            self.features_data = merged_data
        
        # 5. 构建模型
        print("\n步骤 4: 构建机器学习模型...")
        model_results = self.build_regression_models()
        results['model_results'] = model_results
        
        # 6. 生成可视化
        print("\n步骤 5: 生成可视化图表...")
        self.create_visualization()
        
        print("\n" + "=" * 60)
        print("分析完成！")
        
        # 打印总结
        if self.models and self.models['models']:
            best_model_name = max(
                [name for name, result in self.models['models'].items() if result['success']], 
                key=lambda x: self.models['models'][x]['test_r2']
            )
            best_r2 = self.models['models'][best_model_name]['test_r2']
            best_rmse = self.models['models'][best_model_name]['test_rmse']
            
            print(f"最佳模型: {best_model_name}")
            print(f"预测精度: R² = {best_r2:.4f}")
            print(f"预测误差: RMSE = {best_rmse:.2f} m")
        
        print("=" * 60)
        
        return results


def main():
    """
    主程序入口
    """
    print("机场能见度回归建模系统")
    print("基于大雾背景视频学习的能见度分析")
    print("=" * 50)
    
    # 创建分析器实例
    analyzer = AirportVisibilityAnalyzer()
    
    try:
        # 执行完整分析
        results = analyzer.run_complete_analysis()
        
        # 保存结果（可选）
        # 可以在这里添加结果保存功能
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
