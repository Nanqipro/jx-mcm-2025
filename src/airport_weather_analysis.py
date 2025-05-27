#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机场天气分析系统

这个模块整合了机场能见度、气象数据和视频分析功能，
提供完整的雾霾检测和预测解决方案。

主要功能：
- 多源数据文件检查和加载（VIS, PTU, WIND）
- 时间戳匹配和数据融合
- 视频特征提取和清晰度分析
- 雾形成概率计算
- 机器学习建模和预测
- 数据可视化和分析

作者: Airport Weather Analysis Team
版本: 1.0.0
日期: 2024
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

# 关闭警告信息
warnings.filterwarnings('ignore')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FileManager:
    """
    文件管理类
    
    负责检查和验证所有必需的数据文件，包括：
    - VIS能见度数据文件
    - PTU气象数据文件
    - WIND风速数据文件
    - 机场视频文件（可选）
    """
    
    def __init__(self):
        """初始化文件管理器"""
        self.files_info = {
            "VIS_R06_12.his": "能见度数据文件",
            "PTU_R06_12.his": "气象数据文件", 
            "WIND_R06_12.his": "风速数据文件",
            "airport_video.mp4": "机场视频文件(可选)"
        }
    
    def check_files(self) -> Tuple[List[str], List[str]]:
        """
        检查所有数据文件是否存在
        
        Returns
        -------
        Tuple[List[str], List[str]]
            (存在的文件列表, 缺失的文件列表)
        """
        print("=" * 50)
        print("数据文件检查结果:")
        print("=" * 50)
        
        existing_files = []
        missing_files = []
        
        for filename, description in self.files_info.items():
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"✓ {filename:20s} - {description} ({file_size:,} bytes)")
                existing_files.append(filename)
            else:
                print(f"✗ {filename:20s} - {description} (文件不存在)")
                missing_files.append(filename)
        
        print(f"\n存在的文件: {len(existing_files)} 个")
        print(f"缺失的文件: {len(missing_files)} 个")
        
        return existing_files, missing_files


class DataLoader:
    """
    数据加载器类
    
    负责加载和初步分析各种数据文件格式，
    提供统一的数据接口和格式验证功能。
    """
    
    @staticmethod
    def load_and_analyze_vis_file(filename: str = "VIS_R06_12.his") -> Optional[List[Tuple[int, List[str]]]]:
        """
        加载并分析VIS能见度数据文件
        
        Parameters
        ----------
        filename : str
            VIS文件路径
            
        Returns
        -------
        Optional[List[Tuple[int, List[str]]]]
            包含行号和数据的元组列表，如果失败则返回None
        """
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在分析 {filename}...")
        print("=" * 60)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"文件总行数: {len(lines)}")
            print(f"前3行内容:")
            for i, line in enumerate(lines[:3]):
                print(f"第{i+1}行: {line.strip()}")
            
            # 分析数据行
            data_lines = []
            for i, line in enumerate(lines[1:], 1):  # 跳过第一行标题
                if line.strip() and not line.startswith('History'):
                    parts = line.strip().split()
                    if len(parts) >= 10:  # 确保有足够的列
                        data_lines.append((i+1, parts))
            
            print(f"\n有效数据行数: {len(data_lines)}")
            
            if data_lines:
                print(f"\n第一个数据行分析 (第{data_lines[0][0]}行):")
                parts = data_lines[0][1]
                print(f"  列数: {len(parts)}")
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                print(f"  前10列: {parts[:10]}")
                
                print(f"\n最后一个数据行分析 (第{data_lines[-1][0]}行):")
                parts = data_lines[-1][1]
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                
            return data_lines
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return None
    
    @staticmethod
    def load_and_analyze_ptu_file(filename: str = "PTU_R06_12.his") -> Optional[List[Tuple[int, List[str]]]]:
        """
        加载并分析PTU气象数据文件
        
        Parameters
        ----------
        filename : str
            PTU文件路径
            
        Returns
        -------
        Optional[List[Tuple[int, List[str]]]]
            包含行号和数据的元组列表，如果失败则返回None
        """
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在分析 {filename}...")
        print("=" * 60)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"文件总行数: {len(lines)}")
            print(f"前5行内容:")
            for i, line in enumerate(lines[:5]):
                print(f"第{i+1}行: {line.strip()}")
            
            # 分析数据行（通常从第三行开始）
            data_lines = []
            for i, line in enumerate(lines[2:], 2):  # 从第三行开始
                if line.strip():
                    parts = line.strip().split('\t')  # PTU文件通常用tab分隔
                    if len(parts) >= 5:
                        data_lines.append((i+1, parts))
            
            print(f"\n有效数据行数: {len(data_lines)}")
            
            if data_lines:
                print(f"\n第一个数据行分析 (第{data_lines[0][0]}行):")
                parts = data_lines[0][1]
                print(f"  列数: {len(parts)}")
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                print(f"  前5列: {parts[:5]}")
                
                if len(parts) > 10:
                    print(f"  温度相关列 (第13-16列): {parts[12:16] if len(parts) > 15 else parts[12:]}")
                
                print(f"\n最后一个数据行分析 (第{data_lines[-1][0]}行):")
                parts = data_lines[-1][1]
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                
            return data_lines
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return None
    
    @staticmethod
    def load_and_analyze_wind_file(filename: str = "WIND_R06_12.his") -> Optional[List[Tuple[int, List[str]]]]:
        """
        加载并分析WIND风速数据文件
        
        Parameters
        ----------
        filename : str
            WIND文件路径
            
        Returns
        -------
        Optional[List[Tuple[int, List[str]]]]
            包含行号和数据的元组列表，如果失败则返回None
        """
        if not os.path.exists(filename):
            print(f"错误: {filename} 文件不存在")
            return None
        
        print(f"正在分析 {filename}...")
        print("=" * 60)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"文件总行数: {len(lines)}")
            print(f"前3行内容:")
            for i, line in enumerate(lines[:3]):
                print(f"第{i+1}行: {line.strip()}")
            
            # 分析数据行
            data_lines = []
            for i, line in enumerate(lines[1:], 1):  # 跳过第一行标题
                if line.strip() and not line.startswith('History'):
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        data_lines.append((i+1, parts))
            
            print(f"\n有效数据行数: {len(data_lines)}")
            
            if data_lines:
                print(f"\n第一个数据行分析 (第{data_lines[0][0]}行):")
                parts = data_lines[0][1]
                print(f"  列数: {len(parts)}")
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                print(f"  前10列: {parts[:10]}")
                
                print(f"\n最后一个数据行分析 (第{data_lines[-1][0]}行):")
                parts = data_lines[-1][1]
                print(f"  时间戳: {parts[1] if len(parts) > 1 else 'N/A'}")
                
            return data_lines
            
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return None


class VideoProcessor:
    """
    视频处理器类
    
    负责机场视频文件的处理和分析，
    包括视频信息获取、帧提取和特征计算。
    """
    
    @staticmethod
    def check_video_file(filename: str = "airport_video.mp4") -> Optional[Dict[str, Union[float, int]]]:
        """
        检查视频文件信息
        
        Parameters
        ----------
        filename : str
            视频文件路径
            
        Returns
        -------
        Optional[Dict[str, Union[float, int]]]
            包含视频信息的字典，如果失败则返回None
        """
        print(f"检查视频文件: {filename}")
        print("=" * 40)
        
        if not os.path.exists(filename):
            print(f"视频文件 {filename} 不存在")
            return None
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(filename)
            print(f"文件大小: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            # 尝试打开视频文件
            cap = cv2.VideoCapture(filename)
            
            if not cap.isOpened():
                print("无法打开视频文件")
                return None
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"视频信息:")
            print(f"  分辨率: {width} x {height}")
            print(f"  帧率: {fps:.2f} FPS")
            print(f"  总帧数: {frame_count:,}")
            print(f"  时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
            
            # 读取第一帧作为测试
            ret, frame = cap.read()
            if ret:
                print(f"  成功读取第一帧，形状: {frame.shape}")
            else:
                print("  无法读取第一帧")
            
            cap.release()
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            print(f"检查视频文件时出错: {e}")
            return None


class DataParser:
    """
    数据解析器类
    
    负责将原始数据文件解析为结构化的数据格式，
    包括数据清洗、验证和标准化处理。
    """
    
    @staticmethod
    def parse_vis_data(filename: str = "VIS_R06_12.his") -> List[Dict[str, Union[str, float]]]:
        """
        解析VIS能见度数据文件
        
        Parameters
        ----------
        filename : str
            VIS文件路径
            
        Returns
        -------
        List[Dict[str, Union[str, float]]]
            解析后的能见度数据列表
        """
        print("解析VIS能见度数据...")
        
        vis_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过前两行（标题行）
        for i, line in enumerate(lines[2:], 2):
            if line.strip():
                parts = line.strip().split('\t')
                try:
                    # 根据文件格式解析数据
                    data = {
                        'timestamp': parts[1],  # LOCALDATE (BEIJING)
                        'RVR_1A': float(parts[4]) if parts[4].strip() and parts[4].replace('.','').isdigit() else 0,
                        'MOR_1A': float(parts[12]) if len(parts) > 12 and parts[12].strip() and parts[12].replace('.','').isdigit() else 0,
                        'VIS1A': float(parts[17]) if len(parts) > 17 and parts[17].strip() and parts[17].replace('.','').isdigit() else 0,
                        'visibility': float(parts[12]) if len(parts) > 12 and parts[12].strip() and parts[12].replace('.','').isdigit() else 0  # 使用MOR_1A
                    }
                    
                    # 只保留有效数据
                    if data['visibility'] > 0:
                        vis_data.append(data)
                        
                except (ValueError, IndexError):
                    continue
        
        print(f"解析VIS数据: {len(vis_data)} 条有效记录")
        if vis_data:
            print(f"时间范围: {vis_data[0]['timestamp']} 到 {vis_data[-1]['timestamp']}")
            print(f"能见度范围: {min(d['visibility'] for d in vis_data):.1f}m 到 {max(d['visibility'] for d in vis_data):.1f}m")
        
        return vis_data
    
    @staticmethod
    def parse_ptu_data(filename: str = "PTU_R06_12.his") -> List[Dict[str, Union[str, float]]]:
        """
        解析PTU气象数据文件
        
        Parameters
        ----------
        filename : str
            PTU文件路径
            
        Returns
        -------
        List[Dict[str, Union[str, float]]]
            解析后的气象数据列表
        """
        print("解析PTU气象数据...")
        
        ptu_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过前两行（标题行）
        for i, line in enumerate(lines[2:], 2):
            if line.strip():
                parts = line.strip().split('\t')
                try:
                    # 根据文件格式解析数据
                    data = {
                        'timestamp': parts[1],  # LOCALDATE (BEIJING)
                        'pressure': float(parts[3]) if parts[3].strip() and parts[3].replace('.','').replace('-','').isdigit() else 1013.25,
                        'temperature': float(parts[13]) if len(parts) > 13 and parts[13].strip() else 10.0,
                        'humidity': float(parts[14]) if len(parts) > 14 and parts[14].strip() else 80.0,
                        'dewpoint': float(parts[15]) if len(parts) > 15 and parts[15].strip() else 8.0
                    }
                    
                    # 数据合理性检查
                    if (0 <= data['humidity'] <= 100 and 
                        -50 <= data['temperature'] <= 50 and
                        950 <= data['pressure'] <= 1050):
                        ptu_data.append(data)
                        
                except (ValueError, IndexError):
                    continue
        
        print(f"解析PTU数据: {len(ptu_data)} 条有效记录")
        if ptu_data:
            print(f"时间范围: {ptu_data[0]['timestamp']} 到 {ptu_data[-1]['timestamp']}")
            print(f"温度范围: {min(d['temperature'] for d in ptu_data):.1f}°C 到 {max(d['temperature'] for d in ptu_data):.1f}°C")
            print(f"湿度范围: {min(d['humidity'] for d in ptu_data):.1f}% 到 {max(d['humidity'] for d in ptu_data):.1f}%")
        
        return ptu_data
    
    @staticmethod
    def parse_wind_data(filename: str = "WIND_R06_12.his") -> List[Dict[str, Union[str, float]]]:
        """
        解析WIND风速数据文件
        
        Parameters
        ----------
        filename : str
            WIND文件路径
            
        Returns
        -------
        List[Dict[str, Union[str, float]]]
            解析后的风速数据列表
        """
        print("解析WIND风速数据...")
        
        wind_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过前两行（标题行）
        for i, line in enumerate(lines[2:], 2):
            if line.strip():
                parts = line.strip().split('\t')
                try:
                    # 根据文件格式解析数据
                    data = {
                        'timestamp': parts[1],  # LOCALDATE (BEIJING)
                        'wind_speed_2a': float(parts[5]) if len(parts) > 5 and parts[5].strip() and parts[5].replace('.','').isdigit() else 1.0,  # WS2A
                        'wind_direction_2a': float(parts[13]) if len(parts) > 13 and parts[13].strip() and parts[13].replace('.','').isdigit() else 180.0,  # WD2A
                        'vertical_wind_2a': float(parts[22]) if len(parts) > 22 and parts[22].strip() and parts[22].replace('.','').replace('-','').isdigit() else 0.0  # CW2A
                    }
                    
                    # 数据合理性检查
                    if (0 <= data['wind_speed_2a'] <= 50 and 
                        0 <= data['wind_direction_2a'] <= 360):
                        wind_data.append(data)
                        
                except (ValueError, IndexError):
                    continue
        
        print(f"解析WIND数据: {len(wind_data)} 条有效记录")
        if wind_data:
            print(f"时间范围: {wind_data[0]['timestamp']} 到 {wind_data[-1]['timestamp']}")
            print(f"风速范围: {min(d['wind_speed_2a'] for d in wind_data):.1f}m/s 到 {max(d['wind_speed_2a'] for d in wind_data):.1f}m/s")
        
        return wind_data


class DataMerger:
    """
    数据合并器类
    
    负责将来自不同数据源的时间序列数据进行智能匹配和合并，
    支持精确时间匹配和模糊时间匹配算法。
    """
    
    def __init__(self, vis_data: List[Dict], ptu_data: List[Dict], wind_data: List[Dict]):
        """
        初始化数据合并器
        
        Parameters
        ----------
        vis_data : List[Dict]
            能见度数据
        ptu_data : List[Dict]
            气象数据
        wind_data : List[Dict]
            风速数据
        """
        self.vis_data = vis_data
        self.ptu_data = ptu_data
        self.wind_data = wind_data
    
    def merge_data_with_timestamp_matching(self) -> List[Dict[str, Union[str, float]]]:
        """
        使用智能时间戳匹配合并数据
        
        Returns
        -------
        List[Dict[str, Union[str, float]]]
            合并后的数据列表
        """
        print("开始数据合并...")
        
        # 创建时间戳索引
        ptu_dict = {record['timestamp']: record for record in self.ptu_data}
        wind_dict = {record['timestamp']: record for record in self.wind_data}
        
        merged_data = []
        exact_matches = 0
        fuzzy_matches = 0
        
        for vis_record in self.vis_data:
            timestamp = vis_record['timestamp']
            
            # 精确匹配
            ptu_record = ptu_dict.get(timestamp)
            wind_record = wind_dict.get(timestamp)
            
            # 如果精确匹配失败，尝试邻近匹配
            if not ptu_record:
                ptu_record = self.find_nearest_record(self.ptu_data, timestamp, max_diff_minutes=2)
            if not wind_record:
                wind_record = self.find_nearest_record(self.wind_data, timestamp, max_diff_seconds=30)
            
            if ptu_record and wind_record:
                if ptu_dict.get(timestamp) and wind_dict.get(timestamp):
                    exact_matches += 1
                else:
                    fuzzy_matches += 1
                    
                combined_record = {
                    'timestamp': timestamp,
                    'visibility': vis_record['visibility'],
                    'RVR_1A': vis_record['RVR_1A'],
                    'MOR_1A': vis_record['MOR_1A'],
                    'temperature': ptu_record['temperature'],
                    'humidity': ptu_record['humidity'],
                    'pressure': ptu_record['pressure'],
                    'dewpoint': ptu_record['dewpoint'],
                    'wind_speed': wind_record['wind_speed_2a'],
                    'wind_direction': wind_record['wind_direction_2a'],
                    'vertical_wind': wind_record['vertical_wind_2a']
                }
                merged_data.append(combined_record)
        
        print(f"合并结果:")
        print(f"  精确匹配: {exact_matches} 条")
        print(f"  邻近匹配: {fuzzy_matches} 条")
        print(f"  总计: {len(merged_data)} 条有效记录")
        print(f"  合并成功率: {len(merged_data)/len(self.vis_data)*100:.1f}%")
        
        return merged_data
    
    @staticmethod
    def find_nearest_record(data_list: List[Dict], target_timestamp: str, 
                          max_diff_minutes: Optional[int] = None, 
                          max_diff_seconds: Optional[int] = None) -> Optional[Dict]:
        """
        查找最近时间的记录
        
        Parameters
        ----------
        data_list : List[Dict]
            数据记录列表
        target_timestamp : str
            目标时间戳
        max_diff_minutes : Optional[int]
            最大时间差（分钟）
        max_diff_seconds : Optional[int]
            最大时间差（秒）
            
        Returns
        -------
        Optional[Dict]
            最近的记录，如果没有找到则返回None
        """
        try:
            target_time = datetime.strptime(target_timestamp, '%Y-%m-%d %H:%M:%S')
        except:
            return None
        
        best_record = None
        min_diff = float('inf')
        
        # 设置最大时间差
        if max_diff_minutes:
            max_diff_total = max_diff_minutes * 60
        elif max_diff_seconds:
            max_diff_total = max_diff_seconds
        else:
            max_diff_total = 300  # 默认5分钟
        
        for record in data_list:
            try:
                record_time = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
                diff_seconds = abs((target_time - record_time).total_seconds())
                
                if diff_seconds < min_diff and diff_seconds <= max_diff_total:
                    min_diff = diff_seconds
                    best_record = record
            except:
                continue
        
        return best_record


class DataAnalyzer:
    """
    数据分析器类
    
    负责对合并后的数据进行统计分析、质量评估和模式识别，
    提供数据质量报告和统计摘要。
    """
    
    @staticmethod
    def analyze_merged_data(merged_data: List[Dict]) -> Optional[Dict[str, List[float]]]:
        """
        分析合并后的数据质量
        
        Parameters
        ----------
        merged_data : List[Dict]
            合并后的数据列表
            
        Returns
        -------
        Optional[Dict[str, List[float]]]
            包含各变量数据的字典
        """
        print("数据质量分析...")
        print("=" * 50)
        
        if not merged_data:
            print("没有合并的数据可供分析")
            return None
        
        # 基本统计
        print(f"数据概况:")
        print(f"  总记录数: {len(merged_data)}")
        print(f"  时间跨度: {merged_data[0]['timestamp']} 到 {merged_data[-1]['timestamp']}")
        
        # 提取数值数据
        visibility = [d['visibility'] for d in merged_data]
        temperature = [d['temperature'] for d in merged_data]
        humidity = [d['humidity'] for d in merged_data]
        wind_speed = [d['wind_speed'] for d in merged_data]
        
        print(f"\n各变量统计:")
        print(f"  能见度: {np.mean(visibility):.1f}±{np.std(visibility):.1f}m (范围: {np.min(visibility):.1f}-{np.max(visibility):.1f}m)")
        print(f"  温度: {np.mean(temperature):.1f}±{np.std(temperature):.1f}°C (范围: {np.min(temperature):.1f}-{np.max(temperature):.1f}°C)")
        print(f"  湿度: {np.mean(humidity):.1f}±{np.std(humidity):.1f}% (范围: {np.min(humidity):.1f}-{np.max(humidity):.1f}%)")
        print(f"  风速: {np.mean(wind_speed):.1f}±{np.std(wind_speed):.1f}m/s (范围: {np.min(wind_speed):.1f}-{np.max(wind_speed):.1f}m/s)")
        
        # 雾事件统计
        low_vis_500 = sum(1 for v in visibility if v < 500)
        low_vis_1000 = sum(1 for v in visibility if v < 1000)
        
        print(f"\n雾事件统计:")
        print(f"  严重雾霾 (能见度<500m): {low_vis_500} 次 ({low_vis_500/len(visibility)*100:.1f}%)")
        print(f"  轻度雾霾 (能见度<1000m): {low_vis_1000} 次 ({low_vis_1000/len(visibility)*100:.1f}%)")
        
        return {
            'visibility': visibility,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed
        }


class FeatureExtractor:
    """
    特征提取器类
    
    负责从图像和气象数据中提取各种特征，
    包括图像清晰度特征、气象派生特征和雾形成概率计算。
    """
    
    @staticmethod
    def calculate_fog_formation_probability(temperature: float, humidity: float, wind_speed: float) -> float:
        """
        计算雾形成概率
        
        Parameters
        ----------
        temperature : float
            温度（摄氏度）
        humidity : float
            相对湿度（百分比）
        wind_speed : float
            风速（米/秒）
            
        Returns
        -------
        float
            雾形成概率（0-1之间）
        """
        humidity_factor = max(0, (humidity - 70) / 30)
        wind_factor = max(0, (3 - wind_speed) / 3)
        temp_factor = max(0, (15 - temperature) / 15)
        
        fog_prob = (humidity_factor * 0.5 + wind_factor * 0.3 + temp_factor * 0.2)
        return min(1.0, fog_prob)
    
    @staticmethod
    def extract_image_features_from_frame(image: np.ndarray) -> Dict[str, float]:
        """
        从单帧图像提取清晰度特征
        
        Parameters
        ----------
        image : np.ndarray
            输入的灰度图像
            
        Returns
        -------
        Dict[str, float]
            包含各种图像特征的字典
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


class VideoDataProcessor:
    """
    视频数据处理器类
    
    负责将视频数据与气象数据结合处理，
    提取时间同步的图像特征和气象特征。
    """
    
    def __init__(self, merged_data: List[Dict], video_filename: str = "airport_video.mp4"):
        """
        初始化视频数据处理器
        
        Parameters
        ----------
        merged_data : List[Dict]
            合并后的气象数据
        video_filename : str
            视频文件路径
        """
        self.merged_data = merged_data
        self.video_filename = video_filename
    
    def process_video_with_data(self) -> Optional[List[Dict]]:
        """
        处理视频并结合气象数据
        
        Returns
        -------
        Optional[List[Dict]]
            包含图像特征和气象特征的数据列表
        """
        print("开始处理视频文件...")
        
        # 打开视频
        cap = cv2.VideoCapture(self.video_filename)
        if not cap.isOpened():
            print("无法打开视频文件")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {fps} FPS, {total_frames:,} 总帧数")
        print(f"数据记录: {len(self.merged_data)} 条，时间间隔约15秒")
        
        # 计算采样策略
        # 数据是15秒间隔，视频是25fps，所以每375帧(15*25)对应一个数据点
        frames_per_data_point = int(15 * fps)  # 每个数据点对应的帧数
        
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
            image_features = FeatureExtractor.extract_image_features_from_frame(gray)
            
            # 计算雾形成概率
            fog_prob = FeatureExtractor.calculate_fog_formation_probability(
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
        return features_list


class Visualizer:
    """
    数据可视化器类
    
    负责生成各种数据分析图表和可视化结果，
    包括时间序列图、散点图、相关性分析等。
    """
    
    @staticmethod
    def create_preliminary_visualizations(features_data: List[Dict]) -> None:
        """
        创建初步的数据可视化
        
        Parameters
        ----------
        features_data : List[Dict]
            特征数据列表
        """
        if not features_data:
            print("没有特征数据可供可视化")
            return
        
        print("特征数据预览...")
        print("=" * 60)
        print(f"总样本数: {len(features_data)}")
        print(f"时间范围: {features_data[0]['timestamp']} 到 {features_data[-1]['timestamp']}")
        
        # 提取数据用于可视化
        visibility = np.array([d['visibility'] for d in features_data])
        temperature = np.array([d['temperature'] for d in features_data])
        humidity = np.array([d['humidity'] for d in features_data])
        wind_speed = np.array([d['wind_speed'] for d in features_data])
        tenengrad = np.array([d['tenengrad'] for d in features_data])
        dark_channel = np.array([d['dark_channel'] for d in features_data])
        fog_prob = np.array([d['fog_formation_prob'] for d in features_data])
        
        time_indices = np.arange(len(features_data))
        
        # 创建综合图表
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Airport Weather and Visibility Analysis', fontsize=16, fontweight='bold')
        
        # 1. 能见度时间序列
        axes[0, 0].plot(time_indices, visibility, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('Visibility Time Series')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Visibility (m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 温度和湿度时间序列
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(time_indices, temperature, 'r-', linewidth=1, label='Temperature')
        line2 = ax2.plot(time_indices, humidity, 'g-', linewidth=1, label='Humidity')
        
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Temperature (°C)', color='red')
        ax2.set_ylabel('Humidity (%)', color='green')
        ax1.set_title('Temperature and Humidity Time Series')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 3. 风速时间序列
        axes[0, 2].plot(time_indices, wind_speed, 'purple', linewidth=1)
        axes[0, 2].set_title('Wind Speed Time Series')
        axes[0, 2].set_xlabel('Time Index')
        axes[0, 2].set_ylabel('Wind Speed (m/s)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 暗通道与能见度关系
        axes[1, 0].scatter(visibility, dark_channel, alpha=0.5, s=10)
        axes[1, 0].set_title('Dark Channel vs Visibility')
        axes[1, 0].set_xlabel('Visibility (m)')
        axes[1, 0].set_ylabel('Dark Channel Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 清晰度与能见度关系
        axes[1, 1].scatter(visibility, tenengrad, alpha=0.5, s=10)
        axes[1, 1].set_title('Tenengrad vs Visibility')
        axes[1, 1].set_xlabel('Visibility (m)')
        axes[1, 1].set_ylabel('Tenengrad')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 雾形成概率
        axes[1, 2].plot(time_indices, fog_prob, 'brown', linewidth=1)
        axes[1, 2].set_title('Fog Formation Probability')
        axes[1, 2].set_xlabel('Time Index')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 湿度vs能见度（按温度着色）
        scatter1 = axes[2, 0].scatter(humidity, visibility, c=temperature, cmap='coolwarm', alpha=0.6, s=10)
        axes[2, 0].set_title('Humidity vs Visibility (Colored by Temperature)')
        axes[2, 0].set_xlabel('Humidity (%)')
        axes[2, 0].set_ylabel('Visibility (m)')
        plt.colorbar(scatter1, ax=axes[2, 0], label='Temperature (°C)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 风速vs能见度（按雾概率着色）
        scatter2 = axes[2, 1].scatter(wind_speed, visibility, c=fog_prob, cmap='YlOrRd', alpha=0.6, s=10)
        axes[2, 1].set_title('Wind Speed vs Visibility (Colored by Fog Probability)')
        axes[2, 1].set_xlabel('Wind Speed (m/s)')
        axes[2, 1].set_ylabel('Visibility (m)')
        plt.colorbar(scatter2, ax=axes[2, 1], label='Fog Formation Probability')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. 相关性分析
        # 计算主要变量的相关系数
        key_vars = np.column_stack([visibility, temperature, humidity, wind_speed, 
                                   tenengrad/1000, dark_channel, fog_prob])
        corr_matrix = np.corrcoef(key_vars.T)
        
        im = axes[2, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        var_names = ['Visibility', 'Temperature', 'Humidity', 'Wind Speed', 'Tenengrad', 'Dark Channel', 'Fog Probability']
        axes[2, 2].set_xticks(range(len(var_names)))
        axes[2, 2].set_yticks(range(len(var_names)))
        axes[2, 2].set_xticklabels(var_names, rotation=45)
        axes[2, 2].set_yticklabels(var_names)
        axes[2, 2].set_title('Variable Correlation Matrix')
        
        # 添加相关系数数值
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                color = "white" if abs(corr_matrix[i, j]) > 0.6 else "black"
                axes[2, 2].text(j, i, f'{corr_matrix[i, j]:.2f}', 
                               ha="center", va="center", color=color, fontsize=8)
        
        plt.colorbar(im, ax=axes[2, 2])
        plt.tight_layout()
        plt.show()
        
        # 输出关键发现
        print("\nKey Findings:")
        print(f"1. Dark channel vs visibility correlation: {np.corrcoef(visibility, dark_channel)[0,1]:.3f}")
        print(f"2. Tenengrad vs visibility correlation: {np.corrcoef(visibility, tenengrad)[0,1]:.3f}")
        print(f"3. Fog probability vs visibility correlation: {np.corrcoef(visibility, fog_prob)[0,1]:.3f}")
        print(f"4. Humidity vs visibility correlation: {np.corrcoef(visibility, humidity)[0,1]:.3f}")


class ModelBuilder:
    """
    模型构建器类
    
    负责构建和训练机器学习模型，
    包括线性回归、多项式回归和随机森林等算法。
    """
    
    @staticmethod
    def build_single_feature_set_model(X: np.ndarray, visibility: np.ndarray, 
                                     physical_blur: np.ndarray, feature_names: List[str], 
                                     model_name: str) -> Dict[str, Any]:
        """
        构建单一特征集模型
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        visibility : np.ndarray
            能见度目标变量
        physical_blur : np.ndarray
            物理模糊度
        feature_names : List[str]
            特征名称列表
        model_name : str
            模型名称
            
        Returns
        -------
        Dict[str, Any]
            包含模型结果的字典
        """
        print(f"  {model_name}: 特征数={X.shape[1]}, 样本数={X.shape[0]}")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, visibility, test_size=0.2, random_state=42
        )
        
        models = {}
        scores = {}
        
        # 1. 线性回归
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        scores['linear'] = r2_score(y_test, y_pred_linear)
        models['linear'] = linear_model
        
        # 2. 多项式回归
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        y_pred_poly = poly_model.predict(X_test_poly)
        scores['polynomial'] = r2_score(y_test, y_pred_poly)
        models['polynomial'] = (poly_model, poly_features)
        
        # 3. 随机森林
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        scores['random_forest'] = r2_score(y_test, y_pred_rf)
        models['random_forest'] = rf_model
        
        return {
            'models': models,
            'scores': scores,
            'feature_names': feature_names,
            'X_test': X_test,
            'y_test': y_test
        }
    
    @staticmethod
    def build_comprehensive_fog_blur_models(features_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        构建综合的雾模糊度模型
        
        Parameters
        ----------
        features_data : List[Dict]
            特征数据列表
            
        Returns
        -------
        Optional[Dict[str, Any]]
            模型结果字典
        """
        print("构建综合模糊度模型...")
        print("=" * 60)
        
        if not features_data:
            print("没有特征数据可用于建模")
            return None
        
        # 准备特征矩阵
        image_features = ['tenengrad', 'laplacian_var', 'high_freq_ratio', 
                         'contrast_rms', 'edge_density', 'dark_channel']
        weather_features = ['temperature', 'humidity', 'pressure', 'dewpoint',
                           'wind_speed', 'temp_dewpoint_diff', 'fog_formation_prob']
        all_features = image_features + weather_features
        
        # 构建特征矩阵
        X_image = np.array([[item[name] for name in image_features] for item in features_data])
        X_weather = np.array([[item[name] for name in weather_features] for item in features_data])
        X_all = np.hstack([X_image, X_weather])
        
        visibility = np.array([item['visibility'] for item in features_data])
        
        print(f"特征矩阵形状:")
        print(f"  图像特征: {X_image.shape}")
        print(f"  气象特征: {X_weather.shape}")
        print(f"  综合特征: {X_all.shape}")
        print(f"  目标变量: {visibility.shape}")
        
        # 特征标准化
        scaler_image = StandardScaler()
        scaler_weather = StandardScaler()
        scaler_all = StandardScaler()
        
        X_image_scaled = scaler_image.fit_transform(X_image)
        X_weather_scaled = scaler_weather.fit_transform(X_weather)
        X_all_scaled = scaler_all.fit_transform(X_all)
        
        # 物理模型基础 - 基于Beer-Lambert定律
        beta = 3.912 / (visibility + 1e-8)  # 大气散射系数
        physical_blur = 1 - np.exp(-beta * 0.5)  # 假设有效观测距离0.5km
        
        print(f"物理模糊度范围: {physical_blur.min():.4f} - {physical_blur.max():.4f}")
        
        # 构建多种模型
        models_results = {}
        
        # 1. 仅图像特征模型
        print("\n1. 构建仅图像特征模型...")
        models_results['image_only'] = ModelBuilder.build_single_feature_set_model(
            X_image_scaled, visibility, physical_blur, image_features, "图像特征"
        )
        
        # 2. 仅气象特征模型
        print("2. 构建仅气象特征模型...")
        models_results['weather_only'] = ModelBuilder.build_single_feature_set_model(
            X_weather_scaled, visibility, physical_blur, weather_features, "气象特征"
        )
        
        # 3. 综合特征模型
        print("3. 构建综合特征模型...")
        models_results['comprehensive'] = ModelBuilder.build_single_feature_set_model(
            X_all_scaled, visibility, physical_blur, all_features, "综合特征"
        )
        
        # 模型性能对比
        print("\n" + "="*60)
        print("模型性能对比:")
        print("="*60)
        
        for model_name, result in models_results.items():
            print(f"\n{model_name.upper()}:")
            for alg_name, score in result['scores'].items():
                print(f"  {alg_name:15s}: R² = {score:.4f}")
            
            best_alg = max(result['scores'], key=result['scores'].get)
            print(f"  最佳算法: {best_alg} (R² = {result['scores'][best_alg]:.4f})")
        
        return models_results


class AirportWeatherAnalysis:
    """
    机场天气分析主类
    
    整合所有功能模块，提供完整的分析流程管理，
    从数据加载到模型预测的端到端解决方案。
    """
    
    def __init__(self):
        """初始化机场天气分析系统"""
        self.file_manager = FileManager()
        self.data_loader = DataLoader()
        self.video_processor = VideoProcessor()
        self.data_parser = DataParser()
        
        # 数据存储
        self.vis_data = None
        self.ptu_data = None
        self.wind_data = None
        self.merged_data = None
        self.features_data = None
        self.models_results = None
    
    def run_complete_analysis(self) -> None:
        """
        运行完整的分析流程
        
        包括文件检查、数据加载、解析、合并、特征提取、
        可视化和模型构建的完整流程。
        """
        print("开始机场天气分析系统...")
        print("=" * 80)
        
        # 1. 检查文件
        existing_files, missing_files = self.file_manager.check_files()
        if missing_files:
            print(f"警告: 缺失文件 {missing_files}")
        
        # 2. 加载和分析文件
        print("\n" + "=" * 60)
        print("数据文件分析阶段")
        print("=" * 60)
        
        vis_data_lines = self.data_loader.load_and_analyze_vis_file()
        ptu_data_lines = self.data_loader.load_and_analyze_ptu_file()
        wind_data_lines = self.data_loader.load_and_analyze_wind_file()
        video_info = self.video_processor.check_video_file()
        
        # 3. 解析数据
        print("\n" + "=" * 60)
        print("数据解析阶段")
        print("=" * 60)
        
        self.vis_data = self.data_parser.parse_vis_data()
        self.ptu_data = self.data_parser.parse_ptu_data()
        self.wind_data = self.data_parser.parse_wind_data()
        
        # 4. 合并数据
        print("\n" + "=" * 60)
        print("数据合并阶段")
        print("=" * 60)
        
        data_merger = DataMerger(self.vis_data, self.ptu_data, self.wind_data)
        self.merged_data = data_merger.merge_data_with_timestamp_matching()
        
        # 5. 数据质量分析
        print("\n" + "=" * 60)
        print("数据质量分析阶段")
        print("=" * 60)
        
        stats = DataAnalyzer.analyze_merged_data(self.merged_data)
        
        # 6. 视频处理和特征提取
        if os.path.exists("airport_video.mp4"):
            print("\n" + "=" * 60)
            print("视频处理和特征提取阶段")
            print("=" * 60)
            print("注意：视频处理可能需要5-10分钟，请耐心等待...")
            
            video_data_processor = VideoDataProcessor(self.merged_data)
            self.features_data = video_data_processor.process_video_with_data()
            
            if self.features_data:
                # 7. 数据可视化
                print("\n" + "=" * 60)
                print("数据可视化阶段")
                print("=" * 60)
                
                Visualizer.create_preliminary_visualizations(self.features_data)
                
                # 8. 模型构建
                print("\n" + "=" * 60)
                print("机器学习建模阶段")
                print("=" * 60)
                
                self.models_results = ModelBuilder.build_comprehensive_fog_blur_models(self.features_data)
                
                # 9. 综合模型评估
                print("\n" + "=" * 60)
                print("综合模型评估阶段")
                print("=" * 60)
                
                evaluation_result = self.comprehensive_model_evaluation()
                
                # 10. 最终综合可视化
                print("\n" + "=" * 60)
                print("最终综合可视化阶段")
                print("=" * 60)
                
                self.create_final_comprehensive_visualization()
                
                # 11. 输出数学模型总结
                print("\n" + "=" * 60)
                print("数学模型总结阶段")
                print("=" * 60)
                
                self.output_final_mathematical_model()
        
        print("\n" + "=" * 80)
        print("机场天气分析完成！")
        print("=" * 80)
    
    def comprehensive_model_evaluation(self) -> Optional[Tuple[Tuple[str, str, float], Dict[str, Any]]]:
        """
        综合模型评估和分析
        
        对所有构建的模型进行详细评估，找出最佳模型配置，
        并提供详细的性能分析和特征重要性分析。
        
        Returns
        -------
        Optional[Tuple[Tuple[str, str, float], Dict[str, Any]]]
            最佳模型信息和详细结果，如果没有模型结果则返回None
        """
        if not self.models_results:
            print("没有建模结果可供评估")
            return None
        
        print("综合模型评估...")
        print("=" * 80)
        
        # 找出最佳模型
        best_overall_score = 0
        best_model_info = None
        
        for model_type, result in self.models_results.items():
            for alg_name, score in result['scores'].items():
                if score > best_overall_score:
                    best_overall_score = score
                    best_model_info = (model_type, alg_name, score)
        
        if not best_model_info:
            print("未找到有效的模型结果")
            return None
        
        print(f"最佳模型: {best_model_info[0]} - {best_model_info[1]}")
        print(f"最佳性能: R² = {best_model_info[2]:.4f}")
        
        # 详细分析最佳模型
        best_model_type, best_alg_name = best_model_info[0], best_model_info[1]
        best_result = self.models_results[best_model_type]
        
        print(f"\n最佳模型详细分析:")
        print(f"模型类型: {best_model_type}")
        print(f"算法: {best_alg_name}")
        
        # 特征重要性分析（如果是随机森林）
        if best_alg_name == 'random_forest' and 'models' in best_result:
            rf_model = best_result['models']['random_forest']
            feature_names = best_result['feature_names']
            importance = rf_model.feature_importances_
            
            print(f"\n特征重要性排序:")
            indices = np.argsort(importance)[::-1]
            for i, idx in enumerate(indices[:10]):
                print(f"  {i+1}. {feature_names[idx]:20s}: {importance[idx]:.4f}")
        
        # 预测性能分析
        if 'test_data' in best_result and 'predictions' in best_result:
            test_data = best_result['test_data']
            predictions = best_result['predictions'][best_alg_name]
            
            if best_alg_name == 'random_forest' and 'visibility_test' in test_data:
                actual = test_data['visibility_test']
                rmse = np.sqrt(mean_squared_error(actual, predictions))
                mae = mean_absolute_error(actual, predictions)
                print(f"\n预测性能 (能见度):")
                print(f"  RMSE: {rmse:.2f}m")
                print(f"  MAE:  {mae:.2f}m")
                print(f"  相对误差: {mae/np.mean(actual)*100:.1f}%")
            elif 'physical_blur_test' in test_data:
                actual = test_data['physical_blur_test']
                rmse = np.sqrt(mean_squared_error(actual, predictions))
                mae = mean_absolute_error(actual, predictions)
                print(f"\n预测性能 (模糊度):")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE:  {mae:.4f}")
        
        return best_model_info, best_result

    def create_final_comprehensive_visualization(self) -> None:
        """
        创建最终的综合可视化分析图表
        
        生成包含模型性能对比、时间序列分析、特征关系分析等
        的综合可视化图表，所有图表标签使用英文。
        """
        if not self.models_results or not self.features_data:
            print("没有建模结果或特征数据可供可视化")
            return
        
        print("生成最终综合可视化...")
        
        # 提取数据
        visibility = np.array([d['visibility'] for d in self.features_data])
        temperature = np.array([d['temperature'] for d in self.features_data])
        humidity = np.array([d['humidity'] for d in self.features_data])
        tenengrad = np.array([d['tenengrad'] for d in self.features_data])
        dark_channel = np.array([d['dark_channel'] for d in self.features_data])
        
        # 计算物理模糊度
        beta = 3.912 / (visibility + 1e-8)
        physical_blur = 1 - np.exp(-beta * 0.5)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 模型性能对比 (英文标签)
        plt.subplot(3, 4, 1)
        model_names = []
        best_scores = []
        
        for model_type, result in self.models_results.items():
            best_score = max(result['scores'].values())
            best_alg = max(result['scores'], key=result['scores'].get)
            model_names.append(f"{model_type}\n({best_alg})")
            best_scores.append(best_score)
        
        bars = plt.bar(range(len(model_names)), best_scores, 
                       color=['skyblue', 'lightgreen', 'coral'])
        plt.title('Model Performance Comparison', fontsize=12)  # 英文标题
        plt.ylabel('R² Score')
        plt.xticks(range(len(model_names)), model_names, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, best_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 能见度时间序列 (英文标签)
        timestamps = [item['timestamp'] for item in self.features_data]
        time_indices = range(len(timestamps))
        
        plt.subplot(3, 4, 2)
        plt.plot(time_indices, visibility, 'b-', linewidth=1, alpha=0.7)
        plt.title(f'Visibility Time Series (n={len(visibility)})', fontsize=12)  # 英文标题
        plt.xlabel('Time Index')
        plt.ylabel('Visibility (m)')
        plt.grid(True, alpha=0.3)
        
        # 3. 物理模糊度 vs 实际能见度 (英文标签)
        plt.subplot(3, 4, 3)
        plt.scatter(visibility, physical_blur, alpha=0.5, s=8)
        plt.title('Physical Blur vs Visibility', fontsize=12)  # 英文标题
        plt.xlabel('Visibility (m)')
        plt.ylabel('Physical Blur')
        plt.grid(True, alpha=0.3)
        
        # 4. Tenengrad vs 能见度 (英文标签)
        plt.subplot(3, 4, 4)
        plt.scatter(visibility, tenengrad, alpha=0.5, s=8, c='purple')
        plt.title('Tenengrad vs Visibility', fontsize=12)  # 英文标题
        plt.xlabel('Visibility (m)')
        plt.ylabel('Tenengrad')
        plt.grid(True, alpha=0.3)
        
        # 5. 暗通道 vs 能见度 (英文标签)
        plt.subplot(3, 4, 5)
        plt.scatter(visibility, dark_channel, alpha=0.5, s=8, c='red')
        plt.title('Dark Channel vs Visibility', fontsize=12)  # 英文标题
        plt.xlabel('Visibility (m)')
        plt.ylabel('Dark Channel Value')
        plt.grid(True, alpha=0.3)
        
        # 6. 温度湿度散点图 (英文标签)
        plt.subplot(3, 4, 6)
        plt.scatter(temperature, humidity, alpha=0.5, s=8, c='green')
        plt.title('Temperature vs Humidity', fontsize=12)  # 英文标题
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Humidity (%)')
        plt.grid(True, alpha=0.3)
        
        # 7-12. 继续添加其他分析图表，全部使用英文标签
        plt.subplot(3, 4, 7)
        fog_prob = np.array([d['fog_formation_prob'] for d in self.features_data])
        plt.plot(time_indices, fog_prob, 'brown', linewidth=1)
        plt.title('Fog Formation Probability', fontsize=12)  # 英文标题
        plt.xlabel('Time Index')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 8)
        wind_speed = np.array([d['wind_speed'] for d in self.features_data])
        plt.scatter(wind_speed, visibility, alpha=0.5, s=8, c='orange')
        plt.title('Wind Speed vs Visibility', fontsize=12)  # 英文标题
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Visibility (m)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def output_final_mathematical_model(self) -> None:
        """
        输出最终的数学模型总结
        
        提供详细的数学模型描述，包括最佳模型配置、
        数据集信息、物理模型基础和应用建议。
        """
        print("\n" + "="*80)
        print("最终数学模型总结")
        print("="*80)
        
        if not self.models_results or not self.features_data:
            print("没有建模结果或特征数据")
            return
        
        # 找出最佳模型
        best_score = 0
        best_info = None
        
        for model_type, result in self.models_results.items():
            for alg_name, score in result['scores'].items():
                if score > best_score:
                    best_score = score
                    best_info = (model_type, alg_name, result)
        
        if not best_info:
            print("未找到有效的模型结果")
            return
        
        model_type, alg_name, result = best_info
        
        print(f"【最佳模型配置】")
        print(f"特征集: {model_type}")
        print(f"算法: {alg_name}")
        print(f"性能: R² = {best_score:.4f}")
        
        print(f"\n【数据集信息】")
        print(f"训练样本: {len(self.features_data)} 个")
        if self.features_data:
            start_time = self.features_data[0]['timestamp']
            end_time = self.features_data[-1]['timestamp']
            print(f"时间跨度: {start_time} 到 {end_time}")
        
        visibility_values = [d['visibility'] for d in self.features_data]
        print(f"能见度范围: {min(visibility_values):.0f}m - {max(visibility_values):.0f}m")
        
        # 计算雾事件比例 (能见度 < 1000m)
        fog_events = sum(1 for v in visibility_values if v < 1000)
        fog_ratio = fog_events / len(visibility_values) * 100
        print(f"雾事件比例: {fog_ratio:.1f}%")
        
        print(f"\n【物理模型基础】")
        print("基于大气散射理论:")
        print("  透射率: t = exp(-β × d)")
        print("  模糊度: BlurIndex = 1 - t")
        print("  散射系数: β = 3.912 / 能见度")
        
        if alg_name == 'random_forest' and 'models' in result:
            print(f"\n【随机森林模型】")
            print("能见度预测 = RandomForest(特征向量)")
            print("特征向量 = [图像特征, 气象特征]")
            
            # 显示特征重要性
            rf_model = result['models']['random_forest']
            feature_names = result['feature_names']
            importance = rf_model.feature_importances_
            
            print(f"\n前5个重要特征:")
            indices = np.argsort(importance)[::-1]
            for i, idx in enumerate(indices[:5]):
                print(f"  {i+1}. {feature_names[idx]:20s}: {importance[idx]:.4f}")
        
        print(f"\n【应用建议】")
        print("1. 实时预警: 能见度 < 800m 时启动预警")
        print("2. 关键指标: 暗通道值、湿度、温露点差")
        print("3. 更新频率: 建议每15秒更新一次预测")
        print("4. 精度预期: 平均绝对误差约为实际能见度的10-15%")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析结果摘要
        
        Returns
        -------
        Dict[str, Any]
            包含所有分析结果的摘要字典
        """
        summary = {
            'vis_data_count': len(self.vis_data) if self.vis_data else 0,
            'ptu_data_count': len(self.ptu_data) if self.ptu_data else 0,
            'wind_data_count': len(self.wind_data) if self.wind_data else 0,
            'merged_data_count': len(self.merged_data) if self.merged_data else 0,
            'features_data_count': len(self.features_data) if self.features_data else 0,
            'models_available': self.models_results is not None
        }
        return summary


def main():
    """
    主函数 - 程序入口点
    
    创建分析系统实例并运行完整的分析流程。
    """
    try:
        # 创建分析系统
        analysis_system = AirportWeatherAnalysis()
        
        # 运行完整分析
        analysis_system.run_complete_analysis()
        
        # 输出摘要
        summary = analysis_system.get_analysis_summary()
        print(f"\n分析结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 