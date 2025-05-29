#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能见度预测系统测试脚本

该脚本用于测试系统的各个模块是否正常工作。

Author: Data Analysis Team
Date: 2024
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

# 添加src目录到Python路径
sys.path.append('src')

from three import VisibilityAnalyzer
from config import get_config, DevelopmentConfig, ProductionConfig


class TestVisibilityAnalyzer(unittest.TestCase):
    """测试VisibilityAnalyzer类的各种功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.sample_data = cls._create_test_data()
        cls.test_file = 'test_data.csv'
        cls.sample_data.to_csv(cls.test_file, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        if os.path.exists('sample_data.csv'):
            os.remove('sample_data.csv')
    
    @staticmethod
    def _create_test_data() -> pd.DataFrame:
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'laplacian_var': np.random.normal(100, 30, n_samples),
            'high_freq_ratio': np.random.normal(0.15, 0.05, n_samples),
            'edge_density': np.random.normal(0.25, 0.1, n_samples),
            'contrast_std': np.random.normal(15, 5, n_samples),
            'weather_humidity_pct': np.random.normal(70, 20, n_samples),
            'weather_temperature_c': np.random.normal(15, 10, n_samples),
            'wind_wind_speed_10m': np.random.normal(5, 3, n_samples),
            'visibility_mor_raw': np.random.uniform(500, 8000, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def setUp(self):
        """每个测试前的设置"""
        self.analyzer = VisibilityAnalyzer()
    
    def test_data_loading(self):
        """测试数据加载功能"""
        self.analyzer.load_data(self.test_file)
        self.assertIsNotNone(self.analyzer.data)
        self.assertEqual(len(self.analyzer.data), 100)
        self.assertIn('visibility_mor_raw', self.analyzer.data.columns)
    
    def test_data_exploration(self):
        """测试数据探索功能"""
        self.analyzer.load_data(self.test_file)
        basic_info = self.analyzer.explore_data()
        
        self.assertIn('shape', basic_info)
        self.assertEqual(basic_info['shape'], (100, 8))
        self.assertIn('missing_values', basic_info)
        self.assertIn('target_stats', basic_info)
    
    def test_feature_analysis(self):
        """测试特征分析功能"""
        self.analyzer.load_data(self.test_file)
        correlation_results = self.analyzer.analyze_features()
        
        self.assertIsInstance(correlation_results, dict)
        # 应该包含图像特征和气象特征的分析
        self.assertTrue(len(correlation_results) > 0)
    
    def test_feature_selection(self):
        """测试特征选择功能"""
        self.analyzer.load_data(self.test_file)
        selected_features = self.analyzer.select_features(method='manual')
        
        self.assertIsInstance(selected_features, list)
        self.assertTrue(len(selected_features) > 0)
        # 检查选择的特征是否在数据中存在
        for feature in selected_features:
            self.assertIn(feature, self.analyzer.data.columns)
    
    def test_multicollinearity_detection(self):
        """测试多重共线性检测"""
        self.analyzer.load_data(self.test_file)
        selected_features = self.analyzer.select_features(method='manual')
        vif_results = self.analyzer.detect_multicollinearity(selected_features)
        
        self.assertIsInstance(vif_results, pd.DataFrame)
        self.assertIn('特征', vif_results.columns)
        self.assertIn('VIF', vif_results.columns)
    
    def test_data_preparation(self):
        """测试数据准备功能"""
        self.analyzer.load_data(self.test_file)
        X_train, X_test, y_train, y_test = self.analyzer.prepare_data()
        
        self.assertEqual(len(X_train.shape), 2)
        self.assertEqual(len(X_test.shape), 2)
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)
        
        # 检查数据划分比例
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.2, places=1)
    
    def test_model_training(self):
        """测试模型训练功能"""
        self.analyzer.load_data(self.test_file)
        X_train, X_test, y_train, y_test = self.analyzer.prepare_data()
        model_results = self.analyzer.train_models(X_train, X_test, y_train, y_test)
        
        self.assertIsInstance(model_results, dict)
        self.assertTrue(len(model_results) > 0)
        
        # 检查每个模型是否有必要的评估指标
        for model_name, results in model_results.items():
            self.assertIn('test_r2', results)
            self.assertIn('test_rmse', results)
            self.assertIn('model', results)
        
        # 检查是否选择了最佳模型
        self.assertIsNotNone(self.analyzer.best_model)
    
    def test_prediction(self):
        """测试预测功能"""
        self.analyzer.load_data(self.test_file)
        X_train, X_test, y_train, y_test = self.analyzer.prepare_data()
        self.analyzer.train_models(X_train, X_test, y_train, y_test)
        
        # 创建新数据进行预测
        new_data = pd.DataFrame({
            'laplacian_var': [100.0],
            'high_freq_ratio': [0.15],
            'edge_density': [0.25],
            'contrast_std': [15.0],
            'weather_humidity_pct': [70.0],
            'weather_temperature_c': [15.0],
            'wind_wind_speed_10m': [5.0]
        })
        
        predictions = self.analyzer.predict(new_data)
        
        self.assertEqual(len(predictions), 1)
        self.assertGreater(predictions[0], 0)  # 能见度应该大于0
    
    def test_model_equation(self):
        """测试模型公式生成"""
        self.analyzer.load_data(self.test_file)
        X_train, X_test, y_train, y_test = self.analyzer.prepare_data()
        self.analyzer.train_models(X_train, X_test, y_train, y_test)
        
        equation = self.analyzer.get_model_equation()
        
        self.assertIsInstance(equation, str)
        self.assertIn('sqrt(visibility_mor_raw)', equation)


class TestConfiguration(unittest.TestCase):
    """测试配置系统"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = get_config()
        
        self.assertIsNotNone(config.DEFAULT_DATA_FILE)
        self.assertEqual(config.TARGET_COLUMN, 'visibility_mor_raw')
        self.assertIsInstance(config.MANUAL_FEATURES, list)
        self.assertTrue(len(config.MANUAL_FEATURES) > 0)
    
    def test_development_config(self):
        """测试开发环境配置"""
        config = get_config('development')
        
        self.assertIsInstance(config, DevelopmentConfig)
        self.assertTrue(config.VERBOSE)
        self.assertEqual(config.MIN_R2_SCORE, 0.3)
    
    def test_production_config(self):
        """测试生产环境配置"""
        config = get_config('production')
        
        self.assertIsInstance(config, ProductionConfig)
        self.assertFalse(config.VERBOSE)
        self.assertEqual(config.MIN_R2_SCORE, 0.7)
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_environment_variable(self):
        """测试环境变量配置"""
        config = get_config()
        self.assertIsInstance(config, ProductionConfig)


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 创建测试数据
        test_data = TestVisibilityAnalyzer._create_test_data()
        test_file = 'integration_test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        try:
            # 初始化分析器
            analyzer = VisibilityAnalyzer(test_file)
            
            # 执行完整流程
            basic_info = analyzer.explore_data()
            correlation_results = analyzer.analyze_features()
            selected_features = analyzer.select_features(method='manual')
            vif_results = analyzer.detect_multicollinearity(selected_features)
            
            X_train, X_test, y_train, y_test = analyzer.prepare_data()
            model_results = analyzer.train_models(X_train, X_test, y_train, y_test)
            
            # 测试预测
            new_data = pd.DataFrame({
                'laplacian_var': [120.0],
                'high_freq_ratio': [0.18],
                'edge_density': [0.28],
                'contrast_std': [18.0],
                'weather_humidity_pct': [75.0],
                'weather_temperature_c': [20.0],
                'wind_wind_speed_10m': [6.0]
            })
            
            predictions = analyzer.predict(new_data)
            
            # 验证结果
            self.assertIsNotNone(basic_info)
            self.assertIsNotNone(correlation_results)
            self.assertTrue(len(selected_features) > 0)
            self.assertIsNotNone(vif_results)
            self.assertTrue(len(model_results) > 0)
            self.assertEqual(len(predictions), 1)
            self.assertGreater(predictions[0], 0)
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)


def run_performance_test():
    """运行性能测试"""
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)
    
    # 创建较大的测试数据集
    np.random.seed(42)
    n_samples = 1000
    
    large_data = {
        'laplacian_var': np.random.normal(100, 30, n_samples),
        'high_freq_ratio': np.random.normal(0.15, 0.05, n_samples),
        'edge_density': np.random.normal(0.25, 0.1, n_samples),
        'contrast_std': np.random.normal(15, 5, n_samples),
        'weather_humidity_pct': np.random.normal(70, 20, n_samples),
        'weather_temperature_c': np.random.normal(15, 10, n_samples),
        'wind_wind_speed_10m': np.random.normal(5, 3, n_samples),
        'visibility_mor_raw': np.random.uniform(500, 8000, n_samples)
    }
    
    large_df = pd.DataFrame(large_data)
    test_file = 'performance_test_data.csv'
    large_df.to_csv(test_file, index=False)
    
    try:
        import time
        
        start_time = time.time()
        
        # 执行完整分析
        analyzer = VisibilityAnalyzer(test_file)
        analyzer.explore_data()
        analyzer.analyze_features()
        analyzer.select_features(method='manual')
        
        X_train, X_test, y_train, y_test = analyzer.prepare_data()
        analyzer.train_models(X_train, X_test, y_train, y_test)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ 处理 {n_samples} 个样本耗时: {execution_time:.2f} 秒")
        print(f"平均每个样本处理时间: {(execution_time/n_samples)*1000:.2f} 毫秒")
        
        if execution_time < 30:  # 30秒内完成认为性能良好
            print("🚀 性能测试通过")
        else:
            print("⚠️ 性能可能需要优化")
    
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """主测试函数"""
    print("能见度预测系统测试套件")
    print("="*60)
    
    # 运行单元测试
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestVisibilityAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    test_suite.addTest(unittest.makeSuite(TestSystemIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 显示测试结果摘要
    print(f"\n测试结果摘要:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # 运行性能测试
    run_performance_test()
    
    # 总体评估
    if len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n🎉 所有测试通过！系统运行正常。")
        return True
    else:
        print(f"\n❌ 发现问题，请检查上述失败和错误。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 