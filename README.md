# 能见度预测分析系统

基于图像特征和气象数据的能见度预测模型，适用于航空气象、交通安全等应用场景。

## 项目概述

该项目实现了一个完整的能见度预测分析流程，包含：

- **数据探索分析**：基本统计信息、缺失值检查、分布分析
- **特征工程**：图像特征、气象特征、风速特征的相关性分析
- **多重共线性检测**：VIF分析避免冗余特征
- **模型训练**：多种回归模型（线性回归、岭回归、Lasso、弹性网络）
- **模型验证**：残差分析、交叉验证、性能评估
- **预测应用**：新数据的能见度预测

## 技术特点

- ✅ **现代Python语法**：类型注解、异步支持
- ✅ **完整中文注释**：遵循numpy风格文档规范
- ✅ **环境变量配置**：安全的敏感信息管理
- ✅ **模块化设计**：清晰的代码结构和功能分层
- ✅ **可视化图表**：英文标签的专业图表展示

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from src.three import VisibilityAnalyzer

# 初始化分析器
analyzer = VisibilityAnalyzer('complete_synced_data.csv')

# 执行完整分析流程
basic_info = analyzer.explore_data()
correlation_results = analyzer.analyze_features()
analyzer.visualize_correlations()

# 训练模型
X_train, X_test, y_train, y_test = analyzer.prepare_data()
model_results = analyzer.train_models(X_train, X_test, y_train, y_test)

# 残差分析
analyzer.analyze_residuals(X_test, y_test)

# 获取预测公式
equation = analyzer.get_model_equation()
```

### 预测新数据

```python
# 使用训练好的模型预测
new_data = pd.DataFrame({
    'laplacian_var': [100.5],
    'high_freq_ratio': [0.15],
    'edge_density': [0.25],
    'contrast_std': [15.2],
    'weather_humidity_pct': [75.0],
    'weather_temperature_c': [20.5],
    'wind_wind_speed_10m': [5.2]
})

predicted_visibility = analyzer.predict(new_data)
print(f"预测能见度: {predicted_visibility[0]:.1f} 米")
```

## 数据特征说明

### 图像特征 (9个)
- `laplacian_var`: 拉普拉斯方差 (图像清晰度)
- `sobel_mean`: Sobel算子均值 (边缘强度)
- `high_freq_ratio`: 高频分量比例
- `edge_density`: 边缘密度
- `contrast_std`: 对比度标准差
- 等...

### 气象特征 (3个)
- `weather_humidity_pct`: 相对湿度 (%)
- `weather_temperature_c`: 温度 (°C)
- `weather_pressure_hpa`: 气压 (hPa)

### 风速特征 (1个)
- `wind_wind_speed_10m`: 10米高度风速 (m/s)

### 目标变量
- `visibility_mor_raw`: 原始能见度测量值 (米)

## 模型性能

最终模型采用Lasso回归，具有以下性能：

- **R² = 0.8791** (原始尺度)
- **RMSE = 917.7米**
- **MAE = 568.9米**
- **MAPE = 34.7%**

## 核心算法

模型使用平方根变换改善目标变量的偏态分布：

```
sqrt(visibility_mor_raw) = β₀ + Σ(βᵢ × Xᵢ_standardized)
```

其中特征已进行Z-score标准化，最终预测值通过平方运算还原到原始尺度。

## 项目结构

```
├── src/
│   ├── three.py           # 主要分析模块
│   └── three.ipynb        # 原始Jupyter notebook
├── requirements.txt       # 项目依赖
├── README.md             # 项目说明
└── complete_synced_data.csv # 数据文件 (需要提供)
```

## 注意事项

1. **数据文件**：需要将数据文件 `complete_synced_data.csv` 放在项目根目录
2. **中文字体**：如果图表中文显示异常，请安装SimHei字体
3. **环境变量**：敏感配置请使用环境变量而非硬编码
4. **内存使用**：大数据集建议分批处理避免内存溢出

## 开发规范

- 所有函数必须包含完整的中文文档字符串
- 使用类型注解提高代码可读性
- 图表标题和标签必须使用英文
- 遵循PEP 8代码风格规范

## 扩展功能

可基于此框架扩展：

- **实时预测系统**：接入实时气象和图像数据
- **Web界面**：Flask/Django开发在线预测服务
- **移动应用**：集成到气象类移动应用
- **模型优化**：集成深度学习模型提升精度

## 作者

Data Analysis Team - 2024

## 许可证

本项目遵循MIT许可证开源。 