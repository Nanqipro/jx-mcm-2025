# 基于大雾背景视频学习的图像模糊程度分析系统

## 项目简介

本项目基于2020年华为杯数学建模竞赛E题的思路，实现江西省数学建模比赛第一题的解决方案：建立反映大雾导致的视频图像模糊程度的数学模型。

## 核心功能

### 1. 多维度图像模糊程度评价
- **拉普拉斯方差**: 经典的图像清晰度评价指标
- **梯度幅值**: 基于边缘信息的清晰度评价  
- **Sobel方差**: Sobel算子的方差计算
- **Tenengrad**: 基于Sobel算子的清晰度评价
- **Brenner梯度**: 基于相邻像素差值的清晰度评价
- **暗通道先验**: 基于何凯明暗通道先验算法，评估雾的浓度
- **能见度指数**: 综合对比度、边缘密度和亮度分布熵的综合评价

### 2. 数学模型构建
- 基于主成分分析和方差贡献度的权重计算
- 标准化特征处理
- 综合模糊度指数建模

### 3. 时间序列分析
- 支持视频帧序列的连续分析
- 模糊程度时间变化趋势可视化
- 统计特征分析

## 技术特点

### 基于2020E.md的核心思路：
1. **多指标融合**: 参考原文中的"多元多项式拟合"思想，采用多种图像质量指标
2. **特征工程**: 借鉴原文的"特征重要性"分析方法
3. **暗通道先验**: 采用原文第三题中提到的暗通道先验算法
4. **标准化处理**: 参考原文的数据预处理方法

### 数学模型：

综合模糊度指数计算公式：
```BlurIndex = Σ(wi × Fi_normalized)
```

其中：
- `wi`: 第i个特征的权重（基于方差贡献度）
- `Fi_normalized`: 第i个特征的标准化值
- 权重计算：`wi = var(Fi) / Σvar(Fj)`

## 安装和使用

### 环境要求
- Python 3.8+
- 推荐使用conda或virtualenv创建虚拟环境

### 安装依赖
```bash
pip install -r requirements.txt
```

### 使用方法

#### 1. 直接运行演示程序
```bash
python fog_image_blur_analysis.py
```

程序将自动：
- 生成测试用的雾化图像序列
- 分析图像模糊程度
- 建立数学模型
- 生成可视化结果
- 导出Excel分析报告

#### 2. 使用自己的图像数据
```python
from fog_image_blur_analysis import FogBlurAnalyzer

# 创建分析器
analyzer = FogBlurAnalyzer()

# 分析图像序列
results_df = analyzer.process_image_sequence("your_image_folder")

# 建立数学模型
model_params = analyzer.create_blur_model(results_df)

# 可视化结果
analyzer.visualize_blur_analysis(results_df, save_path="results.png")

# 导出结果
analyzer.export_results(results_df, model_params, "analysis_results.xlsx")
```

#### 3. 分析单张图像
```python
analyzer = FogBlurAnalyzer()
metrics = analyzer.analyze_single_image("path/to/image.jpg")
print(metrics)
```

## 输出结果

### 1. 控制台输出
- 模型参数和权重
- 统计信息摘要
- 处理进度信息

### 2. 可视化图表
- 6个子图展示不同指标随时间的变化
- 趋势线分析
- 保存为高分辨率PNG图像

### 3. Excel分析报告
包含以下工作表：
- **原始数据**: 所有图像的详细指标数据
- **统计摘要**: 描述性统计信息
- **模型参数**: 各特征的权重系数
- **相关性矩阵**: 特征间的相关性分析

## 核心算法详解

### 1. 暗通道先验算法
```python
def calculate_dark_channel_prior(self, image, patch_size=15):
    """
    基于何凯明的暗通道先验理论：
    在无雾的户外图像中，至少有一个颜色通道具有很低的强度值
    雾的存在会增加这些低强度值
    """
    min_channel = np.min(image, axis=2)  # RGB三通道最小值
    kernel = np.ones((patch_size, patch_size), np.uint8)
    dark_channel = cv2.erode(min_channel, kernel)  # 局部最小值
    return np.mean(dark_channel) / 255.0
```

### 2. 综合能见度指数
```python
def calculate_visibility_index(self, image):
    """
    基于三个维度评估能见度：
    1. 对比度（标准差）
    2. 边缘密度（Canny边缘检测）
    3. 亮度分布熵（信息熵）
    """
    contrast = np.std(image)
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # 加权综合
    visibility_index = 0.4 * contrast + 0.3 * edge_density * 1000 + 0.3 * entropy
    return visibility_index
```

## 模型验证

### 合成数据验证
程序自动生成具有不同雾化程度的图像序列，验证模型的有效性：
- 雾化强度从0逐渐增加到最大值
- 模型指标应呈现相应的变化趋势
- 模糊度指数与雾化强度呈正相关

### 真实数据应用
适用于：
- 机场监控视频
- 高速公路监控视频  
- 气象观测图像
- 其他户外监控场景

## 技术创新点

1. **多指标融合建模**: 综合7种不同的图像质量评价指标
2. **自适应权重计算**: 基于数据特征自动确定各指标权重
3. **时间序列分析**: 支持连续视频帧的模糊程度变化分析
4. **标准化处理**: 确保不同量纲指标的可比性
5. **完整工作流**: 从图像处理到模型建立到结果导出的完整解决方案

## 扩展应用

本系统可进一步扩展用于：
- 实时能见度监测
- 大雾预警系统
- 交通安全评估
- 气象观测自动化
- 图像质量自动评价

## 参考文献

1. 2020年华为杯数学建模竞赛E题优秀论文
2. K. He, J. Sun, X. Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009
3. S. K. Nayar, S. G. Narasimhan, "Vision in bad weather", ICCV 1999
4. R. T. Tan, "Visibility in bad weather from a single image", CVPR 2008 