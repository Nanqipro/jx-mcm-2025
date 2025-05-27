# 视频特征提取系统 - 环境配置指南

## 必需库安装

### 方法1: 使用 requirements.txt 批量安装 (推荐)
```bash
pip install -r requirements.txt
```

### 方法2: 逐个安装
```bash
# 核心依赖库
pip install opencv-python>=4.5.0
pip install scikit-image>=0.18.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install openpyxl>=3.0.0

# 可视化库 (可选)
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0

# 辅助库
pip install scipy>=1.7.0
pip install pillow>=8.0.0
```

### 方法3: 使用 conda 安装
```bash
# 如果使用 Anaconda/Miniconda
conda install opencv pandas numpy scikit-image openpyxl matplotlib seaborn scipy pillow
```

## 库说明

### 核心库 (必需)
- **opencv-python**: 计算机视觉库，用于视频读取和图像处理
- **scikit-image**: 图像处理库，提供纹理分析和特征提取功能
- **numpy**: 数值计算库，矩阵运算和数学函数
- **pandas**: 数据处理库，用于数据分析和Excel文件操作
- **openpyxl**: Excel文件读写库

### 可视化库 (建议安装)
- **matplotlib**: 基础绘图库
- **seaborn**: 统计可视化库，用于特征相关性分析

### 辅助库
- **scipy**: 科学计算库，提供额外的数学函数
- **pillow**: 图像处理库，PIL的增强版本

## 验证安装

运行以下命令验证关键库是否正确安装：

```python
# 验证脚本
import cv2
import numpy as np
import pandas as pd
import skimage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import openpyxl

print("✅ 所有必需库已成功安装!")
print(f"OpenCV 版本: {cv2.__version__}")
print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")
print(f"Scikit-image 版本: {skimage.__version__}")
```

## 常见问题解决

### 1. OpenCV 安装问题
```bash
# 如果 opencv-python 安装失败，尝试:
pip install --upgrade pip
pip install opencv-contrib-python
```

### 2. Windows 系统特殊处理
```bash
# 在 Windows 上可能需要安装 Visual C++ Build Tools
# 或者使用预编译版本:
pip install --only-binary=all opencv-python scikit-image
```

### 3. macOS 系统注意事项
```bash
# 如果遇到权限问题，使用:
pip install --user [库名]
```

## 运行测试

安装完成后，可以运行代码中的验证函数：

```python
# 导入测试
from getSP import VideoFeatureExtractor

# 创建测试对象 (无需真实视频文件)
print("导入成功! 可以开始使用视频特征提取功能。")
``` 