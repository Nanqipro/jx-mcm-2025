#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体测试脚本
================

用于验证matplotlib中文字体显示是否正常
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties, fontManager
import platform
import numpy as np

def setup_chinese_fonts():
    """设置中文字体支持"""
    system = platform.system()
    
    if system == "Windows":
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun', 'FangSong']
    elif system == "Darwin":  # macOS
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei']
    
    font_list.extend(['DejaVu Sans', 'Arial Unicode MS', 'sans-serif'])
    
    matplotlib.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    return font_list

def list_available_fonts():
    """列出可用的中文字体"""
    chinese_fonts = []
    for font in fontManager.ttflist:
        font_name = font.name
        if any(keyword in font_name for keyword in [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
            'PingFang', 'Hiragino', 'STHeiti', 'WenQuanYi', 'Noto Sans CJK'
        ]):
            chinese_fonts.append(font_name)
    
    return list(set(chinese_fonts))

def test_chinese_display():
    """测试中文字体显示"""
    print("=" * 50)
    print("中文字体测试")
    print("=" * 50)
    
    # 设置字体
    font_list = setup_chinese_fonts()
    available_fonts = list_available_fonts()
    
    print(f"系统平台: {platform.system()}")
    print(f"配置的字体列表: {font_list[:3]}...")
    print(f"检测到的中文字体: {available_fonts}")
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 测试1: 基础中文显示
    ax1 = axes[0, 0]
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax1.plot(x, y, label='正弦波')
    ax1.set_title('基础中文显示测试')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('振幅')
    ax1.legend()
    ax1.grid(True)
    
    # 测试2: 使用FontProperties
    ax2 = axes[0, 1]
    if available_fonts:
        font_prop = FontProperties()
        font_prop.set_family(available_fonts[0])
        ax2.plot(x, np.cos(x), label='余弦波', color='red')
        ax2.set_title('使用FontProperties测试', fontproperties=font_prop)
        ax2.set_xlabel('时间 (秒)', fontproperties=font_prop)
        ax2.set_ylabel('振幅', fontproperties=font_prop)
        ax2.legend(prop=font_prop)
    else:
        ax2.plot(x, np.cos(x), label='余弦波', color='red')
        ax2.set_title('使用FontProperties测试')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('振幅')
        ax2.legend()
    ax2.grid(True)
    
    # 测试3: 中文标题和说明
    ax3 = axes[1, 0]
    data = np.random.normal(100, 15, 1000)
    ax3.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(data), color='red', linestyle='--', 
               label=f'均值: {np.mean(data):.1f}')
    ax3.set_title('数据分布图')
    ax3.set_xlabel('数值')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 测试4: 复杂中文文本
    ax4 = axes[1, 1]
    categories = ['温度', '湿度', '气压', '风速', '能见度']
    values = [25, 60, 1013, 15, 8000]
    ax4.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax4.set_title('气象参数监测')
    ax4.set_ylabel('测量值')
    
    # 旋转x轴标签
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 设置整体标题
    fig.suptitle('matplotlib中文字体显示测试', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图片
    plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight')
    print("\n✓ 测试图表已保存为 'chinese_font_test.png'")
    print("请检查图片中的中文是否正常显示")
    
    plt.show()

if __name__ == "__main__":
    test_chinese_display() 