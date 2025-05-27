#!/usr/bin/env python3
"""
GPU环境修复脚本
自动检测并修复GPU加速环境的常见问题
"""

import subprocess
import sys
import importlib.util


def run_command(cmd, description=""):
    """运行系统命令"""
    print(f"🔧 {description}")
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
            return True
        else:
            print("❌ 失败")
            print(f"错误: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def check_cuda_available():
    """检查CUDA是否可用"""
    print("\n🔍 === 检查CUDA环境 ===")
    
    # 检查nvidia-smi
    if run_command("nvidia-smi", "检查NVIDIA驱动"):
        print("✅ NVIDIA驱动已安装")
    else:
        print("❌ NVIDIA驱动未安装或不可用")
        print("请先安装NVIDIA显卡驱动: https://www.nvidia.com/drivers/")
        return False
    
    # 检查nvcc
    if run_command("nvcc --version", "检查CUDA编译器"):
        print("✅ CUDA Toolkit已安装")
        return True
    else:
        print("⚠️ CUDA Toolkit未安装")
        print("请安装CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        return False


def fix_pytorch():
    """修复PyTorch CUDA支持"""
    print("\n🔧 === 修复PyTorch ===")
    
    # 卸载现有PyTorch
    print("卸载现有PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", "卸载CPU版PyTorch")
    
    # 安装GPU版PyTorch
    print("安装GPU版PyTorch...")
    success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "安装PyTorch GPU版 (CUDA 12.1)"
    )
    
    if not success:
        print("尝试CUDA 11.8版本...")
        success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
            "安装PyTorch GPU版 (CUDA 11.8)"
        )
    
    return success


def fix_cupy():
    """修复CuPy"""
    print("\n🔧 === 修复CuPy ===")
    
    # 尝试不同版本的CuPy
    cupy_versions = [
        ("cupy-cuda12x", "CUDA 12.x"),
        ("cupy-cuda11x", "CUDA 11.x"),
        ("cupy", "通用版本")
    ]
    
    for package, desc in cupy_versions:
        print(f"尝试安装 {package} ({desc})...")
        if run_command(f"pip install {package}", f"安装{desc}"):
            return True
    
    return False


def test_gpu_imports():
    """测试GPU库导入"""
    print("\n🧪 === 测试库导入 ===")
    
    # 测试CuPy
    try:
        import cupy as cp
        print("✅ CuPy导入成功")
        try:
            # 简单测试
            a = cp.array([1, 2, 3])
            print(f"✅ CuPy基本功能正常: {a}")
        except Exception as e:
            print(f"⚠️ CuPy功能测试失败: {e}")
    except ImportError as e:
        print(f"❌ CuPy导入失败: {e}")
    
    # 测试PyTorch CUDA
    try:
        import torch
        print(f"✅ PyTorch导入成功: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA可用: {torch.cuda.get_device_name()}")
        else:
            print("❌ PyTorch CUDA不可用")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")


def create_simplified_gpu_version():
    """创建简化的GPU版本"""
    print("\n📝 === 创建兼容版本 ===")
    
    # 读取原始GPU代码
    try:
        with open('src/getSP_gpu.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建更兼容的版本
        simplified_content = content.replace(
            'self.use_gpu = use_gpu and CUPY_AVAILABLE',
            'self.use_gpu = use_gpu and CUPY_AVAILABLE and torch.cuda.is_available()'
        )
        
        # 保存简化版本
        with open('src/getSP_gpu_fixed.py', 'w', encoding='utf-8') as f:
            f.write(simplified_content)
        
        print("✅ 创建了修复版本: src/getSP_gpu_fixed.py")
        return True
        
    except Exception as e:
        print(f"❌ 创建修复版本失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 GPU环境自动修复工具")
    print("=" * 50)
    
    # 1. 检查CUDA环境
    cuda_ok = check_cuda_available()
    
    # 2. 修复PyTorch
    if cuda_ok:
        pytorch_ok = fix_pytorch()
    else:
        print("⚠️ 跳过PyTorch修复 (CUDA不可用)")
        pytorch_ok = False
    
    # 3. 修复CuPy
    if cuda_ok:
        cupy_ok = fix_cupy()
    else:
        print("⚠️ 跳过CuPy修复 (CUDA不可用)")
        cupy_ok = False
    
    # 4. 测试导入
    test_gpu_imports()
    
    # 5. 创建兼容版本
    create_simplified_gpu_version()
    
    # 总结
    print("\n📋 === 修复总结 ===")
    if cuda_ok and pytorch_ok and cupy_ok:
        print("🎉 GPU环境修复完成！可以使用GPU加速")
        print("运行: python src/getSP_gpu.py")
    elif cuda_ok:
        print("⚠️ 部分修复完成，可能仍需手动处理")
        print("请尝试: python src/getSP_gpu_fixed.py")
    else:
        print("❌ GPU环境不可用，建议使用CPU版本")
        print("运行: python src/getSP.py")
    
    print("\n📖 详细说明请查看: gpu_setup_guide.md")


if __name__ == "__main__":
    main() 