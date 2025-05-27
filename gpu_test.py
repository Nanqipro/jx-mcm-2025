import numpy as np
import time


def test_gpu_environment():
    """测试GPU环境配置"""
    print("🔍 === GPU环境检测开始 ===\n")
    results = {}
    
    # 1. 测试CuPy
    print("1️⃣ 测试 CuPy...")
    try:
        import cupy as cp
        # 创建测试数组
        a_gpu = cp.array([1, 2, 3, 4, 5])
        b_gpu = cp.array([2, 3, 4, 5, 6])
        c_gpu = a_gpu + b_gpu
        results['CuPy'] = '✅ 正常工作'
        
        print(f"   CuPy版本: {cp.__version__}")
        
        # 获取GPU设备信息 (修复API调用)
        try:
            device_id = cp.cuda.Device().id
            device_name = f"GPU Device {device_id}"
            print(f"   GPU设备: {device_name}")
        except:
            try:
                # 尝试使用cupy.cuda.runtime
                device_count = cp.cuda.runtime.getDeviceCount()
                print(f"   检测到 {device_count} 个GPU设备")
            except:
                print("   GPU设备: 无法获取设备信息")
        
        # 获取GPU内存信息
        try:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            print(f"   GPU内存: {(total_mem - free_mem) / 1024**3:.1f}GB / {total_mem / 1024**3:.1f}GB (已用/总计)")
        except:
            print("   GPU内存: 无法获取内存信息")
        
    except ImportError:
        results['CuPy'] = '❌ 未安装 CuPy'
        print("   ❌ CuPy未安装，请运行: pip install cupy-cuda12x")
    except Exception as e:
        results['CuPy'] = f'❌ 错误: {e}'
        print(f"   ❌ CuPy错误: {e}")
    
    print()
    
    # 2. 测试PyTorch CUDA
    print("2️⃣ 测试 PyTorch CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
            gpu_count = torch.cuda.device_count()
            results['PyTorch CUDA'] = f'✅ 检测到 {gpu_count} 个GPU: {device}'
            print(f"   GPU设备: {device}")
            print(f"   GPU数量: {gpu_count}")
            print(f"   PyTorch版本: {torch.__version__}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            results['PyTorch CUDA'] = '❌ CUDA不可用'
            print("   ❌ PyTorch未检测到CUDA")
    except ImportError:
        results['PyTorch CUDA'] = '❌ 未安装 PyTorch'
        print("   ❌ PyTorch未安装")
    except Exception as e:
        results['PyTorch CUDA'] = f'❌ 错误: {e}'
        print(f"   ❌ PyTorch错误: {e}")
    
    print()
    
    # 3. 测试OpenCV CUDA支持
    print("3️⃣ 测试 OpenCV CUDA...")
    try:
        import cv2
        print(f"   OpenCV版本: {cv2.__version__}")
        
        # 检查CUDA设备数量
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                results['OpenCV CUDA'] = f'✅ 检测到 {cuda_devices} 个CUDA设备'
                print(f"   CUDA设备数量: {cuda_devices}")
            else:
                results['OpenCV CUDA'] = '⚠️ 未启用CUDA支持'
                print("   ⚠️ OpenCV未启用CUDA支持")
        except AttributeError:
            results['OpenCV CUDA'] = '⚠️ 版本不支持CUDA'
            print("   ⚠️ 当前OpenCV版本不支持CUDA")
            
    except ImportError:
        results['OpenCV CUDA'] = '❌ 未安装 OpenCV'
        print("   ❌ OpenCV未安装")
    except Exception as e:
        results['OpenCV CUDA'] = f'❌ 错误: {e}'
        print(f"   ❌ OpenCV错误: {e}")
    
    print()
    
    # 4. 测试scikit-image
    print("4️⃣ 测试 scikit-image...")
    try:
        import skimage
        from skimage.feature import local_binary_pattern
        results['scikit-image'] = '✅ 正常工作'
        print(f"   scikit-image版本: {skimage.__version__}")
    except ImportError:
        results['scikit-image'] = '❌ 未安装'
        print("   ❌ scikit-image未安装")
    except Exception as e:
        results['scikit-image'] = f'❌ 错误: {e}'
        print(f"   ❌ scikit-image错误: {e}")
    
    print()
    
    # 输出汇总结果
    print("📋 === 环境检测汇总 ===")
    for component, status in results.items():
        print(f"{component}: {status}")
    
    # 性能测试
    if 'CuPy' in results and '✅' in results['CuPy']:
        print("\n⚡ === GPU性能测试 ===")
        test_gpu_performance()
    else:
        print("\n❌ 跳过性能测试 (CuPy不可用)")
        
    return results


def test_gpu_performance():
    """GPU性能对比测试"""
    try:
        import cupy as cp
        
        # 测试不同大小的矩阵
        test_sizes = [1000, 2000, 5000]
        
        print("矩阵乘法性能对比:")
        print("大小\t\tCPU耗时\t\tGPU耗时\t\t加速比")
        print("-" * 60)
        
        for size in test_sizes:
            try:
                # CPU测试
                start_time = time.time()
                a_cpu = np.random.random((size, size)).astype(np.float32)
                b_cpu = np.random.random((size, size)).astype(np.float32)
                c_cpu = np.dot(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                
                # GPU测试
                start_time = time.time()
                a_gpu = cp.random.random((size, size), dtype=cp.float32)
                b_gpu = cp.random.random((size, size), dtype=cp.float32)
                c_gpu = cp.dot(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()  # 等待GPU计算完成
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"{size}x{size}\t\t{cpu_time:.3f}s\t\t{gpu_time:.3f}s\t\t{speedup:.1f}x")
                
                # 清理GPU内存
                del a_gpu, b_gpu, c_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"{size}x{size}\t\t错误: {e}")
        
        print("\n💡 性能说明:")
        print("- 加速比 > 3x: GPU加速效果显著")
        print("- 加速比 2-3x: GPU加速效果良好") 
        print("- 加速比 < 2x: 建议检查GPU配置")
        
    except Exception as e:
        print(f"性能测试失败: {e}")


def test_video_feature_extraction():
    """测试视频特征提取功能"""
    print("\n🎬 === 视频特征提取功能测试 ===")
    
    try:
        # 尝试导入GPU版本
        print("测试GPU版本导入...")
        try:
            from src.getSP_gpu import VideoFeatureExtractorGPU
            print("✅ GPU版本导入成功")
            gpu_available = True
        except ImportError as e:
            print(f"❌ GPU版本导入失败: {e}")
            gpu_available = False
        
        # 尝试导入CPU版本
        print("测试CPU版本导入...")
        try:
            from src.getSP import VideoFeatureExtractor
            print("✅ CPU版本导入成功")
            cpu_available = True
        except ImportError as e:
            print(f"❌ CPU版本导入失败: {e}")
            cpu_available = False
        
        if gpu_available:
            print("\n🚀 推荐使用GPU加速版本:")
            print("from src.getSP_gpu import quick_extract_features_gpu")
            print("extractor, df = quick_extract_features_gpu('your_video.mp4')")
            
        elif cpu_available:
            print("\n💻 可使用CPU版本:")
            print("from src.getSP import quick_extract_features")
            print("extractor, df = quick_extract_features('your_video.mp4')")
        else:
            print("\n❌ 视频特征提取功能不可用，请检查代码文件")
            
    except Exception as e:
        print(f"测试视频特征提取功能时出错: {e}")


def provide_installation_guide():
    """提供安装指导"""
    print("\n📋 === 安装指导 ===")
    
    print("🔧 基础库安装:")
    print("pip install numpy pandas opencv-contrib-python scikit-image openpyxl matplotlib seaborn")
    
    print("\n⚡ GPU加速库安装:")
    print("# 对于CUDA 12.x:")
    print("pip install cupy-cuda12x torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n# 对于CUDA 11.x:")
    print("pip install cupy-cuda11x torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n📦 一键安装:")
    print("pip install -r requirements_gpu.txt")
    
    print("\n📖 详细指南:")
    print("查看 gpu_setup_guide.md 获取完整安装说明")


if __name__ == "__main__":
    print("🚀 GPU环境检测和性能测试工具")
    print("=" * 50)
    
    # 环境检测
    results = test_gpu_environment()
    
    # 视频特征提取功能测试
    test_video_feature_extraction()
    
    # 根据结果提供建议
    gpu_ready = ('CuPy' in results and '✅' in results['CuPy'])
    
    if gpu_ready:
        print("\n🎉 === GPU环境配置完成 ===")
        print("您的系统已准备好使用GPU加速视频特征提取！")
        print("\n快速开始:")
        print("python src/getSP_gpu.py")
    else:
        print("\n⚠️ === GPU环境需要配置 ===")
        provide_installation_guide()
    
    print("\n" + "=" * 50) 