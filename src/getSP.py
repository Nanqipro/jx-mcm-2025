import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SimpleFogVideoAnalyzer:
    def __init__(self, video_path):
        """
        简化版雾天视频分析器 - 直接使用cap.set跳帧

        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.frames = []
        self.blur_indices = []
        self.timestamps = []
        self.features_df = None

    def extract_frames_direct_jump(self, interval_seconds=30):
        """
        直接使用cap.set跳转提取帧 - 最简单的方法

        Args:
            interval_seconds: 提取间隔（秒）
        """
        print("使用cap.set直接跳帧提取...")
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")

        # 计算需要跳转的帧位置
        frame_positions = []
        time_positions = []

        current_time = 0
        while current_time < duration:
            frame_pos = int(current_time * fps)
            if frame_pos < total_frames:
                frame_positions.append(frame_pos)
                time_positions.append(current_time)
            current_time += interval_seconds

        print(f"将提取 {len(frame_positions)} 帧")

        # 直接跳转并提取帧
        success_count = 0
        with tqdm(total=len(frame_positions), desc="跳帧提取") as pbar:
            for frame_pos, time_pos in zip(frame_positions, time_positions):
                # 关键：直接跳转到指定帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if ret:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frames.append(gray_frame)
                    self.timestamps.append(time_pos)
                    success_count += 1
                else:
                    print(f"警告: 无法读取第 {frame_pos} 帧")

                pbar.update(1)

        cap.release()
        print(f"成功提取 {success_count} 帧图像")

    def extract_frames_by_time(self, interval_seconds=30):
        """
        按时间跳转提取帧（另一种方式）

        Args:
            interval_seconds: 提取间隔（秒）
        """
        print("按时间跳转提取帧...")
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")

        # 生成时间点列表
        time_points = np.arange(0, duration, interval_seconds)
        print(f"将提取 {len(time_points)} 帧")

        success_count = 0
        with tqdm(total=len(time_points), desc="时间跳转提取") as pbar:
            for time_point in time_points:
                # 方法1：按毫秒跳转
                cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
                ret, frame = cap.read()

                if ret:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frames.append(gray_frame)
                    self.timestamps.append(time_point)
                    success_count += 1
                else:
                    print(f"警告: 无法读取时间点 {time_point:.1f}s 的帧")

                pbar.update(1)

        cap.release()
        print(f"成功提取 {success_count} 帧图像")

    def calculate_blur_score(self, image):
        """
        计算单个图像的模糊分数

        Args:
            image: 灰度图像

        Returns:
            dict: 包含模糊度相关特征的字典
        """
        features = {}

        # 1. 图像方差（模糊图像方差较小）
        features['variance'] = np.var(image)

        # 2. Laplacian方差（经典的模糊检测方法）
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)

        # 3. Sobel边缘强度
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        features['sobel_mean'] = np.mean(sobel_magnitude)

        # 4. 标准差对比度
        features['std_contrast'] = np.std(image)

        # 5. Brenner梯度（焦点测量）
        features['brenner'] = np.sum((image[:-2, :] - image[2:, :]) ** 2)

        return features

    def analyze_all_frames(self):
        """
        分析所有提取的帧
        """
        print("正在分析所有帧的模糊度...")

        all_features = []

        for i, frame in enumerate(tqdm(self.frames, desc="模糊度分析")):
            # 计算基本信息
            frame_data = {
                'frame_index': i,
                'timestamp': self.timestamps[i]
            }

            # 计算模糊度特征
            blur_features = self.calculate_blur_score(frame)
            frame_data.update(blur_features)

            all_features.append(frame_data)

        # 转换为DataFrame
        self.features_df = pd.DataFrame(all_features)

        # 标准化特征
        feature_cols = ['variance', 'laplacian_var', 'sobel_mean', 'std_contrast', 'brenner']
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(self.features_df[feature_cols])

        # 计算综合模糊指数（值越大越模糊）
        # 这些特征值越小表示越模糊，所以需要反转
        blur_index = 1 - np.mean(normalized_features, axis=1)

        self.features_df['blur_index'] = blur_index
        self.blur_indices = blur_index.tolist()

        print(f"模糊度分析完成！")

    def estimate_visibility(self, blur_index, k=1000):
        """
        根据模糊指数估算能见度

        Args:
            blur_index: 模糊指数 (0-1)
            k: 经验常数

        Returns:
            估算的能见度值(米)
        """
        # 避免数学错误
        blur_index = np.clip(blur_index, 0.01, 0.99)

        # 简单的反比例关系
        visibility = k * (1 - blur_index) / blur_index

        # 限制合理范围
        visibility = np.clip(visibility, 50, 5000)

        return visibility

    def quick_analysis(self, interval_seconds=30, method='frame'):
        """
        快速分析完整流程

        Args:
            interval_seconds: 提取间隔
            method: 'frame' 按帧跳转, 'time' 按时间跳转
        """
        print("=== 开始快速雾天分析 ===")

        # 1. 提取帧
        if method == 'frame':
            self.extract_frames_direct_jump(interval_seconds)
        else:
            self.extract_frames_by_time(interval_seconds)

        if len(self.frames) == 0:
            print("错误：未能提取到任何帧！")
            return None

        # 2. 分析模糊度
        self.analyze_all_frames()

        # 3. 估算能见度
        visibility_values = [self.estimate_visibility(bi) for bi in self.blur_indices]
        self.features_df['estimated_visibility'] = visibility_values

        # 4. 生成报告
        self.print_analysis_report()

        # 5. 可视化
        self.create_simple_plot()

        return self.features_df

    def print_analysis_report(self):
        """打印分析报告"""
        print("\n" + "=" * 50)
        print("📊 雾天能见度分析报告")
        print("=" * 50)

        print(f"🎬 视频文件: {os.path.basename(self.video_path)}")
        print(f"📊 分析帧数: {len(self.frames)}")
        print(f"⏱️  时间跨度: {self.timestamps[-1]:.1f} 秒")

        # 模糊指数统计
        blur_stats = {
            '平均值': np.mean(self.blur_indices),
            '最小值': np.min(self.blur_indices),
            '最大值': np.max(self.blur_indices),
            '标准差': np.std(self.blur_indices)
        }

        print(f"\n🌫️  模糊指数统计:")
        for key, value in blur_stats.items():
            print(f"   {key}: {value:.4f}")

        # 能见度统计
        visibility_values = self.features_df['estimated_visibility'].values
        vis_stats = {
            '平均值': np.mean(visibility_values),
            '最小值': np.min(visibility_values),
            '最大值': np.max(visibility_values),
            '标准差': np.std(visibility_values)
        }

        print(f"\n👁️  估算能见度统计 (米):")
        for key, value in vis_stats.items():
            print(f"   {key}: {value:.1f}")

        # 安全等级评估
        danger = np.sum(visibility_values < 400)
        warning = np.sum((visibility_values >= 400) & (visibility_values < 800))
        safe = np.sum(visibility_values >= 800)
        total = len(visibility_values)

        print(f"\n✈️  航空安全评估:")
        print(f"   🔴 危险级别 (<400m): {danger} 帧 ({danger / total * 100:.1f}%)")
        print(f"   🟡 警告级别 (400-800m): {warning} 帧 ({warning / total * 100:.1f}%)")
        print(f"   🟢 安全级别 (>800m): {safe} 帧 ({safe / total * 100:.1f}%)")

    def create_simple_plot(self):
        """创建简单的可视化图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 时间轴（转换为分钟）
        time_minutes = np.array(self.timestamps) / 60

        # 模糊指数变化
        ax1.plot(time_minutes, self.blur_indices, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('模糊指数随时间变化', fontsize=14, fontweight='bold')
        ax1.set_xlabel('时间 (分钟)')
        ax1.set_ylabel('模糊指数')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 能见度变化
        visibility = self.features_df['estimated_visibility']
        ax2.plot(time_minutes, visibility, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.axhline(y=400, color='red', linestyle='--', alpha=0.8, label='危险线 (400m)')
        ax2.axhline(y=800, color='orange', linestyle='--', alpha=0.8, label='警告线 (800m)')
        ax2.set_title('估算能见度随时间变化', fontsize=14, fontweight='bold')
        ax2.set_xlabel('时间 (分钟)')
        ax2.set_ylabel('能见度 (米)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, output_file="fog_analysis_results.csv"):
        """
        保存结果到CSV文件
        
        Parameters
        ----------
        output_file : str
            输出文件名，支持指定路径
        """
        if self.features_df is not None:
            # 确保目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"📁 创建目录: {output_dir}")
            
            self.features_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"📁 结果已保存到: {output_file}")
        else:
            print("❌ 没有结果可保存，请先运行分析！")


def main():
    """主函数 - 简化版"""
    # 视频文件路径（修改为你的路径）
    video_path = "../机场视频/a.mp4"

    # 检查文件
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频文件: {video_path}")
        print("请修改video_path为正确的路径")
        return

    try:
        # 创建分析器
        analyzer = SimpleFogVideoAnalyzer(video_path)

        # 执行快速分析
        # interval_seconds: 每隔多少秒提取一帧
        # method: 'frame'(按帧跳转) 或 'time'(按时间跳转)
        results = analyzer.quick_analysis(interval_seconds=60, method='frame')

        if results is not None:
            # 保存结果到指定目录
            analyzer.save_results("../results/雾天分析结果.csv")

            # 显示前几个结果
            print("\n📋 前10帧分析结果:")
            print(results[['timestamp', 'blur_index', 'estimated_visibility']].head(10).to_string(index=False))

    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()