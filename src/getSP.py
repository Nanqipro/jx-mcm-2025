import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from skimage import feature, measure, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import os
import warnings

warnings.filterwarnings('ignore')


class VideoFeatureExtractor:
    """
    视频图像模糊特征提取器
    专门用于分析大雾背景下的视频图像特征
    """

    def __init__(self, video_path, start_time_offset=28):
        """
        初始化特征提取器

        Args:
            video_path: 视频文件路径
            start_time_offset: 视频开始时间偏移(秒)，默认28秒
        """
        self.video_path = video_path
        self.start_time_offset = start_time_offset
        self.features_data = []
        self.video_info = {}

        # 初始化视频信息
        self._initialize_video_info()

    def _initialize_video_info(self):
        """初始化视频基本信息"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        # 获取视频的原始帧率信息
        detected_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 根据您的说明，实际帧率是25 FPS
        actual_fps = 25.0

        self.video_info = {
            'fps': actual_fps,  # 使用实际帧率25 FPS
            'detected_fps': detected_fps,  # 保存检测到的帧率用于对比
            'frame_count': frame_count,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': frame_count / actual_fps  # 使用实际帧率计算时长
        }
        cap.release()

        print(f"视频信息:")
        print(f"  分辨率: {self.video_info['width']}x{self.video_info['height']}")
        print(f"  实际帧率: {self.video_info['fps']:.2f} FPS (25帧/秒)")
        print(f"  检测帧率: {self.video_info['detected_fps']:.2f} FPS")
        print(f"  总帧数: {self.video_info['frame_count']}")
        print(f"  实际时长: {self.video_info['duration']:.2f} 秒")

    def calculate_timestamp(self, frame_index):
        """
        计算帧的物理时间戳 (基于25 FPS)

        Args:
            frame_index: 帧索引

        Returns:
            datetime: 物理时间戳
        """
        # 计算从视频开始的时间偏移 (25帧 = 1秒)
        seconds_from_start = frame_index / 25.0  # 使用固定的25 FPS
        total_seconds = self.start_time_offset + seconds_from_start

        # 转换为时:分:秒格式
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        # 创建今天的日期 + 计算的时间
        base_date = datetime.now().replace(hour=hours, minute=minutes, second=int(seconds),
                                           microsecond=int((seconds % 1) * 1000000))

        return base_date

    def extract_sharpness_features(self, image):
        """提取清晰度特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 拉普拉斯方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()

        # 2. Sobel梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mean = np.mean(sobel_magnitude)
        sobel_std = np.std(sobel_magnitude)

        # 3. Tenengrad梯度
        tenengrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) ** 2
        tenengrad_mean = np.mean(tenengrad)

        # 4. Roberts交叉梯度
        roberts_x = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, np.array([[1, 0], [0, -1]]))
        roberts_y = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, np.array([[0, 1], [-1, 0]]))
        roberts_magnitude = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
        roberts_mean = np.mean(roberts_magnitude)

        return {
            'laplacian_variance': laplacian_variance,
            'sobel_magnitude_mean': sobel_mean,
            'sobel_magnitude_std': sobel_std,
            'tenengrad_mean': tenengrad_mean,
            'roberts_mean': roberts_mean
        }

    def extract_contrast_features(self, image):
        """提取对比度特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 全局对比度 (RMS)
        rms_contrast = np.sqrt(np.mean((gray - np.mean(gray)) ** 2))

        # 2. Michelson对比度
        max_intensity = np.max(gray)
        min_intensity = np.min(gray)
        michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity + 1e-10)

        # 3. 局部对比度标准差
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        local_mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_OPEN, kernel)
        local_contrast = np.abs(gray.astype(np.float32) - local_mean)
        local_contrast_std = np.std(local_contrast)

        # 4. 韦伯对比度
        weber_contrast = np.std(gray) / (np.mean(gray) + 1e-10)

        return {
            'rms_contrast': rms_contrast,
            'michelson_contrast': michelson_contrast,
            'local_contrast_std': local_contrast_std,
            'weber_contrast': weber_contrast
        }

    def extract_frequency_features(self, image):
        """提取频域特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # FFT变换
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # 1. 高频能量比
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # 创建低频掩码
        low_freq_mask = np.zeros((rows, cols), np.uint8)
        r_low = min(rows, cols) // 8
        cv2.circle(low_freq_mask, (ccol, crow), r_low, 1, -1)

        # 创建高频掩码
        high_freq_mask = np.ones((rows, cols), np.uint8) - low_freq_mask

        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)

        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

        # 2. 频谱重心
        y_coords, x_coords = np.ogrid[:rows, :cols]
        spectrum_centroid_x = np.sum(x_coords * magnitude_spectrum) / np.sum(magnitude_spectrum)
        spectrum_centroid_y = np.sum(y_coords * magnitude_spectrum) / np.sum(magnitude_spectrum)
        spectrum_centroid_distance = np.sqrt((spectrum_centroid_x - ccol) ** 2 + (spectrum_centroid_y - crow) ** 2)

        # 3. 频域方差
        freq_variance = np.var(magnitude_spectrum)

        return {
            'high_freq_ratio': high_freq_ratio,
            'spectrum_centroid_distance': spectrum_centroid_distance,
            'frequency_variance': freq_variance,
            'low_freq_energy': low_freq_energy,
            'high_freq_energy': high_freq_energy
        }

    def extract_color_features(self, image):
        """提取颜色空间特征"""
        # RGB统计
        b, g, r = cv2.split(image)
        rgb_mean = [np.mean(r), np.mean(g), np.mean(b)]
        rgb_std = [np.std(r), np.std(g), np.std(b)]

        # HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_lab = cv2.split(lab)

        return {
            'rgb_r_mean': rgb_mean[0], 'rgb_g_mean': rgb_mean[1], 'rgb_b_mean': rgb_mean[2],
            'rgb_r_std': rgb_std[0], 'rgb_g_std': rgb_std[1], 'rgb_b_std': rgb_std[2],
            'hsv_h_mean': np.mean(h), 'hsv_s_mean': np.mean(s), 'hsv_v_mean': np.mean(v),
            'hsv_h_std': np.std(h), 'hsv_s_std': np.std(s), 'hsv_v_std': np.std(v),
            'lab_l_mean': np.mean(l), 'lab_a_mean': np.mean(a), 'lab_b_mean': np.mean(b_lab),
            'lab_l_std': np.std(l), 'lab_a_std': np.std(a), 'lab_b_std': np.std(b_lab),
            'color_saturation': np.mean(s),
            'brightness': np.mean(v)
        }

    def extract_texture_features(self, image):
        """提取纹理特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 局部二值模式 (LBP)
        radius = 3
        n_points = 24
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-10)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

        # 2. 灰度共生矩阵特征
        # 降低图像分辨率以减少计算量
        gray_small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))
        gray_small = (gray_small / 32).astype(np.uint8)  # 量化到8级

        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray_small, distances, angles, levels=8, symmetric=True, normed=True)

        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))

        return {
            'lbp_entropy': lbp_entropy,
            'glcm_contrast': contrast,
            'glcm_dissimilarity': dissimilarity,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_correlation': correlation
        }

    def extract_atmospheric_features(self, image):
        """提取大气光学特征"""
        # 1. 暗通道先验
        b, g, r = cv2.split(image)
        dark_channel = np.minimum(np.minimum(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark_channel, kernel)
        dark_channel_mean = np.mean(dark_channel)

        # 2. 大气光估计
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.percentile(gray, 99)
        atmospheric_light = bright_pixels / 255.0

        # 3. 透射率估计
        image_norm = image.astype(np.float32) / 255.0
        transmission_estimate = 1.0 - 0.95 * (dark_channel.astype(np.float32) / 255.0) / (atmospheric_light + 1e-10)
        transmission_estimate = np.clip(transmission_estimate, 0.1, 1.0)
        transmission_mean = np.mean(transmission_estimate)
        transmission_std = np.std(transmission_estimate)

        # 4. 雾浓度指标
        fog_density = 1.0 - transmission_mean

        # 5. 图像清晰度比率
        clarity_ratio = np.sum(dark_channel > 50) / (dark_channel.shape[0] * dark_channel.shape[1])

        return {
            'dark_channel_mean': dark_channel_mean,
            'atmospheric_light': atmospheric_light,
            'transmission_mean': transmission_mean,
            'transmission_std': transmission_std,
            'fog_density': fog_density,
            'clarity_ratio': clarity_ratio
        }

    def extract_information_theory_features(self, image):
        """提取信息论特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 图像熵
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

        # 2. 条件熵（基于邻域）
        # 简化计算：使用2x2邻域
        rows, cols = gray.shape
        joint_hist = np.zeros((256, 256))

        for i in range(rows - 1):
            for j in range(cols - 1):
                pixel1 = gray[i, j]
                pixel2 = gray[i, j + 1]  # 右邻域
                joint_hist[pixel1, pixel2] += 1

        joint_hist_norm = joint_hist / joint_hist.sum()
        joint_entropy = -np.sum(joint_hist_norm * np.log2(joint_hist_norm + 1e-10))

        # 3. 互信息
        marginal_hist1 = np.sum(joint_hist_norm, axis=1)
        marginal_hist2 = np.sum(joint_hist_norm, axis=0)

        mutual_info = 0
        for i in range(256):
            for j in range(256):
                if joint_hist_norm[i, j] > 0:
                    mutual_info += joint_hist_norm[i, j] * np.log2(
                        joint_hist_norm[i, j] / (marginal_hist1[i] * marginal_hist2[j] + 1e-10))

        return {
            'image_entropy': entropy,
            'joint_entropy': joint_entropy,
            'mutual_information': mutual_info
        }

    def extract_single_frame_features(self, image, frame_index, timestamp):
        """提取单帧的所有特征"""
        features = {
            'frame_index': frame_index,
            'timestamp': timestamp,
            'hours': timestamp.hour,
            'minutes': timestamp.minute,
            'seconds': timestamp.second + timestamp.microsecond / 1000000.0
        }

        # 提取各类特征
        features.update(self.extract_sharpness_features(image))
        features.update(self.extract_contrast_features(image))
        features.update(self.extract_frequency_features(image))
        features.update(self.extract_color_features(image))
        features.update(self.extract_texture_features(image))
        features.update(self.extract_atmospheric_features(image))
        features.update(self.extract_information_theory_features(image))

        return features

    def process_video(self, sampling_interval=1):
        """
        处理整个视频，提取特征

        Args:
            sampling_interval: 采样间隔（帧数），1表示每帧都处理
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")

        frame_index = 0
        processed_frames = 0

        print("开始处理视频...")
        print(f"采样间隔: 每{sampling_interval}帧处理一次 (25帧=1秒)")
        print(f"预计处理帧数: {self.video_info['frame_count'] // sampling_interval}帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按采样间隔处理帧
            if frame_index % sampling_interval == 0:
                timestamp = self.calculate_timestamp(frame_index)
                features = self.extract_single_frame_features(frame, frame_index, timestamp)
                self.features_data.append(features)
                processed_frames += 1

                if processed_frames % 100 == 0:
                    print(f"已处理 {processed_frames} 帧...")

            frame_index += 1

        cap.release()
        print(f"视频处理完成！共处理 {processed_frames} 帧")

        return self.features_data

    def save_features_to_excel(self, output_path="video_features.xlsx"):
        """将特征保存到Excel文件"""
        if not self.features_data:
            print("没有特征数据，请先处理视频")
            return

        df = pd.DataFrame(self.features_data)

        # 重新排列列的顺序，将时间相关列放在前面
        time_cols = ['frame_index', 'timestamp', 'hours', 'minutes', 'seconds']
        feature_cols = [col for col in df.columns if col not in time_cols]
        df = df[time_cols + feature_cols]

        # 保存到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 主要特征表
            df.to_excel(writer, sheet_name='视频特征', index=False)

            # 创建特征说明表
            feature_descriptions = {
                '特征类别': ['清晰度特征', '对比度特征', '频域特征', '颜色特征', '纹理特征', '大气光学特征',
                             '信息论特征'],
                '包含特征': [
                    'laplacian_variance, sobel_magnitude_mean, tenengrad_mean',
                    'rms_contrast, michelson_contrast, local_contrast_std',
                    'high_freq_ratio, spectrum_centroid_distance, frequency_variance',
                    'rgb统计, hsv统计, lab统计, color_saturation',
                    'lbp_entropy, glcm特征',
                    'dark_channel_mean, atmospheric_light, transmission_mean, fog_density',
                    'image_entropy, joint_entropy, mutual_information'
                ],
                '用途': [
                    '量化图像锐度和边缘清晰度',
                    '评估图像对比度和细节可见性',
                    '分析高频信息损失程度',
                    '描述颜色分布和饱和度变化',
                    '捕捉纹理细节的模糊程度',
                    '基于物理模型评估雾浓度',
                    '量化图像信息丢失程度'
                ]
            }

            desc_df = pd.DataFrame(feature_descriptions)
            desc_df.to_excel(writer, sheet_name='特征说明', index=False)

            # 创建统计摘要表
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary_df = df[numeric_cols].describe()
            summary_df.to_excel(writer, sheet_name='特征统计摘要')

        print(f"特征数据已保存到: {output_path}")
        print(f"数据形状: {df.shape}")

        return df

    def get_feature_summary(self):
        """获取特征提取摘要"""
        if not self.features_data:
            return "没有特征数据"

        df = pd.DataFrame(self.features_data)

        summary = {
            '视频信息': self.video_info,
            '特征维度': len(df.columns) - 5,  # 减去时间相关列
            '处理帧数': len(df),
            '时间范围': f"{df['timestamp'].min()} 到 {df['timestamp'].max()}",
            '特征类别数': 7,
            '主要特征类别': [
                '清晰度特征 (5个)',
                '对比度特征 (4个)',
                '频域特征 (5个)',
                '颜色特征 (16个)',
                '纹理特征 (6个)',
                '大气光学特征 (6个)',
                '信息论特征 (3个)'
            ]
        }

        return summary


# 使用示例和测试函数
def analyze_video_features(video_path, start_offset=28, sampling_interval=30):
    """
    分析视频特征的主函数

    Args:
        video_path: 视频文件路径
        start_offset: 视频开始时间偏移(秒)
        sampling_interval: 采样间隔(帧数)，建议30(即每秒1帧，假设30fps)
    """
    try:
        # 创建特征提取器
        extractor = VideoFeatureExtractor(video_path, start_offset)

        # 处理视频
        features = extractor.process_video(sampling_interval)

        # 保存到Excel
        df = extractor.save_features_to_excel("video_blur_features.xlsx")

        # 显示摘要
        summary = extractor.get_feature_summary()
        print("\n=== 特征提取摘要 ===")
        for key, value in summary.items():
            if key == '主要特征类别':
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")

        return extractor, df

    except Exception as e:
        print(f"处理视频时出错: {e}")
        return None, None


# 快速使用函数
def quick_extract_features(video_path="root/a.mp4", output_name="video_blur_features.xlsx"):
    """
    快速特征提取函数

    Args:
        video_path: 视频文件路径
        output_name: 输出Excel文件名

    Returns:
        tuple: (特征提取器对象, DataFrame)
    """
    print("=== 视频特征提取系统 ===")
    print(f"视频路径: {video_path}")
    print(f"起始时间: 0时0分28秒")
    print("提取特征类别: 清晰度、对比度、频域、颜色、纹理、大气光学、信息论")

    try:
        extractor = VideoFeatureExtractor(video_path, start_time_offset=28)

        # 根据视频长度自适应采样间隔 (基于25 FPS)
        if extractor.video_info['duration'] > 300:  # 超过5分钟
            sampling_interval = 50  # 每2秒1帧 (50帧 = 2秒)
            print("长视频，采用快速采样模式 (每50帧1次，即每2秒1帧)")
        elif extractor.video_info['duration'] > 60:  # 1-5分钟
            sampling_interval = 25  # 每秒1帧 (25帧 = 1秒)
            print("中等长度视频，采用标准采样模式 (每25帧1次，即每1秒1帧)")
        else:  # 短视频
            sampling_interval = 12  # 每0.48秒1帧 (约每半秒1帧)
            print("短视频，采用高密度采样模式 (每12帧1次，约每0.5秒1帧)")

        # 提取特征
        features = extractor.process_video(sampling_interval)

        # 保存结果
        df = extractor.save_features_to_excel(output_name)

        # 显示结果摘要
        summary = extractor.get_feature_summary()
        print("\n=== 特征提取结果摘要 ===")
        for key, value in summary.items():
            if key == '主要特征类别':
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")

        print(f"\n✅ 成功！特征数据已保存到: {output_name}")
        print(f"📊 数据维度: {df.shape[0]} 帧 × {df.shape[1]} 特征")

        return extractor, df

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None, None


def demonstrate_feature_correlation():
    """演示特征相关性分析"""
    print("\n=== 特征相关性分析示例 ===")
    print("""
    # 加载提取的特征数据
    df = pd.read_excel('video_blur_features.xlsx', sheet_name='视频特征')

    # 选择关键模糊度指标进行相关性分析
    blur_indicators = [
        'laplacian_variance',      # 清晰度
        'rms_contrast',            # 对比度
        'high_freq_ratio',         # 频域特征
        'fog_density',             # 雾浓度
        'transmission_mean',       # 透射率
        'image_entropy'            # 信息熵
    ]

    # 计算相关性矩阵
    correlation_matrix = df[blur_indicators].corr()

    # 可视化相关性
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('视频模糊度特征相关性分析')
    plt.tight_layout()
    plt.show()
    """)


if __name__ == "__main__":
    # 检查视频文件
    video_path = r"D:\GitHub_local\JXSTSXJM-Code\机场视频\a.mp4"  # 根据您的描述，可能需要修改扩展名

    # 尝试不同的视频格式
    possible_paths = [
        r"D:\GitHub_local\JXSTSXJM-Code\机场视频\a.mp4"
    ]

    video_found = False
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            video_found = True
            break

    if video_found:
        print(f"找到视频文件: {video_path}")
        # 执行特征提取
        extractor, df = quick_extract_features(video_path)

        if df is not None:
            # 显示前几行数据预览
            print("\n=== 数据预览 (前5行关键特征) ===")
            key_features = ['timestamp', 'laplacian_variance', 'rms_contrast',
                            'high_freq_ratio', 'fog_density', 'transmission_mean']
            if all(col in df.columns for col in key_features):
                print(df[key_features].head())

            # 显示特征统计
            print("\n=== 关键特征统计信息 ===")
            numeric_cols = ['laplacian_variance', 'rms_contrast', 'fog_density', 'transmission_mean']
            existing_cols = [col for col in numeric_cols if col in df.columns]
            if existing_cols:
                print(df[existing_cols].describe())

        # 显示后续分析建议
        print("\n=== 后续分析建议 ===")
        print("1. 特征与能见度数据时间对齐")
        print("2. 构建模糊度预测模型")
        print("3. 异常检测和质量控制")
        demonstrate_feature_correlation()

    else:
        print("❌ 未找到视频文件")
        print("请确认视频文件路径和格式")
        print("支持的格式: .mp4, .avi, .mov, .mkv")
        print("\n使用方法:")
        print("extractor, df = quick_extract_features('您的视频路径')")