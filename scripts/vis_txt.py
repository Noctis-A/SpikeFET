import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class TrackingVisualizer:
    def __init__(self,
                 txt_path: str,
                 img_dir: str,
                 output_dir: str = "output",
                 video_output: str = None):
        """
        初始化可视化器
        :param txt_path: 跟踪数据文件路径
        :param img_dir: 原始图像文件夹路径
        :param output_dir: 输出图像保存目录
        :param video_output: 视频输出路径（可选）
        """
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.video_output = video_output

        # 自动创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 颜色配置
        self.color_palette = [
            (0, 255, 0),  # 绿色 - 当前框
            (255, 0, 0),  # 蓝色 - 历史轨迹
            (0, 165, 255)  # 橙色 - 预测路径
        ]

        # 初始化数据
        self.tracking_data = self._parse_tracking_file()
        self.image_files = self._get_sorted_images()

        # 验证数据一致性
        if len(self.tracking_data) != len(self.image_files):
            raise ValueError(f"数据与图像数量不匹配（{len(self.tracking_data)} vs {len(self.image_files)}）")

    def _parse_tracking_file(self):
        """解析跟踪数据文件"""
        data = []
        with open(self.txt_path, 'r') as f:
            for line in f:
                # 支持多种分隔符（制表符、空格、逗号）
                parts = list(map(float, line.replace(',', ' ').split()))
                if len(parts) not in [4, 5]:
                    raise ValueError(f"无效数据行: {line.strip()}")

                # 格式: [frame] x y w h (可选帧号)
                if len(parts) == 5:
                    frame, x, y, w, h = parts
                else:
                    x, y, w, h = parts
                    frame = len(data)  # 自动生成帧号

                data.append({
                    'frame': int(frame),
                    'bbox': (float(x), float(y), float(w), float(h))
                })
        return data

    def _get_sorted_images(self):
        """获取排序后的图像文件列表"""
        valid_ext = ['.jpg', '.jpeg', '.png', '.bmp']
        files = [f for f in os.listdir(self.img_dir)
                 if os.path.splitext(f)[1].lower() in valid_ext]

        # 按数字顺序排序（假设文件名包含数字）
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        return [os.path.join(self.img_dir, f) for f in files]

    def _draw_visualization(self, img, current_bbox, history):
        """在图像上绘制可视化元素"""
        # 转换颜色空间
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 绘制当前边界框
        x, y, w, h = current_bbox
        cv2.rectangle(img,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      self.color_palette[0], 2)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def process_sequence(self, show_progress=True):
        """处理整个图像序列"""
        history = []
        video_writer = None

        # 进度条设置
        iterator = enumerate(zip(self.tracking_data, self.image_files))
        if show_progress:
            iterator = tqdm(iterator, total=len(self.tracking_data))

        for idx, data in enumerate(iterator):
            # 读取图像
            img_path = data[1][1]
            data = data[1][0]
            img = Image.open(img_path)

            # 获取当前帧数据
            current_bbox = data['bbox']
            center = (
                current_bbox[0] + current_bbox[2] / 2,
                current_bbox[1] + current_bbox[3] / 2
            )
            history.append(center)

            # 绘制可视化元素
            visualized_img = self._draw_visualization(img, current_bbox, history)

            # 保存结果图像
            output_path = os.path.join(
                self.output_dir,
                f"frame_{idx:04d}.jpg"
            )
            visualized_img.save(output_path)

            # 初始化视频写入器
            if self.video_output and video_writer is None:
                h, w = visualized_img.size
                video_writer = cv2.VideoWriter(
                    self.video_output,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30, (w, h)
                )

            # 写入视频帧
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(np.array(visualized_img), cv2.COLOR_RGB2BGR))

        # 清理视频写入器
        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    # 使用示例
    visualizer = TrackingVisualizer(
        txt_path="traffic_0061.txt",
        img_dir="traffic_0061/vis_imgs",
        output_dir="SpikeFET/vis",
        video_output="output_video.mp4"
    )

    visualizer.process_sequence()
