import cv2
import os
import argparse


def video_to_images(video_path, output_dir, scale_factor_):
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0  # 视频帧计数
    save_count = 1   # 保存图片的计数，用于编号
    scale_factor_img = scale_factor_  # 缩小4倍

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break  # 没有更多帧则结束

        # 每隔两帧提取一张图像
        if frame_count % 20 == 0:
            # 缩小图像
            height, width = frame.shape[:2]
            new_size = (int(width * scale_factor_img), int(height * scale_factor_img))
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            # 保存图像，编号格式为 0001, 0002, ...
            image_name = f"{save_count:04d}.jpg"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, resized_frame)
            save_count += 1  # 更新编号

        frame_count += 1  # 更新帧计数

    # 释放视频资源
    cap.release()
    print(f"提取完成，图像保存在: {output_dir}")

if __name__ == '__main__':
    # 获取参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="", type=str, help="video_path")
    parser.add_argument("--output_dir", default="", type=str, help="output_dir")
    parser.add_argument("--scale_factor", default=0.25, type=float, help="output_dir")
    args = parser.parse_args()

    # 使用示例
    video_path = args.video_path
    output_dir = args.output_dir
    scale_factor = args.scale_factor
    video_to_images(video_path, output_dir, scale_factor)
