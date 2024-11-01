import torch
import argparse
from tqdm import tqdm
import numpy as np
from utils import NeRF, DatasetProvider, NeRFDataset, render_rays
import cv2
import imageio

def make_video360(ckpt_path):
    # 加载预训练的模型权重
    mstate, fstate = torch.load(args.ckpt, map_location="cpu")
    coarse.load_state_dict(mstate)
    fine.load_state_dict(fstate)

    # 将模型和细化模型设为评估模式，以禁用 Dropout 等操作
    coarse.eval()
    fine.eval()

    # 初始化图像列表 `imagelist`，用于存储渲染的每一帧图像
    imagelist = []

    # 遍历每个角度，逐步获取 360 度旋转的光线数据生成器 `gfn`
    for i, gfn in tqdm(enumerate(trainset.get_rotate_360_rays()), desc="Rendering"):
        # 禁用梯度计算，加快推理速度
        with torch.no_grad():
            rgbs = []

            # 调用生成函数 `gfn` 生成当前视角下的光线数据（按批次）
            for raydirs, rayoris in gfn():
                # 渲染每条光线，得到细化渲染的颜色 `rgb2`
                rgb1, rgb2 = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, num_samples2, white_background)

                # 将渲染结果 `rgb2` 添加到列表 `rgbs`
                rgbs.append(rgb2)

            # 将批次的渲染结果拼接成一个完整图像
            rgb = torch.cat(rgbs, dim=0)

        # 将拼接后的图像从张量格式转换为 RGB 图像，并放缩到 [0, 255] 范围
        rgb = (rgb.view(provider.height, provider.width, 3).cpu().numpy() * 255).astype(np.uint8)

        # 将渲染结果保存为 PNG 图像
        file = f"rotate360/{i:03d}.png"
        print(f"Rendering to {file}")
        cv2.imwrite(file, rgb[..., ::-1])

        # 将图像帧添加到图像列表 `imagelist` 中
        imagelist.append(rgb)


    # 设置输出视频文件路径
    video_file = f"videos/rotate360.mp4"
    # 使用 `imageio.mimwrite` 将图像列表写入视频文件
    print(f"Write imagelist to video file {video_file}")
    imageio.mimwrite(video_file, imagelist, fps=30, quality=10)


if __name__ == '__main__':
    # 获取参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpt/100000.pth", type=str, help="model file used to make 360 rotation video")
    parser.add_argument("--half-resolution", action="store_true", help="use half resolution")
    parser.add_argument("--data_path", default="data/nerf_synthetic/lego", help="data_path")
    parser.add_argument("--white_background", default=True, help="white_background")
    parser.add_argument("--transforms_file", default='transforms_train.json', help="transforms_file")
    args = parser.parse_args()

    # 是否使用一半的分辨率采样
    half_resoultion = args.half_resolution

    # 数据集目录
    data_path = args.data_path
    device = 'cuda'


    # 相机数据名称
    transforms_file = args.transforms_file

    # 是否使用白色背景
    white_background = args.white_background

    # 初始化数据集对象
    provider = DatasetProvider(data_path, transforms_file, half_resoultion)

    # 根据前面的provider初始化nerf数据集
    batch_size = 1024
    trainset = NeRFDataset(provider, batch_size, device)

    # 对光线进行粗采样和细采样
    num_samples1 = 64
    num_samples2 = 128
    sample_z_vals = torch.linspace(
        2.0, 6.0, num_samples1, device=device
    ).view(1, num_samples1)

    # 初始化粗模型和细模型
    x_pedim = 10  # 位置编码的采样维度
    view_pedim = 4  #视角编码的采样维度
    coarse = NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)  # 粗模型
    fine= NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)  # 细模型

    # 使用训练好的模型渲染视频
    ckpt_path = args.ckpt
    make_video360(ckpt_path)