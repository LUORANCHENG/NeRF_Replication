

import torch

from utils import DatasetProvider, NeRFDataset, NeRF, render_rays
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import cv2
import os


def train():
    # 初始化进度条，范围为 1 到最大迭代次数 `maxiters`
    pbar = tqdm(range(1, maxiters))

    # 遍历每次迭代步骤
    for global_step in pbar:
        # 随机从训练集 `trainset` 中选取一个样本索引 `idx`
        idx = np.random.randint(0, len(trainset))

        # 选取该样本对应的光线方向 `raydirs`、光线起点 `rayoris` 和图像像素 `imagepixels`
        raydirs, rayoris, imagepixels = trainset[idx]

        # 使用模型渲染光线，得到初步和细化的颜色结果 `rgb1` 和 `rgb2`
        rgb1, rgb2 = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, num_samples2, white_background)

        # 计算 `rgb1` 和 `rgb2` 与真实图像像素的均方误差损失 `loss1` 和 `loss2`
        loss1 = ((rgb1 - imagepixels)**2).mean()
        loss2 = ((rgb2 - imagepixels)**2).mean()

        # 计算 `rgb2` 渲染的峰值信噪比 (PSNR)，用于评估渲染质量
        psnr = -10. * torch.log(loss2.detach()) / np.log(10.)

        # 总损失为 `loss1` 和 `loss2` 的和
        loss = loss1 + loss2

        # 优化步骤：清零梯度，计算反向传播的梯度并更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条描述，显示当前步数、损失和 PSNR
        pbar.set_description(f"{global_step} / {maxiters}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}")

        # 动态调整学习率
        decay_rate = 0.1
        new_lrate = lrate * (decay_rate ** (global_step / lrate_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # 每 5000 步或初始的 500 步保存一次图像和模型
        if global_step % 5000 == 0 or global_step == 500:
            # 设置保存的文件路径
            imgpath = f"imgs/{global_step:02d}.png"
            pthpath = f"ckpt/{global_step:02d}.pth"

            # 将模型设为评估模式，禁用梯度计算
            coarse.eval()
            with torch.no_grad():
                # 用于保存渲染结果和真实图像像素
                rgbs, imgpixels = [], []

                # 获取测试集中的数据项并逐个渲染
                for raydirs, rayoris, imagepixels in trainset.get_test_item():
                    rgb1, rgb2  = render_rays(coarse, fine, raydirs, rayoris, sample_z_vals, num_samples2, white_background)

                    # 收集渲染的最终结果
                    rgbs.append(rgb2)

                    # 收集真实图像像素
                    imgpixels.append(imagepixels)

                # 将渲染结果和真实像素连接成完整图像张量
                rgb = torch.cat(rgbs, dim=0)
                imgpixels = torch.cat(imgpixels, dim=0)

                # 计算验证集的损失和 PSNR
                loss = ((rgb - imgpixels)**2).mean()
                psnr = -10. * torch.log(loss) / np.log(10.)

                # 输出保存的图像信息及其损失和 PSNR
                print(f"Save image {imgpath}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.6f}")
            # 将模型设置为训练模式
            coarse.train()

            # 将渲染图像转换为 numpy 格式并保存为 PNG 文件
            temp_image = (rgb.view(provider.height, provider.width, 3).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(imgpath, temp_image[..., ::-1])

            # 保存模型和细化模型的状态字典到检查点文件
            torch.save([coarse.state_dict(), fine.state_dict()], pthpath)







if __name__ == '__main__':
    # 数据集目录
    # root = "data/nerf_synthetic/lego"
    root = "data/mydata"

    # 相机数据名称
    # transforms_file = "transforms_train.json"
    transforms_file = "transforms.json"


    os.makedirs("imgs",      exist_ok=True)
    os.makedirs("rotate360", exist_ok=True)
    os.makedirs("videos",    exist_ok=True)
    os.makedirs("ckpt",      exist_ok=True)

    # 是否使用一半的分辨率采样
    half_resoultion = False

    # 初始化数据集对象
    provider = DatasetProvider(root, transforms_file, half_resoultion)

    # 根据前面的provider初始化nerf数据集
    batch_size = 1024
    device = 'cuda'
    trainset = NeRFDataset(provider, batch_size, device)


    """初始化nerf的粗模型和细模型"""
    x_pedim = 10  # 位置编码的采样维度
    view_pedim = 4  #视角编码的采样维度
    coarse = NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)  # 粗模型
    params = list(coarse.parameters())  # 获取模型的参数，保存进列表里面
    fine= NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)  # 细模型
    params.extend(list(fine.parameters()))  # 获取模型的参数，保存进列表里面

    """初始化优化器"""
    lrate_decay = 500 * 1000  # 学习率衰减
    lrate = 5e-4  # 初始学习率
    optimizer = optim.Adam(params, lrate)  # 初始化优化器

    # 获取训练集第一张图片的视角数据,ray_dirs和ray_oris的形状都为[1024, 3]
    ray_dirs, ray_oris, img_pixels = trainset[0]


    """对光线进行粗采样和细采样"""
    # 在一条光线上采样64个点,smple_z_vals的shape为(1, 64)
    num_samples1 = 64
    num_samples2 = 128
    white_background = True
    sample_z_vals = torch.linspace(
        2.0, 6.0, num_samples1, device=device
    ).view(1, num_samples1)

    """开始训练"""
    maxiters = 100000 + 1
    train()




