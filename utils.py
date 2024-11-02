import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------图像以及相机数据集初始化---------------------------------------------------------
class DatasetProvider:
    def __init__(self, root, transforms_file, half_resolution=False):

        # 载入相机数据json文件
        self.meta = json.load(open(os.path.join(root, transforms_file), "r"))

        # 数据集目录
        self.root = root

        # 获取所有的帧
        self.frames = self.meta['frames']

        # 用于保存所有图片的列表
        self.images = []

        # 用于保存每一张图像所对应的相机数据
        self.poses = []

        # 相机的水平视场角
        self.camera_angle_x = self.meta['camera_angle_x']



        # 遍历每一帧，取出对应的图片和相机数据
        for frame in self.frames:

            # 判断图像的路径是否已经有后缀名
            if '.' in frame["file_path"].split('/')[-1]:
                image_file = frame["file_path"]
            else:
                # 拼接出图像文件的路径
                image_file = os.path.join(self.root, frame["file_path"] + '.png')

            # 获取图像文件的后缀
            self.suffix = image_file.split('.')[-1]

            # 读取图片
            image = imageio.v3.imread(image_file)

            # 如果使用一半的分辨率采样
            if half_resolution:
                # 将图像的分辨率减半
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # 把图片保存进images列表里面
            self.images.append(image)

            # 把相机数据保存进poses列表里面
            self.poses.append(frame['transform_matrix'])

        # 将相机数据转换为方便计算的np数组
        self.poses = np.stack(self.poses)

        # 将图像数据转换为方便计算的np数组
        self.images = (np.stack(self.images) / 255.).astype(np.float32)

        # 获取图像的宽
        self.width = self.images.shape[2]

        # 获取图像的高
        self.height = self.images.shape[1]

        # 计算焦距
        self.focal = 0.5*self.width / np.tan(0.5*self.camera_angle_x)


        if self.suffix == 'png':
            # 获取图像的透明度（因为载入的图像是rgba格式的，所以我们需要将不透明度去掉）
            alpha = self.images[..., [3]]
            # 获取图像的rgb值
            rgb = self.images[..., :3]
            # 去除图像的不透明度
            self.images = rgb * alpha + (1 - alpha)
        else:
            # 如果图像是jpg格式的，则不需要去除alpha
            pass


# -------------------------------------------------------------图像以及相机数据集初始化---------------------------------------------------------

# ----------------------------------------------------------nerf数据集初始化--------------------------------------------------------
class NeRFDataset:
    def __init__(self, provider, batch_size=1024, device="cuda"):

        self.images        = provider.images  # 获取图像数据
        self.poses         = provider.poses  # 获取相机数据
        self.focal         = provider.focal  # 获取焦距
        self.width         = provider.width  # 获取图像宽度
        self.height        = provider.height  # 获取图像高度
        self.batch_size    = batch_size  # 获取batch_size
        self.num_image     = len(self.images)  # 获取数据集长度
        self.precrop_iters = 500  # 在前500轮，对图像进行裁剪，只取图像中间的部分去训练，目的是为了加快收敛
        self.precrop_frac  = 0.5  # 定义了裁剪的比例（50%）
        self.niter         = 0  # 表示当前的迭代次数
        self.device        = device  # 获取设备名称

        self.initialize()  # 初始化数据集中用于渲染光线的部分


    def initialize(self):
        """
        作用：
            预处理图像平面坐标、计算光线方向和起点，
        """


        """1. 创建图像平面上的网格坐标"""
        # 创建图像宽度方向上的坐标范围（浮点数）
        warange = torch.arange(self.width, dtype=torch.float32, device=self.device)

        # 创建图像高度方向上的坐标范围（浮点数）
        harange = torch.arange(self.height, dtype=torch.float32, device=self.device)

        # 使用网格生成器生成图像平面的网格坐标
        y, x = torch.meshgrid(harange, warange)

        """2. 计算光线的方向"""
        # 计算光线在 x 方向的相对位置并缩放，使其基于相机的焦距 focal
        self.transformed_x = (x - self.width * 0.5) / self.focal

        # 计算光线在 y 方向的相对位置并缩放，使其基于相机的焦距 focal
        self.transformed_y = (y - self.height * 0.5) / self.focal

        """3. 创建用于中心裁剪的索引矩阵"""
        # 创建一个中心裁剪索引矩阵，每个位置的值是一个从 0 到 width * height - 1 的索引
        self.precrop_index = torch.arange(self.width * self.height).view(self.height, self.width)

        # 计算裁剪区域的高度一半
        dH = int(self.height // 2 * self.precrop_frac)

        # 计算裁剪区域的宽度一半
        dW = int(self.width // 2 * self.precrop_frac)

        # 选取图像的中心部分，并将选中的索引展平成一个一维数组
        self.precrop_index = self.precrop_index[
                             self.height // 2 - dH:self.height // 2 + dH,
                             self.width // 2 - dW:self.width // 2 + dW
                             ].reshape(-1)

        """4. 计算每张图像的光线方向和原点"""
        # 将相机位姿数据转换为 GPU 张量
        poses = torch.cuda.FloatTensor(self.poses, device=self.device)
        # 初始化列表来存储所有图像的光线方向和原点
        all_ray_dirs, all_ray_origins = [], []

        # 遍历每张图像，生成每张图像的光线方向和原点
        for i in range(len(self.images)):
            # 使用 make_rays 函数计算光线方向和原点
            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, poses[i])
            # 将结果添加到相应列表中
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)

        """5. 将结果存储为类属性"""
        # 将所有光线方向堆叠为 3D 张量 (num_images, height * width, 3)
        self.all_ray_dirs = torch.stack(all_ray_dirs, dim=0)

        # 将所有光线原点堆叠为 3D 张量 (num_images, height * width, 3)
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)

        # 将图像数据转换为 GPU 张量，并将每张图像展平为 (num_images, height * width, 3)
        self.images = torch.cuda.FloatTensor(self.images, device=self.device).view(self.num_image, -1, 3)

    def make_rays(self, x, y, pose):
        """
        作用：
            生成光线的方向和起点
        :param x:相机视图平面中光线的x坐标
        :param y:相机视图平面中光线的y坐标
        :param pose:相机的位姿矩阵（4x4的张量），包含旋转和平移信息
        :return:
            ray_dirs：光线的方向向量
            ray_origin：光线的起点位置
        """
        # directions表示这些光线在图像坐标系中的初始方向，表示朝向相机前方的方向
        directions = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)

        # 从姿态矩阵pose中提取相机的旋转矩阵
        camera_matrix = pose[:3, :3]

        # 将相机坐标系中的光线方向转换到世界坐标系中
        ray_dirs = directions.reshape(-1, 3) @ camera_matrix.T

        # 为每条光线设置相同的起点
        ray_origin = pose[:3, 3].view(1, 3).repeat(len(ray_dirs), 1)

        return ray_dirs, ray_origin

    def get_test_item(self, index=0):
        """
        作用：
            用于获取用来测试的光线数据
        :param index:要测试的图像索引
        :return:
            ray_dirs：光线方向，形状 [batch_size, 3]
            ray_oris：光线的起点，形状 [batch_size, 3]
            img_pixels：对应的图像像素值，形状 [batch_size, 3]
        """
        # 从 `all_ray_dirs` 中获取第 `index` 个图像的所有光线方向
        ray_dirs = self.all_ray_dirs[index]

        # 从 `all_ray_origins` 中获取第 `index` 个图像的所有光线起点
        ray_oris = self.all_ray_origins[index]

        # 从 `images` 中获取第 `index` 个图像的像素颜色值
        img_pixels = self.images[index]

        # 遍历 `ray_dirs`，按批次返回光线方向、起点和对应像素
        for i in range(0, len(ray_dirs), self.batch_size):
            yield ray_dirs[i:i + self.batch_size], ray_oris[i:i + self.batch_size], img_pixels[i:i + self.batch_size]

    def get_rotate_360_rays(self):
        # 定义一个平移矩阵，将摄像机沿 z 轴移动 t 距离
        def trans_t(t):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # 定义一个绕 x 轴旋转角度 `phi` 的旋转矩阵
        def rot_phi(phi):
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # 定义一个绕 y 轴旋转角度 `theta` 的旋转矩阵
        def rot_theta(th):
            return np.array([
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # 生成摄像机的视角变换矩阵，模拟球面视角
        def pose_spherical(theta, phi, radius):
            # 1. 平移矩阵，将摄像机沿 z 轴移动 radius 距离
            c2w = trans_t(radius)

            # 2. 绕 x 轴旋转 `phi` 角度
            c2w = rot_phi(phi / 180. * np.pi) @ c2w

            # 3. 绕 y 轴旋转 `theta` 角度
            c2w = rot_theta(theta / 180. * np.pi) @ c2w

            # 4. 将视角从标准视角转换为 NeRF 使用的坐标系
            c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
            return c2w

        # 循环生成 360 度旋转的视角矩阵
        for th in np.linspace(-180., 180., 41, endpoint=False):
            # 生成视角矩阵 `pose`，绕目标物体旋转并将其放置在 GPU 上
            pose = torch.cuda.FloatTensor(pose_spherical(th, -30., 4.), device=self.device)

            # 定义一个生成函数 `genfunc`
            def genfunc():
                # 使用生成的 `pose` 计算光线的方向和起点
                ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, pose)

                # 按批次返回光线方向和起点数据
                for i in range(0, len(ray_dirs), 1024):
                    yield ray_dirs[i:i + 1024], ray_origins[i:i + 1024]

            # 使用 `yield` 返回生成函数 `genfunc`，用于逐批生成光线数据
            yield genfunc

    # 获取数据集的长度
    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        # 增加迭代次数计数器，用于判断是否需要进行中心裁剪
        self.niter += 1

        # 根据索引从预处理好的数据中获取该图像的光线方向、光线原点、以及对应的像素值
        ray_dirs = self.all_ray_dirs[index]
        ray_oris = self.all_ray_origins[index]
        img_pixels = self.images[index]

        # 在设定的迭代次数内（`precrop_iters`）进行图像的中心裁剪操作
        if self.niter < self.precrop_iters:
            # 根据 `precrop_index` 裁剪中心区域的光线方向
            ray_dirs = ray_dirs[self.precrop_index]

            # 裁剪中心区域的光线原点
            ray_oris = ray_oris[self.precrop_index]

            # 裁剪中心区域的像素值
            img_pixels = img_pixels[self.precrop_index]

        # 设置每个批次的光线数量（即每次处理的光线数目）
        nrays = self.batch_size

        # 从光线数据中随机选择 `nrays` 个索引以用于采样，避免处理全部数据带来的计算负担
        select_inds = np.random.choice(ray_dirs.shape[0], size=[nrays], replace=False)

        # 根据选择的索引采样光线方向
        ray_dirs = ray_dirs[select_inds]

        # 根据选择的索引采样光线原点
        ray_oris = ray_oris[select_inds]

        # 根据选择的索引采样对应的像素值
        img_pixels = img_pixels[select_inds]

        # 返回该批次采样的光线方向、光线原点和像素值，用于训练或测试
        return ray_dirs, ray_oris, img_pixels

# ----------------------------------------------------------nerf数据集初始化--------------------------------------------------------

# -----------------------------------------------------------位置编码实现类---------------------------------------------------------
class Embedder(nn.Module):
    def __init__(self, positional_encoding_dim):
        # 初始化位置编码维度
        super().__init__()
        self.positional_encoding_dim = positional_encoding_dim

    # 进行位置编码
    def forward(self, x):
        # 初始化位置编码列表，包含原始输入
        positions = [x]

        # 对每一层频率的编码进行遍历
        for i in range(self.positional_encoding_dim):
            # 对应频率层的正弦和余弦变换
            for fn in [torch.sin, torch.cos]:
                # 将当前频率下的正弦或余弦编码值添加到位置列表中
                positions.append(fn((2.0 ** i) * x))
        # 将所有频率编码拼接为一个大的编码向量
        return torch.cat(positions, dim=-1)
# -----------------------------------------------------------位置编码实现类---------------------------------------------------------


# -------------------------------------------------无视图依赖性的 head（不考虑视角的影响）----------------------------------------------
class NoViewDirHead(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        # 定义一个线性层，将输入维度 ninput 映射为输出维度 noutput
        self.head = nn.Linear(ninput, noutput)

    def forward(self, x, view_dirs):
        # 通过线性层映射得到输出
        x = self.head(x)

        # 使用 Sigmoid 激活函数将前 3 个通道转换为 RGB 值
        rgb = x[..., :3].sigmoid()

        # 使用 ReLU 激活函数将第 4 个通道转换为密度值
        sigma = x[..., 3].relu()

        # 返回密度 sigma 和 RGB 值 rgb
        return sigma, rgb
# -------------------------------------------------无视图依赖性的 head（不考虑视角的影响）----------------------------------------------

# -------------------------------------------------有视图依赖性的 head（考虑视角的影响）-----------------------------------------------
class ViewDenepdentHead(nn.Module):
    def __init__(self, ninput, nview):
        # 初始化，视角依赖性输出层
        super().__init__()

        # 公共特征层
        self.feature = nn.Linear(ninput, ninput)

        # 将特征与视角拼接并进行降维
        self.view_fc = nn.Linear(ninput + nview, ninput // 2)

        # 输出密度的线性层
        self.alpha = nn.Linear(ninput, 1)

        # 输出 RGB 的线性层
        self.rgb = nn.Linear(ninput // 2, 3)

    def forward(self, x, view_dirs):
        # 通过特征层提取输入特征
        feature = self.feature(x)

        # 通过 alpha 层获得密度值，并使用 ReLU 激活
        sigma = self.alpha(x).relu()

        # 将特征和视角信息拼接在一起
        feature = torch.cat([feature, view_dirs], dim=-1)

        # 融合视角信息的特征层，并使用 ReLU 激活
        feature = self.view_fc(feature).relu()

        # 通过 RGB 层获得 RGB 输出，并使用 Sigmoid 激活
        rgb = self.rgb(feature).sigmoid()

        # 返回密度 sigma 和 RGB 值 rgb
        return sigma, rgb
# -------------------------------------------------有视图依赖性的 head（考虑视角的影响）-----------------------------------------------

# ----------------------------------------------------------nerf基础模型----------------------------------------------------------
class NeRF(nn.Module):
    def __init__(self, x_pedim=10, nwidth=256, ndepth=8, view_pedim=4):
        """
        作用：
            初始化模型的各层、宽度和深度
        :param x_pedim:空间坐标的编码维度
        :param nwidth:隐藏层的的神经元数量
        :param ndepth:模型的隐藏层数量，即网络的深度
        :param view_pedim:视角编码维度
        """
        super().__init__()

        # 输入维度经过编码后的总维度 (考虑正弦/余弦频率)
        xdim = (x_pedim * 2 + 1) * 3

        # 存储前8层的列表
        layers = []

        # 每层网络的输入维度
        layers_in = [nwidth] * ndepth

        # 首层输入为位置编码维度
        layers_in[0] = xdim

        # 第6层的输入需要拼接上原始的位置编码
        layers_in[5] = nwidth + xdim

        # 定义前8层结构
        for i in range(ndepth):
            # 创建每层的线性层
            layers.append(nn.Linear(layers_in[i], nwidth))

        # 如果视角位置编码维度大于0，则需要将视角进行编码
        if view_pedim > 0:
            # 视角编码后的总维度
            view_dim = (view_pedim * 2 + 1) * 3

            # 初始化视角编码器
            self.view_embed = Embedder(view_pedim)

            # 视角依赖的输出层
            self.head = ViewDenepdentHead(nwidth, view_dim)
        else:
            # 没有视角依赖的输出层
            self.head = NoViewDirHead(nwidth, 4)

        # 初始化输入位置编码器
        self.xembed = Embedder(x_pedim)

        # 使用 nn.Sequential 包含所有线性层
        self.layers = nn.Sequential(*layers)

    def forward(self, x, view_dirs):
        # 获取输入 x 的形状信息
        xshape = x.shape

        # 将输入 x 进行位置编码
        x = self.xembed(x)

        # 如果使用视角依赖性
        if self.view_embed is not None:
            # 将视角扩展至与 x 相同的形状
            view_dirs = view_dirs[:, None].expand(xshape)

            # 对视角方向编码
            view_dirs = self.view_embed(view_dirs)

        # 保存原始 x 编码，用于跳跃连接
        raw_x = x

        # 遍历模型每一层
        for i, layer in enumerate(self.layers):
            # 应用 ReLU 激活函数
            x = torch.relu(layer(x))
            # 在第5层使用跳跃连接，将初始编码拼接到当前特征
            if i == 4:
                # 拼接跳跃连接后的特征
                x = torch.cat([x, raw_x], axis=-1)

        # 调用模型的 head 输出层，计算最终的密度和 RGB 值
        return self.head(x, view_dirs)
# ----------------------------------------------------------nerf基础模型----------------------------------------------------------

# ---------------------------------------------------生成三维空间中光线的采样点位置---------------------------------------------------
def sample_rays(ray_directions, ray_origins, sample_z_vals):
    """
    作用：
        生成三维空间中光线的采样点位置
    :param ray_directions:光线的方向向量
    :param ray_origins:光线的起始点或原点
    :param sample_z_vals:用于沿着光线采样的深度值
    :return:
        rays：形状为 [nrays, N_samples, 3]，包含每条光线在不同深度位置的采样点坐标
        sample_z_vals：重复后的采样深度值，形状为 [nrays, N_samples]
    """
    # 获取当前图像中的光线数量，默认为1024
    nrays = len(ray_origins)

    # 将 sample_z_vals 的深度值扩展到所有光线，生成一个 (1024, 64) 的矩阵
    sample_z_vals = sample_z_vals.repeat(nrays, 1)

    # 计算每个光线在不同深度的采样点坐标，rays的shape为[1024, 64, 3]
    rays = ray_origins[:, None, :] + ray_directions[:, None, :] * sample_z_vals[..., None]

    # 返回采样点坐标和对应的深度值
    return rays, sample_z_vals
# ---------------------------------------------------生成三维空间中光线的采样点位置---------------------------------------------------


# ---------------------------------------------------对光线的方向向量进行归一化处理---------------------------------------------------
def sample_viewdirs(ray_directions):
    """
    作用：
        对光线的方向向量进行归一化处理
    :param ray_directions: ray_directions：形状为 [nrays, 3] 的张量，每一行表示一条光线的方向向量
    :return:返回一个与 ray_directions 形状相同的张量，但每个方向向量的模长为1
    """
    # 计算每条光线方向向量的模长（或范数）
    direction_norms = torch.norm(ray_directions, dim=-1, keepdim=True)

    # 将每个方向向量除以其模长，使每条光线的方向向量归一化
    normalized_directions = ray_directions / direction_norms

    # 返回归一化后的方向向量
    return normalized_directions
# ---------------------------------------------------对光线的方向向量进行归一化处理---------------------------------------------------

# ------------------------------------------将密度 (sigma) 和颜色 (rgb) 信息转化为最终的渲染图像颜色------------------------------------
def predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background=False):
    """
    作用：
        根据体渲染算法将密度 (sigma) 和颜色 (rgb) 信息转化为最终的渲染图像颜色
    :param sigma:密度，形状为[1024, 64, 1]，代表每条光线在不同采样点的密度值
    :param rgb:颜色值，形状为[1024, 64, 3]，每个采样点的 RGB 颜色
    :param z_vals:深度采样值，形状为[1024, 64]
    :param raydirs:光线方向,形状为[1024, 3]
    :param white_background:布尔值，表示是否使用白色背景
    :return:
        rgb：渲染后的颜色，形状为 [N_rays, 3]
        depth：每条光线的深度值，形状为 [N_rays]
        acc_map：累积权重，用于表示图像中的不透明度掩膜，形状为 [N_rays]
        weights：每个采样点的权重，形状为 [N_rays, N_samples]
    """

    # 获取设备信息，用于创建在同一设备上的张量
    device = sigma.device

    """计算两个采样点之间的距离delta"""
    # 计算相邻采样点之间的距离,delta_prefix.shape=[1024, 63]
    delta_prefix = z_vals[..., 1:] - z_vals[..., :-1]

    # 在最后补充一个大的深度值，以模拟光线的终点,delta_addition.shape=[1024, 1]
    delta_addition = torch.full((z_vals.size(0), 1), 1e10, device=device)

    # 合并深度差值，使每个采样点都有一个距离值（最后一个点到“无穷远”）,delta.shape=[1024, 64]
    delta = torch.cat([delta_prefix, delta_addition], dim=-1)

    # 因为delta是根据z_vals计算的，忽略了光线方向的长度，所以将delta乘以 raydirs 的范数（也就是方向向量的长度），将这个距离转换为沿着光线方向的真实空间距离
    delta = delta * torch.norm(raydirs[..., None, :], dim=-1)

    """计算不透明度alpha"""
    # 根据密度和距离计算每个采样点的透明度，1 - exp(-sigma * delta)，alpha.shape=[1024, 64]
    alpha = 1.0 - torch.exp(-sigma * delta)

    """计算采样点对最终颜色的贡献度权重w"""
    # 计算累积的透射率，exp_term 表示在当前采样点的透射率，对应公式中的每个(1−alpha_n)的部分, exp_term.shape=[1024, 64]
    exp_term = 1.0 - alpha

    # 防止数值问题
    epsilon = 1e-10

    # 在开始位置添加1（完全透明）以便累积计算，exp_addition.shape=[1024, 1]
    exp_addition = torch.ones(exp_term.size(0), 1, device=device)
    exp_term = torch.cat([exp_addition, exp_term + epsilon], dim=-1)  # exp_term.shape=[1024, 65]

    # 计算沿光线的累计透明度，cumprod累积透射率，对应公式中的(1-alpha_1)(1-alpha_2)...(1-alpha_i)；transmittance.shape=[1024, 64]
    transmittance = torch.cumprod(exp_term, axis=-1)[..., :-1]

    # 计算每个采样点的权重，weights 表示对最终颜色的贡献
    weights = alpha * transmittance  # weights.shape=[1024, 64]

    """通过将所有采样点颜色累加得到最终的rgb值"""
    # 使用权重对颜色进行加权求和，得到光线的最终RGB颜色, rgb.shape=[1024, 3]
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)

    # 使用权重对深度进行加权求和，得到光线的最终深度，代表光线穿过的“有效”深度
    depth = torch.sum(weights * z_vals, dim=-1)  # depth.shape=[1024]

    # 累加权重，得到不透明度掩膜
    acc_map = torch.sum(weights, -1)


    # 如果是白色背景，将不透明度的贡献加到RGB上，以模拟白色背景效果
    if white_background:
        rgb = rgb + (1.0 - acc_map[..., None])

    # 返回RGB颜色，深度值，不透明度掩膜，以及采样点的权重
    return rgb, depth, acc_map, weights
# ------------------------------------------将密度 (sigma) 和颜色 (rgb) 信息转化为最终的渲染图像颜色------------------------------------

# ---------------------------------------------------------生成新的128个采样点-----------------------------------------------------
def sample_pdf(bins, weights, N_samples, det=False):
    """
    作用：
        根据给定的权重分布（weights）生成新的采样点
    :param bins: 用于采样的区间，[1024, 63]
    :param weights:每个区间的权重, [1024, 62]
    :param N_samples:需要采样的点数
    :param det:是否使用确定性采样。如果为 True，则使用均匀分布的采样点；否则使用随机分布的采样点。
    :return:返回形状为 [batch, N_samples] 的张量，表示在指定区间内根据权重采样得到的新的采样点
    """

    # 获取设备信息，以便在相同设备上进行计算
    device = weights.device

    # 避免权重中出现零，添加一个小数以防止出现NaN，[1024, 62]
    weights = weights + 1e-5

    # 计算概率密度函数 (PDF), [1024, 62]
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # 计算累积分布函数 (CDF), [1024, 62]
    cdf = torch.cumsum(pdf, -1)

    # 在CDF开头添加一个零，以确保CDF从0开始, [1024, 63]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    # 生成均匀分布的采样点（如果是确定性采样）,[1024, 128]
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        # 扩展 u 的形状以匹配CDF的批次大小
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 随机采样，如果det=False
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # 将采样点u的值与CDF进行匹配以实现反向采样
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)  # inds 表示每个采样点在 cdf 中的区间位置索引，[1024, 128]

    # 找到每个采样点的上下界
    below = torch.max(torch.zeros_like(inds-1), inds-1)  # inds - 1 得到下边界索引
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # 上边界索引
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)  # 组合上下界索引

    # 从CDF和区间中获取对应的边界值
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)


    # 计算每个采样点的具体位置
    denom = (cdf_g[...,1]-cdf_g[...,0])  # 计算区间长度 denom
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)  # 避免除零
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    # 返回采样的深度值,[1024, 128]
    return samples
# ---------------------------------------------------------生成新的128个采样点-----------------------------------------------------

# ----------------------------------------通过粗模型和细模型去预测采样点的颜色和密度，并进行体渲染-----------------------------------------
def render_rays(model, fine, raydirs, rayoris, sample_z_vals, importance=0, white_background=False):
    """
    作用：
        通过粗模型和细模型去预测采样点的颜色和密度，并进行体渲染
    :param model: 初步渲染模型
    :param fine:细化模型
    :param raydirs:光线的方向向量
    :param rayoris:光线的起点向量
    :param sample_z_vals:初始的深度采样点
    :param importance:细化采样的数量
    :param white_background:是否使用白色背景
    :return:
        rgb1：初步渲染得到的颜色结果
        rgb2：分层采样得到的最终颜色结果（渲染最终图像）
    """

    # 基于初始采样点生成沿光线方向的采样点坐标 `rays` [1024, 64, 3]和深度 `z_vals`[1024, 64]
    rays, z_vals = sample_rays(raydirs, rayoris, sample_z_vals)

    # 对光线方向进行归一化，以便用于视角依赖的特征计算,[1024, 3]
    view_dirs = sample_viewdirs(raydirs)

    # 使用初步模型预测每个采样点的密度 `sigma` 和颜色 `rgb`
    sigma, rgb = model(rays, view_dirs)

    # 将 sigma 的最后一个维度（如果是1）移除
    sigma = sigma.squeeze(dim=-1)

    # 使用预测的 `sigma` 和 `rgb` 计算初步渲染的 RGB 颜色、深度、累积不透明度和采样权重
    rgb1, depth1, acc_map1, weights1 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)

    # 使用 `weights1` 进行重要区域的重采样，以在关键区域进行更精细的采样
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # 计算每两个相邻采样点之间的中点

    z_samples = sample_pdf(z_vals_mid, weights1[..., 1:-1], importance, det=True)  # 生成新的128个采样点

    z_samples = z_samples.detach()  # 分离计算图，避免反向传播

    # 将初步采样点 `z_vals` 与新采样点 `z_samples` 合并，并按顺序排序
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    # 根据更新后的深度 `z_vals` 计算新的采样点坐标 `rays`,[1024, 192, 3]
    rays = rayoris[..., None, :] + raydirs[..., None, :] * z_vals[..., :, None]  # 每条光线的新采样点坐标

    # 使用 `fine` 模型在新采样点上进行细化预测，得到密度 `sigma` 和颜色 `rgb`
    sigma, rgb = fine(rays, view_dirs)
    sigma = sigma.squeeze(dim=-1)  # 移除 `sigma` 的最后一个维度

    # 使用预测的 `sigma` 和 `rgb` 计算最终的 RGB 颜色、深度、不透明度和权重
    rgb2, depth2, acc_map2, weights2 = predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background)

    # 返回初步渲染结果 `rgb1` 和最终渲染结果 `rgb2`
    return rgb1, rgb2
# ----------------------------------------通过粗模型和细模型去预测采样点的颜色和密度，并进行体渲染-----------------------------------------
