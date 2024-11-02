# NeRF_Replication

**本仓库是将NeRF作者原有的工作自己重新实现了一遍，以加深理解**

## 参考项目
https://github.com/yenchenlin/nerf-pytorch

https://github.com/bmild/nerf

https://github.com/NVlabs/instant-ngp

[【较真系列】讲人话-NeRF全解（原理+代码+公式）](https://www.bilibili.com/video/BV1CC411V7oq/?spm_id_from=333.337.search-card.all.click&vd_source=1a02178b1644ddc9b579739c3c1616b4)

[手写Nerf](https://www.bilibili.com/video/BV1xL411y7Qa/?vd_source=1a02178b1644ddc9b579739c3c1616b4)

[NeRF论文的地址](https://arxiv.org/abs/2003.08934)

[MipNerf 从0开始搭建到训练自己的数据集](https://blog.csdn.net/Faith_Plus/article/details/134123188?spm=1001.2014.3001.5506)




## 配置环境

首先，请确保你的电脑正确安装了CUDA

然后，按照下面的指令配置虚拟环境

```
$ conda create -n nerf python=3.9
$ conda activate nerf
$ pip install -r requirements.txt
```

## 数据集下载

在浏览器打开```http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip```下载数据集

把下载好的数据集解压，然后把nerf_synthetic文件夹放到data目录下。该数据集提供了一个乐高积木的不同角度图像和对应的相机位姿数据，用于NeRF进行训练。

## 训练NeRF

在项目根目录执行``train_NeRF.py```开始训练NeRF

```
python train_NeRF.py --dataset_root ./data/nerf_synthetic/lego --transforms_file transforms_train.json
```
其中：

- --dataset_root为数据集目录路径
- --transforms_file为相机位姿数据文件路径

## 使用训练好的模型进行推理

训练好的模型会保存在项目根目录下的ckpt文件夹下

在项目的根目录执行```make_video.py```进行推理：此脚本会在```rotate360```文件夹下生成重建物体360度的不同视角图，然后在```videos```文件夹下将不同的视角图拼接成视频
```
python make_video.py --ckpt ckpt/100000.pth --data_path "data/nerf_synthetic/lego" --transforms_file transforms_train.json
```
其中：

- --ckpt为模型路径
- --data_path为数据集路径
- --transforms_file为相机位姿数据路径

## 使用自己的数据集进行训练

### 将视频分割成图片
使用自己的设备(手机和相机都行)围绕目标物体拍摄时长大约1分钟的视频，帧数建议选30帧，视频格式建议为```.mp4```

然后使用项目根目录下的```video2img.py```脚本将视频分解为一张张图像
```
python video2img --video_path 视频路径 --output_dir 输出路径 --scale_factor 0.25
```
其中：

- --scale_factor表示缩放倍数，0.25表示将图片等比例缩小4倍。

### 使用图片来估算相机位姿
下载并安装colmap，这个是用来估计相机位姿的：[colmap下载地址](https://demuc.de/colmap/#download)

下一步我们需要使用[instant-ngp](https://github.com/NVlabs/instant-ngp)中的```colmap2nerf.py```来估计相机姿态数据：

[instant-ngp源码](https://github.com/NVlabs/instant-ngp)，下载成功后，在终端输入以下指令

```
$ conda create -n ngp -y python=3.9
$ conda activate ngp
$ pip install -r requirements.txt
```
配置好ngp的环境后需要使用函数```colmap2nerf.py```来估算相机的位姿数据
```
python scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --colmap_camera_model SIMPLE_PINHOLE --images [图片路径]
```
最后会在根目录下生成一个```transforms.json```文件，把这个文件移动到自己的数据集目录下

### 开始训练
```
python train_NeRF.py --dataset_root 数据集路径 --transforms_file transforms.json
```


## 从问题驱动的角度对NeRf论文进行解读

通读整篇NeRF论文，我们可以发现整个NeRF模型的结构如下：

NeRF模型结构：

- 输入：5D向量(x， y， z， theta， phi)

- 输出：4D向量(密度， 颜色)

- 模型：8层MLP

---

那么我们现在就遇到一些疑问：

<div style="text-align: center"><strong>问题1：我们输入模型的不应该是一张张2D的图像吗，为什么输入的是5D的向量？ </strong></div>

<div style="text-align: center"><strong>问题2：模型输出的不应该是一张张2D的图像吗，为什么输出的是4D的向量？ </strong></div>

为了解答这两个问题：我们猜测会有一个从图像转为5D向量的预处理过程。同样的，也会有一个把输出的4D图像转换为2D图像的后过程。

首先，我们先来了解一下这个输入的5D向量和输出的4D向量：

输入的5D向量实际上是粒子的空间位姿(x， y， z， theta， phi)

输出的4D向量实际上是粒子对应的颜色以及密度。

---

看到这里，我们又遇到了另一个问题：
<div style="text-align: center"><strong>问题3：这个粒子又是什么东西？ </strong></div>
回答：粒子是某一条光线上的发光点，其属性有x， y， z和颜色，一条光线上可能会含有多个粒子，对于图片上的某一像素(u， v)的颜色可以看作是沿着某一条光线上无数个粒子的和。

---

相信你们看到这里，又又会遇到一个新的问题：
<div style="text-align: center"><strong>问题4：这个光线又是什么？ </strong></div>
在回答这个问题之前，我们先来了解一下坐标系相关的知识：

我们主要会用到三个坐标系：

世界坐标系：对应下图中的 $\( X_{w}， Y_{w}， Z_{w} \)$

相机坐标系：对应下图中的 $\( X_{c}， Y_{c}， Z_{c} \)$

归一化相机坐标系：对应下图中的 $\( X_{n}， Y_{n} \)$

像素坐标系：对应下图中的 $\( U， V \)$

![clipboard_2024-10-27_10-41](https://github.com/user-attachments/assets/e85b5d96-d544-42be-8712-e62464e1ab23)

在了解完坐标系后我们再来看我们的光线是怎么来的：光线是由一张图像和对应的相机位姿计算出来的，一条光线由原点，方向和距离来表示，记为 $\ r(t) = o + td \$，其中o为射线原点，d为方向，t为距离。

下面我们来详细介绍一下如何从一张图像和对应的相机位姿来计算光线：

![clipboard_2024-10-27_10-54](https://github.com/user-attachments/assets/0ef29dcc-f04f-4f93-bb20-26d5b4583421)

最后，我们用 $\( x_{w}， y_{w}， z_{w} \)$ 表示光线在世界坐标系下的方向，用C2W矩阵的最后一列的前三个数表示光线的原点。

然后整张图像的shape为H * W，每个像素点都有一条光线，所以一共会有H * W条光线，所以最后d的shape为 $\ (H * W， 3) \$，o的shape也为 $\ (H * W， 3) \$。

但是在实际过程中，我们会选取batch_size条光线进行处理，所以实际上d的shape为 $\ (batch size， 3) \$， o的shape为 $\ (batch size， 3) \$。

到现在为止，我们已经确定了o和d，下面我们来看一下t是怎么确定的：

理论上：t从0到 $\infty$的，是连续的

实际上:t在计算处理的时候是离散的

方法:分别设置两个变量near=2和far=6，在near和far之间均匀采样64个点，记作pts，shape为(1024， 64， 3) 

这个pts就是我们前面所说的**粒子**

最后再拼接上前面的到的光线方向d(1024， 3)，拼接成6D的向量输入到模型中(所以实际传入的不是5D向量，而是6D的向量!)

到这里，我们就可以回答问题3和问题4和问题1了

---

我们看到这里，又会遇到一个新的问题，我们从下面的模型结构图中可以看到，输入模型的分别是一个60维的向量和一个24维的向量，如下图所示:

![clipboard_2024-10-27_14-27](https://github.com/user-attachments/assets/c26f7df5-f62f-421e-81cb-074ef8478d42)

<div style="text-align: center"><strong>问题5：根据我们前面所说的，模型的输入应该是pts(BatchSize， 64， 3)和ray_d(BatchSize， 3)，但是模型实际输入的是一个60维的向量和一个24维的向量，这是为什么？ </strong></div>

回答:作者通过实验发现，当只输入粒子的3D位置和3D视角时，建模结果会丢失细节，原因是缺乏高频信息.

![clipboard_2024-10-27_14-33](https://github.com/user-attachments/assets/b38e641d-6c4c-4cb9-b554-8c5afe0eafc3)

作者为了解决这个问题，引入了位置编码:

$\gamma(p) = \left( \sin\left(2^0 \pi p\right)， \cos\left(2^0 \pi p\right)， \dots， \sin\left(2^{L-1} \pi p\right)， \cos\left(2^{L-1} \pi p\right) \right)$

- p需要归一化[-1， 1]
- 对于空间坐标**x**， L=10， $\gamma(X)\$ 是60D
- 对于视角坐标**d**， L=4，  $\gamma(d)\$ 是24D
- 在代码中，加上初始值: $\gamma(X)\$ 是63D， $\gamma(d)\$ 是27D

实际代码中的模型结构图:

```
NeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=63， out_features=256， bias=True)
    (1): Linear(in_features=256， out_features=256， bias=True)
    (2): Linear(in_features=256， out_features=256， bias=True)
    (3): Linear(in_features=256， out_features=256， bias=True)
    (4): Linear(in_features=256， out_features=256， bias=True)
    (5): Linear(in_features=319， out_features=256， bias=True)
    (6): Linear(in_features=256， out_features=256， bias=True)
    (7): Linear(in_features=256， out_features=256， bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=283， out_features=128， bias=True)
  )
  (feature_linear): Linear(in_features=256， out_features=256， bias=True)
  (alpha_linear): Linear(in_features=256， out_features=1， bias=True)
  (rgb_linear): Linear(in_features=128， out_features=3， bias=True)
)
```
到这里，我们就可以回答问题5了

---

好了，现在我们已经有了模型，我们要如何去计算这个模型的loss呢?
<div style="text-align: center"><strong>问题6：如何去计算这个模型的loss？ </strong></div>
回答:模型采用自监督的方式去计算loss，具体来说:

- GT是图片某一像素的RGB
- 将该像素对应光线上的粒子颜色进行求和
- 粒子的颜色和:该像素的预测值
- 粒子的颜色和与该像素颜色做MSE
- $\ L = \sum_{r \in R} \left\| \hat{C}(r) - C(r) \right\|_2^2 \$
- R是每个batch的射线(1024条)

到这里，我们就可以回答问题6了

---

但是我们又遇到一个新的问题:

<div style="text-align: center"><strong>问题7：如何将一条射线上的粒子的颜色进行求和？(体渲染部分) </strong></div>
回答:

![图片](https://github.com/user-attachments/assets/7d12b3ac-7177-4b95-a274-13baa0b45ad2)

$$ \hat{C}(s) = \int_{0}^{+\infty} T(s) \sigma(s) C(s) \ ds $$

$$ T(s) = e^{-\int_{0}^{s} \sigma(t) \ dt} $$

- $\ T(s) \$:在s点之前，光线没有被阻碍的概率
- $\ \sigma(s) \$:在s点处粒子的密度信息，密度越大，光线越有可能被阻拦
- $\ C(s) \$:在s点处，粒子发出颜色光
- 各点的颜色和概率密度已知，先求 $\ T(s) \$

上面这种情况是在连续情况下将一条射线上的粒子的颜色进行求和，但是计算机只能处理离散化的数据，所以下面我们需要将上面的公式进行离散化处理:

离散化:

-  将光线[0， s]划分为N个等间距区间 $\ [T_n \rightarrow T_{n+1}] \$
-  n=0， 1， 2， ...， N
-  间隔长度为 $\ \delta_n \$
-  假设区间内密度 $$\ \sigma(n) \$$ 和颜色 $$\ C(n) \$$ 固定

$$ \hat{C}(r) = \sum_{i=1}^{N} T_i \left(1 - e^{-\sigma_i \delta_i}\right) c_i $$
$$ \text{where } T_i = e^{-\sum_{j=1}^{i-1} \sigma_j \delta_j} $$

关于如何从连续的式子推出离散的式子，这里就不细说了

然后在实际的代码实现中，还需要往前继续化简一步:

$\ \hat{C} = \sum_{n=0}^{N} C_n \ e^{-\sum_{i=0}^{n} \sigma_i \delta_i} \left(1 - e^{-\sigma_n \delta_n}\right)  \$


$\ \text{设 } \alpha_n = 1 - e^{-\sigma_n \delta_n} \$

$\ \Rightarrow \hat{C} = \sum_{n=0}^{N} C_n \alpha_n \ e^{-\sum_{i=0}^{n-1} \sigma_i \delta_i} \$

$\ = \sum_{n=0}^{N} C_n \alpha_n \ e^{-(\sigma_0 \delta_0 + \sigma_1 \delta_1 + \cdots + \sigma_{n-1} \delta_{n-1})} \$

$\ = \sum_{n=0}^{N} C_n \alpha_n \ e^{-\sigma_0 \delta_0} e^{-\sigma_1 \delta_1} \cdots e^{-\sigma_{n-1} \delta_{n-1}} \$

$\ = \sum_{n=0}^{N} C_n \alpha_n (1 - \alpha_0)(1 - \alpha_1) \cdots (1 - \alpha_{n-1}) \$

$\ = C_0 \alpha_0 + C_1 \alpha_1 (1 - \alpha_0) + C_2 \alpha_2 (1 - \alpha_0)(1 - \alpha_1) + \cdots + C_n \alpha_n (1 - \alpha_0)(1 - \alpha_1) \cdots (1 - \alpha_{n-1}) \$

所以我们最终可以推导出将一条光线上的粒子颜色进行累加的公式:

$$ \hat{C}=C_0 \alpha_0 + C_1 \alpha_1 (1 - \alpha_0) + C_2 \alpha_2 (1 - \alpha_0)(1 - \alpha_1) + \cdots + C_n \alpha_n (1 - \alpha_0)(1 - \alpha_1) \cdots (1 - \alpha_{n-1}) $$

下面给出了在代码实现中将粒子颜色累加的完整思路:

1.不透明度 $\alpha_n$: 

- 不透明度 $\alpha_n$ 表示采样点 $n$ 对光线的遮挡程度。具体计算方法是：

$$\alpha_n = 1 - e^{-\sigma_n \delta_n}$$

- 这里，sigma_n是粒子的密度， delta_n是光线在这个点上的步长距离。密度越大，步长越长，透明度就越低（不透明度越高）

2.权重 $W_n$：

- 每个采样点的权重 $W_n$ 表示该点对最终颜色的贡献度。权重 $W_n$ 的计算公式为：

$$W_n = \alpha_n \prod_{i=0}^{n-1} (1 - \alpha_i)$$

- 这里的 $\prod_{i=0}^{n-1} (1 - \alpha_i)$ 表示光线在前 $n-1$ 个点都没有被完全遮挡的概率。
- 也就是说，权重 $W_n$ 结合了当前点的的不透明度 $\alpha_n$ 和之前所有点的透过率。

3.颜色累加 $\hat{C}(r)$：

- 最终的颜色累加是将每个点的颜色 $C_n$ 按照权重 $W_n$ 加权平均求和得到的：

$$\hat{C}(r) = \sum_{n=0}^{N} W_n C_n$$

- 这样累加的结果就是光线最终看到的颜色。

到这里，我们就可以回答问题7和问题2了

---

现在我们还有一个问题没有解决
<div style="text-align: center"><strong>问题8：在前面我们都是对光线进行均匀的采样，但是空间中会存在很多的无效区域，我们希望在无效区域少采样/不采样，在有效区域要多采样，这个问题要怎么解决? </strong></div>

回答:可以通过粗模型输出得到一个概率，然后通过这个概率去重新在这条光线上进行采样128个粒子，与之前的64个粒子加在一起，即每条光线采样192个粒子

- 可以根据概率密度进行再次采样
- 由两个模型组成
- 粗模型:输入均匀采样粒子，输出密度
- 细模型:根据密度，二次采样
- 最后输出:采用模型2的输出
- 粗模型和细模型结构相同

举个例子:

已知条件

- bins（位置）：[0.0， 1.0， 2.0， 3.0， 4.0]
- weights（权重）：[0.1， 0.2， 0.4， 0.15， 0.15]

我们希望在这些位置上重新采样，并将采样集中在权重较高的位置。

**步骤 1：计算PDF（概率密度函数）**

首先，将权重归一化以得到 PDF。这个例子中的权重已经是归一化的（总和为 1），所以 PDF 和权重相同：

$$PDF=[0.1，0.2，0.4，0.15，0.15]$$

**步骤 2：计算CDF（累积分布函数）**

接下来，我们计算 CDF，即 PDF 的累加和：

$$CDF=[0.1，0.3，0.7，0.85，1.0]$$

**步骤 3：生成随机数并找到对应的CDF区间**

假设我们希望采样 2 个点，因此生成 2 个均匀分布在[0， 1]之间的随机数：

- 随机数 $u_1=0.25$
- 随机数 $u_2=0.8$

接下来，使用逆 CDF 方法，找到每个随机数落在哪个 CDF 区间内。

对于随机数 $u_1=0.25$ :

- 查看CDF列表，发现0.25落在CDF[0.1，0.3]区间内，对应的bins区间是[0.0， 1.0]
- 所以， $u_1=0.25$ 对应的bins区间是[0.0， 1.0]。

对于随机数 $u_2=0.8$ :

- 查看CDF列表，发现0.8落在 CDF[0.7，0.85]区间内，对应的bins区间是[2.0， 3.0]
- 所以， $u_2=0.8$​ 对应的 bins 区间是 [2.0， 3.0]。

**步骤 4：插值计算采样位置**

对于每个落入的区间，通过插值计算随机数对应的具体采样位置。

插值计算 $u_1=0.25$ 的采样位置:

1.计算插值比例t：

$$t = \frac{(u_1 - \text{CDF}[i])}{(\text{CDF}[i+1] - \text{CDF}[i])} = \frac{(0.25 - 0.1)}{(0.3 - 0.1)} = \frac{0.15}{0.2} = 0.75$$

2.计算采样位置：

$$\text{sample}_1 = \text{bins}[i] + t \times (\text{bins}[i+1] - \text{bins}[i]) = 0.0 + 0.75 \times (1.0 - 0.0) = 0.75$$

因此，随机数 $u_1=0.25$ 对应的采样位置为0.75。

插值计算 $u_2=0.8$ 的采样位置:

1.计算插值比例t:

$$t = \frac{(u_2 - \text{CDF}[i])}{(\text{CDF}[i+1] - \text{CDF}[i])} = \frac{(0.8 - 0.7)}{(0.85 - 0.7)} = \frac{0.1}{0.15} \approx 0.6667$$

2.计算采样位置：

$$\text{sample}_2 = \text{bins}[i] + t \times (\text{bins}[i+1] - \text{bins}[i]) = 2.0 + 0.6667 \times (3.0 - 2.0) = 2.0 + 0.6667 = 2.6667$$

因此，随机数 $u_2=0.8$ 对应的采样位置为 2.6667

最终结果:通过二次采样，我们得到两个新的采样位置：

- $sample_1=0.75$
- $sample_2=2.6667$

到这里，我们就可以回答问题8了

---

**最后我们再来看看模型是怎么进行推理的**

假设我们输入的图像是400 * 400的，则一共会有400 * 400条光线，即:

输入:
- 400 * 400条光线上分别采样64个点

输出:
- [400 * 400 * 192， 4]
- 进行体渲染

![clipboard_2024-10-27_14-27](https://github.com/user-attachments/assets/c26f7df5-f62f-421e-81cb-074ef8478d42)

---

**总结**

![clipboard_2024-10-27_16-15](https://github.com/user-attachments/assets/49444f34-83d2-4f25-90cf-6b0d01a034e5)

**前处理:**
- 将图片中的每个像素，通过相机模型找到对应的射线；
- 在每条射线上进行采样，得到 64 个粒子；
- 对batch_size * 64个粒子进行位置编码；
- 位置坐标为 63D 和方向向量为 27D。

**模型1:**
- 8层MLP，
- 输入为(batch_size， 64， 63)和(batch_size， 64， 27)
- 输出为(batch_size， 64， 4)

**后处理1:**
- 计算模型1的输出，对射线进行二次采样；
- 每条射线上共采样192个粒子。

**模型2:**
- 8层MLP，
- 输入为(batch_size， 192， 63)和(batch_size， 192， 27)
- 输出为(batch_size， 192， 4)

**后处理2:**
- 将模型 2 输出通过体渲染，转换为像素。

