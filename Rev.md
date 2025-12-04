# Generative Universal AI-Generated Images Detection : From An Image Transformation Perspective

## Introduction
随着近期生成模型（GAN, Diffusion等）能够生成高度逼真的图像，AI生成图像在社交平台上的泛滥也带来了一定的负面影响，因此迫切需要开发高效的检测手段。当前合成图像检测（Synthetic Image Detection, SID）方法主要致力于挖掘通用的伪造痕迹特征，却普遍忽视了 SID 训练范式本身的问题。本文重新审视 SID 任务，发现现有训练范式中存在两大普遍偏差：伪造痕迹被弱化（weakened artifact features）和伪造痕迹被过拟合（overfitted artifact features）。与此同时，我们发现合成图像的成像机制导致像素间存在更强的局部相关性，这提示检测器应具备局部感知能力。
有鉴于此，我们提出一种轻量且高效的检测器，仅通过三种简单图像变换即可显著提升性能：
- 针对“伪造痕迹被弱化”问题，我们在图像预处理阶段用**裁剪(crop)** 替代传统下采样操作，避免伪造痕迹在缩放过程中被破坏；
- 针对“伪造痕迹被过拟合”问题，我们引入 **ColorJitter** 和 **RandomRotation** 作为额外数据增强，缓解有限训练样本带来的颜色偏差和语义差异等无关偏置；
- 针对“局部感知能力不足”问题，我们设计了一种专为 SID 定制的基于补丁的随机遮挡策略（patch-based random masking），在训练时迫使检测器聚焦于局部区域。

## Related Work

### Image Generated
#### GAN
1. **GAN(original)**
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.
2. **DCGAN**
Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
3. **WGAN / WGAN-GP**
Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. arXiv preprint arXiv:1701.07875.
Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of Wasserstein GANs. Advances in Neural Information Processing Systems, 30.
4. **StyleGAN**
Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4401–4410.

#### Diffusion
5. **DDPM**
Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840–6851.
6. **Stable Diffusion**
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.

### AIGC Detection
#### CNN Based
1. **EfficientNet-based Detectors**
Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the International Conference on Machine Learning (ICML), 6105–6114.

#### Frequency Domain Based
2. **F^3-Net**
Durall, R., Keuper, M., & Keuper, J. (2020). Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 250–251.
3. **Spectral Foreground Consistency (SFC)**
Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Wiedemann, S. (2021). Spectral foreground consistency for detecting AI-generated images. arXiv preprint arXiv:2112.01786.

#### GAN Fingerprint Analysis
4. **GAN Fingerprint**
Zhang, Y., Li, R., Qin, L., Li, J., & Yang, Y. (2020). GAN fingerprinting: Training GANs to have unique fingerprints. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2657–2661.
5. **LEAF**
Cozzolino, D., Müller, T., Thies, J., Nießner, M., & Verdoliva, L. (2022). LEAF: Local attribution fingerprints for image source identification. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19055–19064.

#### Transformer Based
6. **Swin Transformer for AI Image Detection**
Liu et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
7. **Vision Transformer-based detectors (ViT / DeiT)**
Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. Proceedings of the International Conference on Machine Learning (ICML), 10347–10357.

#### Multi-Branch
8. **Gram-Net**
Liu, J., Gao, Y., Li, Z., & Wang, H. (2020). Gram-Net: A frequency-aware generative model detector. Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), 1–6.
9. **ManTra-Net**
Wu, Y., Zhu, J., Zhang, Y., & Su, W. (2019). ManTra-Net: Manipulation tracing network for detection and localization of image forgeries with anomalous features. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 955–964.

#### For Diffusion-Image
10. **DAD**
Li, Y., Yang, X., Sun, C., Qi, Q., & Liao, J. (2023). DAD: Diffusion artifacts detector for AI-generated image detection. Proceedings of the ACM International Conference on Multimedia (ACM MM), 7837–7846.
11. **NPR**
Wang, C., Qi, Q., Sun, C., Liao, J., & Zhang, H. (2023). NPR: Noise-print residual for detecting diffusion model generated images. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5.

### Muti-model generalization research
1. **Universal Fake Image Detection (UFD)**
Gragnaniello, D., Marra, F., Cozzolino, D., Verdoliva, L., & Pitas, I. (2023). Universal fake image detection via adversarial domain generalization. IEEE Transactions on Information Forensics and Security, 18, 3432–3445.
2. **Prompt-level Metadata Analysis (NAI, SD)**
Yu, N., Liu, L., & Fritz, M. (2023). Exploiting metadata in AI-generated image detection. Proceedings of the IEEE International Workshop on Information Forensics and Security (WIFS), 1–6.

### Data Acquisition
1. **AIGD**
Wang, Y., Liu, H., Liu, Y., & Wu, X. (2022). AIGD: A benchmark dataset for AI-generated image detection. Proceedings of the ACM Workshop on Information Hiding and Multimedia Security (IH&MMsec), 123–132.
2. **GenImage Dataset**
Zhu, J., Wang, Y., Liu, H., & Liu, Y. (2023). GenImage: A large-scale benchmark for universal AI-generated image detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 21025–21035.

### Guiding literature
Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review
Zhang, H., & Wang, S. (2024). Methods and trends in detecting AI-generated images: A comprehensive review. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(5), 2918–2939.


## Method

### Dataset Acquisition
generator: [StyleGAN3](https://github.com/NVlabs/stylegan3)

为防止模型利用语义差异（比如人脸 vs 风景）做判别，数据集需要进行分类，保证检测器学到的是伪影/噪声/频谱差异而非语义。

#### Real Part
| 类别   | 真实来源 | 数量 | 下载链接                                                                 |
|--------|----------|------|--------------------------------------------------------------------------|
| portrait | FFHQ     | 1500 | [FFHQ — NVIDIA](https://github.com/NVlabs/ffhq-dataset)（使用官方脚本下载） |
| cat      | ImageNet | 1500 | https://image-net.org/data/winter21_whole/n02121620.tar                   |
| dog      | ImageNet | 1500 | https://image-net.org/data/winter21_whole/n02084071.tar                   |
| car      | ImageNet | 1500 | https://image-net.org/data/winter21_whole/n02958343.tar                   |
| church   | ImageNet | 1500 | https://image-net.org/data/winter21_whole/n03028079.tar                   |
| anime_portrait      | Danbooru | 1500 | https://danbooru.donmai.us/posts.json?tags=1girl+solo+rating%3Asafe                   |

#### Fake Part
| 类别   | 生成器 | 数量 | 模型链接 |
|--------|----------|------| --------------------------------------------------------------------------|
| portrait | StyleGAN3-FFHQ     | 1500 | https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl |
| wild      | StyleGAN2-afhqv2 | 3000 | https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl |
| car      | StyleGAN2-Car | 1500 | http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl |
| church   | StyleGAN2-Church | 1500 | http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl |
| anime_portrait | StyleGAN2-Anime | 1500 | https://mega.nz/file/PeIi2ayb#xoRtjTXyXuvgDxSsSMn-cOh-Zux9493zqdxwVMaAzp4 |

18000 train images in total.

#### Directory Structure
```text
data/datasets/  
└── train_ForenSynths/  
    ├── train/  
    │   ├── portrait/  
    │   │   ├── 0_real/  
    │   │   └── 1_fake/  
    │   ├── wild/  
    │   ├── car/  
    │   ├── church/  
    │   └── anime_portrait/  
    └── val/
```
### Model Architecture

#### Input Preprocessing
- Crop
```python
transforms.RandomCrop([256, 256], pad_if_needed=True),
```
- Flip  
```python
transforms.RandomHorizontalFlip(p=0.5)
```
- Rotation
```python
transforms.RandomRotation(180)
```
- ColorJitter
```python
transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
```
- RandomMasking
```python
RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5)
```
经过以上预处理后，将图像进行小波变换并取HH分量的高频方向作为模型输入。
```python
def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
    DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
    Yl, Yh = DWT_filter(x)
    return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])
```
#### Backbone
骨干网络为ResNet-50的变体，四层layer只保留前两层降低模型复杂度。
```python
# input process
self.inplanes = 64
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.relu = nn.ReLU(inplace=True)
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# resnet layers
self.layer1 = self._make_layer(block, 64 , layers[0])
self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
# classification head
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc1 = nn.Linear(512, num_classes)
```
layer采用标准的Bottleneck结构。

### Train and Evaluation
#### Training Settings
- 损失函数  
默认为 torch.nn.CrossEntropyLoss()。启用 Mixup 或标签平滑时，使用 SoftTargetCrossEntropy 或 LabelSmoothingCrossEntropy。,criterion 初始化
- 优化器  
通过 create_optimizer 自定义创建，使用分层学习率衰减（Layer Decay）逻辑。
- 学习率调度  
迭代级调度 (adjust_learning_rate)，在每个训练迭代步进行更新，而非每个 Epoch 结束。包含一个预热期 (warmup_epochs=1)。
- 有效批量大小  
32×1=32（--update_freq，默认为 1）。
- Mixup/Cutmix  
对样本和标签进行混合，生成软标签，提升模型泛化能力。
- 梯度累积  
通过 loss /= update_freq 实现梯度累积。optimizer.step() 和 optimizer.zero_grad() 仅在 (data_iter_step + 1) % update_freq == 0 时执行。
- 超参数设置  
batch_size=32, blr=1e-2, weight_decay=0.01, warmup_epochs=1, epochs=20

#### Evaluation Metrics
- 评估数据流  
使用 torch.utils.data.SequentialSampler 进行顺序采样，使用 torch.cat 收集整个验证集或测试集的 Logits (predictions) 和真实标签 (labels)。
- 评估指标
1. Top-1 准确率(acc1): 基于 Logits 的 Top-1 预测
2. 二分类准确率(acc): 模型最终二分类性能。对 Softmax 后的类别 1 概率进行阈值 (0.5) 判定，计算准确率。
3. 平均精度(ap): 鲁棒性指标。衡量模型在类别不平衡或排序任务上的性能，使用类别 1 的 Softmax 概率作为得分。