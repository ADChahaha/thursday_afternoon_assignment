import torch
from resnet import ResNet
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np


class ImageProcessor:

    def __init__(self, ckpt_path: str) -> None:
        dict = torch.load(ckpt_path, weights_only=False)
        self.model: torch.nn.Module = ResNet()
        self.model.load_state_dict(dict["model"])
        self.transform = transforms.Compose(
            [transforms.CenterCrop([256, 256]), transforms.ToTensor()]
        )
        self.model.eval()
        self.img: torch.tensor

    def set_image(self, img: Image.Image | list[Image.Image]) -> None:
        """设置图像用于后续处理

        Args:
            img (list[PIL.image] | PIL.image): PIL图像或PIL图像列表
        """
        # 转换为tensor且提升batch为1
        if isinstance(img, list):
            tensors = [self.transform(i) for i in img]
            self.img = torch.stack(tensors, dim=0)
        else:
            tensor = self.transform(img)
            self.img = tensor.unsqueeze(dim=0)

    def eval(self) -> list:
        """将内部的image全部识别

        Returns:
            返回所有图像的类别的数组
        """
        y = self.model(self.img)  # (batch_size, num_class)
        idx = torch.argmax(y, dim=1)  # idx shape: (batch_size,)
        return [self.__idx2type__(i.item()) for i in idx]

    def __idx2type__(self, idx: int) -> str:
        """将模型识别的idx转换为文本

        Args:
            idx (int): 模型生成的idx 假设为0或1

        Returns:
            str: 下标对应的文本描述
        """
        if 0 == idx:
            return "HUMAN"
        elif 1 == idx:
            return "AI"
        else:
            return "UNKNOWN"

    def cam(self) -> list[Image.Image]:
        """
        对当前设置的 self.img 生成 CAM 热力图
        返回:
            list[PIL.Image] 带热力图叠加原图
        """
        if self.img is None:
            raise ValueError("请先用 set_image() 设置图像")

        # 转换原图为 [0,1] numpy, 用于 show_cam_on_image
        def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
            # t: [C,H,W]
            img_np = t.permute(1, 2, 0).cpu().numpy()
            # 归一化到 [0,1]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            return img_np

        # 找到最后一个卷积层
        def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
            last_conv = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is None:
                raise ValueError("未找到 Conv2d 层")
            return last_conv

        # 针对多张图片
        imgs = []
        target_layer = find_last_conv_layer(self.model)

        cam = GradCAM(
            model=self.model,
            target_layers=[target_layer],
        )

        for i in range(self.img.shape[0]):
            input_tensor = self.img[i].unsqueeze(0)
            # 获取预测类别
            outputs = self.model(input_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            targets = [ClassifierOutputTarget(pred_idx)]
            # 生成CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # [H,W]
            img_np = tensor_to_numpy(self.img[i])
            cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            imgs.append(Image.fromarray(cam_img))
        return imgs
