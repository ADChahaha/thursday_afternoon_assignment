import torch
from lightningmodel import MyModel
from torchvision import transforms


class ImageProcessor:

    def __init__(self, ckpt_path):
        self.model: torch.nn.Module = MyModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.img: torch.tensor = None

    def set_image(self, img):
        """设置图像用于后续处理

        Args:
            img (list[PIL.image] | PIL.image): PIL图像或PIL图像列表
        """
        # 转换为tensor且提升batch为1
        transform = transforms.ToTensor()
        if isinstance(img, list):
            tensors = [transform(i) for i in img]
            self.img = torch.stack(tensors, dim=0)
        else:
            tensor = transform(img)
            self.img = tensor.unsqueeze(dim=0)

    def eval(self):
        """将内部的image全部识别

        Returns:
            list[str]
        """
        y = self.model(self.img)  # (batch_size, num_class)
        idx = torch.argmax(y, dim=1)  # idx shape: (batch_size,)
        return [self.__idx2type__(i.item()) for i in idx]

    def cam(self):
        # 待实现
        return []

    def __idx2type__(self, idx):
        """将模型识别的idx转换为文本

        Args:
            idx (int): 模型生成的idx

        Returns:
            str: 下标对应的文本描述
        """
        if 0 == idx:
            return "HUMAN"
        elif 1 == idx:
            return "AI"
        else:
            return "UNKNOWN"
