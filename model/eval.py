import torch
from lightningmodel import MyModel
from torchvision import transforms


class ImageProcessor:

    def __init__(self, ckpt_path):
        self.model: torch.nn.Module = MyModel.load_from_checkpoint(ckpt_path)
        self.model.eval()

    def eval(self, img):
        # img2tensor
        img : torch.tensor = transforms.ToTensor()(img)
        # unsqueeze batch_size 1
        img = img.unsqueeze(dim=0)
        y = self.model(img)
        idx = torch.argmax(y, dim=1).item()
        return self.idx2type(idx)

    def idx2type(self, idx):
        if 0 == idx:
            return "HUMAN"
        elif 1 == idx:
            return "AI"
