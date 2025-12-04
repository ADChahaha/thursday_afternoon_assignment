from torch.utils.data import DataLoader
import torchvision
import lightning
from data.dataset import TrainDataset


class MyDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        transform = torchvision.transforms.ToTensor()
        self.train_dataset = TrainDataset(is_train=True, args=args)
        self.val_dataset =TrainDataset(is_train=False, args=args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )
