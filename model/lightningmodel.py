import lightning
import torchmetrics
import torch
from model import Model


class MyModel(lightning.LightningModule):
    def __init__(self, lr=1e-3, num_classes=0):
        super().__init__()
        model = Model()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

    def training_step(self, batch):
        img, label = batch
        y = self.model(img)
        loss = self.criterion(y, label)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def validation_step(self, batch):
        img, label = batch
        y = self.model(img)
        loss = self.criterion(y, label)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y, label), prog_bar=True)

        
