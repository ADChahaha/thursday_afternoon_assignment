import lightning
import torchmetrics
import torchvision
import torch


class MyModel(lightning.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(512, 10)
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
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
