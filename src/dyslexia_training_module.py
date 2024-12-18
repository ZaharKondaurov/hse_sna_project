import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall

from src.dyslexia_model import DyslexiaResNet, DyslexiaCNN

from typing import Literal


class DyslexiaTrainingModule(pl.LightningModule):
    def __init__(self, num_classes: int = 2, model_name: Literal["resnet", "cnn"] = "resnet", img_size: int = 224):
        super().__init__()

        self.num_classes = num_classes
        if model_name == "resnet":
            self.model = DyslexiaResNet(num_classes)
        elif model_name == "cnn":
            self.model = DyslexiaCNN(num_classes, img_size)
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self.model(x)
        ns_logits = F.log_softmax(y_logit, dim=1)
        loss = F.nll_loss(ns_logits, y)

        logs = {"loss": loss, "train_loss": loss}
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return logs

    def validation_step(self, batch, batch_idx):
        """Validation loop"""
        x, y = batch

        y_logit = self.model(x)
        ns_logits = F.log_softmax(y_logit, dim=1)
        loss = F.nll_loss(ns_logits, y)

        logs = {"val_loss": loss}
        self.log_dict(logs, on_step=True, on_epoch=True, logger=True)
        return logs

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logit = self.model(x)
        ns_logits = F.log_softmax(y_logit, dim=1)
        loss = F.nll_loss(ns_logits, y)

        pred = F.softmax(y_logit, dim=1)

        pred_cpu = pred.cpu()
        y_cpu    = y.cpu()
        f1     = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")(pred_cpu, y_cpu)
        recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")(pred_cpu, y_cpu)
        prec   = Precision(task="multiclass", num_classes=self.num_classes, average="macro")(pred_cpu, y_cpu)

        logs = {"test_loss": loss, "test_f1_macro": f1, "test_recall_macro": recall, "test_precision_macro": prec}

        f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="none")(pred_cpu, y_cpu)
        recall = Recall(task="multiclass", num_classes=self.num_classes, average="none")(pred_cpu, y_cpu)
        prec = Precision(task="multiclass", num_classes=self.num_classes, average="none")(pred_cpu, y_cpu)

        for c in range(self.num_classes):
            s = str(c)
            logs.update({f"test_f1_{s}": f1[c]})
            logs.update({f"test_recall_{s}": recall[c]})
            logs.update({f"test_prec_{s}": prec[c]})

        self.log_dict(logs, on_step=True, on_epoch=True, logger=True)
        return logs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=1
        )

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]
