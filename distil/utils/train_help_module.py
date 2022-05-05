import collections
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

import sys

sys.path.append('../')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class TrainModule(pl.LightningModule):
    def __init__(self, net: torch.nn.Module,
                 optimizer: str='sgd',
                 n_epochs: int=None,
                 lr: float=1e-3,
                 is_reset: bool=False,
                 criterion: torch.nn.Module=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_reset = is_reset
        self.criterion = criterion
        self.metric = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        acc = self.metric(y_pred, y)
        self.log(
            'loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.log(
            'acc',
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        acc = self.metric(y_pred, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        acc = self.metric(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss, acc

    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
            return [optimizer, scheduler]
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.)
            return optimizer

    def on_fit_start(self) -> None:
        if self.is_reset:
            self.net = self.net.apply(weight_reset)
        opt = self.optimizers()
        opt.__setstate__({'state': collections.defaultdict(dict)})

        if self.optimizer == 'sgd':
            self.lr_schedulers().__setstate__({'state': collections.defaultdict(dict)})
