
from lightning.pytorch.cli import LightningCLI
import os
import torch
from torch import optim, nn, utils, Tensor
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import lightning.pytorch as pl

from src.pytorch.model.clip import LinearProbe as LinearProbeCLIP

from lightning.pytorch.loggers import WandbLogger


# define the LightningModule
class LitClassifier(pl.LightningModule):
    def __init__(self, output_dim=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbeCLIP(output_dim=output_dim)

    def get_transform(self):
        return self.model.transform

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./datasets/", batch_size: int = 64, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def set_transform(self, transform):
        self.transform = transform

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)


def cli_main():
    checkpoint = None
    cli = LightningCLI(run=False, save_config_callback=None, auto_configure_optimizers=False)

    if (checkpoint):
        cli.model.load_from_checkpoint(checkpoint)
        # load checkpoint with other hyper-parameters
        # classifier = LitClassifier.load_from_checkpoint(checkpoint,out_dim=20)

    # Load datasets
    cli.datamodule.set_transform(cli.model.get_transform())
    cli.trainer.logger.watch(cli.model, "all")

    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=checkpoint)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
    cli.trainer.logger.experiment.unwatch(cli.model)
    cli.trainer.logger.finalize("success")


if __name__ == "__main__":
    cli_main()
