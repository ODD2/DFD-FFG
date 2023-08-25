
import os
import torch
from torch import optim, nn, utils, Tensor
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import lightning.pytorch as pl

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)


# Load datasets
transform = ToTensor()
train_set = MNIST(os.getcwd(), download=True, train=True, transform=transform)
test_set = MNIST(os.getcwd(), download=True, train=False, transform=transform)
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = utils.data.DataLoader(train_set)
val_loader = utils.data.DataLoader(valid_set)
test_loader = utils.data.DataLoader(test_set)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=0.01, max_epochs=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=autoencoder, dataloaders=test_loader)  # TODO: train_dataloaders vs dataloaders?
