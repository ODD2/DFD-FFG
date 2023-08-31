from torch import optim
from functools import partial
import lightning.pytorch as pl


class ODClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.log = partial(self.log, add_dataloader_idx=False)

    @property
    def transform(self):
        raise NotImplementedError()

    @property
    def n_px(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()

    # TODO: isolate optimizer and lr_scheduler configurations
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer
