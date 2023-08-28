import torch.nn as nn
import src.clip as CLIP
from src.model.base import ODClassifier


class LinearProbe(nn.Module):
    def __init__(self, architecture: str = "ViT-B/16", output_dim: int = 2):
        super().__init__()
        self.model, self.transform = CLIP.load(architecture)
        self.model = self.model.float()
        self.linear = nn.Linear(self.model.visual.output_dim, output_dim, bias=False)
        # disable gradients
        for params in self.model.parameters():
            params.requires_grad_(False)

    def forward(self, x):
        if len(x.shape) > 4:
            assert x.shape[1] == 1
            x = x.squeeze(1)
        return self.linear(self.model.encode_image(x))

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
        return self


class CLIPLinearProbe(ODClassifier):
    def __init__(self, output_dim=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbe(output_dim=output_dim)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.model.visual.input_resolution

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            "train/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, dts_name = *batch[:2], batch[-1]
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            f"test/{dts_name}/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # this is the validation loop
        x, y, dts_name = *batch[:2], batch[-1]
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            f"valid/{dts_name}/loss",
            loss,
            batch_size=x.shape[0]
        )
        return loss
