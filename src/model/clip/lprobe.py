import torch
import torch.nn as nn
import src.clip as CLIP

from functools import partial
from src.model.base import ODClassifier, ODBinaryMetricClassifier


class LinearProbe(nn.Module):
    def __init__(self, architecture: str = "ViT-B/16", output_dim: int = 2):
        super().__init__()
        self.model, self.transform = CLIP.load(architecture)
        self.model = self.model.float()
        self.linear = nn.Linear(self.model.visual.output_dim, output_dim, bias=True)
        # disable gradients
        for params in self.model.parameters():
            params.requires_grad_(False)

    def forward(self, x, *args, **kargs):
        if len(x.shape) > 4:
            b, t = x.shape[0:2]
            logits = self.linear(self.model.encode_image(x.flatten(0, 1))).unflatten(0, (b, t)).mean(dim=1)
        else:
            logits = self.linear(self.model.encode_image(x))
        return logits

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


class CLIPBinaryLinearProb(ODBinaryMetricClassifier):
    def __init__(self, architecture="ViT-B/16"):
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbe(output_dim=2, architecture=architecture)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.model.visual.input_resolution
