import torch.nn as nn
import src.clip as CLIP


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
        return self.linear(self.model.encode_image(x))

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
        return self
