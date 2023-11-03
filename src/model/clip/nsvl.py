import wandb
import torch
import pickle
import torch.nn as nn

from typing import List

from src.clip import clip as CLIP
from src.clip.model_vpt import LayerNorm
from src.model.base import ODBinaryMetricClassifier
from src.model.clip import VideoAttrExtractor


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = VideoAttrExtractor(
            *args,
            **kargs,
            ignore_attr=True
        )
        create_proj_module = (
            lambda x: nn.Sequential(
                LayerNorm(x),
                nn.Dropout(),
                nn.Linear(x, 2, bias=False)
            )
        )
        self.proj = create_proj_module(self.encoder.embed_dim)

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, *args, **kargs):
        results = self.encoder(x)
        logits = self.proj(results["synos"])
        return dict(
            logits=logits,
            ** results
        )


class NativeSummaryVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1]
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain
        )
        self.model = BinaryLinearClassifier(**params)

        self.label_weights = torch.tensor(label_weights)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.n_px

    def shared_step(self, batch, stage):
        x, y, z = batch["xyz"]
        indices = batch["indices"]
        dts_name = batch["dts_name"]
        names = batch["names"]

        output = self(x, **z)
        logits = output["logits"]

        # classification loss
        cls_loss = nn.functional.cross_entropy(
            logits,
            y,
            reduction="none",
            weight=(
                self.label_weights.to(y.device)
                if stage == "train" else
                None
            )
        )

        if (stage == "train"):
            self.log(
                f"{stage}/{dts_name}/loss",
                cls_loss.mean(),
                batch_size=logits.shape[0]
            )

        return {
            "logits": logits,
            "labels": y,
            "loss": cls_loss.mean(),
            "dts_name": dts_name,
            "indices": indices,
            "output": output
        }


if __name__ == "__main__":
    frames = 5
    model = NativeSummaryVideoLearner()
    model.to("cuda")
    model(torch.randn(9, frames, 3, 224, 224).to("cuda"))
