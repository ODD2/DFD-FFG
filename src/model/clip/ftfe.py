import wandb
import torch
import pickle
import torch.nn as nn

from src.clip.model_vpt import PromptMode, LayerNorm
from src.model.base import ODBinaryMetricClassifier
from src.model.clip import FrameAttrExtractor


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = FrameAttrExtractor(*args, **kargs)
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
        logits = self.proj(results["embeds"].mean(1))
        return dict(
            logits=logits,
            **results
        )


class LinearMeanVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        prompt_mode: PromptMode = PromptMode.NONE,
        prompt_num: int = 0,
        prompt_layers: int = 0,
        prompt_dropout: float = 0,
        text_embed: bool = False,
        attn_record: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            prompt_mode=prompt_mode,
            prompt_num=prompt_num,
            prompt_layers=prompt_layers,
            prompt_dropout=prompt_dropout,
            text_embed=text_embed,
            attn_record=attn_record
        )
        self.model = BinaryLinearClassifier(**params)

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
            reduction="none"
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
