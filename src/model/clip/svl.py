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
        num_synos,
        *args,
        **kargs,
    ):
        self.num_synos = num_synos
        super().__init__()
        self.encoder = VideoAttrExtractor(
            *args,
            **kargs,
            num_synos=num_synos
        )
        self.post_ln = nn.Sequential(
            LayerNorm(self.encoder.embed_dim),
            nn.Dropout(),
        )

        self.projs = nn.ModuleList(
            [
                nn.Linear(
                    self.encoder.embed_dim,
                    2,
                    bias=False
                )
                for _ in range(self.num_synos)
            ]
        )

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, *args, **kargs):
        results = self.encoder(x)
        synos = results["synos"]
        logits = torch.log(
            sum([
                self.projs[i](synos[:, i]).softmax(dim=-1) / self.num_synos
                for i in range(self.num_synos)
            ]) + 1e-6
        )
        return dict(
            logits=logits,
            ** results
        )


class SynoVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_frames: int,
        num_synos: int = 1,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1],
        store_attrs: List[str] = []
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            num_frames=num_frames,
            num_synos=num_synos,
            store_attrs=store_attrs
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


class FFGSynoVideoLearner(SynoVideoLearner):
    def __init__(
        self,
        face_parts: List[str],
        face_attn_attr: str,
        face_feature_path: str,
        store_attrs: List[str] = ["s_q"],
        *args,
        **kargs
    ):
        self.num_parts = len(face_parts)
        assert self.num_parts > 1
        num_synos = self.num_parts + 1
        super().__init__(
            *args,
            **kargs,
            store_attrs=store_attrs,
            num_synos=num_synos
        )
        self.save_hyperparameters()

        with open(face_feature_path, "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features[face_attn_attr][p][l]
                        for p in face_parts
                    ])
                    for l in range(self.model.encoder.model.transformer.layers)
                ]
            )
            self.face_features = self.face_features.unsqueeze(1)

    def shared_step(self, batch, stage):
        result = super().shared_step(batch, stage)

        if (stage == "train"):
            dts_name = result["dts_name"]
            x = batch["xyz"][0]

            # face feature guided loss
            qs = torch.stack(
                [
                    attrs["s_q"]
                    for attrs in result["output"]["layer_attrs"]
                ]
            )  # qs.shape = [layer,b,t,syno,patch,head]

            # Cosine Alignments

            # qs = qs.mean(2).flatten(-2)
            # face_features = self.face_features.to(dtype=qs.dtype, device=qs.device)

            # cls_sim = (
            #     1 - torch.nn.functional.cosine_similarity(
            #         qs,
            #         face_features,
            #         dim=-1
            #     )
            # ).mean()

            # Contrastive
            qs = qs[:, :, :, 1:].mean(2).flatten(-2)
            l, b, q = qs.shape[:3]
            face_features = self.face_features.to(dtype=qs.dtype, device=qs.device)

            face_features = face_features / face_features.norm(dim=-1, keepdim=True)
            qs = qs / qs.norm(dim=-1, keepdim=True)

            logits = 100 * (qs @ face_features.transpose(-1, -2))

            cls_sim = torch.nn.functional.cross_entropy(
                logits.flatten(0, 2),
                (
                    torch.range(0, self.num_parts - 1)
                    .unsqueeze(0)
                    .repeat((l * b * q, 1))
                    .to(x.device)
                ),
                reduction="none"
            ).mean()

            self.log(
                f"{stage}/{dts_name}/q_sim",
                cls_sim,
                batch_size=x.shape[0]
            )
            result["loss"] += cls_sim

        return result


if __name__ == "__main__":
    frames = 5
    model = SynoVideoLearner(attn_record=True)
    model.to("cuda")
    logit = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))["logits"]
    logit.sum().backward()
    print("done")
