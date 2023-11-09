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
            num_synos=num_synos,
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
        logits = sum([
            self.projs[i](synos[:, i])
            for i in range(self.num_synos)
        ])
        return dict(
            logits=logits,
            ** results
        )


class SynoVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_frames: int = 1,
        num_synos: int = 1,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1],
        store_attrs: List[str] = [],
        align_temper: float = 50,
        align_weight: float = 1.0

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

        self.num_synos = num_synos
        self.align_temper = align_temper
        self.align_weight = align_weight
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

        loss = cls_loss.mean()

        if (stage == "train"):
            clip_video_features = output["embeds"].mean(dim=1)
            syno_video_features = output["synos"].mean(dim=1)

            clip_video_features = clip_video_features / clip_video_features.norm(dim=-1, keepdim=True)
            syno_video_features = syno_video_features / syno_video_features.norm(dim=-1, keepdim=True)

            video_align_logits = self.align_temper * syno_video_features @ clip_video_features.transpose(-2, -1)

            video_align_loss = torch.nn.functional.cross_entropy(
                video_align_logits,
                torch.arange(
                    0,
                    x.shape[0],
                    device=x.device
                )
            )
            self.log(
                f"{stage}/{dts_name}/video_align_loss",
                video_align_loss.mean(),
                batch_size=logits.shape[0]
            )
            loss += video_align_loss.mean() * self.align_weight

        if (stage == "train"):
            self.log(
                f"{stage}/{dts_name}/loss",
                cls_loss.mean(),
                batch_size=logits.shape[0]
            )

        return {
            "logits": logits,
            "labels": y,
            "loss": loss,
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
        # generic
        num_frames: int = 1,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1],
        syno_attn_attr: str = "s_q",
        ffg_temper: float = 5,
        ffg_weight: float = 1e-1,
        ffg_layers: int = -1,
        store_attrs: List[str] = []
    ):
        self.num_face_parts = len(face_parts)
        self.face_attn_attr = face_attn_attr
        self.syno_attn_attr = syno_attn_attr
        self.ffg_temper = ffg_temper
        self.ffg_weight = ffg_weight
        self.ffg_layers = ffg_layers
        super().__init__(
            num_frames=num_frames,
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            label_weights=label_weights,
            num_synos=self.num_face_parts,
            store_attrs=set([*store_attrs, self.syno_attn_attr])
        )
        self.save_hyperparameters()

        with open(face_feature_path, "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features[self.face_attn_attr][p][l]
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
            target_attn_attrs = torch.stack(
                [
                    attrs[self.syno_attn_attr]
                    for attrs in result["output"]["layer_attrs"]
                ]
            )  # qs.shape = [layer,b,t,syno,patch,head]

            if (len(target_attn_attrs.shape) == 4):
                # shape = [l, b, synos, head*width]
                # for: out, emb
                pass
            elif (len(target_attn_attrs.shape) == 5):
                # shape = [l, b, synos, head,width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.flatten(-2)
            elif (len(target_attn_attrs.shape) == 6):
                # shape = [l, b, t, synos, head, width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.mean(2).flatten(-2)
            else:
                raise NotImplementedError()

            face_features = self.face_features.to(
                dtype=target_attn_attrs.dtype,
                device=target_attn_attrs.device
            )

            if self.ffg_layers == -1:
                pass
            elif self.ffg_layers > 0:
                layers = (
                    self.model.encoder.model.transformer.layers -
                    self.ffg_layers
                )
                face_features = face_features[layers:]
                target_attn_attrs = target_attn_attrs[layers:]
            else:
                raise NotImplementedError()

            l, b, q = target_attn_attrs.shape[:3]
            face_features = face_features / face_features.norm(dim=-1, keepdim=True)
            target_attn_attrs = target_attn_attrs / target_attn_attrs.norm(dim=-1, keepdim=True)

            logits = self.ffg_temper * (target_attn_attrs @ face_features.transpose(-1, -2))

            cls_sim = torch.nn.functional.cross_entropy(
                logits.flatten(0, 2),
                (
                    torch.arange(
                        0,
                        self.num_face_parts
                    )
                    .repeat((l * b))
                    .to(x.device)
                ),
                reduction="none"
            ).mean()

            self.log(
                f"{stage}/{dts_name}/syno_sim",
                cls_sim,
                batch_size=x.shape[0]
            )
            result["loss"] += cls_sim * self.ffg_weight

        return result


if __name__ == "__main__":
    frames = 5
    model = SynoVideoLearner(attn_record=True)
    model.to("cuda")
    logit = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))["logits"]
    logit.sum().backward()
    print("done")
