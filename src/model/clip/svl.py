import wandb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

from typing import List


from src.utility.loss import focal_loss
from src.model.clip import VideoAttrExtractor
from src.model.base import ODBinaryMetricClassifier
from src.clip.model_syno import MultiheadAttentionAttrExtract


def call_module(module):
    def fn(*args, **kwargs):
        return module(*args, **kwargs)
    return fn


def get_module(module):
    def fn():
        return module
    return fn


class GlitchBlock(nn.Module):
    def __init__(
        self,
        n_head,
        n_patch,
        n_filt,
        n_frames,
        ksize,
    ):
        super().__init__()

        self.n_head = n_head
        self.ksize = ksize
        self.n_patch = n_patch


class SynoDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        num_frames,
        num_filters,
        kernel_size,
    ):
        super().__init__()
        self.encoder = get_module(encoder)
        h = encoder.transformer.heads
        L = encoder.transformer.layers
        self.t_conv = self.make_2dconv(
            kernel_size,
            h * L,
            num_filters
        )
        self.p_conv = self.make_2dconv(
            kernel_size,
            (num_frames ** 2) * num_filters,
            1
        )

    def make_2dconv(self, ksize, in_c, out_c, groups=1):
        conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            groups=groups,
            bias=True
        )

        nn.init.normal_(conv.weight, std=0.001)
        nn.init.zeros_(conv.bias)

        return conv

    def attn_map(self, attrs):
        # shape = (b,t,l,h,d)
        _q = attrs['q'][:, :, 1:]
        _k = attrs['k'][:, :, 1:]  # ignore cls token

        _q = _q.permute(0, 2, 1, 3, 4)
        _k = _k.permute(0, 2, 1, 3, 4)

        aff = torch.einsum(
            'nlqhc,nlkhc->nlqkh',
            _q / (_q.size(-1) ** 0.5),
            _k
        )

        aff = aff.softmax(dim=-2)  # shape = (n,l,t,t,h)
        aff = aff.permute(0, 1, 4, 2, 3)  # shape = (n,l,h,t,t)

        return aff

    def forward(self, x):
        # first, we prepare the encoder before the transformer layers.
        x = self.encoder()._prepare(x)
        out = []
        # now, we alternate between synoptic and encoder layers
        for enc_blk in self.encoder().transformer.resblocks:
            data = enc_blk(x)
            x = data["emb"]
            out.append(self.attn_map(data))
        # last, we are done with the encoder, therefore skipping the _finalize step.
        # x =  self.encoder()._finalize(x)

        out = torch.cat(out, dim=2)  # shape = (b,l,h*L,t,t)
        b, p = out.shape[0], int(out.shape[1]**0.5)
        out = out.flatten(0, 1)  # shape = (b*l,h*L,t,t)

        out = self.t_conv(out)  # shape = (b*l,r,t,t)
        out = out.unflatten(0, (b, p, p)).flatten(3)  # shape = (n, p, p, r*t*t)
        out = out.permute(0, 3, 1, 2)  # shape = (n, r*t*t, p, p)
        out = self.p_conv(out)  # shape = (n, 1, p, p)

        return out.flatten(1)  # shape = (n, p*p)


class SynoVideoAttrExtractor(VideoAttrExtractor):
    def __init__(
        self,
        # VideoAttrExtractor
        architecture,
        text_embed,
        pretrain=None,
        store_attrs=[],
        attn_record=False,
        # synoptic
        num_frames=1,
        num_filters=10,
        kernel_size=3
    ):
        super(SynoVideoAttrExtractor, self).__init__(
            architecture=architecture,
            text_embed=text_embed,
            store_attrs=store_attrs,
            attn_record=attn_record,
            pretrain=pretrain
        )
        self.decoder = SynoDecoder(
            encoder=self.model,
            num_frames=num_frames,
            num_filters=num_filters,
            kernel_size=kernel_size
        )

        self.feat_dim = self.model.patch_num

    def forward(self, x):
        synos = self.decoder(x=x)

        layer_attrs = []
        for enc_blk in self.model.transformer.resblocks:
            layer_attrs.append(
                {
                    **enc_blk.pop_attr()
                }
            )

        return dict(
            layer_attrs=layer_attrs,
            synos=synos
        )

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
            self.decoder.train()
        return self


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = SynoVideoAttrExtractor(
            *args,
            **kargs
        )

        self.head = self.make_linear(self.encoder.embed_dim)

    def make_linear(self, embed_dim):
        linear = nn.Linear(
            embed_dim,
            2
        )
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            linear
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
        logits = self.head(synos)
        return dict(
            logits=logits,
            ** results
        )


class SynoVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_frames: int = 1,
        num_filters: int = 10,
        kernel_size: int = 3,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        pretrain: str = None,

        attn_record: bool = False,

        store_attrs: List[str] = [],
        is_focal_loss: bool = True,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            num_frames=num_frames,
            num_filters=num_filters,
            kernel_size=kernel_size,
            store_attrs=store_attrs,
        )
        self.model = BinaryLinearClassifier(**params)

        self.label_weights = torch.tensor(label_weights)
        self.cls_weight = cls_weight
        self.is_focal_loss = is_focal_loss

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
        loss = 0
        # classification loss
        if (stage == "train"):
            if self.is_focal_loss:
                cls_loss = focal_loss(
                    logits,
                    y,
                    gamma=4,
                    weight=(
                        self.label_weights.to(y.device)
                        if stage == "train" else
                        None
                    )
                )
            else:
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
            loss += cls_loss.mean() * self.cls_weight
            self.log(
                f"{stage}/{dts_name}/loss",
                cls_loss.mean(),
                batch_size=logits.shape[0]
            )
        else:
            # classification loss
            cls_loss = nn.functional.cross_entropy(
                logits,
                y,
                reduction="none",
                weight=None
            )
            loss += cls_loss.mean()

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
        # ffg
        face_feature_path: str,
        face_parts: List[str] = [
            "lips",
            "skin",
            "eyes",
            "nose"
        ],
        face_attn_attr: str = "k",
        syno_attn_attr: str = "s_q",
        ffg_temper: float = 30,
        ffg_weight: float = 1.5,
        ffg_layers: int = -1,
        ffg_reverse: bool = False,

        # generic
        num_frames: int = 1,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        pretrain: str = None,

        attn_record: bool = False,

        store_attrs: List[str] = [],
        is_focal_loss: bool = True,
        is_syno_adaptor: bool = True,
        is_temporal_conv: bool = True,
        is_temporal_embedding: bool = True,
        mask_ratio: float = 0.3,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        self.num_face_parts = len(face_parts)
        self.face_attn_attr = face_attn_attr
        self.syno_attn_attr = syno_attn_attr
        self.ffg_temper = ffg_temper
        self.ffg_weight = ffg_weight
        self.ffg_layers = ffg_layers
        self.ffg_reverse = ffg_reverse

        super().__init__(
            num_frames=num_frames,
            num_synos=self.num_face_parts,
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            store_attrs=set([*store_attrs, self.syno_attn_attr]),
            mask_ratio=mask_ratio,
            cls_weight=cls_weight,
            label_weights=label_weights,
            is_focal_loss=is_focal_loss,
            is_syno_adaptor=is_syno_adaptor,
            is_temporal_conv=is_temporal_conv,
            is_temporal_embedding=is_temporal_embedding
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
                if (self.ffg_reverse):
                    face_features = face_features[: self.ffg_layers]
                    target_attn_attrs = target_attn_attrs[: self.ffg_layers]
                else:
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
    # # AttrExtractor Test
    model = SynoVideoAttrExtractor(
        "ViT-B/16",
        text_embed=True,
        store_attrs=["s_q"],
        num_frames=frames
    )
    model.to("cuda")
    results = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))
    synos = results["synos"]
    synos.sum().backward()

    # model = GlitchBlock(n_head=12, n_patch=14, n_filt=10, ksize=3, n_frames=frames)
    # model.to("cuda")
    # logits = model({
    #     "q": torch.randn(1, frames, 197, 12, 64).to("cuda"),
    #     "k": torch.randn(1, frames, 197, 12, 64).to("cuda")
    # })
    print("done")
