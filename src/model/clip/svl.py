import wandb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from src.model.base import ODBinaryMetricClassifier
from src.model.clip import VideoAttrExtractor
from src.utility.loss import focal_loss


def call_module(module):
    def fn(*args, **kwargs):
        return module(*args, **kwargs)
    return fn


def get_module(module):
    def fn():
        return module
    return fn


def make_adaptor(d_model):
    ln = nn.LayerNorm(d_model)
    linear1 = nn.Linear(
        d_model,
        d_model // 8,
        bias=False
    )
    ln = nn.LayerNorm(d_model)
    linear2 = nn.Linear(
        d_model // 8,
        d_model,
        bias=False
    )

    nn.init.normal_(linear1.weight, std=0.001)
    nn.init.normal_(linear2.weight, std=0.001)
    return [ln, linear1, linear2]


# class of per-layer decoder block
class SynoBlock(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        n_frame,
        ksize=3,
        n_filter=1,
        store_attrs=[],

    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.s_proj = nn.Linear(d_model, d_model, bias=True)

        # preserve attrs
        self.store_attrs = store_attrs
        self.attr = {}

    def make_tconv(self, d_model, n_filter, ksize):
        conv_1d = nn.Conv1d(
            d_model,
            n_filter,
            kernel_size=ksize,
            stride=1,
            groups=1,
            bias=True
        )

        nn.init.normal_(conv_1d.weight, std=0.001)
        nn.init.zeros_(conv_1d.bias)

        return conv_1d

    def pop_attr(self):
        ret = self.get_attr()
        self.attr.clear()
        return ret

    def get_attr(self):
        return {k: self.attr[k] for k in self.attr}

    def set_attr(self, **attr):
        self.attr = {
            k: attr[k]
            for k in attr
            if k in self.store_attrs
        }

    def attention(self, attrs, s):
        _k = attrs['k'][:, :, 1:]  # ignore cls token
        _v = attrs['emb'][:, :, 1:]  # ignore cls token

        b, t, l, h, d = _k.shape

        #  Synoptic Cross-Attention
        s_q = self.s_proj(s)

        _k = _k.flatten(-2)

        s_aff = torch.einsum(
            'nqw,ntkw->ntqk',
            s_q / (s_q.size(-1) ** 0.5),
            _k
        )

        s_aff = s_aff.softmax(dim=-1)

        s_mix = torch.einsum('ntql,ntlw->ntqw', s_aff, _v)

        s_mix = s_mix.mean(dim=1)  # shape = (b,s,w)
        return dict(s_q=s_q, s_mix=s_mix)

    def forward(self, attrs, s):
        self.pop_attr()

        data = self.attention(attrs, s)

        s = data["s_mix"]

        self.set_attr(
            **data
        )

        return s


class SynoDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        num_synos,
        num_frames,
        kernel_size,
        num_filters,
        store_attrs=[]
    ):
        super().__init__()
        self.encoder = get_module(encoder)
        self.patch_num = encoder.patch_num

        #
        self.decoder_layers = nn.ModuleList([
            SynoBlock(
                ksize=kernel_size,
                n_filter=num_filters,
                n_frame=num_frames,
                n_head=encoder.transformer.heads,
                d_model=encoder.transformer.width,
                store_attrs=store_attrs
            )
            for i in range(encoder.transformer.layers)
        ])

        self.syno_embedding = nn.Parameter(
            torch.zeros(num_synos, encoder.transformer.width)
        )

    def forward(self, x):
        # first, we prepare the encoder before the transformer layers.
        b = x.shape[0]
        x = self.encoder()._prepare(x)
        s = self.syno_embedding.unsqueeze(0).repeat(b, 1, 1)  # shape = (b, synos, width)
        layer_output = []
        # now, we alternate between synoptic and encoder layers
        for enc_blk, dec_blk in zip(self.encoder().transformer.resblocks, self.decoder_layers):
            data = enc_blk(x)
            x = data["emb"]
            layer_output.append(dec_blk(data, s))
        # last, we are done with the encoder, therefore skipping the _finalize step.
        # x =  self.encoder()._finalize(x)
        s = torch.stack(layer_output, 1).sum(1)
        return s


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
        num_synos=1,
        num_frames=1,
        num_filters=1,
        kernel_size=3,
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
            num_synos=num_synos,
            num_frames=num_frames,
            num_filters=num_filters,
            kernel_size=kernel_size,
            store_attrs=store_attrs,
        )

    def forward(self, x):
        synos = self.decoder(x=x)

        layer_attrs = []
        for enc_blk, dec_blk in zip(self.model.transformer.resblocks, self.decoder.decoder_layers):
            layer_attrs.append(
                {
                    **enc_blk.pop_attr(),
                    **dec_blk.pop_attr()
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
        logits = self.head(synos.mean(1))
        return dict(
            logits=logits,
            ** results
        )


class SynoVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_synos: int = 1,
        num_frames: int = 1,
        kernel_size: int = 3,
        num_filters: int = 1,
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
            num_synos=num_synos,
            kernel_size=kernel_size,
            num_filters=num_filters,
            store_attrs=store_attrs
        )
        self.model = BinaryLinearClassifier(**params)

        self.num_synos = num_synos
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
            cls_weight=cls_weight,
            label_weights=label_weights,
            is_focal_loss=is_focal_loss
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
    # model = SynoVideoAttrExtractor(
    #     "ViT-B/16",
    #     text_embed=True,
    #     store_attrs=["s_q"],
    #     mask_ratio=0.3,
    #     is_temporal_conv=False,
    #     is_syno_adaptor=False,
    #     is_temporal_embedding=False,
    #     num_frames=frames,
    #     num_synos=10
    # )
    # model.to("cuda")
    # results = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))
    # synos = results["synos"]
    # synos.sum().backward()

    model = SynoVideoLearner(attn_record=True)
    model.to("cuda")
    logit = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))["logits"]
    logit.sum().backward()

    print("done")
