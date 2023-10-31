import wandb
import torch
import pickle
import logging
import torch.nn as nn
import src.clip as CLIP
from collections import OrderedDict
from typing import List, Optional, Dict
from src.clip.model import LayerNorm, QuickGELU
from src.model.base import ODClassifier, ODBinaryMetricClassifier
from src.model.clip import FrameAttrExtractor
from src.clip.model_vpt import PromptMode


def attenuate_pretrained_module_grads(module, in_grad, out_grad):
    for p in module.parameters():
        p.grad *= 0.1


class MultiheadAttention(nn.Module):
    """
    The modified multihead attention do not project k and v,
    it accepts k, v exported from CLIP ViT layers,

    q is learnable and it attend to ported k, v but not itself (no self attension)

    frames that do not exist at the end of a video are masked out
    to prevent q attends to the pads

    in addition to the original softmax in the original transformer
    we also apply compositional de-attention to the affinity matrix
    """

    def __init__(
        self,
        num_frames,
        embed_dim,
        n_head,
        reference_module,
        attn_record=False
    ):
        super().__init__()
        self.num_frames = num_frames
        self.attn_record = attn_record

        def smax(q, k, m):
            """
            softmax in the original Transformer
            """
            aff = torch.einsum('nqhc,nkhc->nqkh', q / (q.size(-1) ** 0.5), k)
            aff = aff.masked_fill(~m, float('-inf'))
            n, q, k, h = aff.shape
            affinities = []
            affinities.append(aff.view((n, q, self.num_frames, -1, h)).softmax(dim=-2).view((n, q, k, h)))
            return sum(affinities) / len(affinities)

        # create list of activation drivers.
        self.activations = [smax]

        self.n_act = len(self.activations)
        self.in_proj = nn.Linear(embed_dim, self.n_act * embed_dim, bias=False)
        # TODO: For reproduction, remember to remove both weight loading and bias=True.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj.load_state_dict(reference_module.out_proj.state_dict())
        self.out_proj.requires_grad_(False)

        self.embed_dim = embed_dim
        self.n_head = n_head

        self.aff = None
        self.qs = None

    def forward(self, q, k, v, m):
        # qs is a tuple with elements with shape equal to: b, q, h, d
        qs = (
            self.in_proj(q)
            .view(*q.shape[:2], self.n_head, -1)
            .split(self.embed_dim // self.n_head, -1)
        )
        m = m.unsqueeze(1).unsqueeze(-1)

        aff = 0
        for i in range(self.n_act):
            aff += self.activations[i](qs[i], k, m) / self.n_act

        if self.attn_record:
            logging.debug("recording attention results.")
            self.aff = aff
            self.qs = qs

        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        return (
            self.out_proj(mix.flatten(-2)),
            torch.stack([_q.flatten(-2) for _q in qs], dim=1)
        )


class ResidualAttentionBlock(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L171
    """

    def __init__(
        self,
        n_head: int,
        d_model: int,
        num_frames: int,
        block_index: int,
        num_queries: int,
        layer_indices: List[int],
        reference_layers: List[nn.Module],
    ):
        super().__init__()
        self.in_drop = nn.Dropout()
        self.attn = MultiheadAttention(
            num_frames=num_frames,
            embed_dim=d_model,
            n_head=n_head,
            reference_module=reference_layers[block_index].attn
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ])
        )
        self.ln_2 = LayerNorm(d_model)

        self._apply_reference(block_index, layer_indices, reference_layers)
        self.ln_1.requires_grad_(False)
        self.mlp.requires_grad_(False)
        self.ln_2.requires_grad_(False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        return self.attn(q, k, v, m)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor):
        _x, qs = self.attention(
            self.ln_1(self.in_drop(x)),
            k, v, m
        )
        x = x + _x
        x = x + self.mlp(self.ln_2(x))
        return x, qs

    def _apply_reference(self, block_index, layer_indices, reference_layers):
        def fetch_ln1_params(layer_idx):
            return reference_layers[layer_idx].ln_1.state_dict()

        def fetch_ln2_params(layer_idx):
            return reference_layers[layer_idx].ln_2.state_dict()

        def fetch_mlp_params(layer_idx):
            return reference_layers[layer_idx].mlp.state_dict()

        logging.debug("perform normal reference initialization.")
        current_layer = layer_indices[block_index]
        self.ln_1.load_state_dict(fetch_ln1_params(current_layer))
        self.mlp.load_state_dict(fetch_mlp_params(current_layer))
        self.ln_2.load_state_dict(fetch_ln2_params(current_layer))


class Transformer(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L195
    """

    def __init__(
        self,
        width: int,
        heads: int,
        num_frames: int,
        layer_indices: List[int],
        reference_layers,
        num_queries: int
    ):
        super().__init__()
        self.width = width
        self.resblocks = []
        for block_index in range(len(layer_indices)):
            self.resblocks.append(
                ResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    num_frames=num_frames,
                    block_index=block_index,
                    layer_indices=layer_indices,
                    reference_layers=reference_layers,
                    num_queries=num_queries
                )
            )

        self.resblocks = nn.Sequential(*self.resblocks)

    def forward(self, x: torch.Tensor, layer_attrs, video_mask):
        layer_outs = []
        layer_qs = []
        for _, blk, kv in zip(range(len(self.resblocks)), self.resblocks, layer_attrs):
            x, qs = blk(x, kv['k'], kv['v'], video_mask)
            layer_outs.append(x.unsqueeze(1))
            layer_qs.append(qs.unsqueeze(1))

        return torch.cat(layer_outs, dim=1), torch.cat(layer_qs, dim=1)


class ViTSideNetworkVideoLearner(nn.Module):
    """
    The decoder aggregates the keys and values exported from CLIP ViT layers
    and predict the truthfulness of a video clip

    The positional embeddings are shared across patches in the same spatial location
    """

    def __init__(
        self,
        width,
        heads,
        num_frames,
        num_layers,
        num_queries,
        reference_layers,
        reference_ln_post,
        reference_proj=None
    ):
        super().__init__()
        scale = width ** -0.5
        total_layers = len(reference_layers)
        self.num_queries = num_queries
        self.layer_indices = [i for i in range(total_layers - num_layers, total_layers)]
        self.class_embedding = nn.Parameter(scale * torch.randn(self.num_queries, width))
        self.drop_pre = nn.Dropout()
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames, 1, heads, width // heads))

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width,
            heads,
            num_frames,
            layer_indices=self.layer_indices,
            reference_layers=reference_layers,
            num_queries=num_queries
        )
        self.ln_post = LayerNorm(width)
        self.ln_post.load_state_dict(reference_ln_post.state_dict())

        if (type(reference_proj) == type(None)):
            self.embed_dim = width
            self.proj = nn.Parameter(torch.eye(self.embed_dim))
        else:
            self.proj = nn.Parameter(reference_proj.data)
            self.embed_dim = self.proj.shape[1]

        self.proj.requires_grad_(False)
        self.ln_post.requires_grad_(False)

    def forward(self, layer_attrs, video_mask):
        b, t, q, h, d = layer_attrs[0]['k'].shape

        # discard unwanted layers
        layer_attrs = [layer_attrs[i] for i in self.layer_indices]

        # discard attributes except 'k' & 'v'
        for i in range(len(layer_attrs)):
            for k in list(layer_attrs[i].keys()):
                if not k in ["k", "v"]:
                    layer_attrs[i].pop(k)

        # discard CLS token
        for i in range(len(layer_attrs)):
            for k in layer_attrs[i]:
                layer_attrs[i][k] = layer_attrs[i][k][:, :, 1:]

        # video frame masking
        video_mask = video_mask.repeat_interleave(layer_attrs[0]['k'].size(2), dim=-1)

        # add temporal position embedding
        for i in range(len(layer_attrs)):
            for k in layer_attrs[i]:
                layer_attrs[i][k] = layer_attrs[i][k] + self.positional_embedding

        # flatten
        for i in range(len(layer_attrs)):
            for k in layer_attrs[i]:
                layer_attrs[i][k] = layer_attrs[i][k].flatten(1, 2)

        # create class_embeddings(queries) for the batch
        x = self.ln_pre(
            self.drop_pre(
                self.class_embedding.unsqueeze(0).repeat(b, 1, 1)
            )
        )

        layer_logits, layer_qs = self.transformer(
            x,
            layer_attrs,
            video_mask
        )

        x = layer_logits.mean(dim=2)
        x = self.ln_post(x[:, -1]) @ self.proj

        return {
            "embeds": x,
            "qs": layer_qs
        }


class VideoFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_layers,
        num_frames,
        num_queries,
        architecture,
        prompt_mode,
        prompt_num,
        prompt_layers,
        prompt_dropout,
        text_embed=False,
        attn_record=False,
        pretrain=None,
        frozen=False
    ):
        super().__init__()
        self.encoder = FrameAttrExtractor(
            architecture=architecture,
            prompt_mode=prompt_mode,
            prompt_num=prompt_num,
            prompt_layers=prompt_layers,
            prompt_dropout=prompt_dropout,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            frozen=frozen
        )
        self.decoder = ViTSideNetworkVideoLearner(
            width=self.encoder.model.transformer.width,
            heads=self.encoder.model.transformer.heads,
            num_layers=num_layers,
            num_frames=num_frames,
            num_queries=num_queries,
            reference_layers=self.encoder.model.transformer.resblocks,
            reference_ln_post=self.encoder.model.ln_post,
            reference_proj=(
                self.encoder.model.proj
                if text_embed else
                None
            )
        )

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, masks, *args, **kargs):
        encode_data = self.encoder(x)
        decode_data = self.decoder(encode_data["layer_attrs"], masks)
        video_features = decode_data.pop("embeds")
        frame_features = encode_data["embeds"]
        return dict(
            video_features=video_features,
            frame_features=frame_features,
            **decode_data
        )


class BinaryLinearClassifier(VideoFeatureExtractor):
    def __init__(
        self,
        with_frame=False,
        *args,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        self.with_frame = with_frame
        create_proj_module = (
            lambda x: nn.Sequential(
                LayerNorm(x),
                nn.Dropout(),
                nn.Linear(
                    x,
                    2,
                    bias=False
                )
            )
        )
        if (self.with_frame):
            self.cls_proj_video = create_proj_module(self.decoder.embed_dim)
            self.cls_proj_frame = create_proj_module(self.encoder.embed_dim)
        else:
            self.cls_proj_video = create_proj_module(self.decoder.embed_dim)

    def forward(self, *args, **kargs):
        result = super().forward(*args, **kargs)
        if (self.with_frame):
            result["logits"] = (
                self.cls_proj_video(result["video_features"]) +
                self.cls_proj_frame(result["frame_features"][:, 0])
            ) / 2
        else:
            result["logits"] = (
                self.cls_proj_video(result["video_features"])
            )

        return result


class CLIPBinaryVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_layers: int,
        num_frames: int,
        num_queries: int = 1,
        architecture: str = 'ViT-B/16',
        prompt_mode: PromptMode = PromptMode.NONE,
        prompt_num: int = 0,
        prompt_layers: int = 0,
        prompt_dropout: float = 0,
        with_frame: bool = False,
        text_embed: bool = False,
        label_weights: List[float] = [1, 1],
        pretrain=None,
        frozen=False
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            num_layers=num_layers,
            num_frames=num_frames,
            num_queries=num_queries,
            architecture=architecture,
            prompt_mode=prompt_mode,
            prompt_num=prompt_num,
            prompt_layers=prompt_layers,
            prompt_dropout=prompt_dropout,
            with_frame=with_frame,
            text_embed=text_embed,
            pretrain=pretrain,
            frozen=frozen
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

        if stage == "train":
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


class CLIPBinaryVideoLearnerFFG(CLIPBinaryVideoLearner):
    def __init__(
        self,
        face_parts: List[str],
        face_attn_attr: str,
        face_feature_path: str,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.save_hyperparameters()
        with open(face_feature_path, "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features[face_attn_attr][p][l]
                        for p in face_parts
                    ])
                    for l in self.model.decoder.layer_indices
                ]
            )

    def shared_step(self, batch, stage):
        result = super().shared_step(batch, stage)
        if (stage == "train"):
            dts_name = result["dts_name"]
            # face feature guided loss
            qs = result["output"]["qs"]
            cls_sim = torch.mean(
                (
                    1 - torch.nn.functional.cosine_similarity(
                        qs.flatten(2, 3),
                        self.face_features.repeat((1, qs.shape[2], 1)).to(qs.device),
                        dim=-1
                    ).flatten(1)
                ).mean(dim=1)
            )
            self.log(
                f"{stage}/{dts_name}/cls_sim",
                cls_sim,
                batch_size=result["logits"].shape[0]
            )
            result["loss"] += cls_sim
        return result


if __name__ == "__main__":
    import src.clip as CLIP
