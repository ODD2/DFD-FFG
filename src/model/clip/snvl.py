import torch
import pickle
import logging
import torch.nn as nn
import src.clip as CLIP


from collections import OrderedDict
from typing import List, Optional, Dict
from src.clip.model import LayerNorm, QuickGELU
from src.model.base import ODClassifier, ODBinaryMetricClassifier


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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.embed_dim = embed_dim
        self.n_head = n_head

        self.aff = None
        self.qs = None

    def forward(self, q, k, v, m):
        # qs is a tuple with elements with shape equal to: b, q, h, d
        qs = self.in_proj(q).view(*q.shape[:2], self.n_head, -1).split(self.embed_dim // self.n_head, -1)
        m = m.unsqueeze(1).unsqueeze(-1)

        aff = 0
        for i in range(self.n_act):
            aff += self.activations[i](qs[i], k, m) / self.n_act

        if self.attn_record:
            logging.debug("recording attention results.")
            self.aff = aff
            self.qs = qs

        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        return self.out_proj(mix.flatten(-2)), torch.stack([_q.flatten(-2) for _q in qs], dim=1)


class ResidualAttentionBlock(nn.Module):
    """
    Modified from:
    https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L171
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_frames,
        block_index,
        layer_indices,
        reference_layers,
        dropout: float = 0.5
    ):
        super().__init__()

        self.attn = MultiheadAttention(num_frames, d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("dropout", nn.Dropout(dropout)),
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
        _x, qs = self.attention(self.ln_1(x), k, v, m)
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

    def __init__(self, width: int, heads: int, num_frames: int, layer_indices: List[int], reference_layers):
        super().__init__()
        self.width = width
        self.resblocks = []
        for block_index in range(len(layer_indices)):
            self.resblocks.append(
                ResidualAttentionBlock(
                    width, heads, num_frames,
                    block_index, layer_indices, reference_layers
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
        num_queries,
        num_frames,
        layer_indices,
        reference_layers,
        reference_proj=None,
        dropout=0.5
    ):
        super().__init__()
        scale = width ** -0.5
        self.num_queries = num_queries
        self.layer_indices = layer_indices
        self.class_embedding = nn.Parameter(scale * torch.randn(self.num_queries, width))

        self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames, 1, heads, width // heads))

        self.ln_pre = LayerNorm(width)
        self.drop_pre = torch.nn.Dropout(dropout)
        self.transformer = Transformer(
            width,
            heads,
            num_frames,
            layer_indices=layer_indices,
            reference_layers=reference_layers
        )
        if (type(reference_proj) == type(None)):
            self.proj = nn.Parameter(torch.eye(self.embed_dim))
            self.embed_dim = width
        else:
            self.proj = nn.Parameter(reference_proj.data)
            self.embed_dim = self.proj.shape[1]
        self.proj.requires_grad_(False)

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
        x = self.drop_pre(self.ln_pre(
            self.class_embedding.unsqueeze(0).repeat(b, 1, 1)
        ))

        layer_logits, layer_qs = self.transformer(
            x,
            layer_attrs,
            video_mask
        )

        x = layer_logits.mean(dim=2)

        return {
            "embeds": x[:, -1] @ self.proj,
            "qs": layer_qs
        }


class VideoAttrExtractor(nn.Module):
    def __init__(self, architecture: str = "ViT-B/16"):
        super().__init__()
        self.model, self.transform = CLIP.load(architecture, "cpu")
        self.model = self.model.visual.float()
        self.model.requires_grad_(False)

    @property
    def n_px(self):
        return self.model.input_resolution

    @property
    def n_patch(self):
        return self.model.input_resolution // self.model.patch_size

    def forward(self, x):
        # pass throught for attributes
        if (len(x.shape) > 4):
            b, t = x.shape[:2]
            self.model(x.flatten(0, 1))
        else:
            self.model(x)
        # retrieve all layer attributes
        layer_attrs = []
        for blk in self.model.transformer.resblocks:
            attrs = blk.attn.pop_attr()
            # restore temporal dimension
            for attr_name in attrs:
                if (len(x.shape) > 4):
                    attrs[attr_name] = attrs[attr_name].unflatten(0, (b, t))
                else:
                    attrs[attr_name] = attrs[attr_name]
            layer_attrs.append(attrs)
        return layer_attrs

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
        return self


class TextAffinityHead(nn.Module):
    def __init__(self, out_dim=2, architecture: str = "ViT-B/16"):
        super().__init__()
        model, _ = CLIP.load(architecture, "cpu")
        model.visual = None
        model = model.float().requires_grad_(False)
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # Inversion
        tokens = self.token_embedding(CLIP.tokenize(""))[0]
        self.pre_drop = nn.Dropout()
        self.beg_token = nn.Parameter(
            tokens[0].unsqueeze(0),
            requires_grad=False
        )
        self.end_token = nn.Parameter(
            tokens[1].unsqueeze(0),
            requires_grad=False
        )
        self.cls_text_embed = nn.Parameter(
            torch.randn(
                out_dim,
                tokens.shape[0]-2,
                tokens.shape[1]
            ),
            requires_grad=True
        )

    def create_anchors(self):
        x = torch.cat(
            (
                self.beg_token.expand((self.cls_text_embed.shape[0], -1, -1)),
                self.pre_drop(self.cls_text_embed),
                self.end_token.expand((self.cls_text_embed.shape[0], -1, -1))
            ),
            dim=1
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 76] @ self.text_projection

        return x

    def forward(self, features):
        anchors = self.create_anchors()

        # normalized features
        features = features / features.norm(dim=1, keepdim=True)
        anchors = anchors / anchors.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * features @ anchors.t()

        return logits


class VideoFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_frames=20,
        num_queries=1,
        architecture="ViT-B/16",
        layer_indices=[6, 7, 8, 9, 10, 11],
        dropout=0.5
    ):
        super().__init__()
        self.encoder = VideoAttrExtractor(architecture)
        self.layer_indices = layer_indices
        self.decoder = ViTSideNetworkVideoLearner(
            width=self.encoder.model.transformer.width,
            heads=self.encoder.model.transformer.heads,
            num_frames=num_frames,
            num_queries=num_queries,
            layer_indices=layer_indices,
            reference_layers=self.encoder.model.transformer.resblocks,
            reference_proj=self.encoder.model.proj,
            dropout=dropout
        )

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, masks, *args, **kargs):
        return self.decoder(self.encoder(x), masks)


class BinaryLinearClassifier(VideoFeatureExtractor):
    def __init__(
        self,
        *args,
        dropout=0.5,
        **kargs,
    ):
        super().__init__(dropout=dropout, *args, **kargs)

        self.cls_proj = nn.Sequential(
            LayerNorm(self.decoder.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.decoder.embed_dim, 2, bias=False)
        )

    def forward(self, *args, **kargs):
        result = super().forward(*args, **kargs)
        result["logits"] = self.cls_proj(result["embeds"])
        return result


class BinaryTextAffinityClassifier(VideoFeatureExtractor):
    def __init__(
        self,
        *args,
        architecture="ViT-B/16",
        **kargs
    ):
        super().__init__(*args, architecture=architecture, **kargs)
        self.cls_head = TextAffinityHead(
            out_dim=2,
            architecture=architecture
        )

    def forward(self,  *args, **kargs):
        result = super().forward(*args, **kargs)
        result["logits"] = self.cls_head(result["embeds"])
        return result


class CLIPBinaryVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_frames: int,
        num_queries: int = 1,
        architecture='ViT-B/16',
        layer_indices: List[int] = [],
        is_text_affine: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            num_frames=num_frames,
            num_queries=num_queries,
            architecture=architecture,
            layer_indices=layer_indices
        )
        if (not is_text_affine):
            self.model = BinaryLinearClassifier(**params)
        else:
            self.model = BinaryTextAffinityClassifier(**params)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.n_px

    def shared_step(self, batch, stage):
        x, y, mask, indices, dts_name = *batch[:3], *batch[-2:]
        output = self.model(x, mask)
        logits = output["logits"]
        # classification loss
        cls_loss = nn.functional.cross_entropy(logits, y)
        self.log(
            f"{stage}/{dts_name}/loss",
            cls_loss,
            batch_size=logits.shape[0]
        )

        return {
            "logits": logits,
            "labels": y,
            "loss": cls_loss,
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
    m = TextAffinityHead()
    m.forward(torch.randn(10, 512))
