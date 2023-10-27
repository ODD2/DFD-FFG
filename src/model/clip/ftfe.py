import wandb
import torch
import pickle
import torch.nn as nn

from typing import List
from src.clip.model_vpt import PromptMode, LayerNorm
from src.model.base import ODBinaryMetricClassifier
from src.model.clip import FrameAttrExtractor
from src.clip import clip as CLIP


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = FrameAttrExtractor(
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
        logits = self.proj(results["embeds"].mean(1))
        return dict(
            logits=logits,
            ** results
        )


class TextAffinityHead(nn.Module):
    def __init__(
        self,
        out_dim=2,
        text_prompts=1,
        architecture: str = "ViT-B/16",
        gender_text=["a man", "a woman"]
    ):
        super().__init__()
        model, _ = CLIP.load(
            architecture,
            "cpu",
            ignore_attr=True
        )
        model = model.float().requires_grad_(False)
        self.out_dim = out_dim
        self.text_prompts = text_prompts
        self.ln_final = model.ln_final
        self.n_ctx = model.context_length
        self.logit_scale = model.logit_scale
        self.transformer = model.transformer
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.gender_text = gender_text

        self.gender_embeds = nn.Parameter(
            model.encode_text(CLIP.tokenize(gender_text)),
            requires_grad=False
        )
        # generic token and prompts
        tokens = self.token_embedding(CLIP.tokenize(""))[0]
        self.beg_token = nn.Parameter(
            tokens[0].unsqueeze(0).expand(
                len(self.gender_text) * self.out_dim,
                -1,
                -1
            ).clone(),
            requires_grad=False
        )
        self.end_token = nn.Parameter(
            tokens[1].unsqueeze(0).expand(
                len(self.gender_text) * self.out_dim,
                -1,
                -1
            ).clone(),
            requires_grad=False
        )
        self.cls_text_embed = nn.Parameter(
            torch.zeros(
                len(self.gender_text) * self.out_dim,
                self.text_prompts,
                tokens.shape[1]
            ),
            requires_grad=True
        )
        self.null_token = nn.Parameter(
            tokens[2].unsqueeze(0).expand(
                len(self.gender_text) * self.out_dim,
                self.n_ctx - self.text_prompts - 2,
                -1
            ).clone(),
            requires_grad=False
        )

        # remove visual model to save memory
        model.visual = None

    def create_anchors(self):
        x = torch.cat(
            (
                self.beg_token,
                self.cls_text_embed,
                self.end_token,
                self.null_token
            ),
            dim=1
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[
            :,
            (1 + self.text_prompts)
        ] @ self.text_projection

        return x

    def forward(self, features):
        anchors = self.create_anchors()
        gender_anchors = self.gender_embeds

        # normalized features
        features = features / features.norm(dim=1, keepdim=True)
        anchors = anchors / anchors.norm(dim=1, keepdim=True)
        gender_anchors = gender_anchors / gender_anchors.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        gender_logits = logit_scale * features @ gender_anchors.t()
        gender_affinity = gender_logits.softmax(dim=-1).unsqueeze(1).transpose(1, 2)

        logits = logit_scale * features @ anchors.t()
        logits = logits.view(-1, len(self.gender_text), self.out_dim)
        logits = (logits * gender_affinity).mean(dim=-2)

        return logits


class GenderAffineLinearHead(nn.Module):
    def __init__(
        self,
        out_dim=2,
        text_prompts=1,
        architecture: str = "ViT-B/16",
        gender_text=["a man", "a woman"]
    ):
        super().__init__()
        model, _ = CLIP.load(
            architecture,
            "cpu",
            ignore_attr=True
        )
        model = model.float().requires_grad_(False)
        self.out_dim = out_dim
        self.text_prompts = text_prompts
        self.ln_final = model.ln_final
        self.n_ctx = model.context_length
        self.logit_scale = model.logit_scale
        self.transformer = model.transformer
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.gender_text = gender_text

        self.gender_embeds = nn.Parameter(
            model.encode_text(CLIP.tokenize(gender_text)),
            requires_grad=False
        )
        # generic token and prompts
        width = self.gender_embeds.shape[-1]
        self.proj = nn.Sequential(
            nn.Linear(width, int(width * 0.5)),
            nn.GELU(),
            nn.Linear(int(width * 0.5), width),
        )
        self.head = nn.Linear(width, len(self.gender_text) * self.out_dim)

        # remove visual model to save memory
        model.visual = None

    def forward(self, features, mean_mode="gender"):
        gender_anchors = self.gender_embeds

        # normalized features
        gender_anchors = gender_anchors / gender_anchors.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        gender_logits = logit_scale * features @ gender_anchors.t()
        gender_affinity = gender_logits.softmax(dim=-1).unsqueeze(1).transpose(1, 2)

        logits = self.head(features + self.proj(features))
        logits = logits.view(-1, len(self.gender_text), self.out_dim)
        logits = (logits * gender_affinity).mean(dim=-2)
        return logits


class TextAlignClassifier(nn.Module):
    def __init__(
        self,
        architecture,
        text_prompts,
        gender_text,
        * args,
        **kargs,
    ):
        super().__init__()
        self.encoder = FrameAttrExtractor(
            architecture=architecture,
            text_embed=True,
            ignore_attr=True,
            *args,
            **kargs
        )
        self.proj = TextAffinityHead(
            architecture=architecture,
            text_prompts=text_prompts,
            gender_text=gender_text
        )

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


############################################
class LinearMeanVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        prompt_mode: PromptMode = PromptMode.NONE,
        prompt_num: int = 0,
        prompt_layers: int = 0,
        prompt_dropout: float = 0,
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1]
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


class AMGLMVL(LinearMeanVideoLearner):  # Attention Map Guided LMVL
    def __init__(self, *args, architecture, prompt_layers, attn_record, **kargs):
        super().__init__(
            *args,
            prompt_layers=prompt_layers,
            attn_record=True,
            **kargs
        )
        self.sampler = FrameAttrExtractor(
            architecture=architecture,
            text_embed=False,
            prompt_dropout=0,
            prompt_layers=0,
            prompt_mode=PromptMode.NONE,
            prompt_num=0
        )
        self.sampler.requires_grad_(False)
        self.sampler.eval()

        with open("misc/real_semantic_patches_v4_1000.pickle", "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features['k'][p][l]
                        # for p in ["lips"]
                        # for p in ["lips", "skin"]
                        # for p in ["lips", "eyes"]
                        for p in ["lips", "skin", "nose"]
                    ])
                    for l in range(self.model.encoder.model.transformer.layers)
                ]
            )
            heads = self.sampler.model.transformer.heads
            width = self.sampler.model.transformer.width
            self.face_features = self.face_features.unflatten(-1, (heads, width // heads))

        # self.temperate = nn.Parameter(
        #     torch.tensor(
        #         [
        #             3.0 for _ in range(self.model.encoder.model.transformer.layers)
        #         ]
        #     ),
        #     requires_grad=True
        # )

    def shared_step(self, batch, stage):
        result = super().shared_step(batch, stage)
        if (stage == "train"):
            # extract features and attention attributes
            with torch.no_grad():
                x, y, z = batch["xyz"]
                sample_layer_attrs = self.sampler(x.flatten(0, 1))["layer_attrs"]

            amg_loss = 0
            rob_loss = 0
            layer_count = 0
            for i, blk in enumerate(self.model.encoder.model.transformer.resblocks):
                aff = blk.attn.aff
                _p = blk.prompt_num

                n = aff.shape[0]
                p = aff.shape[1] - 1 - _p
                h = aff.shape[-1]
                c = self.face_features.shape[1]
                facial_map = (
                    torch.einsum(
                        "chd,nphd->ncph",
                        self.face_features[i].to(x.device),  # c,h,d
                        sample_layer_attrs[i]["k"][:, 1:]  # b*t,p+1,h,d
                    ) * 5
                ).softmax(dim=2)

                # if (_p == 0):
                #     continue

                ###############################################################
                # target_map = torch.cat(
                #     [
                #         torch.zeros((n, c, 1, h), device=x.device),
                #         facial_map,
                #         torch.zeros((n, c, _p, h), device=x.device)
                #     ],
                #     dim=2
                # ).repeat(
                #     1,
                #     _p // c,
                #     1,
                #     1
                # )

                # v1: guide the affinity of prompt to patch
                # prompt_map = aff[:, 1 + p:]
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # )

                # v2: guide the affinity of patch to prompt
                # prompt_map = (aff[:, :, 1 + p:] * 100).softmax(dim=1).transpose(1, 2)
                # amg_loss.append += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # )

                # v5
                # prompt_map = aff[:, :, 1 + p:].transpose(1, 2)
                # prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # )

                ###############################################################
                # target_map = facial_map.repeat(1, _p // c, 1, 1)

                # v3
                # prompt_map = (aff[:, 1: 1 + p, 1 + p:]).softmax(dim=1).transpose(1, 2)
                # amg_loss.append(
                #     torch.nn.functional.kl_div(
                #         torch.log(prompt_map),
                #         target_map,
                #         reduction="none"
                #     )
                # )

                # v4
                # prompt_map = aff[:, 1: 1 + p, 1 + p:].transpose(1, 2)
                # prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # )

                # v6
                # prompt_map = aff[:, 1 + p:, 1:1 + p]
                # prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # )
                ###############################################################
                # _gp = int(_p * 0.5 // c * c)
                # target_map = facial_map.repeat(1, _gp // c, 1, 1)
                # # v7
                # prompt_map = aff[:, (1 + p):(1 + p + _gp), 1:(1 + p)]
                # prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # ) * 0.5
                # prompt_map = aff[:, 1: 1 + p, (1 + p):(1 + p + _gp)].transpose(1, 2)
                # prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                # amg_loss += torch.nn.functional.kl_div(
                #     torch.log(prompt_map),
                #     target_map,
                #     reduction="none"
                # ) * 0.5
                ###############################################################

                if (not i == len(self.model.encoder.model.transformer.resblocks) - 1):
                    continue
                target_map = facial_map.mean(1).unsqueeze(1)
                prompt_map = aff[:, [0], 1:(1 + p)]
                prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                amg_loss += torch.nn.functional.kl_div(
                    torch.log(prompt_map),
                    target_map,
                    reduction="none"
                )
                # else:
                #     _gp = int(_p * 0.5 // c * c)
                #     target_map = facial_map.repeat(1, _gp // c, 1, 1)
                #     prompt_map = aff[:, 1: 1 + p, (1 + p):(1 + p + _gp)].transpose(1, 2)
                #     prompt_map = prompt_map / torch.sum(prompt_map, dim=2, keepdim=True)
                #     amg_loss += torch.nn.functional.kl_div(
                #         torch.log(prompt_map),
                #         target_map,
                #         reduction="none"
                #     )

                ###############################################################
                layer_count += 1

            if (layer_count > 0):
                amg_loss = (amg_loss / layer_count).mean()
            else:
                amg_loss = 0

            self.log(
                f"{stage}/{result['dts_name']}/amg_loss",
                amg_loss,
                batch_size=result["logits"].shape[0]
            )

            result["loss"] += 10 * amg_loss

        return result

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.sampler.eval()
        return self


#############################################
class TextMeanVideoLearner(LinearMeanVideoLearner):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        prompt_mode: PromptMode = PromptMode.NONE,
        prompt_num: int = 0,
        prompt_layers: int = 0,
        prompt_dropout: float = 0,
        attn_record: bool = False,
        text_prompts: int = 1,
        gender_text: List[str] = ["a man", "a woman"],
        label_weights: List[float] = [1, 1]
    ):
        super().__init__()
        self.save_hyperparameters()

        params = dict(
            architecture=architecture,
            prompt_mode=prompt_mode,
            prompt_num=prompt_num,
            prompt_layers=prompt_layers,
            prompt_dropout=prompt_dropout,
            attn_record=attn_record,
            text_prompts=text_prompts,
            gender_text=gender_text
        )
        self.model = TextAlignClassifier(**params)

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
    TextAffinityHead().create_anchors()
