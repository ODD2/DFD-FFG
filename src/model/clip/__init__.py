import torch
import logging
import torch.nn as nn
import src.clip.clip as CLIP


class FrameAttrExtractor(nn.Module):
    def __init__(
        self,
        architecture,
        prompt_mode,
        prompt_num,
        prompt_layers,
        prompt_dropout,
        text_embed,
        ignore_attr=False,
        attn_record=False,
        pretrain=None,
        frozen=False
    ):
        super().__init__()
        self.model, self.transform = CLIP.load(
            architecture,
            "cpu",
            prompt_mode=prompt_mode,
            prompt_num=prompt_num,
            prompt_layers=prompt_layers,
            prompt_dropout=prompt_dropout,
            ignore_attr=ignore_attr,
            attn_record=attn_record
        )

        self.model = self.model.visual.float()

        if not text_embed:
            self.model.proj = None
            self.feat_dim = self.model.transformer.width
        else:
            self.feat_dim = self.model.output_dim

        if (pretrain):
            logging.info("Loading image encoder pretrain weights...")
            state_dict = torch.load(pretrain, "cpu")
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                conflicts = self.model.load_state_dict(state_dict, strict=False)
                logging.warning(
                    f"during visual pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
                )

        self.model.requires_grad_(False)

        if not frozen:
            for param in self.model.prompt_parameters():
                param.requires_grad_(True)

    @property
    def n_px(self):
        return self.model.input_resolution

    @property
    def n_patch(self):
        return self.model.input_resolution // self.model.patch_size

    @property
    def embed_dim(self):
        return self.feat_dim

    def forward(self, x):
        # pass throught for attributes
        if (len(x.shape) > 4):
            b, t = x.shape[:2]
            embeds = self.model(x.flatten(0, 1)).unflatten(0, (b, t))
        else:
            embeds = self.model(x)
        # retrieve all layer attributes
        layer_attrs = []
        for blk in self.model.transformer.resblocks:
            attrs = blk.pop_attr()
            # restore temporal dimension
            for attr_name in attrs:
                if (len(x.shape) > 4):
                    attrs[attr_name] = attrs[attr_name].unflatten(0, (b, t))
                else:
                    attrs[attr_name] = attrs[attr_name]
            layer_attrs.append(attrs)
        return dict(
            layer_attrs=layer_attrs,
            embeds=embeds
        )

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
            for m in self.model.prompt_dropout_modules():
                m.train()
        return self


if __name__ == "__main__":
    from src.clip.model_vpt import PromptMode
    FrameAttrExtractor(
        "ViT-B/16",
        prompt_mode=PromptMode.DEEP,
        prompt_num=10,
        prompt_layers=12,
        prompt_dropout=0.2,
        text_embed=False,
        pretrain="logs/DFD-FFG/71hfy89x/checkpoints/epoch=38-step=8307_encoder.pth"
    )
