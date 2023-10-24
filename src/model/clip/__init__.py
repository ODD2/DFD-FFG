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
        pretrain=None
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
            self.model.load_state_dict(pretrain)

        self.model.requires_grad_(False)
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
            attrs = blk.attn.pop_attr()
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