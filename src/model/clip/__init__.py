import torch
import random
import logging
import open_clip
import torch.nn as nn
import src.clip.clip as CLIP


def load_model(architecture, **kargs):
    # architecture parameter split
    # e.g: ViT-B/16|laion2b_s34b_b79k
    params = architecture.split('|')

    clip_arch = params[0]
    model, transform = CLIP.load(
        clip_arch,
        "cpu",
        **kargs
    )

    if (len(params) == 2):
        open_pretrain = params[1]

        _model, _, _ = open_clip.create_model_and_transforms(
            clip_arch.replace("/", "-"),
            pretrained=open_pretrain,
            device="cpu"
        )
        try:
            model.load_state_dict(
                _model.state_dict(),
                strict=True)
        except:
            conflicts = model.load_state_dict(
                _model.state_dict(),
                strict=False
            )
            logging.warning(
                f"OpenCLIP pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
            )
        del _model

    elif (len(params) > 2):
        raise NotImplementedError()

    return model, transform


class VideoAttrExtractor(nn.Module):
    def __init__(
        self,
        architecture,
        text_embed,
        num_frames=1,
        num_synos=1,
        store_attrs=[],
        attn_record=False,
        pretrain=None,
        frozen=False,
        mask_ratio=0.0,
        is_temporal_conv=True,
        is_syno_adaptor=True,
        is_temporal_embedding=True,
        enable_syno=True,
        layer_ratio=1.0
    ):
        super().__init__()
        self.model, self.transform = load_model(
            architecture,
            num_frames=num_frames,
            store_attrs=store_attrs,
            attn_record=attn_record,
            num_synos=num_synos,
            is_syno_adaptor=is_syno_adaptor,
            is_temporal_conv=is_temporal_conv,
            is_temporal_embedding=is_temporal_embedding,
            enable_syno=enable_syno
        )
        self.mask_ratio = mask_ratio
        self.model = self.model.visual.float()
        self.model.post_init_tuneables()

        if (pretrain):
            logging.info("Loading image encoder pretrain weights...")
            state_dict = torch.load(pretrain, "cpu")
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                conflicts = self.model.load_state_dict(
                    state_dict, strict=False)
                logging.warning(
                    f"during visual pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
                )

        self.model.requires_grad_(False)

        if (layer_ratio < 1.0):
            truncate_layers = int(self.model.transformer.layers * layer_ratio)
            self.model.transformer.layers = truncate_layers
            self.model.transformer.resblocks = self.model.transformer.resblocks[:truncate_layers]
            self.model.ln_post.requires_grad_(True)
            self.model.proj.requires_grad_(True)

        if not text_embed:
            self.model.proj = None
            self.feat_dim = self.model.transformer.width
        else:
            self.feat_dim = self.model.output_dim

        if not frozen:
            for module in self.model.tuneable_modules():
                module.requires_grad_(True)

    @property
    def n_px(self):
        return self.model.input_resolution

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def patch_num(self):
        return self.model.patch_num

    @property
    def n_patch(self):
        return int(self.n_px // self.patch_size)

    @property
    def embed_dim(self):
        return self.feat_dim

    def forward(self, x):
        b, t = x.shape[:2]

        # training augmentations
        if self.training and self.mask_ratio > 0:
            patch_mask = torch.rand(self.model.patch_num) < self.mask_ratio
            patch_mask = patch_mask.to(dtype=float, device=x.device) * -1e3
            patch_mask = patch_mask.unsqueeze(1)
        else:
            patch_mask = torch.tensor(0, device=x.device)

        # pass throught for attributes
        embeds, synos = self.model(x, syno=True, patch_mask=patch_mask)
        # retrieve all layer attributes
        layer_attrs = []
        for blk in self.model.transformer.resblocks:
            attrs = blk.pop_attr()
            layer_attrs.append(attrs)
        return dict(
            layer_attrs=layer_attrs,
            embeds=embeds,
            synos=synos
        )

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
            for module in self.model.tuneable_modules():
                if (issubclass(type(module), torch.nn.Module)):
                    module.train()
        return self


if __name__ == "__main__":
    VideoAttrExtractor(
        "ViT-B/16",
        text_embed=False,
        layer_ratio=0.5
    )
