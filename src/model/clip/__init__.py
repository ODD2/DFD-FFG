import torch
import random
import logging
import torch.nn as nn
import src.clip.clip as CLIP


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
        mask_ratio=0.0
    ):
        super().__init__()
        self.model, self.transform = CLIP.load(
            architecture,
            "cpu",
            num_frames=num_frames,
            store_attrs=store_attrs,
            attn_record=attn_record,
            num_synos=num_synos
        )
        self.mask_ratio = mask_ratio
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
                conflicts = self.model.load_state_dict(
                    state_dict, strict=False)
                logging.warning(
                    f"during visual pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
                )

        self.model.requires_grad_(False)

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

        if self.training and self.mask_ratio > 0:
            mpi = torch.rand(self.model.patch_num) < self.mask_ratio
            mpi = mpi.to(dtype=float, device=x.device) * -1e4
            mpi = mpi.unsqueeze(1)
        else:
            mpi = torch.tensor(0, device=x.device)

        # pass throught for attributes
        embeds, synos = self.model(
            x,
            syno=True,
            mpi=mpi
        )
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
        pretrain="logs/DFD-FFG/71hfy89x/checkpoints/epoch=38-step=8307_encoder.pth"
    )
