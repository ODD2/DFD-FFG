import os
import sys
import torch


@torch.inference_mode()
def extract_features(encoder, frame):
    assert len(frame.shape) == 4
    # get attributes from each CLIP ViT layer
    kvs = encoder(frame.unsqueeze(1).to("cuda"))["layer_attrs"]
    # discard original CLS token and restore temporal dimension
    for i in range(len(kvs)):
        for k in ["q", "k", "v", "out", "emb"]:
            kvs[i][k] = kvs[i][k][:, :, 1:].to("cpu")
            if (len(kvs[i][k].shape) == 5):
                kvs[i][k] = kvs[i][k].flatten(-2)
            elif (len(kvs[i][k].shape) == 4):
                pass
            else:
                raise NotImplementedError()

    torch.cuda.empty_cache()
    return kvs


def load_model(model_cls, ckpt_path="", extra_params={}):
    if (ckpt_path == ""):
        model = model_cls(
            attn_record=True,
            **extra_params
        )
    else:
        model = model_cls.load_from_checkpoint(
            ckpt_path,
            strict=False,
            attn_record=True,
            map_location="cpu",
            ** extra_params
        )
    return model
