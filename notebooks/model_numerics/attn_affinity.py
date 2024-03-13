# %%
import gc
import cv2
import time
import math
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from notebooks.tools import load_model
from src.model.clip.svl import SynoVideoLearner, FFGSynoVideoLearner

if __name__ == "__main__":
    with torch.no_grad():
        result = load_model(
            FFGSynoVideoLearner,
            "logs/ECCV/0v90yk5x/checkpoints/epoch=11-step=816.ckpt",
            extra_params=dict(
                store_attrs=["s_q"],
                attn_store=True
            )
        )
        model, preprocess = result.model, result.transform
        model.to("cuda")
        model.eval()
        image = Image.open("notebooks/woman.png")
        image = preprocess(image).unsqueeze(0).unsqueeze(0).to("cuda").repeat(1, 10, 1, 1, 1)
        results = model.forward(image)
        n_patch = model.encoder.n_patch
        attn_layers = model.encoder.decoder.decoder_layers
        for i, blk in enumerate(attn_layers):
            aff = blk.aff
            print(aff.shape)
            aff = aff.flatten(0, 2).mean(0).unflatten(-1, (n_patch, n_patch))
            print(
                f"{i}:\n",
                aff
            )

# %%
