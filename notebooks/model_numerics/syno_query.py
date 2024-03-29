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
                store_attrs=["s_q"]
            )
        )
        model, preprocess = result.model, result.transform
        model.to("cuda")
        model.eval()
        image = Image.open("notebooks/woman.png")
        image = preprocess(image).unsqueeze(0).unsqueeze(0).to("cuda").repeat(1, 10, 1, 1, 1)
        results = model.forward(image)
        layer_attrs = results["layer_attrs"]
        for i, l_attr in enumerate(layer_attrs):
            s_q = l_attr["s_q"].flatten(0, 1)
            print(
                f"{i}:\n",
                1 + torch.nn.functional.cosine_similarity(s_q.unsqueeze(1), s_q.unsqueeze(0), dim=-1)
            )

# %%
