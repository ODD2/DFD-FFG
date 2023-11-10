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
    result = load_model(
        FFGSynoVideoLearner,
        "logs/DFD-FFG(Experiment)/44tjfyzc/checkpoints/last.ckpt",
        extra_params=dict(
            store_attrs=["s_q", "q"]
        )
    )
    model, preprocess = result.model, result.transform
    model.to("cuda:1")
    model.eval()
    image1 = Image.open("notebooks/woman.png")
    image1 = preprocess(image1).unsqueeze(0).unsqueeze(0).to("cuda:1")
    image2 = Image.open("notebooks/man.png")
    image2 = preprocess(image2).unsqueeze(0).unsqueeze(0).to("cuda:1")
    results = model.forward(torch.cat([image1, image2]))
    layer_attrs = results["layer_attrs"]
    for i, l_attr in enumerate(layer_attrs):
        s_q = l_attr["s_q"].flatten(0, 1).flatten(-2)
        q = l_attr["q"].flatten(0, 2).flatten(-2)[[0]]
        print(
            f"{i}:\n",
            1 + torch.nn.functional.cosine_similarity(s_q, q, dim=-1)
        )

# %%
