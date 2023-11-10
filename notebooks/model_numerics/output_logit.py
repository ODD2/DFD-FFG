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
        "logs/DFD-FFG(Experiment)/8fahh79n/checkpoints/last.ckpt",
        extra_params=dict(
            store_attrs=["s_q"]
        )
    )
    model, preprocess = result.model, result.transform
    model.to("cuda")
    model.eval()
    image = Image.open("notebooks/woman.png")
    image = preprocess(image).unsqueeze(0).unsqueeze(0).to("cuda")
    results = model.forward(image)
    print(results)

# %%
