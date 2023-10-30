# %%
import os
import sys
import notebooks.tools as tools
import cv2
import torch
import pickle
import random
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import torchvision.transforms as T

from tqdm import tqdm
from notebooks.tools import extract_features
from src.model.clip import FrameAttrExtractor
from src.clip.model_vpt import PromptMode
from src.dataset.ffpp import FFPP, FFPPSampleStrategy, FFPPAugmentation
from notebooks.fetch_facial_feature import fetch_semantic_features


def var(x, patch_num, *args, **kargs):
    # shows the variance of the video patches over the temporal dimension.
    return (
        torch.var(x, dim=0)
        .mean(dim=-1)
        .view((patch_num, patch_num))
        .unsqueeze(-1)
    )


def max_stdev(x, patch_num, *args, **kargs):
    # shows the maximum stdev value of the video patches averaged over the temporal dimension.
    return (
        torch.max(
            torch.abs((x - torch.mean(x, dim=0)) / torch.sqrt(torch.var(x, dim=0))).mean(dim=-1),
            dim=0
        )[0]
        .view(patch_num, patch_num)
        .unsqueeze(-1)
    )


def one_patch_cos_sim(x, t, c, patch_num, *args, **kargs):
    # shows the video patch similarities given a patch location
    return (
        torch.nn.functional.cosine_similarity(x, x[t, c], dim=-1)
        .view((-1, patch_num, patch_num))
        .permute(1, 0, 2)
        .flatten(1, 2)
        .unsqueeze(-1)
    )


def semantic_patch_cos_sim(x, patch_num, part, _s, _l, semantic_patches, s=None, *args, **kargs):
    # shows the video patch similarities given a semantic patch
    # _l -> the layer of the feature x
    # s -> the mandatory subject, overwrites _s
    # _s -> the subject of the feature x
    return (
        (
            (
                torch.nn.functional.cosine_similarity(
                    x,
                    semantic_patches[_s if s == None else s][part][_l],
                    dim=-1
                ) / 2 + 0.5
            )
            .view((-1, patch_num, patch_num))
            .permute(1, 0, 2)
            .flatten(1, 2)
            .unsqueeze(-1)
        )
    )


def plotter(
    features,
    title="",
    mode="subject-layer",
    num_layers=16,
    unit_size=3,
    font_size=12,
    plot_params={}
):
    keys = list(features.keys())
    num_keys = len(keys)
    num_layers = len(features[keys[0]])
    num_frames = features[keys[0]][0].shape[0]

    def create():
        if mode == "subject-layer":
            plt.figure(figsize=(unit_size * num_layers, unit_size * num_keys), layout="constrained")
            plt.suptitle(title, fontsize=font_size)
        elif mode == "layer-frame":
            plt.figure(figsize=(unit_size * num_frames, unit_size * num_layers), layout="constrained")
            plt.suptitle(title, fontsize=font_size)

    def show():
        plt.tight_layout()
        plt.show()

    if mode == "subject-layer":
        create()
        for j, s in enumerate(features.keys()):
            for i, v in enumerate(features[s]):
                plt.subplot(num_keys, num_layers, j * num_layers + i + 1)
                plt.title(f"L{i}-{s.upper()}")
                plt.gca().axis("off")
                plt.imshow(v, **plot_params)
        show()

    elif mode == "layer-frame":
        for j, s in enumerate(features.keys()):
            create()
            for i, v in enumerate(features[s]):
                plt.subplot(num_layers, 1, i + 1)
                plt.title(f"L{i}-{s.upper()}")
                plt.gca().axis("off")
                plt.imshow(v, **plot_params)
            show()
    else:
        raise NotImplementedError()


def driver(features, method, subjects=None, patch_num=14, **kargs):
    if subjects == None:
        subjects = list(features[0].keys())

    assert features[0][subjects[0]].shape[1] == patch_num**2

    r = {
        k: [] for k in subjects
    }
    num_layers = len(features)
    for l in range(num_layers):
        for s in subjects:
            # variance
            r[s].append(method(features[l][s], patch_num=patch_num, _l=l, _s=s, ** kargs).float())

    return r


# %%

encoder = FrameAttrExtractor(
    architecture="ViT-L/14",
    prompt_dropout=0,
    prompt_layers=0,
    prompt_mode=PromptMode.NONE,
    prompt_num=0,
    text_embed=False
)
encoder.eval()
encoder.to("cuda")

patch_num = encoder.n_patch
n_px = encoder.n_px
dataset = FFPP(
    df_types=["REAL", "DF", "FS", "F2F", "NT"],
    compressions=['c23'],
    n_px=n_px,
    num_frames=1,
    clip_duration=1,
    strategy=FFPPSampleStrategy.NORMAL,
    augmentations=FFPPAugmentation.NONE,
    force_random_speed=False,
    split='train',
    data_dir="datasets/ffpp",
    vid_ext=".avi",
    pack=False,
    transform=encoder.transform
)
random.seed(1019)
clip = dataset[random.randint(0, len(dataset))][0]
features = extract_features(encoder, clip[0][0])

# evals = driver(features, var)
# plotter(evals, "", "subject-layer", unit_size=2)

# evals = driver(features, one_patch_cos_sim, t=0, c=21)
# plotter(evals, "", "layer-frame", unit_size=2)

semantic_patches = fetch_semantic_features(
    encoder, 
    df_types=["REAL"], 
    sample_num=100,
    centroid_mode=True
)

# %%
plot_params = dict(vmin=0, vmax=1)
part = "skin"
# %%
evals = driver(features, semantic_patch_cos_sim, part=part, s="q", semantic_patches=semantic_patches,patch_num=16)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="k", semantic_patches=semantic_patches,patch_num=16)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="v", semantic_patches=semantic_patches,patch_num=16)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="out", semantic_patches=semantic_patches,patch_num=16)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

 # %%
