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
from src.model.clip import VideoAttrExtractor
from src.dataset.ffpp import FFPP, FFPPSampleStrategy, FFPPAugmentation
from notebooks.fetch_facial_feature import fetch_semantic_features
# %%


def var(x, n_patch, *args, **kargs):
    # shows the variance of the video patches over the temporal dimension.
    return (
        torch.var(x, dim=0)
        .mean(dim=-1)
        .view((n_patch, n_patch))
        .unsqueeze(-1)
    )


def max_stdev(x, n_patch, *args, **kargs):
    # shows the maximum stdev value of the video patches averaged over the temporal dimension.
    return (
        torch.max(
            torch.abs((x - torch.mean(x, dim=0)) / torch.sqrt(torch.var(x, dim=0))).mean(dim=-1),
            dim=0
        )[0]
        .view(n_patch, n_patch)
        .unsqueeze(-1)
    )


def one_patch_cos_sim(x, t, c, n_patch, *args, **kargs):
    # shows the video patch similarities given a patch location
    return (
        torch.nn.functional.cosine_similarity(x, x[t, c], dim=-1)
        .view((-1, n_patch, n_patch))
        .permute(1, 0, 2)
        .flatten(1, 2)
        .unsqueeze(-1)
    )


def semantic_patch_cos_sim(x, n_patch, part, _s, _l, semantic_patches, s=None, *args, **kargs):
    # shows the video patch similarities given a semantic patch
    # _l -> the layer of the feature x
    # s -> the mandatory subject, overwrites _s
    # _s -> the subject of the feature x
    feat = (
        (
            (
                (
                    torch.nn.functional.cosine_similarity(
                        x,
                        semantic_patches[_s if s == None else s][part][_l],
                        dim=-1
                    ) / 2 + 0.5
                ) * 30
            ).softmax(dim=-1)
            .view((-1, n_patch, n_patch))
            .permute(1, 0, 2)
            .flatten(1, 2)
            .unsqueeze(-1)
        )
    )
    feat = (feat - feat.min()) / (feat.max() - feat.min())
    return feat


def plotter(
    features,
    title="",
    mode="subject-layer",
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


def driver(features, method, subjects=None, n_patch=14, **kargs):
    if subjects == None:
        subjects = list(features[0].keys())
    assert features[0][subjects[0]].shape[1] == 1
    assert features[0][subjects[0]].shape[2] == n_patch**2

    r = {
        k: [] for k in subjects
    }
    num_layers = len(features)
    for l in range(num_layers):
        for s in subjects:
            # variance
            r[s].append(method(features[l][s].flatten(0, 1), n_patch=n_patch, _l=l, _s=s, ** kargs).float())

    return r


evals = driver(features, semantic_patch_cos_sim, part=part, s="q", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

# %%

encoder = VideoAttrExtractor(
    architecture="ViT-L/14",
    text_embed=False,
    store_attrs=["q", "k", "v", "out", "emb"]
)
encoder.eval()
encoder.to("cuda")

n_patch = encoder.n_patch
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
# %%
random.seed(1019)
clip = dataset[random.randint(0, len(dataset))][0][0][0]
print(clip[0][0].shape)
# clip[:, :, 136:196, 72:152] = 0
# %%
features = extract_features(encoder, clip)

# evals = driver(features, var)
# plotter(evals, "", "subject-layer", unit_size=2)

# evals = driver(features, one_patch_cos_sim, t=0, c=21)
# plotter(evals, "", "layer-frame", unit_size=2)

# %%
# semantic_patches = fetch_semantic_features(
#     encoder,
#     df_types=["REAL"],
#     sample_num=1000
# )
with open("misc/L14_real_semantic_patches_v2_2000.pickle", "rb") as f:
    semantic_patches = pickle.load(f)


# %%
plot_params = dict(vmin=0, vmax=1)
part = "lips"
# %%
print(features[0]["q"].shape)
# %%
evals = driver(features, semantic_patch_cos_sim, part=part, s="q", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="k", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="v", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="out", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

evals = driver(features, semantic_patch_cos_sim, part=part, s="emb", semantic_patches=semantic_patches, n_patch=n_patch)
plotter(evals, "", "subject-layer", unit_size=2, plot_params=plot_params)

# %%
