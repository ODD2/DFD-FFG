# %%
import gc
import cv2
import time
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from notebooks.tools import load_model
from src.model.clip.svl import SynoVideoLearner, FFGSynoVideoLearner

DEVICE = "cuda"


# %%
from src.dataset.ffpp import FFPP, FFPPAugmentation, FFPPSampleStrategy
from src.dataset.fsh import FSh

sample_model = SynoVideoLearner()

dataset = FFPP(
    df_types=["NT"],
    compressions=["c23"],
    n_px=sample_model.n_px,
    strategy=FFPPSampleStrategy.NORMAL,
    augmentations=FFPPAugmentation.NONE,
    force_random_speed=False,
    vid_ext=".avi",
    data_dir="datasets/ffpp/",
    num_frames=10,
    clip_duration=4,
    split="train",
    transform=sample_model.transform,
    pack=False,
    ratio=1.0
)

# dataset = FSh(
#     ["c23"],
#     vid_ext=".avi",
#     data_dir="datasets/ffpp/",
#     num_frames=10,
#     clip_duration=4,
#     split="test",
#     transform=sample_model.transform,
#     pack=False,
#     ratio=1.0
# )


def abbrev(value):
    return round(value, 2)

# %%


def interpret(
    clips,
    model,
    query_idx=0,
    logit_index=1,
    mode="all",
    num_layers=-1,
    aspect="frame"
):
    # For now, enforce frame level estimation
    model.zero_grad()
    clips = clips.to(DEVICE)
    num_patch = model.encoder.model.patch_num
    num_patch_per_side = int(math.sqrt(num_patch))
    image_attn_blocks = list(model.encoder.decoder.decoder_layers)

    logits = model(clips)["logits_s"]  # TODO: make sure to select the correct logits
    indicator = logits[:, logit_index].sum(dim=0)

    relevance_storage = []

    if (num_layers == -1):
        beg_layer = 0
        end_layer = len(image_attn_blocks)
    elif (num_layers > 0):
        end_layer = len(image_attn_blocks)
        beg_layer = end_layer - num_layers
    else:
        raise NotImplementedError()

    for target_layer in range(beg_layer, end_layer):
        model.zero_grad()
        aff = image_attn_blocks[target_layer].aff
        # aff.shape = [b,t,synos,patches,head]
        grad = torch.autograd.grad(
            indicator,
            [aff],
            retain_graph=True
        )[0].detach().cpu()

        aff = aff.detach().cpu()
        # aff.shape = [b,t,synos,patches,head]
        cam = aff.detach().cpu()
        # cam.shape = [b,t,synos,patches,head]
        if (mode == "grad"):
            val = grad
        elif (mode == "cam"):
            val = cam
        elif (mode == "all"):
            val = grad * cam
        else:
            raise NotImplementedError()

        ###### This part is for the multi-head attention map visualization ######
        # reshape to restore (h, n, t, q, p^2)
        # val = val.permute((4, 0, 1, 2, 3))
        # average over each heads.
        # val = val.clamp(min=0).mean(dim=0)
        ##########################################################################

        ###### This part is for the single-head attention map visualization ######
        val = val.clamp(min=0)
        ##########################################################################

        # fetch the specified query
        relevance = val[:, :, [query_idx]].mean(dim=0)

        # reshape to match the original patch geometry
        relevance = relevance.unflatten(-1, (num_patch_per_side, num_patch_per_side))

        model.zero_grad()

        relevance_storage.append(relevance)

    if aspect == "frame":
        relevance_storage = sum(relevance_storage) / len(relevance_storage)
    elif aspect == "layer":
        relevance_storage = torch.stack(
            [rel.mean(dim=0) for rel in relevance_storage]
        )

    model.zero_grad()
    del clips
    del logits
    del indicator
    return relevance_storage


def draw_flatten_heatmap(relevance: torch.Tensor, scale=500):
    # process clip relevance map
    relevance = torch.nn.functional.interpolate(
        relevance,
        size=224,
        mode='bilinear'
    ).numpy()

    # convert frame/cam sequence into a large frame
    relevance = relevance.transpose(2, 0, 3, 1).reshape(224, -1, 1)

    if scale > 0:
        relevance = relevance * scale
        relevance[relevance > 1] = 1
    elif scale == -1:
        relevance /= relevance.max()
    elif scale == -2:
        relevance = relevance - relevance.min()
        relevance /= (relevance.max() - relevance.min())
    else:
        raise NotImplementedError()

    relevance = cv2.applyColorMap(np.uint8(255 * relevance), cv2.COLORMAP_JET)
    relevance = cv2.cvtColor(relevance, cv2.COLOR_RGB2BGR)
    return relevance


# show_relevance(torch.randn(1, 1, 14, 14))
# interpret(torch.randn(1, 3, 224, 224), model)


# %%
# interpret prediction
UNIT = 2
FRAMES = 10
NUM_BATCHES = 6
BATCH_SIZE = 5
NUM_LAYERS = 24

# sample videos
random.seed(1019)
clips = torch.cat(
    [
        dataset.get_entity(
            random.randrange(0, len(dataset)),
            with_entity_info=True
        )["clips"]
        for _ in range(NUM_BATCHES * BATCH_SIZE)
    ],
    dim=0
)

model_configs = [
    # ("none", SynoVideoLearner, "logs/DFD-FFG/kjfr7es5/checkpoints/epoch=8-step=535.ckpt"),
    # ("lips", FFGSynoVideoLearner, "logs/DFD-FFG/6vtmspln/checkpoints/epoch=5-step=377.ckpt"),
    # (797, FFGSynoVideoLearner, "logs/DFD-FFG/y5fgtrnm/checkpoints/epoch=4-step=314.ckpt")
    # ("skin", FFGSynoVideoLearner, "logs/DFD-FFG/rad4hm3u/checkpoints/epoch=6-step=440.ckpt")
    # (684, LinearMeanVideoLearner, "logs/DFD-FFG/gfy5j5wf/checkpoints/epoch=49-step=10650.ckpt")
    # ("hd9twd8v", FFGSynoVideoLearner, "logs/DFD-FFG/hd9twd8v/checkpoints/epoch=9-step=598.ckpt"),
    # ("g5m75moo", FFGSynoVideoLearner, "logs/DFD-FFG/g5m75moo/checkpoints/epoch=5-step=346.ckpt"),
    # ("dxcryj5v", FFGSynoVideoLearner, "logs/DFD-FFG/dxcryj5v/checkpoints/last.ckpt"),
    # ("ydgdslgl", FFGSynoVideoLearner, "logs/DFD-FFG/ydgdslgl/checkpoints/epoch=9-step=629.ckpt"),
    # ("hhipjbbh", FFGSynoVideoLearner, "logs/DFD-FFG/hhipjbbh/checkpoints/last.ckpt"),
    # ("3fek0hoo", FFGSynoVideoLearner, "logs/DFD-FFG/3fek0hoo/checkpoints/last.ckpt"),
    # ("zqo6ijqq", FFGSynoVideoLearner, "logs/DFD-FFG/zqo6ijqq/checkpoints/last.ckpt"),
    # ("z94z887d", FFGSynoVideoLearner, "logs/DFD-FFG/z94z887d/checkpoints/last.ckpt"),
    # ("o9jcrwr6", FFGSynoVideoLearner, "logs/DFD-FFG/o9jcrwr6/checkpoints/last.ckpt")
    # ("26txmw8y", FFGSynoVideoLearner, "logs/DFD-FFG/26txmw8y/checkpoints/last.ckpt"),
    # ("248erwy6", FFGSynoVideoLearner, "logs/DFD-FFG/248erwy6/checkpoints/last.ckpt"),
    # ("p5tdrroa", FFGSynoVideoLearner, "logs/DFD-FFG/p5tdrroa/checkpoints/last.ckpt"),
    # ("xwnug1t1", FFGSynoVideoLearner, "logs/DFD-FFG/xwnug1t1/checkpoints/last.ckpt"),
    # ("cnex8ypf", FFGSynoVideoLearner, "logs/DFD-FFG/cnex8ypf/checkpoints/last.ckpt"),
    # ("cnex8ypf", FFGSynoVideoLearner, "logs/DFD-FFG/stnn1s5y/checkpoints/last.ckpt"),
    # ("2avus0qv", FFGSynoVideoLearner, "logs/DFD-FFG/2avus0qv/checkpoints/last.ckpt"),
    # ("rkw6q4zw", FFGSynoVideoLearner, "logs/DFD-FFG/rkw6q4zw/checkpoints/last.ckpt")
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(Experiment)/8iz2btxl/checkpoints/last.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(Experiment)/hfne0qof/checkpoints/last.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(Experiment)/pvdrbg3m/checkpoints/last.ckpt")
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(Experiment)/35g5yvj5/checkpoints/last.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(Experiment)/9uyiv6pc/checkpoints/last.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/DFD-FFG(CVPR-EXP)/ib5mz44a/checkpoints/epoch=6-step=1694.ckpt"),
    # ("125", SynoVideoLearner, "logs/CVPR/vmobdb0k/checkpoints/epoch=9-step=2420.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/CVPR/szfhgbkm/checkpoints/epoch=7-step=2040.ckpt"),
    ("125", FFGSynoVideoLearner, "logs/ECCV/otfsj0qd/checkpoints/epoch=29-step=2040.ckpt"),
    # ("125", FFGSynoVideoLearner, "logs/ECCV/8gbspxsz/checkpoints/epoch=29-step=2040.ckpt"),
    # ("125", SynoVideoLearner, "logs/ECCV/8gbspxsz/checkpoints/epoch=29-step=2040.ckpt")
]

scenarios = [
    (
        syno,  # syno num
        1,  # logit
        # ("all", 50000),  # mode,scaler
        # ("all", -2),
        # ("grad", 500),
        # ("cam", 100),git r
        ("cam", -2),
        -1,  # layers(from tail)
        # "layer"  # aspect
        "frame"
    )
    for syno in [0, 1, 2, 3]
]


def runner(model_config, scenario):
    model_name, model_cls, model_path = model_config
    model = load_model(model_cls, model_path).model
    model.to(DEVICE)
    model.eval()
    syno, logit_index, mode_scale, num_layers, aspect = scenario
    mode = mode_scale[0]
    scale = mode_scale[1]

    relevance = 0
    for b in range(0, NUM_BATCHES * BATCH_SIZE, BATCH_SIZE):
        print(clips.shape)
        print(b, b + BATCH_SIZE)
        relevance += interpret(
            clips[b:b + BATCH_SIZE],
            model,
            query_idx=syno,
            logit_index=logit_index,
            mode=mode,
            num_layers=num_layers,
            aspect=aspect
        ) / NUM_BATCHES
        gc.collect()
        torch.cuda.empty_cache()

    flatten_heatmap = draw_flatten_heatmap(
        relevance,
        scale=scale
    )

    title = "model={}/scale={}/syno={}/logit={}/aspect={}".format(
        model_name,
        scale,
        int(syno),
        int(logit_index),
        aspect
    )

    return dict(
        title=title,
        img=flatten_heatmap
    )


timestr = time.strftime("%H%M%S")
for s_idx, scenario in enumerate(scenarios):
    scenario_images = []

    for config in model_configs:
        scenario_images.append(runner(config, scenario))
        gc.collect()
        torch.cuda.empty_cache()

    rows = len(scenario_images)
    cols = NUM_LAYERS

    plt.figure(
        figsize=(UNIT * cols, UNIT * rows),
        layout="constrained"
    )
    for i, model_results in enumerate(scenario_images):
        plt.subplot(rows, 1, i + 1)
        plt.title(model_results["title"])
        plt.imshow(model_results["img"], vmax=255, vmin=0)
        plt.gca().axis("off")
    plt.savefig(f"./misc/extern/{timestr}-{s_idx}.png")

# %%
