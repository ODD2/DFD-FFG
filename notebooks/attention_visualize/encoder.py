# %%
import gc
import cv2
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from notebooks.tools import load_model
from src.model.clip.ftfe import LinearMeanVideoLearner

DEVICE = "cuda"


# %%
from src.dataset.ffpp import FFPP, FFPPAugmentation, FFPPSampleStrategy

sample_model = LinearMeanVideoLearner()

dataset = FFPP(
    df_types=["NT"],
    compressions=["c23"],
    n_px=sample_model.n_px,
    strategy=FFPPSampleStrategy.NORMAL,
    augmentations=FFPPAugmentation.NONE,
    force_random_speed=False,
    vid_ext=".avi",
    data_dir="datasets/ffpp/",
    num_frames=1,
    clip_duration=4,
    split="train",
    transform=sample_model.transform,
    pack=False,
    ratio=1.0
)


# %%
def interpret(
    clips,
    model,
    query_idx=0,
    target_layer=0,
    logit_index=0,
    mode="all",
    reverse=False
):
    # For now, enforce frame level estimation
    assert clips.shape[1] == 1
    model.zero_grad()
    num_patch = model.encoder.n_patch
    num_queries = 1
    num_tokens = num_patch**2
    image_attn_blocks = list(
        dict(model.encoder.model.transformer.resblocks.named_children()).values()
    )

    logits = model(clips)["logits"]
    model.zero_grad()
    indicator = logits[:, logit_index].sum(dim=0)
    aff = image_attn_blocks[target_layer].attn.aff
    grad = torch.autograd.grad(
        indicator,
        [aff],
        retain_graph=True
    )[0].detach().cpu()

    aff = aff.detach().cpu()
    cam = aff.detach().cpu()

    if (mode == "grad"):
        val = grad
    elif (mode == "cam"):
        val = cam
    elif (mode == "all"):
        val = grad * cam
    else:
        raise NotImplementedError()

    # reshape to restore (n,h,p^2+1,p^2+1)
    # cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
    val = val.permute((0, 3, 1, 2))

    # average over each heads.
    val = val.clamp(min=0).mean(dim=1)

    if reverse:
        val = val.transpose(1, 2)
        aff = aff.transpose(1, 2)

    # fetch the specified query
    relevance = val[:, query_idx, 1:1 + num_tokens].mean(dim=0)

    # reshape to match the original patch geometry
    relevance = relevance.view(
        (num_queries, 1, num_patch, num_patch)
    )

    indicator.backward()
    model.zero_grad()

    head_prob = aff[:, [query_idx], :1 + num_tokens].sum(dim=2).flatten(0, 1).mean(0)

    def abbrev(value):
        return round(value, 2)

    prob_avg = abbrev(torch.mean(head_prob, dim=0).flatten().item())
    prob_var = abbrev(torch.var(head_prob, dim=0).flatten().item())
    prob_max = abbrev(torch.max(head_prob, dim=0)[0].flatten().item())
    prob_min = abbrev(torch.min(head_prob, dim=0)[0].flatten().item())

    return dict(
        relevance=relevance,
        prob_dist=dict(
            avg=prob_avg,
            var=prob_var,
            max=prob_max,
            min=prob_min
        )
    )


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
MAX_LAYER = 10
NUM_SAMPLES = 40

# sample videos
random.seed(1019)
clips = torch.cat(
    [
        dataset.get_entity(
            random.randrange(0, len(dataset)),
            with_entity_info=True
        )["clips"]
        for _ in range(NUM_SAMPLES)
    ],
    dim=0
).to(DEVICE)

model_configs = [
    (0, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=0-step=63.ckpt"),
    (1, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=1-step=126.ckpt"),
    (2, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=2-step=189.ckpt"),
    (3, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=3-step=252.ckpt"),
    (4, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=4-step=315.ckpt"),
    (5, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=5-step=378.ckpt"),
    (10, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=10-step=693.ckpt"),
    (15, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=15-step=1008.ckpt"),
    (20, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=20-step=1323.ckpt"),
    (29, LinearMeanVideoLearner, "logs/DFD-FFG/o2y68y49/checkpoints/epoch=29-step=1890.ckpt")
    # (550, LinearMeanVideoLearner, "logs/DFD-FFG/ozeorw7b/checkpoints/epoch=49-step=8500.ckpt")

]

scenarios = [
    (
        False,  # rev
        prompt,
        800
    )
    for prompt in [0]
]


def runner(model_config, scenario):
    model_name, model_cls, model_path = model_config
    model = load_model(model_cls, model_path).model
    model.to(DEVICE)
    model.eval()
    model_images = []
    rev, prompt, scale = scenario

    for layer in range(MAX_LAYER):
        results = interpret(
            clips,
            model,
            query_idx=prompt,
            target_layer=layer,
            reverse=rev
        )

        relevance = results["relevance"]

        flatten_heatmap = draw_flatten_heatmap(
            relevance,
            scale=scale
        )
        model_images.append(dict(
            flatten_heatmap=flatten_heatmap,
            prob_dist=results["prob_dist"]
        ))

    title = "model={}/layer={}/rev={}/scale={}".format(
        model_name,
        int(MAX_LAYER),
        int(rev),
        scale
    )
    return dict(
        title=title,
        img=np.concatenate(
            [
                i["flatten_heatmap"]
                for i in model_images
            ],
            axis=1
        ),
        prob_infos=[
            "avg={}\nvar={}\nmin={}\nmax={}".format(
                i["prob_dist"]["avg"],
                i["prob_dist"]["var"],
                i["prob_dist"]["min"],
                i["prob_dist"]["max"],
            )
            for i in model_images
        ]
    )


plt.ioff()
timestr = time.strftime("%H%M%S")
for s_idx, scenario in enumerate(scenarios):
    scenario_images = []

    for config in model_configs:
        scenario_images.append(runner(config, scenario))
        gc.collect()
        torch.cuda.empty_cache()

    rows = len(model_configs)
    cols = MAX_LAYER

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
    ###############################################
    plt.figure(
        figsize=(UNIT * cols, UNIT * rows),
        layout="constrained"
    )
    for i, model_results in enumerate(scenario_images):
        for j, prob_info in enumerate(model_results["prob_infos"]):
            plt.subplot(
                rows,
                cols,
                i * cols + j + 1
            )
            plt.text(0.5, 0.5, prob_info)
            plt.gca().axis("off")
    plt.savefig(f"./misc/extern/{timestr}-{s_idx}(probs).png")
    plt.show()

# %%
