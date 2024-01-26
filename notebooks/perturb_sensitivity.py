# %%
import os
import torch
import random

from tqdm import tqdm
from functools import partial
from matplotlib import pyplot as plt
from torchvision import transforms as T
from notebooks.tools import extract_features
from src.model.clip import VideoAttrExtractor
from src.dataset.ffpp import FFPP, FFPPAugmentation, FFPPSampleStrategy

# %%


def sensitivity_for_perturbations(
    encoder,
    num_samples,
    attributes=['q', 'k', 'v', 'out', "emb"]
):

    n_patch = encoder.n_patch
    n_px = encoder.n_px
    n_layers = encoder.model.transformer.layers
    dataset_partial = partial(
        FFPP,
        df_types=["REAL", "DF", "FS", "F2F", "NT"],
        compressions=['c23'],
        n_px=n_px,
        num_frames=1,
        clip_duration=3,
        strategy=FFPPSampleStrategy.NORMAL,
        force_random_speed=False,
        split='train',
        data_dir="datasets/ffpp",
        vid_ext=".avi",
        pack=False,
        transform=encoder.transform
    )

    dataset1 = dataset_partial(augmentations=FFPPAugmentation.NONE)
    dataset2 = dataset_partial(augmentations=FFPPAugmentation.PERTURBATION)

    # storage for patch-wise cosine distance of attention attributes
    storage = [{k: torch.zeros(n_patch * n_patch) for k in attributes} for _ in range(n_layers)]
    random.seed(1019)
    for i in tqdm(range(num_samples)):
        idx = random.randrange(0, len(dataset1))
        data1 = dataset1[idx]
        data2 = dataset2[idx]
        #######################################
        # plt.figure(figsize=(50, 5))
        # plt.subplot(2, 1, 1)
        # plt.imshow(
        #     np.stack(
        #         (data1[0]["c23"][:30]).numpy().transpose((0, 2, 3, 1)), axis=1
        #     ).reshape((n_px, -1, 3))
        # )
        # plt.subplot(2, 1, 2)
        # plt.imshow(
        #     np.stack(
        #         (data2[0]["c23"][:30]).numpy().transpose((0, 2, 3, 1)), axis=1
        #     ).reshape((n_px, -1, 3))
        # )
        #######################################
        features1 = extract_features(encoder, data1[0][0][0])
        features2 = extract_features(encoder, data2[0][0][0])
        for i in range(n_layers):
            for attr in attributes:
                storage[i][attr] += (
                    1 - torch.nn.functional.cosine_similarity(
                        features1[i][attr], features2[i][attr], dim=-1
                    )
                ).view(-1) / 2 / num_samples
    return storage


# %%
encoder = VideoAttrExtractor(
    architecture="ViT-L/14",
    text_embed=False,
    store_attrs=["q", "k", "v", "out", "emb"]
)
encoder.eval()
encoder.to("cuda")
n_patch = encoder.n_patch
# %%
target_folder = "./misc/attn_attr_sens/"
os.makedirs(target_folder, exist_ok=True)
scenario_storages = {}
for aug_type in [
    "perturbations",
    # "dev-mode+force-rgb",
    # "dev-mode+force-hue",
    # "dev-mode+force-bright",
    # "dev-mode+force-comp",
    # "dev-mode+force-dscale",
    # "dev-mode+force-sharpen",
]:
    scenario_storages[aug_type] = sensitivity_for_perturbations(
        encoder,
        3000
    )

# %%
# find max & min
for storage in scenario_storages.values():
    for j, attr in enumerate(["q", "k", "v", "out", "emb"]):
        global_max = -10
        global_min = 10
        for i, attrs in enumerate(storage):
            global_min = min(storage[i][attr].min().item(), global_min)
            global_max = max(storage[i][attr].max().item(), global_max)
        for i, attrs in enumerate(storage):
            storage[i][attr] = (storage[i][attr] - global_min) / (global_max - global_min)


for aug_type, storage in scenario_storages.items():
    plt.figure(figsize=(len(storage) * 1, len(storage[0]) * 1), layout="constrained")
    for i, attrs in enumerate(storage):
        for j, attr in enumerate(["q", "k", "v", "out", "emb"]):
            plt.subplot(len(storage[0]), len(storage), j * len(storage) + i + 1)
            # data = (storage[i][attr].view(n_patch, n_patch).numpy() - global_min) / (global_max - global_min)
            data = storage[i][attr].view(n_patch, n_patch).numpy()
            plt.imshow(data, vmin=0, vmax=0.5)
            plt.gca().axis("off")
    plt.savefig(os.path.join(target_folder, f"{aug_type}.pdf"))
    plt.show()
    plt.close()
# %%
