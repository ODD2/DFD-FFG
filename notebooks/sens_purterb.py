
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
from src.model.clip.snvl import CLIPVideoAttrExtractor
from src.dataset.ffpp import FFPP, FFPPSampleStrategy, FFPPAugmentation


def sensitivity_for_perturbations(
    encoder,
    num_samples,
    attributes=['q', 'k', 'v', 'out'],
    augmentation=FFPPAugmentation.PERTURBATION,
):
    n_px = encoder.n_px
    patch_num = encoder.n_patch
    transform = encoder.transform

    # create dataset & models
    params = dict(
        df_types=["REAL", "DF", "FS", "F2F", "NT"],
        n_px=n_px,
        num_frames=1,
        compressions=['c23'],
        clip_duration=3,
        strategy=FFPPSampleStrategy.NORMAL,
        augmentation=FFPPAugmentation.NONE,
        force_random_speed=False,
        split='train',
        data_dir="datasets/ffpp",
        vid_ext=".avi",
        pack=False,
        transform=transform
    )

    dataset1 = FFPP(**params)
    params["augmentation"] = augmentation
    dataset2 = FFPP(**params)

    # storage for patch-wise cosine distance of attention attributes
    storage = [
        {
            k: torch.zeros(patch_num * patch_num)
            for k in attributes
        }
        for _ in range(encoder.model.transformers)
    ]

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
        features1 = extract_features(encoder, data1[0]["c23"])
        features2 = extract_features(encoder, data2[0]["c23"])
        for i in range(12):
            for attr in attributes:
                storage[i][attr] += (
                    1 - torch.nn.functional.cosine_similarity(
                        features1[i][attr], features2[i][attr], dim=-1
                    )
                ).squeeze(0) / 2 / num_samples
    return storage


if __name__ == "__main__":
    encoder = CLIPVideoAttrExtractor()
    encoder.eval()
    encoder.to("cuda")

    target_folder = "./misc/attn_attr_sens/"
    os.makedirs(target_folder, exist_ok=True)
    scenario_storages = {}

    for aug_type in [
        FFPPAugmentation.PERTURBATION,
        FFPPAugmentation.DEV | FFPPAugmentation.RGB,
        FFPPAugmentation.DEV | FFPPAugmentation.HUE,
        FFPPAugmentation.DEV | FFPPAugmentation.BRIGHT,
        FFPPAugmentation.DEV | FFPPAugmentation.COMP,
        FFPPAugmentation.DEV | FFPPAugmentation.DSCALE,
        FFPPAugmentation.DEV | FFPPAugmentation.SHARPEN,
    ]:
        scenario_storages[aug_type] = sensitivity_for_perturbations(
            encoder,
            100,
            augmentation=aug_type
        )

    # find max & min
    global_max = -10
    global_min = 10
    for storage in scenario_storages.values():
        for i, attrs in enumerate(storage):
            for j, attr in enumerate(attrs):
                # record
                global_min = min(storage[i][attr].min().item(), global_min)
                global_max = max(storage[i][attr].max().item(), global_max)
    for aug_type, storage in scenario_storages.items():
        plt.figure(figsize=(len(storage) * 1, len(storage[0]) * 1), layout="constrained")
        for i, attrs in enumerate(storage):
            for j, attr in enumerate(attrs):
                plt.subplot(len(storage[0]), len(storage), j * len(storage) + i + 1)
                data = (storage[i][attr].view(14, 14).numpy() - global_min) / (global_max - global_min)
                plt.imshow(data, vmin=0, vmax=1)
                plt.gca().axis("off")
        plt.savefig(os.path.join(target_folder, f"{aug_type}.pdf"))
        plt.close()
