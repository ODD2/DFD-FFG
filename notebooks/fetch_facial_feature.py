# %%
from torchvision.transforms import Compose
# %%
import os
import sys
import cv2
import torch
import pickle
import random
import torchvision
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import torchvision.transforms as T


from tqdm import tqdm
from src.clip.model_vpt import PromptMode
from notebooks.tools import extract_features
from src.model.clip.snvl import FrameAttrExtractor
from src.dataset.ffpp import FFPP, FFPPSampleStrategy, FFPPAugmentation
from sklearn.cluster import KMeans

SAMPLE_NUM = 10


def fetch_semantic_features(
    encoder,
    df_types=["REAL", "NT", "DF", "FS", "F2F"],
    subjects=['q', 'k', 'v', 'out', "emb"],
    seconds=3,
    frames=1,
    sample_num=100,
    save_path="",
    visualize=False,
    seed=None,
    centroid_mode=False
):
    if (os.path.exists(save_path)):
        raise Exception(f"{save_path}, file exists!")

    transform = encoder.transform
    patch_num = encoder.n_patch
    n_px = encoder.n_px
    # create dataset & models
    dataset = FFPP(
        df_types=df_types,
        n_px=n_px,
        num_frames=frames,
        compressions=['c23'],
        clip_duration=seconds,
        strategy=FFPPSampleStrategy.NORMAL,
        augmentations=FFPPAugmentation.NONE,
        force_random_speed=False,
        split='train',
        data_dir="datasets/ffpp",
        vid_ext=".avi",
        pack=False,
        transform=T.Compose([
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC)
        ]),
        max_clips=3
    )

    # the following patch locations are based on a 14x14 grid.
    semantic_locations = {
        "eyes": [
            [3, 3], [3, 4], [3, 9], [3, 10],
            [4, 3], [4, 4], [4, 9], [4, 10]
        ],
        "nose": [
            [7, 6], [6, 6], [5, 6],
            [7, 7], [6, 7], [5, 7]
        ],
        "lips": [
            [10, 5], [10, 6], [10, 7], [10, 8]
        ],
        "eyebrows": [
            # [2, 3], [2, 4],
            # [2, 9], [2, 10]
        ],
        "skin": [
            [0, 6], [0, 7],
            [1, 6], [1, 7],
            [7, 3], [7, 10],
            # [12, 6], [12, 7],
            # [13, 6], [13, 7]
        ]
    }

    # part keypoint data, prepared for frame augmentations
    part_keypoints = []
    part_labels = []

    for part in semantic_locations:
        for loc in semantic_locations[part]:
            part_keypoints.append(loc)
            part_labels.append(part)
    part_keypoints = (np.array(part_keypoints) + 0.5) / 14 * n_px

    # frame augmentations
    augmentations = alb.Compose(
        [
            alb.Flip(p=1.0),
            alb.RandomRotate90(p=1.0),
            alb.RandomResizedCrop(
                n_px, n_px,
                scale=(0.5, 0.7), ratio=(1, 1),
                p=1.0,
            )
        ],
        keypoint_params=alb.KeypointParams(format='yx', remove_invisible=True, label_fields=["part_name"])
    )

    # container
    layer_num = len(encoder.model.transformer.resblocks)
    semantic_patches = {
        s: {
            k: [[] for _ in range(layer_num)]
            for k in semantic_locations.keys()
        }
        for s in subjects
    }

    if (seed):
        random.seed(seed)

    indices = random.sample(range(len(dataset)), sample_num)
    # random samples
    for idx in tqdm(indices):
        ########## random select index ############
        idx = random.randint(0, len(dataset))
        data = dataset[idx]

        ######### extract video features ###########
        for clip in data[0]:
            # sample augmentation
            frame = clip[0][0].permute((1, 2, 0)).numpy()
            result = augmentations(image=frame, keypoints=part_keypoints, part_name=part_labels)
            rrc_frame, rrc_kp, rrc_lb = result["image"], result["keypoints"], result["part_name"]

            rrc_semantic_locations = {
                part: []
                for part in semantic_locations.keys()
            }

            for loc, part in zip(rrc_kp, rrc_lb):
                rrc_semantic_locations[part].append(loc)

            for part in rrc_semantic_locations.keys():
                rrc_semantic_locations[part] = np.round(
                    np.clip(np.array(rrc_semantic_locations[part]) / n_px * 14 - 0.5, a_min=0, a_max=14)
                )

            if visualize:
                # visualization for debugging
                # > visualize ordinary frame
                plt.imshow(frame)
                plt.scatter(part_keypoints[..., 1], part_keypoints[..., 0])
                plt.show()
                # > visualize frame after augmentation
                rrc_kp = np.array(rrc_kp)
                plt.imshow(rrc_frame)
                plt.scatter(rrc_kp[..., 1], rrc_kp[..., 0])
                plt.show()
                # > visualize frame and keypoints at patch level
                plt.imshow(cv2.resize(rrc_frame, (14, 14)))
                for part in rrc_semantic_locations.keys():
                    if (len(rrc_semantic_locations[part]) > 0):
                        plt.scatter(rrc_semantic_locations[part][..., 1], rrc_semantic_locations[part][..., 0])
                plt.show()

            # extract frame features
            features = extract_features(
                encoder,
                transform(
                    torch.from_numpy(rrc_frame).permute((2, 0, 1)).unsqueeze(0)
                )
            )

            # post-process semantic locations
            rrc_semantic_locations = {
                k: [
                    int(_v[0] / 13 * (patch_num - 1)) * patch_num + int(_v[1] / 13 * (patch_num - 1))
                    for _v in v
                ]
                for k, v in rrc_semantic_locations.items()
            }

            ######### extract video features ###########
            for l in range(layer_num):
                for s in subjects:
                    for p, loc in rrc_semantic_locations.items():
                        semantic_patches[s][p][l].extend(
                            features[l][s][0, loc].tolist()
                        )

    def centroid(points):
        if (len(points) == 0):
            return None

        points = torch.tensor(points)
        # sort_point_l2_idx = torch.nn.functional.mse_loss(
        #     # points.unsqueeze(0).repeat(points.shape[0], 1, 1),
        #     # points.unsqueeze(1).repeat(1, points.shape[0], 1),
        #     points.unsqueeze(0),
        #     points.unsqueeze(1),
        #     reduction="none"
        # ).flatten(1).sum(1).sort(descending=True)[1]
        sort_point_l2_idx = (
            (
                1 - torch.nn.functional.cosine_similarity(
                    # points.unsqueeze(0).repeat(points.shape[0], 1, 1),
                    # points.unsqueeze(1).repeat(1, points.shape[0], 1),
                    # reduction="none",
                    points.unsqueeze(0),
                    points.unsqueeze(1),
                    dim=-1
                )
            ).flatten(1).sum(1).sort(descending=True)[1]
        )
        skip_num = int(sort_point_l2_idx.shape[0] * 0.9)
        target_indices = sort_point_l2_idx[skip_num:]
        return points[target_indices].mean(0)

    semantic_patches = {
        s: {
            p: [
                (
                    centroid(semantic_patches[s][p][l])
                    if centroid_mode else
                    torch.tensor(semantic_patches[s][p][l]).mean(dim=0)
                )
                for l in range(layer_num)
            ]
            for p in semantic_locations.keys()
        }
        for s in subjects
    }

    if (len(save_path) > 0):
        with open(save_path, "wb") as f:
            pickle.dump(semantic_patches, f)

    return semantic_patches


# %%
if __name__ == "__main__":
    # encoder = FrameAttrExtractor(
    #     architecture="ViT-B/16",
    #     prompt_dropout=0,
    #     prompt_layers=0,
    #     prompt_mode=PromptMode.NONE,
    #     prompt_num=0,
    #     text_embed=False,
    #     pretrain="misc/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
    # )

    encoder = FrameAttrExtractor(
        architecture="ViT-B/16",
        prompt_dropout=0.2,
        prompt_layers=12,
        prompt_mode=PromptMode.DEEP,
        prompt_num=10,
        text_embed=False,
        pretrain="logs/DFD-FFG/9smr6fw7/checkpoints/epoch=38-step=8307_encoder.pth"
    )

    encoder.eval()
    encoder.to("cuda")

    fetch_semantic_features(
        encoder, df_types=["REAL"],
        sample_num=SAMPLE_NUM,
        visualize=False,
        save_path=f"./misc/9smr6fw7_real_semantic_patches_v1_{SAMPLE_NUM}.pickle",
        seed=1019
    )

    # fetch_semantic_features(
    #     encoder, df_types=["REAL"],
    #     sample_num=SAMPLE_NUM,
    #     visualize=False,
    #     save_path=f"./misc/FL_real_semantic_patches_v1_{SAMPLE_NUM}.pickle",
    #     seed=1019
    # )

# %%
