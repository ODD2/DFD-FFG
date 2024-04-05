# %%
import os
import sys
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
from sklearn.cluster import KMeans

import copy as capypare

TYPE_DIRS = {
    'REAL': 'real',
    'DF': 'DF',
    'FS': 'FS',
    'F2F': 'F2F',
    'NT': 'NT'
}


class FaceLandmark():
    def __init__(self, init_px, n_px, grid=14, data_offset=17, offset=0) -> None:
        self.init_px = init_px
        self.n_px = n_px
        self.landmarks_array = None
        self.data_offset = data_offset
        self.grid = grid
        self.offset = offset
        self.LANDMARKS = {
            # 'l_eyebrows': [[18, 22]],
            # 'r_eyebrows': [[23, 27]],
            # 'brows': [[18, 22], [23, 27]],
            # 'l_eye': [[37, 42]],
            # 'r_eye': [[43, 48]],
            'eyes': [[37, 42], [43, 48]],
            'nose': [[28, 31]],
            'lips': [[62, 64], [66, 68]],
            'skin': [[1, 17]]

        }

    def set_landmarks_array(self, landmarks_array):
        self.landmarks_array = (landmarks_array * self.n_px / self.init_px).astype(int)

    def fetch_semantic_landmarks(self, key: str, grid_like=False, visualization=False):
        if key not in self.LANDMARKS.keys():
            raise NotImplementedError()

        result = np.empty((0, 2))
        for (si, ei) in self.LANDMARKS[key]:
            si, ei = si - 1 - self.data_offset, ei - self.data_offset
            result = np.append(result, self.landmarks_array[self.offset + si:self.offset + ei, :], axis=0)

        if grid_like:
            return self.convert_grid_map(result, visualization)

        return np.flip(result, axis=-1)

    def convert_grid_map(self, array, visualization=False):
        step = int(self.n_px // self.grid)
        result = []
        for (x, y) in array:
            result.append([np.ceil(x / step) - 1, np.ceil(y / step) - 1])  # grid idx

        if visualization:
            return (np.array(result) + 0.5) * step
        else:
            return np.array(result)


def fetch_semantic_features(
    encoder,
    df_types=["REAL", "NT", "DF", "FS", "F2F"],
    subjects=['q', 'k', 'v', 'out', "emb"],
    seconds=2,
    frames=1,
    sample_num=100,
    save_path="",
    visualize=False,
    seed=None,
    centroid_mode=False
):
    if save_path != None and (os.path.exists(save_path)):
        raise Exception(f"{save_path}, file exists!")

    transform = encoder.transform
    patch_num = encoder.n_patch
    n_px = encoder.n_px
    FFPP.prepare_data(
        data_dir="datasets/ffpp/",
        compressions=['c23'],
        vid_ext='.avi'
    )
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
        max_clips=5
    )

    landmarker = FaceLandmark(224, n_px, 16, 0, 0)

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
            k: [dict(embed=0, count=0) for _ in range(layer_num)]
            for k in landmarker.LANDMARKS.keys()
        }
        for s in subjects
    }

    if (seed):
        random.seed(seed)

    indices = random.sample(range(len(dataset)), sample_num)
    # random samples
    for idx in tqdm(indices):
        ########## random select index ############
        idx = random.randrange(0, len(dataset))
        data = dataset.get_entity(idx, with_entity_info=True)

        ######### extract video features ###########
        for f_idx, clip in zip(data['indices'], data['clips']):
            # sample augmentation
            frame = clip[0].permute((1, 2, 0)).numpy()

            # landmark frame
            _df = TYPE_DIRS[str(data['df_type'])]
            _video_name = data['video_name']
            landmark_path = f'/scratch1/users/od/FaceForensicC23/cropped/frame_data/{_df}/{_video_name}.pickle'

            with open(landmark_path, 'rb') as f:
                landmark = pickle.load(f)

            if (len(landmark[f_idx[0]]['landmarks']) == 0):
                continue

            landmark_keypoints = landmark[f_idx[0]]['landmarks'][0]

            landmarker.set_landmarks_array(landmark_keypoints)

            # part keypoint data, prepared for frame augmentations
            part_keypoints = []
            part_labels = []
            for region in landmarker.LANDMARKS.keys():
                kps = landmarker.fetch_semantic_landmarks(region, grid_like=False)
                for kp in kps:
                    part_keypoints.append(kp)
                    part_labels.append(region)
            part_keypoints = np.clip(np.array(part_keypoints), a_min=0, a_max=n_px - 1)

            result = augmentations(image=frame, keypoints=part_keypoints, part_name=part_labels)
            rrc_frame, rrc_kp, rrc_lb = result["image"], result["keypoints"], result["part_name"]

            rrc_semantic_locations = {
                region: []
                for region in landmarker.LANDMARKS.keys()
            }

            for loc, region in zip(rrc_kp, rrc_lb):
                rrc_semantic_locations[region].append(loc)

            rrc_semantic_locations_copy = capypare.deepcopy(rrc_semantic_locations)
            for region in rrc_semantic_locations_copy.keys():
                rrc_semantic_locations_copy[region] = np.array(rrc_semantic_locations_copy[region])

            for region in rrc_semantic_locations.keys():
                rrc_semantic_locations[region] = np.clip(
                    np.floor(np.array(rrc_semantic_locations[region]) / n_px * patch_num - 0.5),
                    a_min=0,
                    a_max=patch_num - 1
                )

            if visualize:
                # visualization for debugging
                plt.imshow(frame)
                plt.axis('off')
                plt.show()

                # # > visualize frame and keypoints at patch level
                plt.imshow(cv2.resize(rrc_frame, (patch_num, patch_num)))
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
                    int(_v[0]) * patch_num + int(_v[1])
                    for _v in v
                ]
                for k, v in rrc_semantic_locations.items()
            }

            ######### extract video features ###########
            for l in range(layer_num):
                for s in subjects:
                    for p, loc in rrc_semantic_locations.items():
                        location_embeds = features[l][s][:, :, loc].flatten(0, -2).cpu()
                        location_embeds /= location_embeds.norm(dim=-1, keepdim=True)
                        semantic_patches[s][p][l]["embed"] += location_embeds.sum(dim=0)
                        semantic_patches[s][p][l]["count"] += location_embeds.shape[0]

    semantic_patches = {
        s: {
            p: [

                (
                    semantic_patches[s][p][l]["embed"] /
                    semantic_patches[s][p][l]["count"]
                )
                for l in range(layer_num)
            ]
            for p in landmarker.LANDMARKS.keys()
        }
        for s in subjects
    }

    if save_path != None:
        if (len(save_path) > 0):
            with open(save_path, "wb") as f:
                pickle.dump(semantic_patches, f)

    return semantic_patches


SAMPLE_NUM = 2000
# %%
if __name__ == "__main__":
    encoder = VideoAttrExtractor(
        architecture="ViT-B/16",
        text_embed=False,
        store_attrs=["q", "k", "v", "out", "emb"]
    )

    encoder.eval()
    encoder.to("cuda")

    fetch_semantic_features(
        encoder, df_types=["REAL"],
        sample_num=SAMPLE_NUM,
        visualize=False,
        # save_path=None,
        save_path=f"./misc/B16_real_semantic_patches_v4_{SAMPLE_NUM}.pickle",
        seed=1019
    )

    # fetch_semantic_features(alb.RandomResizedCrop(
    #     n_px, n_px,
    #     scale=(0.5, 0.7), ratio=(1, 1),
    #     p=1.0,
    # )
    # )

# %%
