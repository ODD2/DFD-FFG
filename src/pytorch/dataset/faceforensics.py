from time import time
import logging
from .base import *


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, ratio_list, always_apply=False, p=0.5):
        super(RandomDownScale, self).__init__(always_apply, p)
        self.ratio_list = ratio_list

    def apply(self, image, scale=1.0, **params):
        return self.randomdownscale(image, scale)

    def randomdownscale(self, img, scale, **params):
        keep_input_shape = True
        H, W, C = img.shape
        img_ds = cv2.resize(
            img,
            (int(W / scale), int(H / scale)),
            interpolation=cv2.INTER_CUBIC
        )
        logging.debug(f"Downscale Ratio: {scale}")
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_ds

    def get_params(self):
        return {
            "scale": np.random.randint(self.ratio_list[0], self.ratio_list[1] + 1)
        }

    def get_transform_init_args_names(self):
        return ("ratio_list",)


class FFPPSampleStrategy(IntEnum):
    NORMAL = auto()
    CONTRAST_RAND = auto()
    QUALITY_PAIR = auto()
    CONTRAST_PAIR = auto()


class FFPPAugmentation(IntFlag):
    NONE = auto()
    DEV = auto()
    NORMAL = auto()
    ROBUSTNESS = auto()
    PERTURBATION = auto()
    # > NORMAL
    VIDEO = auto()
    VIDEO_RRC = auto()
    FRAME = auto()
    FRAME_NOISE = auto()
    # > DEV
    RGB = auto()
    HUE = auto()
    BRIGHT = auto()
    COMP = auto()
    DSCALE = auto()
    SHARPEN = auto()


class DeepFakeDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 24,
            num_workers: int = 8,
            vid_ext: str = ".avi",
            clip_duration: int = 4,
            num_frames: int = 10,
            pack: bool = False
    ):
        super().__init__()
        # generic parameters
        self.transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # dataset metadata
        self.data_dir = data_dir
        self.vid_ext = vid_ext
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.pack = pack

        # dataset splits
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._predict_dataset = None

    def create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )

    def affine_model(self, model):
        self.transform = model.transform

    def prepare_data(self):
        raise Exception("Implementation Required")

    def setup(self, stage: str):
        raise Exception("Implementation Required")

    def train_dataloader(self):
        return self.create_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self._val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self._test_dataset)

    def predict_dataloader(self):
        return self.create_dataloader(self._predict_dataset)


class FFPP(Dataset):
    TYPE_DIRS = {
        'REAL': 'real/',
        'DF': 'DF/',
        'FS': 'FS/',
        'F2F': 'F2F/',
        'NT': 'NT/'
    }

    COMPRESSIONS = {'c23', 'raw'}

    @classmethod
    def get_cache_dir(cls, df_type, comp):
        return path.expanduser(
            f'./.cache/{cls.__name__}-{df_type}-{comp}.pkl'
        )

    def __init__(
        self,
        data_dir: str,
        vid_ext: str,
        df_types: List[str],
        compressions: List[str],
        num_frames: int,
        clip_duration: int,
        transform: Optional[Callable],
        n_px: int,
        split: str,
        pack: bool,
        strategy: FFPPSampleStrategy,
        augmentation: FFPPAugmentation,
        force_random_speed: Optional[bool] = None
    ):
        # configurations
        self.data_dir = data_dir
        self.vid_ext = vid_ext
        self.df_types = df_types
        self.compressions = compressions
        self.num_frames = num_frames
        self.clip_duration = clip_duration
        self.transform = transform
        self.n_px = n_px
        self.split = split
        self.pack = pack
        self.strategy = strategy
        self.augmentation = augmentation
        self.train = (True if split == "train" else False)
        self.random_speed = (
            force_random_speed
            if not force_random_speed == None else
            (
                True
                if self.train else
                False
            )
        )

        # record missing videos in the csv file for further usage.
        self.stray_videos = {}

        # stacking data clips
        self.stack_video_clips = []

        # build video metadata structure for fast retreival
        self._build_video_table()
        self._build_video_list()

        # augmentation selections
        logging.debug(f"Augmentations: {self.augmentation}")
        if FFPPAugmentation.NONE in self.augmentation:
            self.frame_augmentation = None
            self.video_augmentation = None

        elif FFPPAugmentation.DEV in self.augmentation:
            self.frame_augmentation = None

            if FFPPAugmentation.RGB in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=1.)
                    ],
                    p=1.
                )
            elif FFPPAugmentation.HUE in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3),
                            sat_shift_limit=(-0.3, 0.3),
                            val_shift_limit=(-0.3, 0.3),
                            p=1.
                        ),
                    ],
                    p=1.
                )
            elif FFPPAugmentation.BRIGHT in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.RandomBrightnessContrast(
                            brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.
                        ),
                    ],
                    p=1.
                )
            elif FFPPAugmentation.COMP in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.ImageCompression(
                            quality_lower=40, quality_upper=100, p=1.
                        )
                    ],
                    p=1.
                )
            elif FFPPAugmentation.DSCALE in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        RandomDownScale((2, 3), p=1)
                    ],
                    p=1.
                )
            elif FFPPAugmentation.SHARPEN in self.augmentation:
                self.video_augmentation = alb.ReplayCompose(
                    [
                        alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
                    ],
                    p=1.
                )
            else:
                raise NotImplementedError()

        elif FFPPAugmentation.ROBUSTNESS in self.augmentation:
            self.frame_augmentation = None
            self.video_augmentation = alb.ReplayCompose(
                [
                    alb.Resize(
                        self.n_px, self.n_px, cv2.INTER_CUBIC
                    ),
                    alb.RandomResizedCrop(
                        self.n_px, self.n_px, scale=(0.7, 0.9), ratio=(1, 1)
                    ),
                    alb.HorizontalFlip(),
                    alb.ToGray()
                ],
                p=1.
            )

        elif FFPPAugmentation.PERTURBATION in self.augmentation:
            augmentations = [
                alb.Resize(
                    self.n_px, self.n_px, cv2.INTER_CUBIC
                ),
                alb.RGBShift(
                    (-20, 20), (-20, 20), (-20, 20), p=0.7
                ),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.7
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.7
                ),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.7
                ),
                alb.OneOf([
                    RandomDownScale((2, 3), p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ], p=0.5)
            ]
            self.video_augmentation = alb.ReplayCompose(
                augmentations,
                p=1.
            )

        elif FFPPAugmentation.NORMAL in self.augmentation:
            self.frame_augmentation = None
            self.video_augmentation = None

            if FFPPAugmentation.VIDEO in self.augmentation:
                augmentations = [
                    alb.Resize(
                        self.n_px, self.n_px, cv2.INTER_CUBIC
                    ),
                    alb.RGBShift(
                        (-20, 20), (-20, 20), (-20, 20), p=0.3
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3
                    ),
                    alb.ImageCompression(
                        quality_lower=40, quality_upper=100, p=0.5
                    ),
                    alb.HorizontalFlip()
                ]

                if FFPPAugmentation.VIDEO_RRC in self.augmentation:
                    augmentations += [
                        alb.RandomResizedCrop(
                            self.n_px, self.n_px, scale=(0.7, 0.9), ratio=(1, 1), p=0.3
                        ),
                        alb.Compose(
                            [
                                alb.RandomScale(
                                    (-0.5, -0.1), always_apply=True
                                ),
                                alb.Resize(
                                    self.n_px, self.n_px, cv2.INTER_CUBIC, always_apply=True
                                )
                            ], p=0.3
                        )
                    ]
                self.video_augmentation = alb.ReplayCompose(
                    augmentations,
                    p=1.
                )

            if FFPPAugmentation.FRAME in self.augmentation:
                augmentations = [
                    alb.Resize(
                        self.n_px, self.n_px, cv2.INTER_CUBIC
                    ),
                    alb.RGBShift(
                        (-5, 5), (-5, 5), (-5, 5), p=0.3
                    ),
                    alb.HueSaturationValue(
                        hue_shift_limit=(-0.05, 0.05), sat_shift_limit=(-0.05, 0.05), val_shift_limit=(-0.05, 0.05), p=0.3
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.3
                    ),
                    alb.ImageCompression(
                        quality_lower=80, quality_upper=100, p=0.5
                    ),
                ]
                if FFPPAugmentation.FRAME_NOISE in self.augmentation:
                    augmentations += [
                        alb.OneOf(
                            [
                                alb.GaussNoise(
                                    per_channel=True,
                                    p=1.0
                                ),
                                alb.MultiplicativeNoise(
                                    per_channel=False,
                                    elementwise=True,
                                    always_apply=False,
                                    p=1.0
                                ),
                            ],
                            p=0.3
                        )
                    ]

                self.frame_augmentation = alb.ReplayCompose(
                    augmentations,
                    p=1.0
                )

            if (self.frame_augmentation == None and self.video_augmentation == None):
                raise NotImplementedError()

        else:
            raise NotImplementedError()

        # construct augmentation driver
        if (self.frame_augmentation == None and self.frame_augmentation == None):
            def driver(x, replay=None):
                return x, replay

        else:
            def driver(x, replay=None):
                # transform to numpy, the alb required format
                x = [_x.numpy().transpose((1, 2, 0)) for _x in x]

                # initialize replay data
                if (replay == None):
                    replay = {}

                # frame augmentation
                if (not self.frame_augmentation == None):
                    if ("frame" in replay):
                        assert len(replay["frame"]) == len(x), "Error! frame replay should match the number of frames"
                        x = [
                            alb.ReplayCompose.replay(
                                _r,
                                image=_x
                            )["image"]
                            for _x, _r in zip(x, replay["frame"])
                        ]

                    else:
                        replay["frame"] = [None for _ in x]
                        for i, _x in enumerate(x):
                            result = self.frame_augmentation(image=_x)
                            x[i] = result["image"]
                            replay["frame"][i] = result["replay"]
                # sequence augmentation
                if (not self.video_augmentation == None):
                    if ("video" in replay):
                        x = [
                            alb.ReplayCompose.replay(
                                replay["video"],
                                image=_x
                            )["image"]
                            for _x in x
                        ]
                    else:
                        replay["video"] = self.video_augmentation(image=x[0])["replay"]
                        x = [
                            alb.ReplayCompose.replay(
                                replay["video"],
                                image=_x
                            )["image"]
                            for _x in x
                        ]
                # revert to tensor
                x = [torch.from_numpy(_x.transpose((2, 0, 1))) for _x in x]
                return x, replay

        self.augmentation = driver

        self.video_readers = dict()

    def _build_video_table(self):
        self.video_table = {}
        for df_type in self.df_types:
            self.video_table[df_type] = {}
            for comp in self.compressions:
                # compose cache directory path
                meta_cache_path = path.expanduser(FFPP.get_cache_dir(df_type, comp))

                # load metadatas
                with open(meta_cache_path, 'rb') as f:
                    video_metas = pickle.load(f)

                # post process the video path
                for idx in video_metas:
                    video_metas[idx]["path"] = path.join(self.data_dir, video_metas[idx]["path"]) + self.vid_ext

                # store in video table.
                self.video_table[df_type][comp] = video_metas

    def _build_video_list(self):
        self.video_list = []

        with open(path.join(self.data_dir, 'splits', f'{self.split}.json')) as f:
            idxs = json.load(f)

        logging.debug(f"DF TYPES:{self.df_types}")
        logging.debug(f"DF TYPES:{self.compressions}")

        for df_type in self.df_types:
            for comp in self.compressions:
                comp_videos = []
                adj_idxs = sorted(
                    [i for inner in idxs for i in inner]
                    if df_type == 'REAL' else
                    ['_'.join(idx) for idx in idxs] + ['_'.join(reversed(idx)) for idx in idxs]
                )

                for idx in adj_idxs:
                    if idx in self.video_table[df_type][comp]:
                        clips = int(self.video_table[df_type][comp][idx]["duration"] // self.clip_duration)
                        if (clips > 0):
                            comp_videos.append((df_type, comp, idx, clips))
                    else:
                        logging.warning(
                            f'Video {path.join(self.data_dir, self.TYPE_DIRS[df_type], comp, "videos", idx)} does not present in the processed dataset.'
                        )
                        self.stray_videos[idx] = (0 if df_type == "REAL" else 1)
                self.video_list += comp_videos

        # stacking up the amount of data clips for further usage
        self.stack_video_clips = [0]
        self.real_clip_idx = {}
        for df_type, _, idx, i in self.video_list:
            self.stack_video_clips.append(self.stack_video_clips[-1] + i)
            if df_type == "REAL":
                self.real_clip_idx[idx] = [self.stack_video_clips[-2], self.stack_video_clips[-1] - 1]
        self.stack_video_clips.pop(0)

    def __len__(self):
        if (self.pack):
            return len(self.video_list)
        else:
            return self.stack_video_clips[-1]

    def __getitem__(self, idx):
        desire_item_indices = []
        if self.strategy == FFPPSampleStrategy.NORMAL:
            desire_item_indices = [idx]
        elif self.strategy == FFPPSampleStrategy.CONTRAST_PAIR:
            raise NotImplementedError()
        elif self.strategy == FFPPSampleStrategy.CONTRAST_RAND:
            raise NotImplementedError()
        elif self.strategy == FFPPSampleStrategy.QUALITY_PAIR:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        item_dicts = [
            self.get_entity(desire_item_index)
            for desire_item_index in desire_item_indices
        ]

        return [[item_dict[k] for item_dict in item_dicts] for k in item_dicts[0].keys()]

    # The 'idx' here represents the entity index from the __getitem__.
    # Depending on self.pack, the entity index either indicates a clip or the video.
    def get_entity(self, idx, replay=None, with_entity_info=False):
        video_idx, df_type, comp, video_name, num_clips = self.video_info(idx)
        video_meta = self.video_table[df_type][comp][video_name]
        logging.debug(f"Entity/Video Index:{idx}/{video_idx}")
        logging.debug(f"Entity DF/COMP:{df_type}/{comp}")

        # - video path
        vid_path = video_meta["path"]
        # - create video reader
        vid_reader = VideoReader(vid_path, "video")
        # - frames per second
        video_sample_freq = vid_reader.get_metadata()["video"]["fps"][0]

        entity_clips = []
        entity_masks = []

        # desire all clips in the video under pack mode, else fetch only the clip of index.
        if (self.pack):
            clips_desire = range(num_clips)
        else:
            clips_desire = [idx - (0 if video_idx == 0 else self.stack_video_clips[video_idx - 1])]

        for clip_of_video in clips_desire:
            # video frame processing
            frames = []

            # derive the video offset
            video_offset_duration = clip_of_video * self.clip_duration

            # augment the data only while training.
            if (self.random_speed):
                # the slow motion factor for video data augmentation
                video_speed_factor = random.random() * 0.5 + 0.5
                video_shift_factor = random.random() * (1 - video_speed_factor)
            else:
                video_speed_factor = 1
                video_shift_factor = 0
            logging.debug(f"Video Speed Motion Factor: {video_speed_factor}")
            logging.debug(f"Video Shift Factor: {video_shift_factor}")

            # the amount of frames to skip
            video_sample_offset = int(
                video_offset_duration + self.clip_duration * video_shift_factor
            )
            # the amount of frames for the duration of a clip
            video_clip_samples = int(
                video_sample_freq * self.clip_duration * video_speed_factor
            )
            # the amount of frames to skip in order to meet the num_frames per clip.(excluding the head & tail frames )
            if (self.num_frames == 1):
                video_sample_stride = 0
            else:
                video_sample_stride = (
                    (video_clip_samples - 1) / (self.num_frames - 1)
                ) / video_sample_freq

            logging.debug(f"Loading Video: {vid_path}")
            logging.debug(f"Sample Offset: {video_sample_offset}")
            logging.debug(f"Sample Stride: {video_sample_stride}")

            # fetch frames of clip duration
            for sample_idx in range(self.num_frames):
                vid_reader.seek(video_sample_offset + sample_idx * video_sample_stride, keyframes_only=True)
                frame = next(vid_reader)
                frames.append(frame["data"])

            # augment the data only while training.
            frames, replay = self.augmentation(frames, replay)
            logging.debug("Augmentations Applied.")

            # stack list of torch frames to tensor
            frames = torch.stack(frames)

            # transformation
            if (self.transform):
                frames = self.transform(frames)

            # padding and masking missing frames.
            mask = torch.tensor(
                [1.] * len(frames) +
                [0.] * (self.num_frames - len(frames)),
                dtype=torch.bool
            )
            if frames.shape[0] < self.num_frames:
                diff = self.num_frames - len(frames)
                padding = torch.zeros(
                    (diff, *frames.shape[1:]),
                    dtype=frames.dtype
                )
                frames = torch.concatenate(
                    frames,
                    padding
                )

            entity_clips.append(frames)
            entity_masks.append(mask)
            logging.debug(
                "Video Clip: {}({}s~{}s), Completed!".format(
                    vid_path,
                    self.clip_duration*clip_of_video,
                    (self.clip_duration+1)*clip_of_video
                )
            )

        entity_clips = torch.stack(entity_clips)
        entity_masks = torch.stack(entity_masks)

        entity_info = {
            "comp": comp,
            "video_name": video_name,
            "df_type": df_type,
            "vid_path": vid_path
        }
        entity_data = {
            "clips": entity_clips,
            "label": 0 if (df_type == "REAL") else 1,
            "masks": entity_masks,
            "idx": idx
        }

        if with_entity_info:
            return {**entity_data, **entity_info}
        else:
            return entity_data

    def video_info(self, idx):
        if (self.pack):
            video_idx = idx
        else:
            video_idx = next(i for i, x in enumerate(self.stack_video_clips) if idx < x)
        return video_idx, *self.video_list[video_idx]

    def video_meta(self, idx):
        df_type, comp, name = self.video_info(idx)[1:4]
        return self.video_table[df_type][comp][name]

    def collate_fn(self, batch):
        _clips, _label, _masks, _index = list(zip(*batch))

        _clips = [i for l in _clips for i in l]
        _label = [i for l in _label for i in l]
        _masks = [i for l in _masks for i in l]
        _index = [i for l in _index for i in l]

        clips = torch.cat(_clips)
        masks = torch.cat(_masks)

        if self.pack:
            # post-process the label & index to match the shape of corresponding clips
            num_video_clips = torch.tensor([video_clips.shape[0] for video_clips in _clips])
            label = torch.tensor(_label).repeat_interleave(num_video_clips)
            index = torch.tensor(_index).repeat_interleave(num_video_clips)
        else:
            label = torch.tensor(_label)
            index = torch.tensor(_index)

        assert clips.shape[0] == masks.shape[0] == label.shape[0] == index.shape[0]

        return [clips, label, masks, index]


class FFPPDataModule(DeepFakeDataModule):
    def __init__(
            self,
            df_types: List[str] = [],
            compressions: List[str] = [],
            strategy: FFPPSampleStrategy = FFPPSampleStrategy.NORMAL,
            augmentations: FFPPAugmentation = FFPPAugmentation.NONE,
            force_random_speed: bool = None,
            *args, **kargs

    ):
        super().__init__(*args, **kargs)
        self.df_types = sorted(
            set([i for i in df_types if i in FFPP.TYPE_DIRS]),
            reverse=True
        )
        self.compressions = sorted(
            set([i for i in compressions if i in FFPP.COMPRESSIONS]),
            reverse=True
        )
        self.strategy = strategy
        self.augmentations = augmentations
        self.force_random_speed = force_random_speed

    def affine_model(self, model):
        super().affine_model(model)
        self.n_px = model.n_px

    def prepare_data(self):
        progress_bar = tqdm(FFPP.TYPE_DIRS.keys())
        for df_type in progress_bar:
            for comp in self.compressions:
                video_metas = {}

                # description for progress bar
                progress_bar.set_description(f"{df_type}: {comp}/videos")

                # compose the path for metadata cache
                meta_cache_path = path.expanduser(FFPP.get_cache_dir(df_type, comp))

                # next entity if cache exists
                if path.exists(meta_cache_path):
                    continue

                # video directory for df_type of compression
                video_dir = path.join(self.data_dir, FFPP.TYPE_DIRS[df_type], f'{comp}/videos')

                # build metadata
                for f in scandir(video_dir):
                    if self.vid_ext in f.name:
                        vid_reader = torchvision.io.VideoReader(
                            f.path,
                            "video"
                        )
                        try:
                            fps = vid_reader.get_metadata()["video"]["fps"][0]
                            duration = vid_reader.get_metadata()["video"]["duration"][0]
                            video_metas[f.name[:-len(self.vid_ext)]] = {
                                "fps": fps,
                                "frames": round(duration * fps),
                                "duration": duration,
                                "path": f.path[len(self.data_dir):-len(self.vid_ext)]
                            }
                        except:
                            logging.error(f"Error Occur During Video Table Creation: {f.path}")

                # cache the metadata
                makedirs(path.dirname(meta_cache_path), exist_ok=True)
                with open(meta_cache_path, 'wb') as f:
                    pickle.dump(video_metas, f)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        data_cls = partial(
            FFPP,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            df_types=self.df_types,
            compressions=self.compressions,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            n_px=self.n_px,
            force_random_speed=self.force_random_speed
        )

        if stage == "fit":
            self._train_dataset = data_cls(
                split="train",
                pack=self.pack,
                strategy=self.strategy,
                augmentation=self.augmentations
            )
        elif stage == "validate":
            self._val_dataset = data_cls(
                split="val",
                pack=self.pack,
                strategy=FFPPSampleStrategy.NORMAL,
                augmentation=FFPPAugmentation.NONE
            )

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self._test_dataset = data_cls(
                split="test",
                pack=self.pack,
                strategy=FFPPSampleStrategy.NORMAL,
                augmentation=FFPPAugmentation.NONE
            )


if __name__ == "__main__":
    from src.utility.od.visualize import dataset_entity_visualize

    class Dummy():
        pass

    dtm = FFPPDataModule(
        ["REAL", "DF", "FS", "F2F", "NT"],
        ["c23"],
        data_dir="datasets/ffpp/",
        batch_size=8,
        num_workers=0,
        force_random_speed=False,
        augmentations=(
            FFPPAugmentation.NORMAL |
            FFPPAugmentation.VIDEO |
            FFPPAugmentation.VIDEO_RRC |
            FFPPAugmentation.FRAME
        ),
        pack=True,
    )
    model = Dummy()
    model.n_px = 224
    model.transform = lambda x: x
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtm.setup("validate")
    dtm.setup("test")

    # iterate the whole dataset for visualization and sanity check
    # for split, iterable in {
    #     'train': dtm._train_dataset, 'val': dtm._val_dataset, 'test': dtm._test_dataset
    # }.items():
    #     save_folder = f"./misc/extern/dump_dataset/ffpp/{split}/"
    #     for entity_idx in tqdm(range(len(iterable))):
    #         if (entity_idx > 100):
    #             break
    #         dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    # iterate the all dataloaders for debugging.
    # for fn in [dtm.train_dataloader, dtm.val_dataloader, dtm.test_dataloader]:
    #     iterable = fn()
    #     for batch in tqdm(iterable):
    #         pass
