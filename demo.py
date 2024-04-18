import os
import cv2
import sys
import yaml
import json
import math
import torch
import pickle
import shutil
import logging
import warnings
import argparse
import numpy as np


from os import path
from datetime import datetime
from torchmetrics.classification import AUROC, Accuracy
from src.utility.builtin import ODTrainer, ODLightningCLI
from torchvision.io import VideoReader

# TODO: this is a scratchy method to render a video with bounding boxes and prediction results.
# TODO: need to refactor for versatility and reusability.


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg_path", type=str)
    parser.add_argument("data_cfg_path", type=str)
    parser.add_argument("model_ckpt_path", type=str)
    parser.add_argument("video_path", type=str)
    parser.add_argument("--precision", type=str, default="16")
    return parser.parse_args(args=args)


def configure_logging():
    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
    logging.basicConfig(level="INFO", format=logging_fmt)
    warnings.filterwarnings(action="ignore")


@torch.inference_mode()
def demo_driver(cli, cfg_dir, ckpt_path, video_path):
    # setup model
    model = cli.model

    try:
        model = model.__class__.load_from_checkpoint(ckpt_path)
    except Exception as e:
        print(f"Unable to load model from checkpoint in strict mode: {e}")
        print(f"Loading model from checkpoint in non-strict mode.")
        model = model.__class__.load_from_checkpoint(ckpt_path, strict=False)

    model.eval()
    transforms = model.transform

    BATCH = 30
    stride = 0.333

    # load original video
    vid_reader = VideoReader(video_path, "video", num_threads=1)
    vid_ext = os.path.splitext(video_path)[-1]
    vid_name = os.path.split(video_path)[1].replace(vid_ext, "")
    fps = vid_reader.get_metadata()["video"]["fps"][0]

    frames = []
    for frame_data in vid_reader:
        frames.append(frame_data["data"])
    frames = torch.stack(frames)
    del vid_reader
    _, H, W = frames[0].shape

    # load bboxes of original video
    with open(video_path.replace("videos", "frame_data").replace(vid_ext, ".pickle"), "rb") as f:
        fdata = pickle.load(f)
        bboxes = []
        for data in fdata:
            data["bboxes"] = [
                bbox.reshape(2, -1)
                if len(bbox.shape) == 1 else bbox
                for bbox in data["bboxes"]
            ]
            face_idx = np.argsort([
                np.linalg.norm((bbox[0] - bbox[1])) for bbox in data["bboxes"]
            ])[-1]
            bboxes.append(data["bboxes"][face_idx])

    # load face cropped video
    vid_reader = VideoReader(
        video_path.replace("/videos", "/cropped/videos").replace(vid_ext, ".avi"),
        "video",
        num_threads=1
    )
    cropped_frames = []
    for frame_data in vid_reader:
        cropped_frames.append(frame_data["data"])
    cropped_frames = torch.stack(cropped_frames)
    del vid_reader

    # sample frames and inference
    indices = torch.tensor([int(math.floor(i * stride * fps)) for i in range(10)], dtype=torch.long)
    probs = []
    i = 0
    clip_count = len(cropped_frames) - indices[-1]
    while (i < clip_count):
        batch = min(clip_count - i, BATCH)
        clips = torch.stack([
            transforms(cropped_frames[indices + i + j]) for j in range(batch)
        ]).to("cuda")
        results = model.evaluate(clips)
        probs.extend(results["logits"].softmax(dim=-1)[:, 1].flatten().cpu().tolist())
        i += batch

    # draw and write to video
    bbox_frames = []
    for frame, bbox, prob in zip(frames[indices[-1]:], bboxes[indices[-1]:], probs):
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        thickness = int(np.linalg.norm(bbox[0] - bbox[1]) * 0.01)
        color = (0, 255, 0) if prob < 0.5 else (0, 0, 255)
        category = "REAL" if prob < 0.5 else "FAKE"
        frame = cv2.rectangle(
            frame,
            bbox[0].astype(int),
            bbox[1].astype(int),
            color,
            thickness
        )
        frame = cv2.putText(
            frame,
            f'{round(prob,2)}',
            [int(bbox[0][0]), int(bbox[1][1] - thickness)],
            cv2.FONT_HERSHEY_SIMPLEX,
            1, color, thickness, cv2.LINE_AA
        )

        frame = cv2.putText(
            frame,
            category,
            [int(bbox[0][0]), int(bbox[0][1] - thickness)],
            cv2.FONT_HERSHEY_SIMPLEX,
            1, color, thickness, cv2.LINE_AA
        )

        bbox_frames.append(frame)

    writer = cv2.VideoWriter(
        f'pred_{vid_name}.avi',
        cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
        fps,
        (W, H)
    )

    for frame in bbox_frames:
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    # stride = 0.333
    # video_path = "/scratch2/users/od/HeyGen/videos/D11949002 - Assignment W5 - Frontier of AIoT.mp4"
    # video_path = "/scratch2/users/od/HeyGen/cropped/videos/D11949002 - Assignment W5 - Frontier of AIoT.avi"

    configure_logging()

    params = parse_args()

    cli = ODLightningCLI(
        run=False,
        trainer_class=ODTrainer,
        save_config_callback=None,
        parser_kwargs={
            "parser_mode": "omegaconf"
        },
        auto_configure_optimizers=False,
        seed_everything_default=1019,
        args=[
            '-c', params.model_cfg_path,
            '-c', params.data_cfg_path,
            '--trainer.logger=null',
            f'--trainer.devices=1',
            f'--trainer.precision={params.precision}',
        ],
    )

    cfg_dir = os.path.split(params.model_cfg_path)[0]
    ckpt_path = params.model_ckpt_path
    video_path = params.video_path

    demo_driver(
        cli=cli,
        cfg_dir=cfg_dir,
        ckpt_path=ckpt_path,
        video_path=video_path
    )
