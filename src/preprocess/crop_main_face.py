import os
import cv2
import pickle
import argparse
import numpy as np
import torchvision
from glob import glob
from tqdm import tqdm
from pathlib import Path
from os.path import exists


def load_args():
    parser = argparse.ArgumentParser(
        description='Pre-processing'
    )
    parser.add_argument(
        '--root-dir', default=None, help='video directory'
    )
    parser.add_argument(
        '--mean-face', default='./misc/20words_mean_face.npy', help='mean face path'
    )
    parser.add_argument(
        '--crop-width', default=150, type=int, help='width of face crop'
    )
    parser.add_argument(
        '--crop-height', default=150, type=int, help='height of face crop'
    )
    parser.add_argument(
        '--start-idx', default=15, type=int, help='start of landmark frame_idx'
    )
    parser.add_argument(
        '--stop-idx', default=68, type=int, help='end of landmark frame_idx'
    )
    parser.add_argument(
        '--window-margin', default=12, type=int, help='window margin for smoothed landmarks'
    )
    parser.add_argument(
        '--video-dir', default="videos", type=str, help='video folder'
    )
    parser.add_argument(
        '--fdata-dir', default="frame_data", type=str, help='frame data folder'
    )
    parser.add_argument(
        '--glob-exp', default="*/*", type=str, help='additional glob expressions.'
    )
    parser.add_argument(
        '--crop-dir', default="cropped", type=str, help="folder destination to save the process results."
    )
    parser.add_argument(
        '--replace', action="store_true", default=False
    )
    args = parser.parse_args()

    return args


class FaceData:
    def __init__(self, _lm, _bbox, _idx):
        self.ema_lm = _lm  # shape = (68, 2)
        self.ema_bbox = _bbox  # shape = (2, 2)
        self.lm = [_lm]
        self.bbox = [_bbox]
        self.idx = [_idx]

    def last_landmark(self):
        return self.ema_lm

    def last_bbox(self):
        return self.ema_bbox

    def d_lm(self, landmarks):
        return np.sum(
            np.linalg.norm(landmarks - self.last_landmark(), axis=-1),
            axis=1
        ) / landmarks.shape[1]

    def d_bbox(self, bboxes):
        return np.sum(
            np.linalg.norm(bboxes - self.last_bbox(), axis=-1),
            axis=1
        ) / bboxes.shape[1]

    def add(self, _lm, _bbox, _idx):
        self.ema_lm = self.ema_lm * 0.5 + _lm * 0.5
        self.ema_bbox = self.ema_bbox * 0.5 + _bbox * 0.5
        self.lm.append(_lm)
        self.bbox.append(_bbox)
        self.idx.append(_idx)

    def __len__(self):
        return len(self.lm)


def get_main_face_data(frame_landmarks, frame_bboxes):
    # post-process the extracted frame faces.
    # create face identity database to track landmark motion.
    face_dbs = []
    num_frames = len(frame_landmarks)
    for frame_idx, landmarks, bboxes in zip(range(num_frames), frame_landmarks, frame_bboxes):

        if (
            landmarks == None or len(landmarks) == 0 or
            bboxes == None or len(bboxes) == 0
        ):
            continue
        assert len(landmarks) == len(bboxes)
        num_faces = len(landmarks)
        landmarks = np.stack(landmarks)
        bboxes = np.stack(bboxes)

        matched_indices = {}

        # find and connect with the closest face in the database.
        for db_idx, db_face in enumerate(face_dbs):
            # face landmark and bbox motion distance.
            d = db_face.d_bbox(bboxes) + db_face.d_lm(landmarks)

            # the motion continues if the landmark motion distance is lower than 100.
            if (np.min(d) > 100):
                continue
            # get the closest face in the database.
            closest_idx = np.argmin(d)
            proximity = d[closest_idx]

            if (
                (not closest_idx in matched_indices) or
                (matched_indices[closest_idx]["d"] > proximity)
            ):
                matched_indices[closest_idx] = dict(d=proximity, db_idx=db_idx)

        # finalize and update the database entity.
        for face_idx, save_data in matched_indices.items():
            face_dbs[save_data["db_idx"]].add(landmarks[face_idx], bboxes[face_idx], frame_idx)

        # create new database entity for untracked landmarks.
        for face_idx, landmark, bbox in zip(range(num_faces), landmarks, bboxes):
            if face_idx in matched_indices:
                continue
            else:
                face_dbs.append(FaceData(landmark, bbox, frame_idx))

    # report only the most consistant face in the video.
    main_face = sorted(face_dbs, key=lambda x: len(x), reverse=True)[0]

    return main_face.lm, main_face.bbox


def save_video(
    filename,
    frames,
    fps
):
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        writer.write(frame)
    writer.release()  # close the writer


def affine_transform(
    frame,
    bboxes,
    landmarks,
    reference,
    target_size=(256, 256),
    reference_size=(256, 256),
    stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0
):
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
    stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

    # Warp the face patch and the landmarks
    transform = cv2.estimateAffinePartial2D(
        np.vstack([landmarks[x] for x in stable_points]),
        stable_reference, method=cv2.LMEDS
    )[0]

    transformed_frame = cv2.warpAffine(
        frame,
        transform,
        dsize=(target_size[0], target_size[1]),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value
    )
    transformed_landmarks = np.matmul(
        landmarks,
        transform[:, :2].transpose()
    ) + transform[:, 2].transpose()

    transformed_bboxes = np.matmul(
        bboxes,
        transform[:, :2].transpose()
    ) + transform[:, 2].transpose()

    return transformed_frame, transformed_landmarks, transformed_bboxes


def crop_driver(
    img,
    bboxes,
    landmarks,
    height,
    width
):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height + 1
    elif (center_y + height) > img.shape[0]:
        center_y = img.shape[0] - height - 1

    if center_x - width < 0:
        center_x = width + 1
    elif (center_x + width) > img.shape[1]:
        center_x = img.shape[1] - width - 1

    uy, by = int(center_y - height), int(center_y + height)
    lx, rx = int(center_x - width), int(center_x + width)
    cutted_img = np.copy(img[uy:by, lx:rx])
    cutted_landmarks = np.copy(landmarks) - [lx, uy]
    cutted_bboxes = np.copy(bboxes) - [lx, uy]

    return cutted_img, cutted_landmarks, cutted_bboxes


def crop_patch(
    frames,
    landmarks,
    bboxes,
    reference,
    window_margin,
    start_idx,
    stop_idx,
    crop_height,
    crop_width
):
    crop_frames = []
    crop_bboxes = []
    crop_landmarks = []

    assert len(frames) == len(landmarks) == len(bboxes)

    length = len(frames)

    for frame_idx in range(length):
        frame = frames[frame_idx]
        margin = min(window_margin // 2, frame_idx, length - 1 - frame_idx)
        # smoothed landmarks
        smoothed_landmarks = np.mean(
            landmarks[frame_idx - margin: frame_idx + margin + 1],
            axis=0
        )
        smoothed_landmarks += (landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0))
        # smoothed bboxes
        smoothed_bboxes = np.mean(
            bboxes[frame_idx - margin: frame_idx + margin + 1],
            axis=0
        )
        smoothed_bboxes += (bboxes[frame_idx].mean(axis=0) - smoothed_bboxes.mean(axis=0))
        # affine transform
        transformed_frame, transformed_landmarks, transformed_bboxes = affine_transform(
            frame,
            smoothed_bboxes,
            smoothed_landmarks,
            reference,
        )
        crop_frame, crop_landmark, crop_bbox = crop_driver(
            transformed_frame,
            transformed_bboxes,
            transformed_landmarks[start_idx:stop_idx],
            crop_height // 2,
            crop_width // 2
        )
        crop_frames.append(crop_frame)
        crop_landmarks.append(crop_landmark)
        crop_bboxes.append(crop_bbox)

    # convert to numpy array for better extensibility.
    crop_frames = np.array(crop_frames)
    crop_landmarks = np.array(crop_landmarks)
    crop_bboxes = np.array(crop_bboxes)

    return crop_frames, crop_landmarks, crop_bboxes


def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()
    return fps, frames


def get_video_frame_data(fdata_path):
    with open(fdata_path, "rb") as f:
        frame_data = pickle.load(f)
        frame_landmarks = [frame["landmarks"] for frame in frame_data]
        frame_bboxes = [frame["bboxes"] for frame in frame_data]

    assert len(frame_landmarks) == len(frame_bboxes)

    if (len(frame_landmarks[0][0]) == 98):
        _98_to_68_mapping = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44,
            45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76,
            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
            89, 90, 91, 92, 93, 94, 95
        ]

        frame_landmarks = [[lm[_98_to_68_mapping] for lm in landmarks]for landmarks in frame_landmarks]

    assert len(frame_landmarks[0][0]) == 68, "landmark should be 68 points."

    if (len(frame_bboxes[0][0].shape) == 1):
        frame_bboxes = [[bbox.reshape((2, 2)) for bbox in bboxes]for bboxes in frame_bboxes]

    return frame_landmarks, frame_bboxes


def main():
    args = load_args()
    reference = np.load(args.mean_face)

    video_root = os.path.join(args.root_dir, args.video_dir)
    fdata_root = os.path.join(args.root_dir, args.fdata_dir)
    crop_root = os.path.join(args.root_dir, args.crop_dir)

    video_files = sorted(glob(os.path.join(video_root, args.glob_exp)))

    for video_path in tqdm(video_files):
        try:
            rel_video_path = os.path.splitext(os.path.relpath(video_path, video_root))[0]
            fdata_path = os.path.join(fdata_root, rel_video_path) + ".pickle"
            crop_video_path = os.path.join(crop_root, args.video_dir, rel_video_path) + ".avi"
            crop_fdata_path = os.path.join(crop_root, args.fdata_dir, rel_video_path) + ".pickle"

            if (exists(f"{crop_video_path}") and exists(f"{crop_fdata_path}") and not args.replace):
                continue

            fps, frames = get_video_frames(video_path)

            frame_landmarks, frame_bboxes = get_video_frame_data(fdata_path)

            assert len(frames) == len(frame_landmarks) == len(frame_bboxes)

            landmarks, bboxes = get_main_face_data(
                frame_landmarks=frame_landmarks,
                frame_bboxes=frame_bboxes
            )

            if (not len(landmarks) == len(frames)):
                raise Exception("landmark and frame count mismatch.")

            crop_frames, crop_landmarks, crop_bboxes = crop_patch(
                frames,
                landmarks,
                bboxes,
                reference,
                window_margin=args.window_margin,
                start_idx=args.start_idx,
                stop_idx=args.stop_idx,
                crop_height=args.crop_height,
                crop_width=args.crop_width
            )

            # save video
            os.makedirs(os.path.dirname(crop_video_path), exist_ok=True)

            save_video(crop_video_path, crop_frames, fps)

            # save frame data
            os.makedirs(os.path.dirname(crop_fdata_path), exist_ok=True)

            with open(crop_fdata_path, "wb") as f:
                crop_bboxes = crop_bboxes.reshape(-1, 2, 2)
                pickle.dump(
                    [
                        dict(landmarks=[landmarks], bboxes=[bboxes])
                        for landmarks, bboxes in zip(crop_landmarks, crop_bboxes)
                    ],
                    f
                )
        except Exception as e:
            print(video_path, e)


if __name__ == "__main__":
    main()
