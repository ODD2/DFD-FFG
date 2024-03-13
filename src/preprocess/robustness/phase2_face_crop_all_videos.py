import os
import argparse
from glob import glob
from src.preprocess.crop_main_face import main as crop_entrance


def parse_args():
    parser = argparse.ArgumentParser(description='Add a distortion to video.')

    parser.add_argument("action", type=str)

    parser.add_argument(
        '--dts_root',
        type=str,
        default="/scratch1/users/od/FaceForensicC23/"
    )

    parser.add_argument(
        '--rob_dir',
        type=str,
        default="robustness"
    )

    parser.add_argument(
        '--fd_dir',
        type=str,
        default="frame_data"
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    rob_root = os.path.join(args.dts_root, args.rob_dir)
    fd_root = os.path.join(args.dts_root, args.fd_dir)

    dftypes = ["DF", "NT", "FS", "F2F", "real"]
    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC']
    level_list = ['1', '2', '3', '4', '5']

    if args.action == "setup":
        for t in type_list:
            for l in level_list:
                for df in dftypes:
                    p1 = os.path.join(fd_root, df)
                    p2 = os.path.join(fd_root, t, l)
                    os.makedirs(p2, exist_ok=True)
                    p2 = os.path.join(p2, df)
                    os.system(f"ln -s {p1} {p2}")
    elif args.action == "run":
        crop_entrance(
            [
                "--root-dir", args.dts_root,
                "--video-dir", args.rob_dir,
                "--mean-face", "./misc/20words_mean_face.npy",
                "--glob-exp", "*/*/*/*.mp4",
                "--crop-dir", "cropped_robust"
            ]
        )
    elif args.action == "clean":
        for t in type_list:
            p = os.path.join(fd_root, t)
            os.system(f"rm -rf {p}")
