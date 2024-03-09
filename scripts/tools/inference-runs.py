import os
import argparse
from typing import List
from glob import glob


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", type=str, nargs='+')
    parser.add_argument("--log-dir", type=str, default="./logs/ECCV/")
    parser.add_argument("--cfg-path", type=str, default="./configs/generic/inference.yaml")
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--devices", type=int, default=-1)
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = parse_args()
    log_dir = args.log_dir
    cfg_path = args.cfg_path
    for run in args.ids:
        setting_path = os.path.join(log_dir, run, "setting.yaml")
        ckpt_paths = glob(os.path.join(log_dir, run, "checkpoints", "epoch*.ckpt"))
        assert len(ckpt_paths) == 1, "in the current setting, only save the best model"
        ckpt_path = ckpt_paths[0]
        result = os.system(
            f"python -m inference {setting_path} {cfg_path} {ckpt_path} --precision={args.precision} --devices={args.devices} --notes={run}"
        )

        if (not result == 0):
            print(f"failed run: {run}")
            break
        else:
            print(f"complete run: {run}")
