import os
import yaml
import json
import torch
import pickle
import logging
import warnings
import argparse

from os import path
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch.cli import LightningCLI
from torchmetrics.classification import AUROC, Accuracy


class StatsRecorder:
    def __init__(self, label):
        self.label = label
        self.prob = 0
        self.count = 0

    def update(self, prob, label):
        assert label == self.label
        self.prob += prob
        self.count += 1

    def compute(self):
        return {
            "label": self.label,
            "prob": self.prob / self.count
        }


def configure_logging():
    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
    logging.basicConfig(level="INFO", format=logging_fmt)
    warnings.filterwarnings(action="ignore")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg_path", type=str)
    parser.add_argument("data_cfg_path", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(args=args)


@torch.inference_mode()
def main(args=None):
    params = parse_args(args=args)
    device = params.device
    if params.precision == 32:
        dtype = torch.float32
    elif params.precision == 16:
        dtype = torch.float16
    else:
        raise NotImplementedError()

    configure_logging()

    with open(params.model_cfg_path) as f:
        config = yaml.safe_load(f)
        model_config = {k: config[k] for k in ["model"]}

    with open(params.data_cfg_path) as f:
        config = yaml.safe_load(f)
        data_config = {k: config[k] for k in ["data"]}

    cli = LightningCLI(
        run=False,
        save_config_callback=None,
        parser_kwargs={
            "parser_mode": "omegaconf"
        },
        trainer_defaults={
            "logger": False
        },
        auto_configure_optimizers=False,
        seed_everything_default=1019,
        args={
            **model_config,
            **data_config,
        },

    )

    # setup model
    model = cli.model
    model = model.__class__.load_from_checkpoint(params.ckpt_path, strict=False)
    model = model.to(device)
    model.eval()

    # setup dataset
    datamodule = cli.datamodule
    datamodule.prepare_data()
    datamodule.affine_model(model)
    datamodule.setup('test')

    stats = {}
    report = {}

    test_dataloaders = datamodule.test_dataloader()

    for dts_name, dataloader in test_dataloaders.items():
        # iterate all videos
        auc_calc = AUROC(task="BINARY", num_classes=2)
        acc_calc = Accuracy(task="BINARY", num_classes=2)
        dataset = dataloader.dataset
        dts_stats = {}
        for batch in tqdm(dataloader):
            x, y, z = batch["xyz"]
            names = batch["names"]
            x = x.to(device)
            z = {
                _k: z[_k].to(device)
                for _k in z
            }
            y = y.tolist()
            with torch.autocast(device_type=device, dtype=dtype):
                results = model.evaluate(
                    x,
                    **z
                )
            probs = results["logits"].softmax(dim=-1)[:, 1].detach().cpu().tolist()

            for prob, label, name in zip(probs, y, names):
                if (not name in dts_stats):
                    dts_stats[name] = StatsRecorder(label)
                dts_stats[name].update(prob, label)

        # compute the average probability.
        for k in dts_stats:
            dts_stats[k] = dts_stats[k].compute()

        # add straying videos into metric calculation
        for k, v in dataset.stray_videos.items():
            dts_stats[k] = dict(
                label=v,
                prob=0.5,
                stray=1
            )

        # compute the metric scores
        dataset_labels = []
        dataset_probs = []
        for v in dts_stats.values():
            dataset_labels.append(v["label"])
            dataset_probs.append(v["prob"])
        dataset_labels = torch.tensor(dataset_labels)
        dataset_probs = torch.tensor(dataset_probs)
        accuracy = acc_calc(dataset_probs, dataset_labels).item()
        roc_auc = auc_calc(dataset_probs, dataset_labels).item()
        accuracy = round(accuracy, 3)
        roc_auc = round(roc_auc, 3)
        logging.info(f'[{dts_name}] accuracy: {accuracy}, roc_auc: {roc_auc}')
        stats[dts_name] = dts_stats
        report[dts_name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }

    # save report and stats.
    root = path.split(params.model_cfg_path)[0]
    timestamp = datetime.utcnow().strftime("%m%dT%H%M")
    with open(path.join(root, f'report_{timestamp}.json'), "w") as f:
        json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

    with open(path.join(root, f'stats_{timestamp}.pickle'), "wb") as f:
        pickle.dump(stats, f)

    # TODO: send notification after completing the inference.

    return report


if __name__ == "__main__":
    main()
