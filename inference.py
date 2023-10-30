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
from main import configure_logging
from torchmetrics.classification import AUROC, Accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_cfg_path", type=str)
    parser.add_argument("data_cfg_path", type=str)
    parser.add_argument("ckpt_path", type=str)
    return parser.parse_args()


N = 8


@torch.inference_mode()
def main(params):
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
    model.eval()
    device = model.device

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
        stats[dts_name] = {"label": [], "prob": [], "meta": []}
        for batch in tqdm(dataloader):
            x, y, z = batch["xyz"]
            indices = batch["indices"]
            x = x.to(device)
            z = {
                _k: z[_k].to(device)
                for _k in z
            }

            probs = []
            for beg in range(0, x.shape[0], N):
                results = model.evaluate(
                    x[beg:beg + N],
                    **{
                        _k: z[_k][beg:beg + N]
                        for _k in z
                    }
                )
                probs.append(results["logits"].softmax(dim=-1).detach().cpu())
            prob = torch.cat(probs).mean(dim=0)
            label = y[0]
            index = indices[0]

            # statistic recordings
            stats[dts_name]["label"].append(label.item())
            stats[dts_name]["prob"].append(prob[1].item())
            stats[dts_name]["meta"].append(dataset.video_meta(index))

            acc_calc.update(prob[1].unsqueeze(0), label.unsqueeze(0))
            auc_calc.update(prob[1].unsqueeze(0), label.unsqueeze(0))

        # add straying videos into metric calculation
        stray_names = list(dataset.stray_videos.keys())
        stray_labels = [dataset.stray_videos[name] for name in stray_names]
        stray_preds = [0 if i > 0.5 else 1 for i in stray_labels]
        stray_probs = [0.5] * len(stray_labels)
        stats[dts_name]["label"] += stray_preds
        stats[dts_name]["prob"] += stray_probs
        stats[dts_name]["meta"] += [{"path": name, "stray": 1} for name in stray_names]
        acc_calc.update(torch.tensor(stray_probs), torch.tensor(stray_labels))
        auc_calc.update(torch.tensor(stray_probs), torch.tensor(stray_labels))

        accuracy = round(acc_calc.compute().item(), 3)
        roc_auc = round(auc_calc.compute().item(), 3)
        logging.info(f'[{dts_name}] accuracy: {accuracy}, roc_auc: {roc_auc}')
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


if __name__ == "__main__":
    main(parse_args())
