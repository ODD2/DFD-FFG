import os
import wandb
import torch
import logging
import warnings
import lightning.pytorch as pl

from typing import Optional
from torch import optim, nn
from functools import partial


from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import EarlyStopping

from src.model.clip import CLIPLinearProbe
from src.dataset.base import ODDataModule
from src.dataset.ffpp import FFPPDataModule
from src.dataset.cdf import CDFDataModule
from src.dataset.dfdc import DFDCDataModule
from src.utility.builtin import ODTrainer, ODModelCheckpoint

torch.set_float32_matmul_precision('high')


class ODLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stop")
        parser.set_defaults(
            {
                "early_stop.patience": 10,
            }
        )
        parser.add_lightning_class_args(ODModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                'checkpoint.save_last': True,
                'checkpoint.save_top_k': 1,
            }
        )


def configure_logging():
    logging_fmt = "[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s"
    logging.basicConfig(level="INFO", format=logging_fmt)
    warnings.filterwarnings(action="ignore")

    # disable warnings from the xformers efficient attention module due to torch.user_deterministic_algorithms(True,warn_only=True)
    warnings.filterwarnings(
        action="ignore",
        message=".*efficient_attention_forward_cutlass.*",
        category=UserWarning
    )

    # logging.basicConfig(level="DEBUG", format=logging_fmt)


def cli_main():
    # logging configuration
    configure_logging()

    # initialize cli
    cli = ODLightningCLI(
        run=False,
        trainer_class=ODTrainer,
        save_config_kwargs={
            'config_filename': 'setting.yaml'
        },
        auto_configure_optimizers=False,
        seed_everything_default=1019
    )

    # monitor model gradient and parameter histograms
    cli.trainer.logger.experiment.watch(cli.model, log='all', log_graph=False)

    # Load datasets
    cli.datamodule.affine_model(cli.model)

    # run
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, verbose=False)

    # save the running config
    cli.trainer.logger.experiment.save(
        glob_str=os.path.join(cli.trainer.log_dir, 'setting.yaml'),
        base_path=cli.trainer.log_dir,
        policy="now"
    )

    # ending
    cli.trainer.logger.experiment.unwatch(cli.model)
    cli.trainer.logger.experiment.finish()


if __name__ == "__main__":
    cli_main()
