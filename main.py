import os
import torch
import logging
import warnings
import lightning.pytorch as pl

from src.utility.builtin import ODTrainer,  ODLightningCLI

torch.set_float32_matmul_precision('high')


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


def configure_cli():
    return ODLightningCLI(
        run=False,
        trainer_class=ODTrainer,
        save_config_kwargs={
            'config_filename': 'setting.yaml'
        },
        auto_configure_optimizers=False,
        seed_everything_default=1019
    )


def cli_main():
    # logging configuration
    configure_logging()

    # initialize cli
    cli = configure_cli()

    # monitor model gradient and parameter histograms
    cli.trainer.logger.experiment.watch(cli.model, log='all', log_graph=False)

    # Load & configure datasets
    cli.datamodule.affine_model(cli.model)
    cli.datamodule.affine_trainer(cli.trainer)

    # run
    cli.trainer.fit(
        cli.model,
        datamodule=cli.datamodule
    )
    cli.trainer.test(
        cli.model,
        datamodule=cli.datamodule,
        verbose=False,
        ckpt_path="best"
    )

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
