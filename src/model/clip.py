import torch
import torch.nn as nn
import src.clip as CLIP

from functools import partial
from src.model.base import ODClassifier
from torchmetrics.classification import AUROC, Accuracy


class LinearProbe(nn.Module):
    def __init__(self, architecture: str = "ViT-B/16", output_dim: int = 2):
        super().__init__()
        self.model, self.transform = CLIP.load(architecture)
        self.model = self.model.float()
        self.linear = nn.Linear(self.model.visual.output_dim, output_dim, bias=True)
        # disable gradients
        for params in self.model.parameters():
            params.requires_grad_(False)

    def forward(self, x):
        if len(x.shape) > 4:
            assert x.shape[1] == 1
            x = x.squeeze(1)
        return self.linear(self.model.encode_image(x))

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
        return self


class CLIPLinearProbe(ODClassifier):
    def __init__(self, output_dim=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = LinearProbe(output_dim=output_dim)

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.model.visual.input_resolution

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, 'train')
        return result['loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'test')
        return result['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'valid')
        return result['loss']

    def shared_step(self, batch, stage):
        x, y, dts_name = *batch[:2], batch[-1]
        x = self.model(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log(
            f"{stage}/{dts_name}/loss",
            loss,
            batch_size=x.shape[0]
        )
        return {
            "logits": x,
            "labels": y,
            "loss": loss,
            "dts_name": dts_name
        }


class CLIPBinaryLinearProb(CLIPLinearProbe):
    def __init__(self):
        super().__init__(2)
        self.dts_metrics = {}
        self.metric_map = {
            "auc": partial(AUROC, task="BINARY", num_classes=2),
            "acc": partial(Accuracy, task="BINARY", num_classes=2)
        }

    def get_metric(self, dts_name, metric_name):
        if (not dts_name in self.dts_metrics):
            self.dts_metrics[dts_name] = {}
        if (not metric_name in self.dts_metrics[dts_name]):
            self.dts_metrics[dts_name][metric_name] = self.metric_map[metric_name]()
        return self.dts_metrics[dts_name][metric_name]

    def reset_metrics(self):
        self.dts_metrics.clear()

    # shared procedures
    def shared_metric_update_procedure(self, result):
        # save metrics
        logits = result['logits'].detach().cpu()
        labels = result['labels'].detach().cpu()
        self.get_metric(result['dts_name'], 'auc').update(logits[:, 1], labels)
        self.get_metric(result['dts_name'], 'acc').update(logits[:, 1], labels)

    def shared_beg_epoch_procedure(self, phase):
        self.reset_metrics()

    def shared_end_epoch_procedure(self, phase):
        self.log_dict(
            {
                f'{phase}/{dts_name}/{metric_name}': metric_obj.compute()
                for dts_name, metrics in self.dts_metrics.items()
                for metric_name, metric_obj in metrics.items()
            }
        )
        self.reset_metrics()

    # validation
    def on_validation_epoch_start(self) -> None:
        self.shared_beg_epoch_procedure('valid')

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'valid')
        self.shared_metric_update_procedure(result)
        return result['loss']

    def on_validation_epoch_end(self) -> None:
        self.shared_end_epoch_procedure('valid')

    # test
    def on_test_epoch_start(self) -> None:
        self.shared_beg_epoch_procedure('test')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'test')
        self.shared_metric_update_procedure(result)
        return result['loss']

    def on_test_epoch_end(self) -> None:
        self.shared_end_epoch_procedure('test')
