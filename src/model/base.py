import torch.nn as nn
import lightning.pytorch as pl

from torch import optim
from functools import partial
from torchmetrics.classification import AUROC, Accuracy


class ODClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.log = partial(self.log, add_dataloader_idx=False)
        self.model = None

    @property
    def transform(self):
        raise NotImplementedError()

    @property
    def n_px(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        results = [self.shared_step(batch[dts_name], 'train') for dts_name in batch]
        return sum([_results['loss'] for _results in results])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'test')
        return result['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        result = self.shared_step(batch, 'valid')
        return result['loss']

    def shared_step(self, batch, stage):
        x, y, mask, indices, dts_name = *batch[:3], *batch[-2:]
        x = self.model(x, mask)
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
            "dts_name": dts_name,
            "indices": indices
        }

    def evaluate(self, batch, pack=True):
        return self.shared_step(batch, 'eval')

    # TODO: isolate optimizer and lr_scheduler configurations
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


class ODBinaryMetricClassifier(ODClassifier):
    def __init__(self):
        super().__init__()
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
        logits = result['logits'].detach().cpu().softmax(dim=-1)
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
