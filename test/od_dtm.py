from tqdm import tqdm
from src.dataset.base import AggregateDataModule
from src.dataset.ffpp import FFPPDataModule, FFPPAugmentation, FFPPSampleStrategy
from src.dataset.cdf import CDFDataModule


aggr = AggregateDataModule(
    [
        FFPPDataModule(
            ["REAL", "DF", "FS", "F2F", "NT"],
            ["c23"],
            data_dir="datasets/ffpp/",
            batch_size=24,
            num_workers=16,
            force_random_speed=False,
            strategy=FFPPSampleStrategy.NORMAL,
            augmentations=(
                FFPPAugmentation.NORMAL |
                FFPPAugmentation.VIDEO |
                FFPPAugmentation.VIDEO_RRC |
                FFPPAugmentation.FRAME
            ),
            pack=False
        )
    ],
    [
        FFPPDataModule(
            ["REAL", "DF", "FS", "F2F", "NT"],
            ["c23"],
            data_dir="datasets/ffpp/",
            batch_size=24,
            num_workers=16,
            force_random_speed=False,
            strategy=FFPPSampleStrategy.NORMAL,
            augmentations=(
                FFPPAugmentation.NORMAL |
                FFPPAugmentation.VIDEO |
                FFPPAugmentation.VIDEO_RRC |
                FFPPAugmentation.FRAME
            ),
            pack=False
        ),
        CDFDataModule(
            data_dir="datasets/cdf/",
            batch_size=24,
            num_workers=16,
            pack=True
        )
    ]
)


class Dummy():
    pass


model = Dummy()
model.n_px = 224
model.transform = lambda x: x
aggr.prepare_data()
aggr.affine_model(model)
# aggr.setup("fit")
aggr.setup("validate")


# for fn in [aggr.train_dataloader, aggr.val_dataloader, aggr.test_dataloader]:
for fn in [aggr.val_dataloader]:
    for iterable in fn():
        for batch in tqdm(iterable):
            pass
