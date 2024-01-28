python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/evl.yaml \
--data.init_args.train_datamodules.init_args.batch_size=40 \
--notes="full evl_l14"
