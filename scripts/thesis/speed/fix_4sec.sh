python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--data.init_args.train_datamodules.init_args.force_random_speed=false \
--notes="disable random speed"
