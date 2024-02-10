python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--data.init_args.train_datamodules.init_args.force_random_speed=false \
--data.init_args.clip_duration=1 \
--data.init_args.num_frames=20 \
--data.init_args.batch_size=10 \
--data.init_args.train_datamodules.init_args.batch_size=16 \
--model.init_args.num_frames=20 \
--notes="disable random speed, 1 sec 20 fps, batch size 20."
