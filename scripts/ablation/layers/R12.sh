python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.ffg_layers=R12 \
--model.init_args.ffg_reverse=true \
--notes="ffg layer=R12"
