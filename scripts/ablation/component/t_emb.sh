python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.is_temporal_embedding=false \
--notes="no temporal embedding"
