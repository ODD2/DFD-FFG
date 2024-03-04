python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.t_attrs=["q","k","v","out","emb"] \
--notes="t_attrs=ALL"
