python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/svl.yaml \
--model.init_args.op_mode=["T"] \
--notes="no spatial branch"
