python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.face_attn_attr="out" \
--model.init_args.s_k_attr="out" \
--notes="ablation_guide s_k_attr=out"