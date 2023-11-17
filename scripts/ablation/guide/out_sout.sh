python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.face_attn_attr="out" \
--model.init_args.syno_attn_attr="s_out" \
--notes="ablation_guide out->s_out"