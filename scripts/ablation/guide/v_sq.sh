python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.face_attn_attr="v" \
--model.init_args.syno_attn_attr="s_q" \
--notes="ablation_guide v->s_q"