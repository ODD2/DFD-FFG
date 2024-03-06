python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--config configs/version/Share/final/scenario/compression/raw.yaml \
--model.init_args.face_feature_path="misc/L14_real_semantic_patches_v3_2000_raw.pickle" \
--notes="full ffg_l14 raw "
