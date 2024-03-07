python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--config configs/version/Share/final/scenario/compression/c40.yaml \
--model.init_args.face_feature_path="misc/L14_real_semantic_patches_v3_2000_c40.pickle" \
--notes="full ffg_l14 c40"
