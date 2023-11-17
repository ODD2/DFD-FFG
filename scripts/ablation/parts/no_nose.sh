python -m main \
--config configs/version/Share/final/base.yaml \
--config configs/version/Share/final/clip/L14/ffg.yaml \
--model.init_args.face_parts=["lips","skin","eyes"] \
--notes="albate_parts_no_nose"
