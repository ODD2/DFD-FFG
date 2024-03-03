TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
ROOT="/scratch1/users/od/FaceForensicC23/cropped"
FOLDER="robustness"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
            mkdir -p "./$T/$L/F2F/c23/"
            mkdir -p "./$T/$L/FSh/c23/"
            mkdir -p "./$T/$L/DF/c23/"
            mkdir -p "./$T/$L/NT/c23/"
            mkdir -p "./$T/$L/real/c23/"
            mkdir -p "./$T/$L/FS/c23/"
            mkdir -p "./$T/$L/splits/"


            ln -sfn "$ROOT/$FOLDER/$T/$L/F2F/" "./$T/$L/F2F/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/FSh/" "./$T/$L/FSh/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/DF/"  "./$T/$L/DF/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/NT/"  "./$T/$L/NT/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/real/"  "./$T/$L/real/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/FS/"  "./$T/$L/FS/c23/videos"
            ln -sf "$ROOT/videos/test.json" "./$T/$L/splits/test.json"
            ln -sf "$ROOT/videos/val.json" "./$T/$L/splits/val.json"
            ln -sf "$ROOT/videos/train.json" "./$T/$L/splits/train.json"
    done
done