TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
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


            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/F2F/" "./$T/$L/F2F/c23/videos"
            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/FSh/" "./$T/$L/FSh/c23/videos"
            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/DF/"  "./$T/$L/DF/c23/videos"
            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/NT/"  "./$T/$L/NT/c23/videos"
            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/real/"  "./$T/$L/real/c23/videos"
            ln -sfn "/stock/FaceForensicC23/robust_cropped_faces/$T/$L/FS/"  "./$T/$L/FS/c23/videos"
            ln -sf "/stock/FaceForensicC23/videos/test.json" "./$T/$L/splits/test.json"
            ln -sf "/stock/FaceForensicC23/videos/val.json" "./$T/$L/splits/val.json"
            ln -sf "/stock/FaceForensicC23/videos/train.json" "./$T/$L/splits/train.json"
    done
done