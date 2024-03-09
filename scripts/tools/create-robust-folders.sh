TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
ROOT="/scratch1/users/od/FaceForensicC23/cropped"
CSV_FOLDER="/scratch1/users/od/FaceForensicC23/csv_files"
FOLDER="robustness"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
            mkdir -p "datasets/$FOLDER/$T/$L/F2F/c23/"
            mkdir -p "datasets/$FOLDER/$T/$L/FSh/c23/"
            mkdir -p "datasets/$FOLDER/$T/$L/DF/c23/"
            mkdir -p "datasets/$FOLDER/$T/$L/NT/c23/"
            mkdir -p "datasets/$FOLDER/$T/$L/real/c23/"
            mkdir -p "datasets/$FOLDER/$T/$L/FS/c23/"


            ln -sfn "$ROOT/$FOLDER/$T/$L/F2F/" "datasets/$FOLDER/$T/$L/F2F/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/FSh/" "datasets/$FOLDER/$T/$L/FSh/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/DF/"  "datasets/$FOLDER/$T/$L/DF/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/NT/"  "datasets/$FOLDER/$T/$L/NT/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/real/"  "datasets/$FOLDER/$T/$L/real/c23/videos"
            ln -sfn "$ROOT/$FOLDER/$T/$L/FS/"  "datasets/$FOLDER/$T/$L/FS/c23/videos"
            ln -sfn $CSV_FOLDER "datasets/$FOLDER/$T/$L/csv_files"
    done
done