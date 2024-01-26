TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
CKPT="logs/CVPR/hwpsq3j0/checkpoints/epoch=8-step=3060.ckpt"
SETTING="logs/CVPR/hwpsq3j0/setting.yaml"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
            python -m inference \
            $SETTING \
            "./configs/generic/robustness/$T($L).yaml" \
            $CKPT \
            --notes "$T($L)"
    done
done