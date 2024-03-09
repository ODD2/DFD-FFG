TYPES="CS CC BW GNC GB JPEG VC"
LEVELS="1 2 3 4 5"
CKPT="logs/ECCV/otfsj0qd/checkpoints/epoch=29-step=2040.ckpt"
SETTING="logs/ECCV/otfsj0qd/setting.yaml"
for T in $TYPES; 
do
    for L in $LEVELS; 
        do
            python -m inference \
            $SETTING \
            "./configs/generic/robustness/$T($L).yaml" \
            $CKPT \
            --notes "$T($L)" \
            --devices -1 
    done
done