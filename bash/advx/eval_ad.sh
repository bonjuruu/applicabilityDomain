#!/bin/bash


DATANAMES=("abalone" "australian" "banknote" "breastcancer" "htru2" "phoneme" "ringnorm" "texture")
INDICES=(1 2 3 4 5)
ATTACKS=("fgsm" "apgd" "cw2")
ADS=("gamma" "kappa" "delta" "boundingbox" "prob")

for IDX in "${INDICES[@]}"; do
    for DATA in "${DATANAMES[@]}"; do
        for ATT in "${ATTACKS[@]}"; do
            for AD in "${ADS[@]}"; do
                echo "-------------------------------------------------------------"
                echo "[$IDX] Running on Data: $DATA Attack: $ATT Defence: $AD..."
                python ./experiments/advx/step4_defence.py -d $DATA -o "./results/numeric/run_$IDX/" -a $ATT --ad $AD
            done
        done
    done
done
