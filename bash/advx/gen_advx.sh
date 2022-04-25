#!/bin/bash

DATANAMES=("abalone" "australian" "banknote" "breastcancer" "htru2" "phoneme" "ringnorm" "texture")
INDICES=(1 2 3 4 5)

for IDX in "${INDICES[@]}"; do
    for DATA in "${DATANAMES[@]}"; do
        echo "-------------------------------------------------------------"
        echo "[$IDX] Generating advx exmaples on $DATA..."
        python ./experiments/advx/step3_attack.py -d $DATA -i $IDX -a fgsm -e 0.1 0.3 0.6 1.0 1.5
        python ./experiments/advx/step3_attack.py -d $DATA -i $IDX -a apgd -e 0.1 0.3 0.6 1.0 1.5
        python ./experiments/advx/step3_attack.py -d $DATA -i $IDX -a cw2 -e 0 5 10
    done
done