#!/bin/bash

DATANAMES=("abalone" "australian" "banknote" "breastcancer" "htru2" "phoneme" "ringnorm" "texture")
INDICES=(1 2 3 4 5)

for IDX in "${INDICES[@]}"; do
    echo "---------------------------------------------------------------------"
    echo "Run #: $IDX"
    for DATA in "${DATANAMES[@]}"; do
        python ./experiments/advx/step2_train_clf.py -d $DATA -i $IDX
    done
done