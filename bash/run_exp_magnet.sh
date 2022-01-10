#!/bin/bash

DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "hERG" "HIV" "Liver")
CLASSIFIERS=("rf" "svm" "knn" "nn")
CLF_ARGS=("{\"n_estimators\": 200, \"n_jobs\": -1}" "{\"C\": 10}" "{\"n_neighbors\": 5, \"n_jobs\": -1}" "{\"batch_size\": 128, \"hidden_dim\": 512}")

# Example:
# python ./experiments/pipeline_magnet.py -d Ames --clf svm --clfArg "{\"C\": 10}"
# python ./experiments/pipeline_magnet.py -d Ames --clf nn --clfArg "{\"batch_size\": 128, \"hidden_dim\": 512}"

for DATA in ${DATALIST[@]}; do
    for ((I=0; I<${#CLASSIFIERS[@]}; I++)); do
        python ./experiments/pipeline_magnet.py -d $DATA --clf ${CLASSIFIERS[I]} --clfArg "${CLF_ARGS[I]}"
    done
done 

echo "Done! Program has completed."
