#!/bin/bash

DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "hERG" "HIV" "Liver")
CLASSIFIERS=("rf" "svm" "knn")
CLF_ARGS=("{\"n_estimators\": 200}" "{\"C\": 10, \"probability\": true}" "{\"n_neighbors\": 5}")
ADS=("gamma" "kappa" "delta" "boundingbox" "prob")
AD_ARGS=("{\"dist_metric\": \"jaccard\"}" "{\"dist_metric\": \"jaccard\"}" "{\"dist_metric\": \"jaccard\"}" "{\"n_pc\": 5}" "{}")

for DATA in ${DATALIST[@]}; do
    for ((I=0; I<${#CLASSIFIERS[@]}; I++)); do
        for ((J=0; J<${#ADS[@]}; J++)); do
            python ./experiments/pipeline_ad.py -d $DATA --clf ${CLASSIFIERS[I]} --clfArg "${CLF_ARGS[I]}" --ad ${ADS[J]} --adArg "${AD_ARGS[J]}"
        done
    done
done 

echo "Done! Program has completed."
