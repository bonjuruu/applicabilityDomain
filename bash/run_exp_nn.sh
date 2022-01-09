#!/bin/bash

DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "hERG" "HIV" "Liver")
ADS=("gamma" "kappa" "delta" "boundingbox" "prob")
AD_ARGS=("{\"dist_metric\": \"jaccard\"}" "{\"dist_metric\": \"jaccard\"}" "{\"dist_metric\": \"jaccard\"}" "{\"n_components\": 5}" "{}")

# Example:
# python ./experiments/pipeline_nn.py -d Ames --ad gamma --adArg "{\"dist_metric\": \"jaccard\"}"

for DATA in ${DATALIST[@]}; do
    for ((J=0; J<${#ADS[@]}; J++)); do
        python ./experiments/pipeline_nn.py -d $DATA --ad ${ADS[J]} --adArg "${AD_ARGS[J]}"
    done
done 

echo "Done! Program has completed."
