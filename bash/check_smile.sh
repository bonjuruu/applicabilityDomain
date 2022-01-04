#!/bin/bash

DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "hERG" "HIV" "Liver")

for DATA in ${DATALIST[@]}; do
    python ./experiments/check_smile.py -f ./data/smiles/$DATA\_smiles.csv -o ./data/subsets/smiles
done
