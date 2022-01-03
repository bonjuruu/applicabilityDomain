#!/bin/bash

DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "FXa" "hERG" "HIV" "Liver")

for DATA in ${DATALIST[@]}; do
    python ./experiments/check_smile.py -f ./data/smiles/$DATA\_smiles.csv -o ./data2/smiles/$DATA\_smiles.csv
done
