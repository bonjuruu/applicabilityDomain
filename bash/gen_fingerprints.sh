#!bin/bash

METHODS=('ecfp' 'rdkit' 'maccs')
DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "hERG" "HIV" "Liver")

for METHOD in ${METHODS[@]}; do
    for DATA in ${DATALIST[@]}; do
        echo Running $METHOD on $DATA...
        python ./experiments/gen_fingerprints.py -f ./data/smiles/$DATA\_smiles.csv -o ./data/$METHOD/$DATA\_$METHOD.npy -m $METHOD
    done
done
