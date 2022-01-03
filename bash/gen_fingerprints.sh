#!bin/bash

METHODS=('ecfp' 'rdkit')
DATALIST=("Ames" "BBBP" "Cancer" "CYP1A2" "FXa" "hERG" "HIV" "Liver")

for METHOD in ${METHODS[@]}; do
    for DATA in ${DATALIST[@]}; do
        python ./experiments/gen_fingerprints.py -f ./data2/smiles/$DATA\_smiles.csv -m ecfp -o ./data2/$METHOD/$DATA\_$METHOD.npy
    done
done
