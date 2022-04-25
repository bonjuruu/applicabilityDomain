#!/bin/bash

INDICES=(1 2 3 4 5)

for IDX in "${INDICES[@]}"; do
    rm -r ./results/numeric/run_$IDX
done