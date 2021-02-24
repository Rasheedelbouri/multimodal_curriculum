#!/bin/bash

TYPES=("m" "c" "w")
VALS=(10 5 3)
MIX=("gaussian" "k-means" "class")
BOOLS=("False" "True")

for ctype in "${TYPES[@]}"
do
    for batch in "${VALS[@]}"
    do
        for cluster in "${VALS[@]}"
        do
            for mix in "${MIX[@]}"
            do 
                for depth in "${BOOLS[@]}"
                do
                    python3 main.py --curricType="${ctype}" --curricBatches="${batch}" --numClusters="${cluster}" --mixType="${mix}" --depthFirst="${depth}"
                done
            done
        done
    done
done