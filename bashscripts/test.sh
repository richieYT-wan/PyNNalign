#!/bin/bash

strings=("icore_aliphatic_index" "icore_boman" "icore_hydrophobicity" "icore_isoelectric_point" 
"icore_dissimilarity_score" "icore_blsm_mut_score" "ratio_rank" "EL_rank_wt_aligned" "foreignness_score" 
"Total_Gene_TPM")

length=${#strings[@]}
total_combinations=$((2**length - 1))

# Generate all possible combinations of strings
for ((i=1; i<=total_combinations; i++)); do
    combination=""
    for ((j=0; j<length; j++)); do
        if (( (i & (1<<j)) != 0 )); then
            if [ -z "$combination" ]; then
                combination="${strings[$j]}"
            else
                combination="$combination, ${strings[$j]}"
            fi
        fi
    done
    echo "Combination: $combination"
done

echo "Total combinations: $total_combinations"

