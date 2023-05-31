#! /usr/bin/bash

#HOMEDIR="/home/projects/vaccine/people/yatwan/PyNNalign/"
HOMEDIR="/Users/riwa/Documents/code/PyNNalign/"
PYDIR="${HOMEDIR}pyscripts/"
OUTDIR="${HOMEDIR}output/"
OUTDIRFINAL="${OUTDIR}230601_hyperparameters_tuning_gridsearch/"
mkdir -p ${OUTDIRFINAL}

# stupid loop to combine features

ENC="BL50LO"
PAD=-15
COMBINATION="EL_rank_mut icore_selfsimilarity"
NH=25
STD=true
BN=true
DO=0.15
WS=5
EFNH=5
EFBN=true
EFDO=0.15
LR=5e-5
WD=1e-2
BS=64
strings=("icore_aliphatic_index" "icore_boman" "icore_hydrophobicity" "icore_isoelectric_point" "icore_selfsimilarity" "icore_blsm_mut_score" "ratio_rank" "EL_rank_wt_aligned" "foreignness_score" "Total_Gene_TPM")
length=${#strings[@]}
# Generate all possible combinations of strings
for ((i=1; i<2**$length; i++)); do
    COMBINATION=""
    for ((j=0; j<length; j++)); do
        if (( (i & (1<<j)) != 0 )); then
            if [ -z "$COMBINATION" ]; then
                COMBINATION="${strings[$j]}"
            else
                COMBINATION="${COMBINATION} ${strings[$j]}"
            fi
        fi
    done
        # Saving stuff at the end of the 10fold kcv run ; CREATING FILENAME
    # Replace space in COMBINATION with dash
    COMBI=${COMBINATION// /-}
    # Create a list of values
    values=("$ENC" "$PAD" "$NH" "$STD" "$BN" "$DO" "$WS" "$EFNH" "$EFBN" "$EFDO" "$LR" "$WD" "$BS")

    # Replace '.' or '-' with 'm' and join the values with underscores
    FILENAME=$(IFS=_; echo "${values[*]//['-']/xx}")
    FILENAME=${FILENAME//0./zp}
    # Append COMBINATION to the end of the result string
    FILENAME+="_${COMBI//_/XX}"
    FILENAME=${FILENAME//icore_/}

    cd ${PYDIR}
    pids=()
    for fold in $(seq 0 9);
    do
      # TODO: FILENAME WORKS PROPERLY. JUST NEED TO RUN THE PROGRAM NOW AND CREATE THE LOOP/GRIDSEARCH
      python3 ./train_test_model_ef.py -trf "${HOMEDIR}data/aligned_icore/230530_cedar_aligned.csv" -tef "${HOMEDIR}data/aligned_icore/230530_prime_aligned.csv"  -ml 12 -ne 450 -x mutant -y target -o ${FILENAME} -kf ${fold} -enc ${ENC} -pad ${PAD} -fc ${COMBINATION} -nh ${NH} -std ${STD} -bn ${BN} -do ${DO} -ws ${WS} -efnh ${EFNH} -efbn ${EFBN} -efdo ${EFDO} -lr ${LR} -wd ${WD} -bs ${BS} &
      pids+=($!)
    done

    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    mkdir -p "${OUTDIRFINAL}${FILENAME}/"
    cd "${OUTDIR}"
    mv "${FILENAME}"_* "${OUTDIRFINAL}${FILENAME}/"
done
