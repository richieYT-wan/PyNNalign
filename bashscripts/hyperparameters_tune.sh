#!/bin/bash

# Making extra columns combinations :
#strings=("icore_aliphatic_index" "icore_boman" "icore_hydrophobicity" "icore_isoelectric_point" "icore_dissimilarity_score" "icore_blsm_mut_score" "ratio_rank" "EL_rank_wt_aligned" "foreignness_score" "Total_Gene_TPM")

strings=("EL_rank_mut" "icore_blsm_mut_score" "icore_selfsimilarity" "hydrophobicity")

length=${#strings[@]}
total_combinations=$((2**length - 1))

COMBINATION_VALUES=()
# Generate all possible combinations of strings
for ((i=1; i<=total_combinations; i++)); do
    combination=""
    for ((j=0; j<length; j++)); do
        if (( (i & (1<<j)) != 0 )); then
            if [ -z "$combination" ]; then
                combination="${strings[$j]}"
            else
                combination="$combination ${strings[$j]}"
            fi
        fi
    done
    COMBINATION_VALUES+=("$combination")
done
# Define the list of values for each variable
ENC_VALUES=("BL50LO" "BL62LO" "onehot")
PAD_VALUES=(0)
NH_VALUES=(10 25 50 75)
STD_VALUES=(true)
BN_VALUES=(true false)
DO_VALUES=(0.0 0.25)
WS_VALUES=(5 6)
EFNH_VALUES=(2 5 10)
EFBN_VALUES=(true false)
EFDO_VALUES=(0.0 0.25)
LR_VALUES=(1e-4 5e-4 1e-3)
WD_VALUES=(1e-2 1e-5 0)
BS_VALUES=(128 256)

# Iterate over the variables and their values
for ENC in "${ENC_VALUES[@]}"; do
  for PAD in "${PAD_VALUES[@]}"; do
    for NH in "${NH_VALUES[@]}"; do
      for STD in "${STD_VALUES[@]}"; do
        for BN in "${BN_VALUES[@]}"; do
          for DO in "${DO_VALUES[@]}"; do
            for WS in "${WS_VALUES[@]}"; do
              for EFNH in "${EFNH_VALUES[@]}"; do
                for EFBN in "${EFBN_VALUES[@]}"; do
                  for EFDO in "${EFDO_VALUES[@]}"; do
                    for LR in "${LR_VALUES[@]}"; do
                      for WD in "${WD_VALUES[@]}"; do
                        for BS in "${BS_VALUES[@]}"; do
                          for COMBINATION in "${COMBINATION_VALUES[@]}"; do
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
                            # Write the code snippet to a new file
                            echo "#!/bin/bash" > "${FILENAME}.sh"
                            echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >>"${FILENAME}.sh"
                            echo "source activate pynn" >> "${FILENAME}.sh"
                            echo 'HOMEDIR="/home/projects/vaccine/people/yatwan/PyNNalign/"' >> "${FILENAME}.sh"
                            echo 'PYDIR="${HOMEDIR}pyscripts/"' >> "${FILENAME}.sh"
                            echo 'OUTDIR="${HOMEDIR}output/"' >> "${FILENAME}.sh"
                            echo 'OUTDIRFINAL="${OUTDIR}230601_hyperparameters_tuning_gridsearch/"' >> "${FILENAME}.sh"
                            echo 'mkdir -p ${OUTDIRFINAL}' >> "${FILENAME}.sh"
                            echo "COMBINATION=\"${COMBINATION}\"" >> "${FILENAME}.sh"
                            echo 'COMBI=${COMBINATION// /-}' >> "${FILENAME}.sh"
                            echo "values=(" >> "${FILENAME}.sh"
                            for value in "${values[@]}"; do
                              echo "  \"$value\"" >> "${FILENAME}.sh"
                            done
                            echo ")" >> "${FILENAME}.sh"
                            echo 'FILENAME=$(IFS=_; echo "${values[*]//['\''-'\'' ]/xx}")' >> "${FILENAME}.sh"
                            echo 'FILENAME=${FILENAME//0./zp}' >> "${FILENAME}.sh"
                            echo '# Append COMBINATION to the end of the result string' >> "${FILENAME}.sh"
                            echo 'FILENAME+="_${COMBI//_/XX}"' >> "${FILENAME}.sh"
                            echo 'FILENAME=${FILENAME//icore_/}' >> "${FILENAME}.sh"
                            echo 'cd ${PYDIR}' >> "${FILENAME}.sh"
                            echo 'pids=()' >> "${FILENAME}.sh"
                            echo 'for fold in $(seq 0 9);' >> "${FILENAME}.sh"
                            echo 'do' >> "${FILENAME}.sh"
                            echo "python3 ./train_test_model_ef.py -trf \"\${HOMEDIR}data/aligned_icore/230530_cedar_aligned.csv\" -tef \"\${HOMEDIR}data/aligned_icore/230530_prime_aligned.csv\" -ml 12 -ne 750 -x mutant -y target -o ${FILENAME} -kf \${fold} -enc ${ENC} -pad ${PAD} -fc ${COMBINATION} -nh ${NH} -std ${STD} -bn ${BN} -do ${DO} -ws ${WS} -efnh ${EFNH} -efbn ${EFBN} -efdo ${EFDO} -lr ${LR} -wd ${WD} -bs ${BS} &" >> "${FILENAME}.sh"
                            # echo "python3 ./train_test_model_ef.py -trf \"\${HOMEDIR}data/aligned_icore/230530_cedar_aligned.csv\" -tef \"\${HOMEDIR}data/aligned_icore/230530_prime_aligned.csv\" -ml 12 -ne 11 -x mutant -y target -o ${FILENAME} -kf \${fold} -enc ${ENC} -pad ${PAD} -fc ${COMBINATION} -nh ${NH} -std ${STD} -bn ${BN} -do ${DO} -ws ${WS} -efnh ${EFNH} -efbn ${EFBN} -efdo ${EFDO} -lr ${LR} -wd ${WD} -bs ${BS} &" >> "${FILENAME}.sh"
                            echo '  pids+=($!)' >> "${FILENAME}.sh"
                            echo 'done' >> "${FILENAME}.sh"
                            echo 'for pid in "${pids[@]}"; do' >> "${FILENAME}.sh"
                            echo '  wait "$pid"' >> "${FILENAME}.sh"
                            echo 'done' >> "${FILENAME}.sh"
                            echo 'mkdir -p "${OUTDIRFINAL}${FILENAME}/"' >> "${FILENAME}.sh"
                            echo 'cd "${OUTDIR}"' >> "${FILENAME}.sh"
                            echo 'mv "${FILENAME}"_* "${OUTDIRFINAL}${FILENAME}/"' >> "${FILENAME}.sh"

                            # Submit the script for execution
                            qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=10:thinnode,mem=46gb,walltime=01:00:00 "${FILENAME}.sh"
                            # qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=10:thinnode,mem=46gb,walltime=00:05:00 "${FILENAME}.sh"
                            # rm "${FILENAME}.sh"
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
