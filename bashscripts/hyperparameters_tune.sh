#!/bin/bash

# Define the list of values for each variable
ENC_VALUES=("BL50LO")
PAD_VALUES=(-17)
NH_VALUES=(15)
STD_VALUES=(true)
BN_VALUES=(true)
DO_VALUES=(0.15)
WS_VALUES=(5)
EFNH_VALUES=(5)
EFBN_VALUES=(true)
EFDO_VALUES=(0.15)
LR_VALUES=(1e-4)
WD_VALUES=(1e-2)
BS_VALUES=(64)
COMBINATION_VALUES=("EL_rank_mut icore_selfsimilarity")
fold=0
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
                            echo 'HOMEDIR="/Users/riwa/Documents/code/PyNNalign/"' >> "${FILENAME}.sh"
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
                            echo "python3 ./train_test_model_ef.py -trf \"\${HOMEDIR}data/aligned_icore/230530_cedar_aligned.csv\" -tef \"\${HOMEDIR}data/aligned_icore/230530_prime_aligned.csv\" -ml 12 -ne 20 -x mutant -y target -o ${FILENAME} -kf \${fold} -enc ${ENC} -pad ${PAD} -fc ${COMBINATION} -nh ${NH} -std ${STD} -bn ${BN} -do ${DO} -ws ${WS} -efnh ${EFNH} -efbn ${EFBN} -efdo ${EFDO} -lr ${LR} -wd ${WD} -bs ${BS} &" >> "${FILENAME}.sh"

                            echo '  pids+=($!)' >> "${FILENAME}.sh"
                            echo 'done' >> "${FILENAME}.sh"
                            echo 'for pid in "${pids[@]}"; do' >> "${FILENAME}.sh"
                            echo '  wait "$pid"' >> "${FILENAME}.sh"
                            echo 'done' >> "${FILENAME}.sh"
                            echo 'mkdir -p "${OUTDIRFINAL}${FILENAME}/"' >> "${FILENAME}.sh"
                            echo 'cd "${OUTDIR}"' >> "${FILENAME}.sh"
                            echo 'mv "${FILENAME}"_* "${OUTDIRFINAL}${FILENAME}/"' >> "${FILENAME}.sh"

                            # Submit the script for execution
                            qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=10:thinnode,mem=46.5gb,walltime=00:10:00 ${FILENAME}
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
