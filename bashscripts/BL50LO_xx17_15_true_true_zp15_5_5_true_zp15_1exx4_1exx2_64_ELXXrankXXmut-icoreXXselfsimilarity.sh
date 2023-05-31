#!/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd
HOMEDIR="/home/projects/vaccine/people/yatwan/PyNNalign/"
PYDIR="${HOMEDIR}pyscripts/"
OUTDIR="${HOMEDIR}output/"
OUTDIRFINAL="${OUTDIR}230601_hyperparameters_tuning_gridsearch/"
mkdir -p ${OUTDIRFINAL}
COMBINATION="EL_rank_mut icore_selfsimilarity"
COMBI=${COMBINATION// /-}
values=(
  "BL50LO"
  "-17"
  "15"
  "true"
  "true"
  "0.15"
  "5"
  "5"
  "true"
  "0.15"
  "1e-4"
  "1e-2"
  "64"
)
FILENAME=$(IFS=_; echo "${values[*]//['-' ]/xx}")
FILENAME=${FILENAME//0./zp}
# Append COMBINATION to the end of the result string
FILENAME+="_${COMBI//_/XX}"
FILENAME=${FILENAME//icore_/}
cd ${PYDIR}
pids=()
for fold in $(seq 0 9);
do
python3 ./train_test_model_ef.py -trf "${HOMEDIR}data/aligned_icore/230530_cedar_aligned.csv" -tef "${HOMEDIR}data/aligned_icore/230530_prime_aligned.csv" -ml 12 -ne 20 -x mutant -y target -o BL50LO_xx17_15_true_true_zp15_5_5_true_zp15_1exx4_1exx2_64_ELXXrankXXmut-icoreXXselfsimilarity -kf ${fold} -enc BL50LO -pad -17 -fc EL_rank_mut icore_selfsimilarity -nh 15 -std true -bn true -do 0.15 -ws 5 -efnh 5 -efbn true -efdo 0.15 -lr 1e-4 -wd 1e-2 -bs 64 &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
mkdir -p "${OUTDIRFINAL}${FILENAME}/"
cd "${OUTDIR}"
mv "${FILENAME}"_* "${OUTDIRFINAL}${FILENAME}/"
