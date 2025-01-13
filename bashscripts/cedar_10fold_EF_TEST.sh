#! /usr/bin/bash

cd /Users/riwa/Documents/code/PyNNalign/pyscripts/test_debug/
DATADIR=/Users/riwa/Documents/code/PyNNalign/data/aligned_icore/

pids=()
# SINGLE
for f in $(seq 0 9);
do
  python3 ./train_model_ef_single_double_debug.py -f "${DATADIR}230530_cedar_aligned.csv" -kf ${f} -o 230531_EF_SINGLE -el single -enc BL50LO -x mutant -y target -fc EL_rank_mut icore_selfsimilarity -ml 12 -pad -15 -nh 25 -std True -bn True -do 0.15 -ws 5 -efnh 5 -efbn True -efdo 0.15 -lr 1e-4 -bs 128 -ne 300 -tol 0.0001 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

pids=()
# DOUBLE
for f in $(seq 0 9);
do
  python3 ./train_model_ef_single_double_debug.py -f "${DATADIR}230530_cedar_aligned.csv" -kf ${f} -o 230531_EF_DOUBLE -el double -enc BL50LO -x mutant -y target -fc EL_rank_mut icore_selfsimilarity -ml 12 -pad -15 -nh 25 -std True -bn True -do 0.15 -ws 5 -efnh 5 -efbn True -efdo 0.15 -lr 1e-4 -bs 128 -ne 300 -tol 0.0001 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done


cd /Users/riwa/Documents/code/PyNNalign/output/
mkdir -p 230531_EF_TEST
for dir in $(ls ./ | grep "230531_EF_DOUBLE\|230531_EF_SINGLE");
do
  mv "${dir}" "230531_EF_TEST/${dir}"
done

