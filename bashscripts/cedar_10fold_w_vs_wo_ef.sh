#! /usr/bin/bash

cd /Users/riwa/Documents/code/PyNNalign/pyscripts/

pids=()
# First do the standard (no EF)
for f in $(seq 0 9);
do
  python3 ./train_model.py -f ../data/aligned_icore/230530_cedar_aligned.csv -kf ${f} -o 230530_CEDAR_NO_EF -enc BL50LO -x mutant -y target -ml 12 -pad -15 -nh 25 -std True -bn True -do 0.15 -ws 5 -lr 1e-4 -bs 128 -ne 300 -tol 0.0001 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

pids=()
# Then do the extra feat (with EF). here maybe only adding %Rank
for f in $(seq 0 9);
do
  python3 ./train_model_ef.py -f ../data/aligned_icore/230530_cedar_aligned.csv -kf ${f} -o 230530_CEDAR_WITH_EF -enc BL50LO -x mutant -y target -fc EL_rank_mut icore_selfsimilarity icore_blsm_mut_score -ml 12 -pad -15 -nh 25 -std True -bn True -do 0.15 -ws 5 -efnh 5 -efbn True -efdo 0.0 -lr 1e-4 -bs 128 -ne 300 -tol 0.0001 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done


cd ../output/
mkdir 230530_EF_COMPARISON
for dir in $(ls ./ | grep "230530_CEDAR_NO_EF\|230530_CEDAR_WITH_EF");
do
  mv "${dir}" "230530_EF_COMPARISON/${dir}"
done

