#! /usr/bin/bash

cd /Users/riwa/Documents/code/PyNNalign/pyscripts/
pids=()
for f in $(seq 0 4);
do
  python3 ./train_model.py -f ../data/NetMHCIIpan_train/drb1_0301.csv -kf ${f} -o 230529_burnin_test -enc BL50LO -ml 21 -pad -15 -nh 50 -br 10 -std True -bn True -do 0. -ws 9 -lr 1e-4 -bs 256 -ne 500 -tol 0.0001 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

cd ../output/
mkdir 230529_Burn_In_Fix

for dir in $(ls ./ | grep 230529_burnin_test);
do
  mv "${dir}" "230529_Burn_In_Fix/${dir}"
done

