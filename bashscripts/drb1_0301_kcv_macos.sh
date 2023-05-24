#! /usr/bin/bash

cd /Users/riwa/Documents/code/PyNNalign/pyscripts/
for f in $(seq 0 4);
do
  python3 ./train_model.py -f ../data/NetMHCIIpan_train/drb1_0301.csv -kf ${f} -o 230524_run_MHC -enc BL50LO -ml 21 -pad -15 -nh 50 -std True -bn True -do 0.0 -ws 9 -lr 1e-4 -bs 256 -ne 500 -tol 0.0001
done
cd ../output/
mkdir 230524_MHCII_DRB1_0301_kcvs

for dir in $(ls ./ | grep 230524_run_MHC);
do mv "${dir}" "230524_MHCII_DRB1_0301_kcvs/${f}"
done

