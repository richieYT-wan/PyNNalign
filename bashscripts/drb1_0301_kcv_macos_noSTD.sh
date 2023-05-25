#! /usr/bin/bash

cd /Users/riwa/Documents/code/PyNNalign/pyscripts/
for f in $(seq 0 4);
do
  python3 ./train_model.py -f ../data/NetMHCIIpan_train/drb1_0301.csv -kf ${f} -o 230524_FixedPadding_DRB1_NOSTD -enc BL50LO -ml 21 -pad -15 -nh 50 -std False -bn True -do 0. -ws 9 -lr 1e-4 -bs 256 -ne 500 -tol 0.001 &
done
cd ../output/
mkdir 230525_NOSTD_FixedPadding_MHCII_DRB1_0301_kcvs

for dir in $(ls ./ | grep DRB1_NOSTD);
do mv "${dir}" "230525_NOSTD_FixedPadding_MHCII_DRB1_0301_kcvs/${f}"
done

