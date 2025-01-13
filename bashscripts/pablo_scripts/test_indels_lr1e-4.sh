#!/bin/bash


source /home/projects/vaccine/people/pasbes/PyNNalign/myenv/bin/activate


cd /home/projects/vaccine/people/pasbes/PyNNalign/pyscripts

# Run your Python scripts with the specified parameters
python train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/mhc1_el_5M.csv -tef ../data/mhc1_el_subsample/test_data.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 50 -br 10 -otf True -cuda True -bs 1024 -lr 1e-4 -ne 200 -indel True -add_ps True -kf 0 -o add-ps_True_indel_True_lr_1e-4
