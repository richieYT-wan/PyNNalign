#!/bin/bash


source /home/projects/vaccine/people/pasbes/PyNNalign/myenv/bin/activate


cd /home/projects/vaccine/people/pasbes/PyNNalign/pyscripts

# Run your Python scripts with the specified parameters
mprof run python train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/mhc1_el_500k_subsample.csv -tef ../data/mhc1_el_sub10k/test_data.csv -o NO_INDEL_NO_ADD-PS -ml 13 -ws 9 -pad=-20 -y 'target' -x 'sequence' -std False -bn False -nh 10 -indel False -add_ps False -ne 500 -efnh 10 -br 15

