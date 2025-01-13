#!/bin/bash


source /home/projects/vaccine/people/pasbes/PyNNalign/myenv/bin/activate


cd /home/projects/vaccine/people/pasbes/PyNNalign/pyscripts

# Run your Python scripts with the specified parameters
python resume_training_singlepass.py -trf ../data/mhc1_el_subsample/mhc1_el_5M.csv -tef ../data/mhc1_el_subsample/test_data.csv -model_folder ../output/addps_True_indel_True_KFold_0_240307_2216_hi8OS/ -ml 13 -pad -20 -y target -x sequence -bs 1024 -otf True -cuda True -lr 1e-5 -ne 180 -kf 0 -add_ps True -indel True -o 240307_addps_True_indel_True -rid hi8OS
