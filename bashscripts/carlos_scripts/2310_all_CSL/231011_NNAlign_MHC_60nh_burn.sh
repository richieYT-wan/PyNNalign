#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231011_NNAlign_all_60nh_fold1
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=80gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=10:00:00


id=231011_nostd_MHC_burn_60nh_fold1
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

echo "Running iteration 1"
python3 ./train_test_model_efsinglepass.py -trf ../data/carlos/train_all5.txt -tef ../data/carlos/short_random_peptides_fakemhc.txt -ml 37 -nh 60 -std f -bn True -efnh 5 -o CSL_mhc_nostd_burn_60nh_all -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -br 10 -kf 1
