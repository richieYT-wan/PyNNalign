#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231005_NNAlign_80nh
### Number of nodes
#PBS -l nodes=1:ppn=20
### Memory
#PBS -l mem=10gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=16:00:00


id=231005_nostd_MHC_burn_80nh
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"
python3 ./train_test_model_efsinglepass.py -trf ../data/carlos/Train_1_MHCps.txt -tef ../data/carlos/Test_1_MHCps.txt -ml 35 -nh 80 -std f -bn True -efnh 5 -o CSL_mhc_nostd_burn_80nh -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -br 10
