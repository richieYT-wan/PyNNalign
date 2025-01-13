#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231011_NNAlign_ran_DRB10301
### Number of nodes
#PBS -l nodes=1:ppn=20
### Memory
#PBS -l mem=60gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=4:00:00


id=231011_nostd_MHC_burn_60nh_ran_DRB1_0301
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

train_file="../data/carlos/train_all5_DRB1_0301.txt"
test_file="../data/carlos/peps_13-21_DRB1_0301.txt"

for i in {1..5};
do
    echo "Running iteration ${i}"
    python3 ./train_test_model_efsinglepass.py -trf $train_file -tef $test_file -ml 35 -nh 60 -std f -bn True -efnh 5 -o CSL_mhc_nostd_burn_60nh_ran_DRB1_0301 -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -br 10 -kf ${i}
done

echo "Script finished"







