#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231014_NNAlign_ran_DRB10401
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=20gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=3:00:00


id=231014_nostd_MHC_burn_60nh_ran_DRB1_0401
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

train_file="../data/carlos/train_all5_DRB1_0401.txt"
test_file="../data/carlos/peps_13-21_DRB1_0401.txt"

for i in {1..5};
do
    echo "Running iteration ${i}"
    python3 ./train_test_model_efrandom.py -trf $train_file -tef $test_file -ml 37 -nh 60 -std f -bn True -efnh 5 -o CSL_mhc_nostd_burn_60nh_ran_DRB1_0401 -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -br 10 -kf ${i}
done

echo "Script finished"







