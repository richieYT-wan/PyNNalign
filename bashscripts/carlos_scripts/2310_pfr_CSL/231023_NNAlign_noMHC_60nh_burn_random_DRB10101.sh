#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231023_NNAlign_nomhc_ran_DRB10101
### Number of nodes
#PBS -l nodes=1:ppn=20
### Memory
#PBS -l mem=20gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=2:00:00

out_dir=CSL_nomhc_nostd_burn_60nh_ran_DRB1_0101
id=231023_nostd_noMHC_burn_60nh_ran_DRB1_0101
#PBS -e $out_dir/$id.err
#PBS -o $out_dir/$id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

train_file="../data/carlos/train_all5_DRB1_0101.txt"
test_file="../data/carlos/peps_13-21_DRB1_0101.txt"

for i in {1..5};
do
    echo "Running iteration ${i}"
    python3 ./train_test_model_efrandom.py -trf $train_file -tef $test_file -ml 37 -nh 60 -std f -bn True -efnh 5 -o $out_dir -x Sequence -y BA -enc BL50LO -add_ps False -ws 9 -br 10 -kf ${i}
done

echo "Script finished"







