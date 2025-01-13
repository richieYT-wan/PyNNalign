#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231114_PyNNAlign
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=40gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=10:00:00


id=231114_nostd_nomhc_burn_60nh
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

for i in {1..5};
do
    echo "Running iteration ${i}"
    python3 ./train_test_model_efsinglepass.py -trf ../data/carlos/train_all5_correct.txt -tef ../data/carlos/short_random_peptides_fakemhc.txt -ml 37 -nh 60 -std False -bn False -efnh 5 -o CSL_nostd_nomhc_noburn_60nh_all -x Sequence -y BA -enc BL50LO -add_ps False -ws 9 -br 10 -add_pfr False -add_fr_len False -add_pep_len False -kf ${i}
    echo "Iteration $i completed"
done

echo "Script finished"
