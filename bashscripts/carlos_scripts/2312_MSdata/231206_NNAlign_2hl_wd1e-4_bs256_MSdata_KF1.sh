#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231206_PyNNAlign_KF1
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=120gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=180:00:00


id=231206_2hl_wd1e-4_bs256_MSdata_KF1
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
python3 ./train_test_model_efsinglepass.py -trf ../data/carlos/MS_C00_data_noUZB.txt -tef ../data/carlos/short_random_peptides_fakemhc.txt -ml 21 -nh 60 -std f -bn False -efnh 5 -o CSL_2hl_wd1e-4_bs256_MSdata1 -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -add_pfr True -add_fr_len True -add_pep_len True -add_hl True -nh2 30 -wd 1e-4 -bs 256 -br 10 -kf 1
echo "Iteration 1 completed"


echo "Script finished"
