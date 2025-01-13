#!/bin/bash
### Note: No commands may be executed until after the #PBS lines
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 180923_NNAlign_2
### Output files (comment out the next 2 lines to get the job name used instead)
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=20
### Memory
#PBS -l mem=40gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=3:00:00

id=180923_std_MHC_noburn
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"
python3 ./train_test_model_efsinglepass.py -trf ../data/carlos/Train_1_MHCps.txt -tef ../data/carlos/Test_1_MHCps.txt -ml 35 -nh 10 -std True -bn True -efnh 5 -o CSL_mhc_std_noburn_10nh -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9

