#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231215_nofeat_pred
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=80gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=20:00:00


id=nofeatures_pred_ranpeps
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

python3 ./predict_random.py -tef ../data/carlos/15mer_ranpep_DRB1_0301.txt -dir ../output/231214_report_nofeatures_AUC/ -od /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2312_report/231214_nofeatures_ranpeps/DRB1_0301/ -of '15mer_ranpep_pred' -ef 0 -add_hl False -nh2 0 -bs 64 -add_ps False -add_pfr False -add_fr_len False -add_pep_len False

echo "Script finished"
