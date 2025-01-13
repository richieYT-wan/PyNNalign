#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231217_MS_pred
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=80gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=20:00:00


id=MSred_pred_validations
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

python3 ./predict_validations.py -trf ../data/carlos/MS_C00_data_noUZB.txt -dir ../output/231201_2hl_wd1e-4_bs256_MSdata0.15/ -od /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2312_report/MSreduced_valid_preds/ -of 'MSreduced_valid_preds' -ef 734 -add_hl True -nh2 30 -bs 256 -add_ps True -add_pfr True -add_fr_len True -add_pep_len True

echo "Script finished"
