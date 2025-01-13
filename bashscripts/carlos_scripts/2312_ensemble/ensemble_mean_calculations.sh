#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N Ensemble_mean
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=40gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=10:00:00


id=Ensemble_mean
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

python3 ./ensemble_mean_group.py -in /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/output/231201_2hl_wd1e-4_bs256_do0.3_ensemble -id 'valid_preds_ensemble_concat' -out '/home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/output/231201_2hl_wd1e-4_bs256_do0.3_ensemble/valid_preds_ensemble_concat_grouped.csv'

echo "Script finished"
