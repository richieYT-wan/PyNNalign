#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231215_bestpreds
### Number of nodes
#PBS -l nodes=1:ppn=10
### Memory
#PBS -l mem=80gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=20:00:00


id=231215_bestpred_nofeat
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2312_report/231214_nofeatures_ranpeps/DRB1_0301
pwd

echo \"Starting PyScript\"

python3 ../../../../pyscripts/testpreds_selection.py -d /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2312_report/231214_nofeatures_ranpeps/DRB1_0301 -id 15mer -of 15mer_ranpep_final_pred

echo "Script finished"
