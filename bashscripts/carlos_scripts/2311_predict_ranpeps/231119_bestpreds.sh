#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231119_qsub_best_pred
### Number of nodes
#PBS -l nodes=1:ppn=10
### Memory
#PBS -l mem=80gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=20:00:00


id=231119_best_pred
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2311_predict_ranpeps/231116_complete_wd1e-4_ranpeps/DRB1_1101
pwd

echo \"Starting PyScript\"

python3 ../../../../pyscripts/testpreds_selection.py -d /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/bashscripts/2311_predict_ranpeps/231116_complete_wd1e-4_ranpeps/DRB1_1101 -id 15mer -of 15mer_ranpep_final_pred

echo "Script finished"
