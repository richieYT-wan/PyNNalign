#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

HOMEDIR="/home/projects/vaccine/people/yatwan/PyNNalign/"
PYDIR="${HOMEDIR}pyscripts/"
OUTDIR="${HOMEDIR}output/"
OUTDIRFINAL="${OUTDIR}230601_hyperparameters_tuning_gridsearch/"

cd ${PYDIR}
python3 ./hyperparams_condition_analysis.py -d ${OUTDIRFINAL} -x 'pred' -y 'target' -nc 39