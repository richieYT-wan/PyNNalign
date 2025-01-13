#! /usr/bin/bash

#$ -N NNAlign_Carlos          # Give your job a name

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Activate the "base" Conda environment
conda activate base

HOMEDIR=/home/projects/vaccine/people/yatwan/PyNNalign/
PYDIR=\${HOMEDIR}pyscripts/
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./train_model.py -f ../data/NetMHCIIpan_train/drb1_0301.csv -kf ${f} -o 230425_run_MHC -enc BL50LO -ml 21 -pad -15 -nh 50 -std True -bn True -do 0.0 -ws 9 -lr 1e-4 -bs 256 -ne 500 -tol 0.0001" > "230524_DRB1_kcv_f${f}.sh"
chmod +x "230524_DRB1_kcv_f${f}.sh"