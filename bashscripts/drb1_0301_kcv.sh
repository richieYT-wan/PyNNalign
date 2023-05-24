#! /usr/bin/bash

for f in $(seq 0 4);
do
  echo "#! /usr/bin/bash

  source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
  source activate phd

  HOMEDIR=/home/projects/vaccine/people/yatwan/PyNNalign/
  PYDIR=\${HOMEDIR}pyscripts/
  cd \${PYDIR}
  pwd
  echo \"Starting PyScript\"
  python3 python3 train_model.py -f ../data/NetMHCIIpan_train/drb1_0301.csv -kf ${f} -enc BL50LO -ml 21 -pad -15 -nh 50 -std True -bn True -do 0.0 -ws 9 -bs 256 -ne 25 -tol 0.0001" > "230524_DRB1_kcv_f${f}.sh"
  chmod +x "230524_DRB1_kcv_f${f}.sh"
  qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=1:thinnode,mem=10gb,walltime=00:10:00 "230524_DRB1_kcv_f${f}.sh"
done

