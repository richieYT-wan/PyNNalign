#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts/

python3 ./train_model_benchmark.py -f ../data/NetMHCIIpan_train/drb_python.csv -kf 1 -o 230804_BENCH_PYTHON -enc BL50LO -ml 21 -pad 0 -nh 50 -std True -bn True -do 0. -ws 9 -lr 1e-4 -bs 256 -ne 300 -tol 0.0001 -br 10
