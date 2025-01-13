#! /usr/bin/bash
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate pynn

cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts/
pids=()
for f in $(seq 0 4);
do
  python3 ./train_model.py -f ../data/NetMHCIIpan_train/drb_python.csv -kf ${f} -o 230804_BENCH_PYTHON -enc BL50LO -ml 21 -pad 0 -nh 50 -std True -bn True -do 0. -ws 9 -lr 1e-4 -bs 256 -ne 100 -tol 0.0001 -br 10 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

cd ../output/
ODIR='230804_speed_benchmark_python/'
mkdir $ODIR

for dir in $(ls ./ | grep 230804_BENCH_PYTHON);
do
  mv "${dir}" "${ODIR}/${dir}"
done

