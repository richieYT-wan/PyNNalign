#!/bin/bash

### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 231205_readdata
### Number of nodes
#PBS -l nodes=1:ppn=40
### Memory
#PBS -l mem=120gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 1 hour)
#PBS -l walltime=20:00:00


id=231205_readMSdata
#PBS -e $id.err
#PBS -o $id.log

# Load anaconda modules
module load tools
module load anaconda3/4.4.0

# Change the directory
cd /home/projects/vaccine/people/cadsal/NNAlign_SpecialCourse/PyNNalign/pyscripts
pwd

echo \"Starting PyScript\"

for i in {1..1};
do
    echo "Running iteration ${i}"
    python3 ./read_data.py -trf ../data/carlos/MS_C00_data_noUZB.txt -tef ../data/carlos/short_random_peptides_fakemhc.txt -ml 21 -nh 60 -std f -bn False -efnh 5 -o read_data_try -x Sequence -y BA -enc BL50LO -fc pseudoseq -add_ps True -ps pseudoseq -ws 9 -add_pfr True -add_fr_len True -add_pep_len True -add_hl True -nh2 30 -wd 1e-4 -bs 256 -br 10 -kf ${i}
    echo "Iteration $i completed"
done

echo "Script finished"
