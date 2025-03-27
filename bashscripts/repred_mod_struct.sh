#!/bin/bash
source /home/people/riwa/anaconda3/etc/profile.d/conda.sh
source activate cuda
cd ../pyscripts
python3 ensemble_predictions.py -cuda True -tef ../data/NetMHCII_EL_jonas/modified_struct/250327_HLA_DR_test_top10k_modified_disorder.csv -model_folder ../models/HLA_DR_PerPositionStructure_IleIo/ -o mod_disorder
python3 ensemble_predictions.py -cuda True -tef ../data/NetMHCII_EL_jonas/modified_struct/250327_HLA_DR_test_top10k_modified_rsa.csv -model_folder ../models/HLA_DR_PerPositionStructure_IleIo/ -o mod_rsa
python3 ensemble_predictions.py -cuda True -tef ../data/NetMHCII_EL_jonas/modified_struct/250327_HLA_DR_test_top10k_modified_pq3_H.csv -model_folder ../models/HLA_DR_PerPositionStructure_IleIo/ -o mod_pq3_H
python3 ensemble_predictions.py -cuda True -tef ../data/NetMHCII_EL_jonas/modified_struct/250327_HLA_DR_test_top10k_modified_pq3_E.csv -model_folder ../models/HLA_DR_PerPositionStructure_IleIo/ -o mod_pq3_E
python3 ensemble_predictions.py -cuda True -tef ../data/NetMHCII_EL_jonas/modified_struct/250327_HLA_DR_test_top10k_modified_pq3_C.csv -model_folder ../models/HLA_DR_PerPositionStructure_IleIo/ -o mod_pq3_C