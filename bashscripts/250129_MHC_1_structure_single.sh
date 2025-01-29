#!/bin/bash


basecommand="python3 train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/250128_MHCI_EL_structure_train_900k.csv -tef ../data/mhc1_el_subsample/250128_MHCI_EL_structure_test_100k.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 64 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel True --add_pfr False --add_fr_len False --add_pep_len False -lr 5e-5 --add_structure True --add_mean_structure False "

# Run 3 conditions :
# add nothing (BASELINE)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="MHCI_SingleStruct_rsa_only"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf -o $output_name -rid $rid -scols 'rsa'"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done

# 2 Add per position structure (variant 1)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="MHCI_SingleStruct_disorder_only"
# shellcheck disable=SC1009
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf -o $output_name -rid $rid -scols 'disorder'"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done
# 3 Add mean structure values (variant 2)

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="MHCI_SingleStruct_pq3"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf -o $output_name -rid $rid -scols pq3_H pq3_E pq3_C"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done