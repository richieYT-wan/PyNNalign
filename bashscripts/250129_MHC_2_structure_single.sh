#!/bin/bash


basecommand="python3 train_test_model_efsinglepass.py -trf ../data/NetMHCII_EL_jonas/HLA_DR_subsample_all_partitions_922k.csv -tef ../data/NetMHCII_EL_jonas/HLA_DR_500K_unseen_sequence_test_set.csv -ml 21 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 64 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel True --add_pfr False --add_fr_len False --add_pep_len False -lr 5e-5 --add_structure True --add_mean_structure False"

add_mean_str=(True False)
add_pos_str=(True False)

# Run 3 conditions :
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="HLA_DR_SingleStruct_rsa_only"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf -o $output_name -rid $rid -scols 'rsa'"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="HLA_DR_SingleStruct_disorder_only"
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

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="HLA_DR_SingleStruct_pq3"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf -o $output_name -rid $rid -scols pq3_H pq3_E pq3_C"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done