#!/bin/bash
basecommand="python3 train_test_model_efsinglepass.py -trf ../data/NetMHCII_EL_jonas/HLA_DR_subsample_all_partitions_922k.csv -tef ../data/NetMHCII_EL_jonas/HLA_DR_500K_unseen_sequence_test_set.csv -ml 21 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 64 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel False --add_pfr True --add_fr_len True --add_pep_len True -lr 5e-5"

on="RERUNS_FEB3_CORRECT_HYPERPARAMS"
# Run 3 conditions :
# add nothing (BASELINE)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="$on-HLA_DR_baseline"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure False --add_mean_structure False --two_stage False -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done

# 2 Add per position structure (variant 1)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="$on-HLA_DR_PerPositionStructure"
# shellcheck disable=SC1009
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure True --add_mean_structure False --two_stage False -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done
# 3 Add mean structure values (variant 2)

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="$on-HLA_DR_AddMeanStructure"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure False --add_mean_structure True --two_stage True -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
done