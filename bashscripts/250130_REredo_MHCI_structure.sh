#!/bin/bash
basecommand="python3 train_test_model_efsinglepass.py -trf /home/projects/vaccine/people/yatwan/PyNNalign/data/netmhci4.1_wcontext/250130_MHCI_structure_train_900k_fixed_partitions.csv -tef /home/projects/vaccine/people/yatwan/PyNNalign/data/netmhci4.1_wcontext/250130_MHCI_structure_test_86944_fixed_overlap.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 64 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel True --add_pfr False --add_fr_len False --add_pep_len False -lr 5e-5"

# Run 3 conditions :
# add nothing (BASELINE)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="REDO_MHCI_structRedo_baseline"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure False --add_mean_structure False --two_stage False -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
  $QSUB nodes=1:ppn=40:gpus=1,mem=180gb,walltime=15:00:00 $script_name
done

# 2 Add per position structure (variant 1)
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="REDO_MHCI_structRedo_PerPositionStructure"
# shellcheck disable=SC1009
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure True --add_mean_structure False --two_stage False -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
  $QSUB nodes=1:ppn=40:gpus=1,mem=180gb,walltime=15:00:00 $script_name

done
# 3 Add mean structure values (variant 2)

rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
output_name="REDO_MHCI_structRedo_AddMeanStructure"
for kf in {0..4};do
  script_name="job_${output_name}_kf${kf}.sh"
  command="$basecommand -kf $kf --add_structure False --add_mean_structure True --two_stage True -o $output_name -rid $rid"
  echo "#!/bin/bash" > $script_name
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
  echo "source activate cuda" >> $script_name
  echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
  echo "$command" >> $script_name
  $QSUB nodes=1:ppn=40:gpus=1,mem=180gb,walltime=15:00:00 $script_name
done