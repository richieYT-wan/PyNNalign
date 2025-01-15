#!/bin/bash

# Base command without the parameters that will change
base_command="python train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/mhc1_el_1M_structure.csv -tef ../data/mhc1_el_subsample/test_structure.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 64 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel True --add_pfr True --add_fr_len True --add_pep_len True --add_structure False -lr 1e-5"

# Parameters to vary
add_str_values=(True False)

for add_str in "${add_str_values[@]}"; do
  # Generate a random ID for the job
  rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
  output_name="ReBenchmark_NH54_ExtData_AddMeanStruct_$add_str"
    
  for kf in {0..4}; do
    # Construct the full command with varying parameters
    command="$base_command -kf $kf --add_mean_structure $add_str --two_stage $add_str -o $output_name -rid $rid"
      
    # Create a script file for this job
    script_name="job_${output_name}_kf${kf}.sh"
    echo "#!/bin/bash" > $script_name
    echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
    echo "source activate cuda" >> $script_name
    echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
    echo "$command" >> $script_name
      
    # Submit the job script to qsub
    qsub_command="qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:gpus=1,mem=180gb,walltime=16:00:00 $script_name"
    echo "Submitting job: $script_name"
    eval $qsub_command
  done
done


base_command="python train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/mhc1_el_1M_structure.csv -tef ../data/mhc1_el_subsample/test_structure.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 54 -br 10 -otf True -cuda True -bs 1024 -ne 500 -wd 0 -add_ps True -indel True --add_pfr True --add_fr_len True --add_pep_len True --add_structure False -lr 1e-5 -bn True -do 0.25 -add_hl True -nh2 27"


# Parameters to vary
add_str_values=(True False)

for add_str in "${add_str_values[@]}"; do
  # Generate a random ID for the job
  rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
  output_name="ReBenchmark_NH54_ExtData_BNDOextraLayer_AddMeanStruct_$add_str"

  for kf in {0..4}; do
    # Construct the full command with varying parameters
    command="$base_command -kf $kf --add_mean_structure $add_str --two_stage $add_str -o $output_name -rid $rid"

    # Create a script file for this job
    script_name="job_${output_name}_kf${kf}.sh"
    echo "#!/bin/bash" > $script_name
    echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
    echo "source activate cuda" >> $script_name
    echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
    echo "$command" >> $script_name

    # Submit the job script to qsub
    qsub_command="qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:gpus=1,mem=180gb,walltime=16:00:00 $script_name"
    echo "Submitting job: $script_name"
    eval $qsub_command
  done
done
