#!/bin/bash

# Base command without the parameters that will change
base_command="python train_test_model_efsinglepass.py -trf ../data/mhc1_el_subsample/mhc1_el_5M.csv -tef ../data/mhc1_el_subsample/test_data.csv -ml 13 -ws 9 -pad -20 -y target -x sequence -std False -bn False -nh 50 -efnh 10 -br 10 -otf True -cuda True -bs 1024 -lr 1e-5 -ne 180"

# Parameters to vary
add_ps_values=(True False)
indel_values=(True False)

for add_ps in "${add_ps_values[@]}"; do
  for indel in "${indel_values[@]}"; do
    # Generate a random ID for the job
    rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
    output_name="addps_${add_ps}_indel_${indel}"
    
    for kf in {0..4}; do
      # Construct the full command with varying parameters
      command="$base_command -kf $kf -add_ps $add_ps -indel $indel -o $output_name -rid $rid"
      
      # Create a script file for this job
      script_name="job_${output_name}_kf${kf}.sh"
      echo "#!/bin/bash" > $script_name
      echo "source /home/projects/vaccine/people/pasbes/PyNNalign/myenv/bin/activate" >> $script_name
      echo "cd /home/projects/vaccine/people/pasbes/PyNNalign/pyscripts" >> $script_name
      echo "$command" >> $script_name
      
      # Submit the job script to qsub
      qsub_command="qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:gpus=1,mem=180gb,walltime=16:00:00 $script_name"
      echo "Submitting job: $script_name"
      eval $qsub_command
    done
  done
done

echo "All tasks submitted."

