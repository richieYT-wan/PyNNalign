#!/bin/bash

# Base command without the parameters that will change

o1="ensemble_preds_Nh64_BaseModel_NoMeanStruct"
c1="python3 ensemble_predictions.py -tef ../data/test_sets_ligands_abelin_wo_context/merged_test_sets_ligands_abelin_no_U.csv -model_folder ../models/Nh64_BaseModel_NoMeanStruct/ -o $o1"

# Generate a random ID for the job
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
# Construct the full command with varying parameters
command="$c1 $rid"

# Create a script file for this job
script_name="job_$o1.sh"
echo "#!/bin/bash" > $script_name
echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
echo "source activate cuda" >> $script_name
echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
echo "$command" >> $script_name

# Submit the job script to qsub
qsub_command="qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:gpus=1,mem=180gb,walltime=16:00:00 $script_name"
echo "Submitting job: $script_name"
eval $qsub_command



o2="ensemble_preds_Nh64_BnDoEL_NoMeanStruct"
c2="python3 ensemble_predictions.py -tef ../data/test_sets_ligands_abelin_wo_context/merged_test_sets_ligands_abelin_no_U.csv -model_folder ../models/Nh64_BnDoEL_NoMeanStruct/ -o $o2"

# Generate a random ID for the job
rid=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)
# Construct the full command with varying parameters
command="$c2 $rid"

# Create a script file for this job
script_name="job_$o2.sh"
echo "#!/bin/bash" > $script_name
echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> $script_name
echo "source activate cuda" >> $script_name
echo "cd /home/projects/vaccine/people/yatwan/PyNNalign/pyscripts" >> $script_name
echo "$command" >> $script_name

# Submit the job script to qsub
qsub_command="qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:gpus=1,mem=180gb,walltime=16:00:00 $script_name"
echo "Submitting job: $script_name"
eval $qsub_command

echo "All tasks submitted."

