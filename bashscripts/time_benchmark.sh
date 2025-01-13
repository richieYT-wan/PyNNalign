#!/bin/bash

cd /home/projects/vaccine/people/yatwan/PyNNalign/bashscripts/

# Number of times to run each script
NUM_RUNS=5

# Script 1
SCRIPT1="./benchmark_python.sh"

# Script 2
SCRIPT2="./benchmark_morni.sh"


# Function to run a script and calculate average execution time
run_script() {
  local script=$1
  local total_time=0

  for ((i=1; i<=NUM_RUNS; i++)); do
    #echo "Running $script - Run $i..."
    # Capture the output of 'time' command to a temporary file
    tmp_file=$(mktemp)
    { time ./"$script" ; } 2> "$tmp_file" # Redirect time's output to /dev/null
    # Extract the real time value from the temporary file using grep and awk
    real_time=$(grep "real" "$tmp_file" | awk '{print $2}')
    rm "$tmp_file"  # Clean up temporary file
    #!/bin/bash
    # Extract minutes and seconds using awk
    minutes=$(echo "$real_time" | awk -F'm' '{print $1}')
    seconds=$(echo "$real_time" | awk -F'm' '{print $2}' | sed 's/[^0-9.]//g')

    # Convert minutes to seconds and add the seconds
    total_seconds=$(echo "$minutes * 60 + $seconds" | bc)

    #echo "Total seconds: $total_seconds"
    total_time=$(awk "BEGIN {print $total_time + $total_seconds}")
  done

  # Calculate the average execution time
  average_time=$(awk "BEGIN {print $total_time / $NUM_RUNS}")
  echo "Total execution time for $script: $total_time seconds"
  echo "Average execution time for $script: $average_time seconds"
}

# Run the scripts and get average execution time
run_script "$SCRIPT1" > /home/projects/vaccine/people/yatwan/PyNNalign/benchmark_python.txt
run_script "$SCRIPT2" > /home/projects/vaccine/people/yatwan/PyNNalign/benchmark_morni.txt

