#!/bin/bash
source pyvenv/bin/activate
module load cuda/12.1

# Configuration arrays
overlap_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through all combinations
for overlap in "${overlap_values[@]}"; do
    echo "------------------------------------------------------------"
    echo "Starting run: overlap=$overlap"
    echo "------------------------------------------------------------"
    
    # Run the Python script with current parameters
    python experiments/student_teacher/run_cl.py \
        --overlap "$overlap" \
        --g_type "overlap" \

    echo "Completed: overlap=$overlap"
    echo ""
done


echo "All experiments completed!"