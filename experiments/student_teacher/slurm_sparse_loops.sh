#!/bin/bash
# slurm_sparse_loops.sh > sparse_params.txt
human_date=$(date +"%a_%d_%b_%Y")    # Human-friendly: Thu-05-Oct-2023
TALAPAS_PATH="/home/mtrappet/tau/StudentTeacher/sparse_run_date_${human_date}/"


params=(
    "4 2 0"
    "4 3 1"
    "3 2 1"
    "6 4 2"
)

for tuple in "${params[@]}"; do
    read d_hs n_active shared mask_type <<< "$tuple"
    echo "$d_hs $n_active $shared "determ" \"$TALAPAS_PATH\""
    echo "$d_hs $n_active $shared "random" \"$TALAPAS_PATH\""
done
