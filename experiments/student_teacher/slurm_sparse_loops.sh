#!/bin/bash
# slurm_sparse_loops.sh > sparse_params.txt
human_date=$(date +"%a_%d_%b_%Y_%H%M" | tr -d '\r')    # Human-friendly: Thu-05-Oct-2023
TALAPAS_PATH="/home/mtrappet/tau/StudentTeacher/single_layer_sparse/run_date_${human_date}/"

d_hs=200
sparsities=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) #(0.0 0.5) # 

for sparsity in "${sparsities[@]}"; do
    echo " $d_hs $sparsity determ \"$TALAPAS_PATH\""
    echo " $d_hs $sparsity random \"$TALAPAS_PATH\""
done
