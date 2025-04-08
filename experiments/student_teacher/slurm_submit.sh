#!/bin/bash

NUM_LINES=$(cat sparse_params.txt | wc -l)

sbatch --array=1-$NUM_LINES slurm_sparse_run.sh