#!/bin/bash

NUM_LINES=$(cat params.txt | wc -l)

sbatch --array=1-$NUM_LINES futures_sbatch.sh

