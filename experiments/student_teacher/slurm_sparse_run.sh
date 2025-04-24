#!/bin/bash

#SBATCH --partition=gpu,gpulong
#SBATCH --job-name=StuTea1layer
#SBATCH --output=StuTea1layer/%x-%A-%a.out
#SBATCH --error=StuTea1layer/%x-%A-%a.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=70G

#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --account=tau  ### Account used for job submission

#SBATCH --mail-type=all
#SBATCH --mail-user=mtrappet@uoregon.edu

module load cuda/12.4.1
source /home/mtrappet/BranchGating/data-science/bin/activate
params=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sparse_params.txt)

python /home/mtrappet/stu_teach/RepresentationSimilarityCL/experiments/student_teacher/sparsity $params