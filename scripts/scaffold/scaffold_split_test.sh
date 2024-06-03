#!/bin/bash

#SBATCH --job-name=scaffold_run
#SBATCH --array=1-2 # Run 2 tasks as a job array
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

# Define the commands for each task
case ${SLURM_ARRAY_TASK_ID} in
    1)
        python scaffold_split_test.py --fs --num_proc 20 > scaffold_fs.txt 2> scaffold_error_fs.txt
        ;;
    2)
        python scaffold_split_test.py --rs --num_proc 20 > scaffold_rs.txt 2> scaffold_error_rs.txt
        ;;
esac