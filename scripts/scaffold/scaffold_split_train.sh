#!/bin/bash
#SBATCH --job-name=scaffold_run
#SBATCH --array=1-2 # Run 2 tasks as a job array
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

# Define the split ratios
split_ratios=(0.5 0.25 0.125 0.0625 0.03125)

# Define the commands for each task
case ${SLURM_ARRAY_TASK_ID} in
    1)
        for ratio in "${split_ratios[@]}"
        do
            python scaffold_split_train.py --fs --split_ratios $ratio >> scaffold_fs_$ratio.txt 2>> scaffold_error_fs_$ratio.txt
        done
        ;;
    2)
        for ratio in "${split_ratios[@]}"
        do
            python scaffold_split_train.py --rs --split_ratios $ratio >> scaffold_rs_$ratio.txt 2>> scaffold_error_rs_$ratio.txt
        done
        ;;
esac