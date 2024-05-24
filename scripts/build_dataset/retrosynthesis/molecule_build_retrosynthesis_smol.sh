#!/bin/bash
#SBATCH --job-name=smolinst_rs
#SBATCH --output=smolinst_rs.txt
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

conda activate bioagent
python molecule_build_retrosynthesis_smol.py --data_dir /home/ys792/data/open-mol/SMolInst-Reactions/raw --repo_id OpenMol/SMol_RS_Filtered_825K_SMILES-MMChat --private --num_proc 20 --output_dir /home/ys792/data/open-mol/SMolInst-Reactions/rs_mmchat_smiles
