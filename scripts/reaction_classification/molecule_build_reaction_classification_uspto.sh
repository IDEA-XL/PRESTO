#!/bin/bash
#SBATCH --job-name=uspto_tpl_mmchat_smiles
#SBATCH --output=uspto_tpl_mmchat_smiles.txt
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

conda activate bioagent
python molecule_build_reaction_classification_uspto.py --data_dir OpenMol/USPTO_1k_TPL-SFT --repo_id OpenMol/USPTO_1k_TPL_SMILES-MMChat --private --num_proc 10 --output_dir /home/ys792/data/open-mol/SMolInst-Reactions/uspto_tpl_mmchat_smiles