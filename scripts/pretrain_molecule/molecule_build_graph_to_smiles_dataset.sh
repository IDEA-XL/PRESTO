#!/bin/bash
#SBATCH --job-name=pubchem_g2s
#SBATCH --output=pubchem_g2s.txt
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

python molecule_build_graph_to_smiles_dataset.py --qa_path /home/ys792/data/open-mol/PubChemSFT/all_clean.json --repo_id OpenMol/PubChem_G2S_300K_SMILES-MMPretrain --output_dir /home/ys792/data/open-mol/SMolInst-NC/g2s_mmchat_smiles --private --num_proc 4