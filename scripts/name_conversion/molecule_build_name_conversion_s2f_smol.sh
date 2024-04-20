#!/bin/bash
#SBATCH --job-name=smolinst_nc_s2f
#SBATCH --output=smolinst_nc_s2f.txt
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

conda activate bioagent
python molecule_build_name_conversion_s2f_smol.py --data_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/SMolInstruct/raw --out_dir /home/ys792/data/open-mol/SMolInst-NC/s2f_mmchat_smiles