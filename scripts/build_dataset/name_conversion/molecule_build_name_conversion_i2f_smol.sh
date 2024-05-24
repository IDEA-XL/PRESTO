#!/bin/bash
#SBATCH --job-name=smolinst_nc_i2f
#SBATCH --output=smolinst_nc_i2f.txt
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=pi_gerstein

conda activate bioagent
python molecule_build_name_conversion_i2f_smol.py --data_dir /gpfs/gibbs/pi/gerstein/xt86/bioagent/data/SMolInstruct/raw --repo_id OpenMol/SMol_I2F_270K-MMChat --private --num_proc 20 --output_dir /home/ys792/data/open-mol/SMolInst-NC/i2f_mmchat_smiles
