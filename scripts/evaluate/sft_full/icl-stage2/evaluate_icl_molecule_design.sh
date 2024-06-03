#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"

TASK=icl_molecule_design
MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH="checkpoints/stage2/llava-moleculestm-vicuna-7b-v1.5-pretrain_all"

# log path
# LOG_DIR="./logs/full/ICL/$TASK/3shot"
# DATA_DIR="/cto_labs/AIDD/DATA/ChemLLMBench/ICL/molecule_design/3shot/test"
# python scripts/evaluate_model.py \
#     --model_name_or_path $BASE_LLM_PATH \
#     --lora_enable False \
#     --dataset_path  $DATA_DIR \
#     --max_new_tokens 256 \
#     --cache_dir $LOG_DIR \
#     --output_dir $LOG_DIR \
#     --evaluator "smiles" \
#     --is_icl True 

LOG_DIR="./logs/full/ICL/$TASK/5shot"
DATA_DIR="/cto_labs/AIDD/DATA/ChemLLMBench/ICL/molecule_design/5shot/test"
python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --lora_enable False \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --is_icl True 