#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"

MODEL_VERSION=vicuna-7b-v1.5
TRAIN_VERSION=sft-skip-stage2
BASE_LLM_PATH="checkpoints/sft/llava-moleculestm-$MODEL_VERSION-$TRAIN_VERSION/epoch-3"
PROJECTOR_DIR="$BASE_LLM_PATH/non_lora_trainables.bin"


# log path
TASK=forward_reaction_prediction
LOG_DIR="./logs/scaffold/$TASK/"
DATA_DIR="/cto_labs/AIDD/DATA/React/MolInstruct/forward_scaffold_mmchat_smiles/"
python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose 


# retro
TASK=retrosynthesis
LOG_DIR="./logs/scaffold/$TASK/"
DATA_DIR="/cto_labs/AIDD/DATA/React/MolInstruct/retro_scaffold_mmchat_smiles/"
python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose