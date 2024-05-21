#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"

DATASET=clintox
TASK=icl_pp_$DATASET
MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH="checkpoints/stage2/llava-moleculestm-vicuna-7b-v1.5-pretrain_rxn_nc"
# eval
N_SHOT=4
LOG_DIR="./logs/full/ICL/$TASK/${N_SHOT}"
DATA_DIR="/cto_labs/AIDD/DATA/ChemLLMBench/ICL/pp/${DATASET}_scaffold_${N_SHOT}shot_int/test"
for((i=0;i<3;i++)); do
    echo "Iteration $i"
    python scripts/evaluate_model.py \
        --model_name_or_path $BASE_LLM_PATH \
        --lora_enable False \
        --dataset_path  $DATA_DIR \
        --max_new_tokens 4 \
        --cache_dir $LOG_DIR/iter_$i \
        --output_dir $LOG_DIR/iter_$i \
        --evaluator "classification" \
        --parser "classification" \
        --is_icl True 
done

N_SHOT=8
LOG_DIR="./logs/full/ICL/$TASK/${N_SHOT}"
DATA_DIR="/cto_labs/AIDD/DATA/ChemLLMBench/ICL/pp/${DATASET}_scaffold_${N_SHOT}shot_int/test"
for((i=0;i<3;i++)); do
    echo "Iteration $i"
    python scripts/evaluate_model.py \
        --model_name_or_path $BASE_LLM_PATH \
        --lora_enable False \
        --dataset_path  $DATA_DIR \
        --max_new_tokens 4 \
        --cache_dir $LOG_DIR/iter_$i \
        --output_dir $LOG_DIR/iter_$i \
        --evaluator "classification" \
        --parser "classification" \
        --is_icl True 
done