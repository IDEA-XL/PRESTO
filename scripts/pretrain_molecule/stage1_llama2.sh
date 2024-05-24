#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"

MODEL_VERSION=llama-2-7b
BASE_LLM_PATH=checkpoints/$MODEL_VERSION
MODEL_CLS=LlamaLMMForCausalLM
DATA_DIR="/cto_labs/AIDD/DATA/MolFM/pubchemsft_desc/stage1"
OUTPUT_DIR="checkpoints/stage1/llava-moleculestm-$MODEL_VERSION-stage1"

NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --dataset_path $DATA_DIR \
    --data_mixture "pubchem_cap" \
    --output_dir $OUTPUT_DIR \
    --pretrain_projectors \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 5 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --logging_steps 1 \
    --report_to none \
    --deepspeed configs/zero2.json