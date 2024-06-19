#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"
MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH="checkpoints/stage2/llava-moleculestm-$MODEL_VERSION-pretrain_rxn_nc"
MODEL_CLS=LlamaLMMForCausalLM

# output path
SFT_VERSION=rxn_nc_subset
OUTPUT_DIR="checkpoints/sft/llava-moleculestm-$MODEL_VERSION-sft-$SFT_VERSION"

NUM_GPUS=8
deepspeed --num_gpus=$NUM_GPUS scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --data_mixture "sft_subset" \
    --output_dir $OUTPUT_DIR \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 3 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --report_to none \
    --deepspeed configs/zero2.json