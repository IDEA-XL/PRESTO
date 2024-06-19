#!/bin/bash

# set as environment variables
export HF_HOME="/cto_labs/AIDD/cache"
export MOLECULE_2D_PATH="checkpoints/MoleculeSTM/"
MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH=checkpoints/$MODEL_VERSION
MODEL_CLS=LlamaLMMForCausalLM
# data path
DATA_DIR="/cto_labs/AIDD/DATA/React/USPTO/Interleaved"
# output path
OUTPUT_DIR="checkpoints/stage2/llava-moleculestm-$MODEL_VERSION-pretrain"
# load stage-1 projector
PROJECTOR_DIR="checkpoints/stage1/llava-moleculestm-$MODEL_VERSION-stage1/lmm_projector.bin"

NUM_GPUS=8
deepspeed --num_gpus=$NUM_GPUS scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --dataset_path $DATA_DIR \
    --data_mixture "uspto_rxn_interleaved" \
    --output_dir $OUTPUT_DIR \
    --pretrained_projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --report_to none \
    --deepspeed configs/zero2.json