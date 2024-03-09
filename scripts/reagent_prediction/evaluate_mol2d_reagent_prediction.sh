#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/gpfs/gibbs/pi/gerstein/xt86/bioagent/checkpoints/MoleculeSTM/"

MODEL_VERSION=lmsys/vicuna-7b-v1.5
MODEL_CLS=LlamaLMMForCausalLM
DATA_DIR="/gpfs/gibbs/pi/gerstein/xt86/bioagent/data/Mol-Instructions/data/Molecule-oriented_Instructions/retrosynthesis/test"
LOG_DIR="./logs/forward_reaction_prediction"
PEFT_MODEL_DIR="/gpfs/gibbs/pi/gerstein/xt86/bioagent/checkpoints/llava-moleculestm-$MODEL_VERSION-retrosynthesis"
PROJECTOR_DIR="/gpfs/gibbs/pi/gerstein/xt86/bioagent/checkpoints/llava-moleculestm-$MODEL_VERSION-pretrain/non_lora_trainables.bin"

python ../evaluate_model.py \
    --model_name_or_path $MODEL_VERSION \
    --model_lora_path $PEFT_MODEL_DIR \
    --pretrained_projectors_path $PROJECTOR_DIR \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 4096 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose \