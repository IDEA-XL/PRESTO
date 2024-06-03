# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from bioagent.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import transformers
import logging

from bioagent.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from bioagent.data import DataArguments
from bioagent.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from bioagent.modalities import MODALITY_BUILDERS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    modalities = MODALITY_BUILDERS[model_args.modality_builder]()
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_modalities(model_cls, training_args, model_args, modalities)
