from typing import List, Type, Dict
import logging
import subprocess
import torch
import os

import transformers
from transformers import PreTrainedModel

from presto.model_utils import (
    make_model_lora,
    fix_tokenizer,
    get_mm_adapter_state,
)
from presto.modalities.base_modality import Modality
from presto.data import LMMDataset
from presto.hparams import TrainingArguments, ModelArguments


README_TEMPLATE = """
---
license: apache-2.0
base_model: {base_model}
dataset: {dataset}
tags:
  - finetuned
  - multimodal
inference: false
---

These are weights for a version of `{base_model}` finetuned for multimodal applications. 

### Modalities

{modalities}

### Usage

GitHub: https://github.com/open-mol/presto (includes training scripts and basic inference server)

### Dataset

{dataset} ({num_examples} examples)

```
{dataset_example}
```

### Training Device(s)

```
{training_devices_dump}
```


### Model

```
{repr_model}
```

"""


def _get_training_devices_dump() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=gpu_name,gpu_bus_id,vbios_version", "--format=csv"]
    )
    return out.decode("utf-8").strip()


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "pretrain_projectors", False):
        # Only save Adapter
        keys_to_match = ['_lmm_projector']

        weight_to_save = get_mm_adapter_state(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                lmm_projector_folder = os.path.join(parent_folder, "lmm_projector")
                os.makedirs(lmm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(lmm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'lmm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def save_model_metadata(model_cls: Type[PreTrainedModel], 
                        training_args, 
                        model_args, 
                        modalities: List[Modality], 
                        data_module: Dict[str, LMMDataset],
                        model: PreTrainedModel) -> None:
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "model_named_parameters.txt"), "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name} {param.shape} {param.requires_grad}\n")

    with open(os.path.join(training_args.output_dir, "README.md"), "w") as f:
        modalities_text = [
            f"* {m.__class__.__name__} (use `{m.token}` in text and provide `{m.data_key}`"
            for m in modalities
        ]
        dataset = data_module["train_dataset"]
        readme_text = README_TEMPLATE.format(
            base_model=model_args.model_name_or_path,
            dataset=training_args.data_mixture or training_args.dataset_name,
            dataset_example=repr(dataset.get_example()),
            num_examples=len(dataset),
            modalities="\n".join(modalities_text),
            training_devices_dump=_get_training_devices_dump(),
            repr_model=f"{model_cls.__name__}.model =\n\n{repr(model)}",
        )
        f.write(readme_text)


def load_model_and_tokenizer_for_training(
    model_cls: Type[PreTrainedModel],
    model_args: ModelArguments,
    training_args: TrainingArguments,
    modalities: List[Modality]
) -> PreTrainedModel:

    for m in modalities:
        m.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    fix_tokenizer(tokenizer)


    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.modalities = modalities
    model.config.use_cache = False
    model.config.model_cls = model_cls.__name__
    model.config.modality_builder = model_args.modality_builder

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.model_lora_path:
        raise ValueError(
            "LoRA path not supported for training -- set the output path to an existing model to resume training"
        )

    if training_args.lora_enable:
        logging.info("Adding LoRA adapters...")
        model = make_model_lora(model, training_args)

    if training_args.pretrained_projectors_path:
        projector_weights = torch.load(
            training_args.pretrained_projectors_path, map_location="cpu"
        )
        projector_weights = {
            k: v for k, v in projector_weights.items() if "_lmm_projector" in k
        }
    else:
        projector_weights = {}

    model.get_model().initialize_modules(modalities, projector_weights)

    if training_args.pretrain_projectors:
        model.requires_grad_(False)
        for m in modalities:
            proj = getattr(model.get_model(), m.name + "_lmm_projector")
            for p in proj.parameters():
                p.requires_grad = True

    return model