from typing import Optional, List
from dataclasses import field, dataclass
import logging
import subprocess
import pathlib
import torch
import shutil
import glob
import os

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer, TrainerCallback

from presto.data import (
    make_supervised_data_module,
)
from presto.model_utils import (
    make_model_lora,
    get_peft_state,
    get_peft_state_non_lora,
    fix_tokenizer,
    get_peft_state,
    get_peft_state_non_lora,
    get_mm_adapter_state,
)
from presto.modalities.base_modality import Modality


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

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    dataset_name: str = field(
        default=None, metadata={"help": "Single dataset to use."}
    )
    data_mixture: str = field(
        default=None, metadata={"help": "Datasets mixture to use."}
    )
    eval_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    pretrain_projectors: bool = field(default=False)
    pretrained_projectors_path: Optional[str] = field(default=None)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.1")
    model_cls: str = field(default="MistralLMMForCausalLM")
    modality_builder: str = field(default="vision_clip")
    model_lora_path: Optional[str] = field(default=None)
    projectors_path: Optional[str] = field(default=None)


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


class LMMTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters())
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )


def _get_training_devices_dump() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=gpu_name,gpu_bus_id,vbios_version", "--format=csv"]
    )
    return out.decode("utf-8").strip()


def train_for_modalities(
    model_cls,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    modalities: List[Modality],
):
    global local_rank
    local_rank = training_args.local_rank
    
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

    data_module = make_supervised_data_module(tokenizer, training_args, modalities)
    data_module = make_supervised_data_module(tokenizer, training_args, modalities)

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

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "model_named_parameters.txt"), "w"
    ) as f:
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
    
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            if args.lora_enable:
                state_dict = get_peft_state(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora(
                    model.named_parameters()
                )
                if args.local_rank in [-1, 0]:
                    model.config.save_pretrained(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))

    trainer = LMMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[SaveCallback()],
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)