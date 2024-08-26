from typing import Optional, List, Type, Dict
from dataclasses import field, dataclass
import pathlib
import torch
import shutil
import glob
import os

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer, TrainerCallback, PreTrainedModel
from trl import DPOTrainer

from presto.data import (
    make_supervised_data_module,
)
from presto.model_utils import (
    get_peft_state,
    get_peft_state_non_lora,
    get_peft_state,
    get_peft_state_non_lora,
)
from presto.modalities.base_modality import Modality
from presto.hparams import TrainingArguments, ModelArguments
from presto.trainer_utils import safe_save_model_for_hf_trainer, save_model_metadata, load_model_and_tokenizer_for_training


local_rank = None

class LMMSupervisedTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMSupervisedTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMSupervisedTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters())
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )

class LMMDPOTrainer(DPOTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMDPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMDPOTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters())
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )


def train_for_modalities(
    model_cls,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    modalities: List[Modality],
):
    global local_rank
    local_rank = training_args.local_rank

    model, tokenizer = load_model_and_tokenizer_for_training(model_cls, model_args, training_args)
    data_module = make_supervised_data_module(tokenizer, training_args, modalities)
    save_model_metadata(model_cls, training_args, model_args, modalities, data_module, model)
    
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

    if training_args.training_mode == "sft":
        trainer = LMMSupervisedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[SaveCallback()],
            **data_module,
        )
    elif training_args.training_mode == "dpo":
        ref_model, _ = load_model_and_tokenizer_for_training(model_cls, model_args, training_args, modalities) if training_args.ref_model else None
        trainer = LMMDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            beta=training_args.dpo_beta,
            tokenizer=tokenizer,
            callbacks=[SaveCallback()],
            **data_module,
        )
    else:
        raise ValueError(f"Unknown training mode: {training_args.training_mode}")

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
