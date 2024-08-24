import transformers
from dataclasses import dataclass, field
from typing import Optional


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
    pretrain_projectors: bool = field(
        default=False,
        metadata={"help": "Whether to pretrain projectors."},
    )
    pretrained_projectors_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained projectors."},
    )
    bits: int = field(
        default=16,
        metadata={"help": "Number of bits to use."},
    )
    lora_enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA adapters."},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Dimension of the LoRA adapters."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Number of heads in the LoRA adapters."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout rate for the LoRA adapters."},
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "Path to the pretrained LoRA adapter weights."},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for the LoRA adapters."},
    )
    training_mode: str = field(
        default="sft",
        metadata={"help": "The training mode to use (sft or dpo)"}
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model for DPO training"}
    )
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for DPO loss"}
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt for DPO training"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length of the entire input sequence for DPO training"}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={"help": "The name or path of the model."}
    )
    model_cls: str = field(
        default="MistralLMMForCausalLM",
        metadata={"help": "The class name of the model."},
    )
    modality_builder: str = field(
        default="vision_clip",
        metadata={"help": "The modality builder."},
    )
    model_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the model LoRA weights."},
    )
    projectors_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the projectors."},
    )
