import logging
import transformers
import torch

from bioagent.modalities import MODALITY_BUILDERS
from bioagent.model_utils import fix_tokenizer
from bioagent.data import (
    Dataset, 
    LMMDataset, 
    DataCollatorForSupervisedLMMDataset, 
    LMMInterleavedDataset,
    make_supervised_data_module,
)
from bioagent.training import TrainingArguments
from tqdm import tqdm

# export CHAT_TEMPLATE_PATH=/cto_labs/AIDD/chat_templates/vicuna.jinja

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_max_length = 2048
    
    data_args = TrainingArguments(
        data_mixture="pretrain_v2",
        output_dir="/tmp/nc",
    )   
    llama_path = "checkpoints/vicuna-7b-v1.5"
    
    # load modalities
    modalities = MODALITY_BUILDERS["molecule_2d"]()
    for m in modalities:
        m.to(
            dtype=torch.bfloat16,
            device=device,
        )
    
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        llama_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    fix_tokenizer(tokenizer)
    
    # load dataset
    data_module = make_supervised_data_module(tokenizer, data_args, modalities)
    dataset = data_module["train_dataset"]
    
    # iterate over the dataset
    miss_match = 0
    for i, sample in enumerate(tqdm(dataset)):
        try:
            pass
        except Exception as e:
            breakpoint()
            miss_match += 1
            continue
    print(f"Miss match: {miss_match}")