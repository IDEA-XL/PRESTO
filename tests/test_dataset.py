import logging
import transformers
import torch
import pytest

from bioagent.modalities import MODALITY_BUILDERS
from bioagent.model_utils import fix_tokenizer
from bioagent.data import (
    Dataset, 
    LMMDataset, 
    DataCollatorForSupervisedLMMDataset, 
    LMMInterleavedDataset,
    make_supervised_data_module,
    _DATASETS,
    _MIXTURES
)
from bioagent.training import TrainingArguments
from tqdm import tqdm



@pytest.fixture
def setup():
    tokenizer = transformers.AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    modalities = []
    data_args = TrainingArguments(
        output_dir="/tmp/sft",
    )
    return tokenizer, modalities, data_args


@pytest.mark.parametrize("dataset_name", _DATASETS)
def test_load_individual_datasets(setup, dataset_name):
    if dataset_name in ["pubchem_cap", "uspto_rxn"]:
        pytest.skip("Skipping dataset")
    tokenizer, modalities, data_args = setup
    data_args.dataset_name = dataset_name
    data_module = make_supervised_data_module(tokenizer, data_args, modalities)
    assert data_module["train_dataset"] is not None
    assert len(data_module["train_dataset"]) > 0
    assert data_module["eval_dataset"] is not None
    assert len(data_module["eval_dataset"]) > 0


@pytest.mark.parametrize("mixture_name", _MIXTURES)
def test_load_dataset_mixtures(setup, mixture_name):
    if mixture_name in ["pubchem_cap", "uspto_rxn_interleaved", "pretrain_v2", "pretrain_v3"]:
        pytest.skip("Skipping dataset")
    tokenizer, modalities, data_args = setup
    data_args.data_mixture = mixture_name
    data_module = make_supervised_data_module(tokenizer, data_args, modalities)
    assert data_module["train_dataset"] is not None
    assert len(data_module["train_dataset"]) > 0

# export CHAT_TEMPLATE_PATH=/cto_labs/AIDD/chat_templates/vicuna.jinja

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_max_length = 2048
    data_args = TrainingArguments(
        data_mixture="sft_subset",
        output_dir="/tmp/sft",
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