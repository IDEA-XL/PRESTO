from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import os

from torch.utils.data import Dataset, ConcatDataset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch

from bioagent.modalities.base_modality import Modality
from bioagent.constants import IGNORE_INDEX
from bioagent.data_tools import encode_chat, encode_interleaved_data
import bioagent.data.data_mixture as datasets_mixture


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_mixture: str = field(
        default="pubchem_cap", metadata={"help": "Datasets mixture to use."}
    )


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")


class LMMDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
    ):
        super(LMMDataset, self).__init__()
        self.dataset = _resolve_dataset(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.modalities = modalities

    def __len__(self):
        return len(self.dataset)

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            return encode_chat(item, self.tokenizer, self.modalities)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)
        

class LMMInterleavedDataset(LMMDataset):
    r"""
    Interleaved dataset for LMM pretraining. Each sample is a naive concatenation of multiple modality tokens
    and the surrounding text (Not Chat Format). The modality tokens are interleaved with the text tokens.
    """
    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            return encode_interleaved_data(item, self.tokenizer, self.modalities)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)
        

class LMMConcatDataset(ConcatDataset):
    def get_example(self) -> Dict:
        return self.datasets[0][0]


@dataclass
class DataCollatorForSupervisedLMMDataset:
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ["input_ids", "labels"]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # (modality, batch_size, instance_idx, x/edge_index/edge_attr)
        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments,
                                modalities: List[Modality],
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and 
    modified by Jason Lu and Haotian Tang."""
    all_datasets = []
    extra_info = []
    datasets_mixture.register_datasets_mixtures()
    mixture = datasets_mixture.DATASETS_MIXTURES[data_args.data_mixture]
    for dataset in mixture:
        dataset_type = dataset.dataset_type
        if os.path.exists(dataset.data_path):
            data_args.dataset_path = dataset.data_path
        if dataset_type == "cap":
            dataset_cls = LMMDataset
        elif dataset_type == "sft":
            dataset_cls = LMMDataset
        elif dataset_type == "interleaved":
            dataset_cls = LMMInterleavedDataset
        else:
            raise NotImplementedError
        train_dataset = dataset_cls(tokenizer=tokenizer, modalities=modalities, data_args=data_args,)
        all_datasets.append(train_dataset)
        extra_info.append(len(train_dataset))
    
    all_datasets = LMMConcatDataset(all_datasets)

    data_collator = DataCollatorForSupervisedLMMDataset(tokenizer=tokenizer, modalities=modalities)
    return dict(train_dataset=all_datasets,
                eval_dataset=None,
                data_collator=data_collator)