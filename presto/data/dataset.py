# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import os

import transformers
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import  ConcatDataset, Subset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset

from presto.modalities.base_modality import Modality
from presto.constants import IGNORE_INDEX
from presto.data_tools import encode_chat, encode_interleaved_data

_DATASETS = {}
_MIXTURES = {}
DATASET_BASE_DIR="/cto_labs/AIDD/DATA/React/InstructChemReact/"

@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data. (Will be deprecated in future versions)"}
    )


class DatasetType(Enum):
    CHAT = "chat"
    INTERLEAVED = "interleaved"


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default=DatasetType.CHAT)
    train_path: str = field(
        default='', metadata={"help": "Path to the training data."}
    )
    eval_path: str = field(
        default='', metadata={"help": "Path to the evaluation data."}
    )
    test_path: str = field(
        default='', metadata={"help": "Path to the test data."}
    )
    repo_id: str = field(
        default='', metadata={"help": "Hugging Face dataset repository ID."}
    )

def _register_dataset(name, type, train_path='', eval_path='', test_path='', repo_id=''):
    dataset = Dataset(dataset_name=name, dataset_type=type, train_path=train_path, eval_path=eval_path, test_path=test_path, repo_id=repo_id)
    _DATASETS[name] = dataset


def _register_mixture(mixture_name, dataset_names: Dict[str, float]):
    fracs = dataset_names.values()
    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")
    if all(name in _DATASETS for name in dataset_names):
        _MIXTURES[mixture_name] = [(_DATASETS[name], frac) for name, frac in dataset_names.items()]
    else:
        raise ValueError("One or more dataset names provided do not exist in the dataset registry.")


def _resolve_dataset(args: Dataset, split: str):
    split_path = getattr(args, f"{split}_path", None)
    if split_path and os.path.exists(split_path):
        return load_from_disk(split_path)
    
    if args.repo_id:
        try:
            dataset = load_dataset(args.repo_id)
            if split in dataset.keys():
                return dataset[split]
            elif split == "eval" and "validation" in dataset.keys():
                return dataset["validation"]
            elif split == "eval" and "valid" in dataset.keys():
                return dataset["valid"]
            elif split == "eval" and "val" in dataset.keys():
                return dataset["val"]
            else:
                return dataset["train"]
        except:
            raise ValueError(f"Dataset {args.dataset_name} not found in the Hugging Face dataset hub.")
    
    raise ValueError(f"Dataset {args.dataset_name} not found.")


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                training_args: transformers.TrainingArguments,
                                modalities: List[Modality],
                                ) -> Dict:
    if training_args.dataset_name is not None:
        print(f"Using dataset: {training_args.dataset_name}")
        assert training_args.dataset_name in _DATASETS, f"Dataset {training_args.dataset_name} not found in registry."
        dataset = _DATASETS[training_args.dataset_name]
        train_dataset = _resolve_dataset(dataset, split="train")
        eval_dataset = _resolve_dataset(dataset, split="eval") if dataset.eval_path or dataset.repo_id else None
    elif training_args.data_mixture is not None:
        print(f"Using dataset mixture: {training_args.data_mixture}")
        assert training_args.data_mixture in _MIXTURES, f"Dataset mixture {training_args.data_mixture} not found in registry."
        mixture = _MIXTURES[training_args.data_mixture]
        train_datasets = []
        eval_datasets = []
        for data_args, frac in mixture:
            dataset_cls = _CLS_MAPPING[data_args.dataset_type]
            train_dataset = dataset_cls(tokenizer=tokenizer, modalities=modalities, data_args=data_args, split="train")
            train_subset = Subset(train_dataset, range(int(len(train_dataset)*frac)))
            train_datasets.append(train_subset)
            if training_args.eval_path:
                eval_datasets.append(dataset_cls(tokenizer=tokenizer, modalities=modalities, data_args=data_args, split="eval"))
        train_dataset = LMMConcatDataset(train_datasets)
        if training_args.eval_path:
            eval_dataset = LMMConcatDataset(eval_datasets)
        else:
            eval_dataset = None
    else:
        raise ValueError("No dataset or dataset mixture specified.")
    data_collator = DataCollatorForSupervisedLMMDataset(tokenizer=tokenizer, modalities=modalities)
    print(f"Train dataset length: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval dataset length: {len(eval_dataset)}")
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class LMMDataset(TorchDataset):
    def __init__(
        self,
        data_args: Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
        split: str
    ):
        super(LMMDataset, self).__init__()

        self.dataset = _resolve_dataset(data_args, split)
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

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch


_CLS_MAPPING = {DatasetType.CHAT: LMMDataset, DatasetType.INTERLEAVED: LMMInterleavedDataset}


## Register datasets
# PubChem 330K Caption Dataset
_register_dataset(
    name="pubchem_cap",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "pubchem_caption"),
)

# USPTO RXN Interleaved Dataset
_register_dataset(
    name="uspto_rxn",
    type=DatasetType.INTERLEAVED,
    train_path=os.path.join(DATASET_BASE_DIR, "uspto_rxn"),
)

 # Yields Regression
_register_dataset(
    name="yields_regression",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "yields_regression", "train"),
    repo_id='OpenMol/BH-SM_YR_10K-MMChat'
)
 
# Forward Prediction
_register_dataset(
    name="forward_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "forward_prediction", "train"),
    repo_id='OpenMol/MolInst_FS_125K_SMILES-MMChat'
)

# Retrosynthesis
_register_dataset(
    name="retrosynthesis",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "retrosynthesis", "train"),
    repo_id='OpenMol/MolInst_RS_125K_SMILES-MMChat'
)

# Reaction Classification
_register_dataset(
    name="reaction_classification",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reaction_classification", "train"),
    repo_id='OpenMol/MolInst_FS_125K_SMILES-MMChat'
)

# Reagent Selection
_register_dataset(
    name="reagent_selection",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reagent_selection", "train"),
    repo_id="OpenMol/HTE_RAS_4K-MMChat",
)

# Reagent Prediction
_register_dataset(
    name="reagent_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reagent_prediction", "train"),
    repo_id="OpenMol/RCR_RP_57K_SMILES-MMChat",
)

# Solvent Prediction
_register_dataset(
    name="solvent_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "solvent_prediction", "train"),
    repo_id='OpenMol/RCR_SP_70K_SMILES-MMChat'
)

# Catalyst Prediction
_register_dataset(
    name="catalyst_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "catalyst_prediction", "train"),
    repo_id="OpenMol/RCR_CP_10K_SMILES-MMChat",
)

## Name Conversion
# s2f
_register_dataset(
    name="s2f",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "s2f_mmchat_smiles", "train"),
    repo_id="OpenMol/SMol_S2F_270K-MMChat",
)

# s2i
_register_dataset(
    name="s2i",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "s2i_mmchat_smiles", "train"),
    repo_id="OpenMol/SMol_S2I_270K-MMChat",
)

# i2s (text-only)
_register_dataset(
    name="i2s",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "i2s_mmchat_smiles", "train"),
    repo_id="OpenMol/SMol_I2S_270K-MMChat",
)

# i2f (text-only)
_register_dataset(
    name="i2f",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "i2f_mmchat_smiles", "train"),
    repo_id="OpenMol/SMol_I2F_270K-MMChat",
)

# g2s (from PubChem)
_register_dataset(
    name="g2s",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "g2s_mmchat_smiles",),
    repo_id="OpenMol/PubChem_G2S_300K_SMILES-MMPretrain",
)

## Register a mixture of datasets
_register_mixture(
    mixture_name = "pubchem_cap",
    dataset_names = {"pubchem_cap":1.0},
)

_register_mixture(
    mixture_name = "uspto_rxn_interleaved",
    dataset_names = {"uspto_rxn":1.0},
)

_register_mixture(
    mixture_name = "pretrain_v2",
    dataset_names = {"uspto_rxn":1.0, "g2s":1.0, "s2f":1.0, "s2i":1.0},
)

_register_mixture(
    mixture_name = "pretrain_v3",
    dataset_names = {"uspto_rxn":1.0, "g2s":1.0, "s2f":1.0, "s2i":1.0, "i2s":1.0, "i2f":1.0},
)

_register_mixture(
    mixture_name = "sft",
    dataset_names = {
        "yields_regression": 1.0,
        "forward_prediction": 1.0,
        "retrosynthesis": 1.0,
        "reaction_classification": 1.0,
        "reagent_selection": 1.0,
        "reagent_prediction": 1.0,
        "solvent_prediction": 1.0,
        "catalyst_prediction": 1.0,
    },
)

_register_mixture(
    mixture_name = "sft_subset",
    dataset_names = {
        "yields_regression": 1.0, # ~9.5k
        "forward_prediction": 0.1, # ~12k
        "retrosynthesis": 0.1, # ~12k
        "reaction_classification": 0.1, # ~54k
        "reagent_selection": 1.0, # ~4k
        "reagent_prediction": 0.2, # ~11k
        "solvent_prediction": 0.2, # ~14k
        "catalyst_prediction": 1.0, # ~10k
    },
)

_register_mixture(
    mixture_name = "nc",
    dataset_names = {
        "s2f": 1.0,
        "s2i": 1.0,
        "i2s": 1.0,
        "i2f": 1.0,
    },
)

_register_mixture(
    mixture_name = "text_only",
    dataset_names = {"i2s": 1.0, "i2f": 1.0},
)