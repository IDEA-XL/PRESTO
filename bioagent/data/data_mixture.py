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

from dataclasses import dataclass, field

@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_path: str = field(
        default=None, metadata={"help": "Path to the training image data."}
    )

DATASETS_MIXTURES = {}
DATASETS = {}

def add_dataset(dataset):
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    # PubChem 330K Caption Dataset
    pubchem_cap = Dataset(
        dataset_name="pubchem_cap",
        dataset_type="cap",
        data_path="/cto_labs/AIDD/DATA/MolFM/pubchemsft_desc/stage1",
    )
    add_dataset(pubchem_cap)
    
    # USPTO RXN Interleaved Dataset
    uspto_rxn_interleaved = Dataset(
        dataset_name="uspto_rxn",
        dataset_type="interleaved",
        data_path="/cto_labs/AIDD/DATA/React/USPTO/Interleaved",
    )
    add_dataset(uspto_rxn_interleaved)
    
    
    DATASETS_MIXTURES.update({'pubchem_cap': [pubchem_cap]})
    DATASETS_MIXTURES.update({'uspto_rxn_interleaved': [uspto_rxn_interleaved]})