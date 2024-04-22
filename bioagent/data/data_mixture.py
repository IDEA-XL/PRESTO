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

import os
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
DATASET_BASE_DIR="/cto_labs/AIDD/DATA/React/InstructChemReact/"

def add_dataset(dataset):
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    # PubChem 330K Caption Dataset
    pubchem_cap = Dataset(
        dataset_name="pubchem_cap",
        dataset_type="cap",
        data_path=os.path.join(DATASET_BASE_DIR, "pubchem_caption"),
    )
    add_dataset(pubchem_cap)
    
    # USPTO RXN Interleaved Dataset
    uspto_rxn_interleaved = Dataset(
        dataset_name="uspto_rxn",
        dataset_type="interleaved",
        data_path=os.path.join(DATASET_BASE_DIR, "uspto_rxn_interleaved"),
    )
    add_dataset(uspto_rxn_interleaved)
    
    # yields regression
    yields_regression = Dataset(
        dataset_name="yields_regression",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "yields_regression", "train"),
    )
    
    # forward prediction
    forward_prediction = Dataset(
        dataset_name="forward_prediction",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "forward_prediction", "train"),
    )
    
    # retrosynthesis
    retrosynthesis = Dataset(
        dataset_name="retrosynthesis",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "retrosynthesis", "train"),
    )
    
    # reaction classification
    reaction_classification = Dataset(
        dataset_name="reaction_classification",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "reaction_classification", "train"),
    )
    
    # reagent selection
    reagent_selection = Dataset(
        dataset_name="reagent_selection",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "reagent_selection", "train"),
    )
    
    # reagent prediction
    reagent_prediction = Dataset(
        dataset_name="reagent_prediction",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "reagent_prediction", "train"),
    )
    
    # solvent prediction
    solvent_prediction = Dataset(
        dataset_name="solvent_prediction",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "solvent_prediction", "train"),
    )
    
    # catalyst prediction
    catalyst_prediction = Dataset(
        dataset_name="catalyst_prediction",
        dataset_type="sft",
        data_path=os.path.join(DATASET_BASE_DIR, "catalyst_prediction", "train"),
    )
    
    DATASETS_MIXTURES.update({'pubchem_cap': [pubchem_cap]})
    DATASETS_MIXTURES.update({'uspto_rxn_interleaved': [uspto_rxn_interleaved]})
    DATASETS_MIXTURES.update({"sft": [yields_regression, forward_prediction, retrosynthesis, reaction_classification, reagent_selection, reagent_prediction, solvent_prediction, catalyst_prediction]})