# PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes

PRESTO (Progressive Pretraining Enhances Synthetic Chemistry Outcomes) is a framework that focuses on pretraining and finetuning large language models (LLMs) for various tasks in synthetic chemistry.

![PRESTO](assets/teaser.png)

## Installation

1. Install the required dependencies:
   ```
   conda create -n presto python=3.10
   pip install -r requirements.txt
   pip install -e .
   ```

2. Set up the necessary environment variables:
   ```
   export MOLECULE_2D_PATH="/path/to/MoleculeSTM/"
   export WANDB_API_KEY="your_wandb_api_key"
   ```

## Pretraining

### Stage 1: Molecule-Text Alignment

To perform Stage 1 pretraining for molecule-text alignment, run the following command:
```bash
bash scripts/pretrain_multi_molecule/stage1.sh
```

This script will pretrain the model using the PubChem caption dataset and save the pretrained model checkpoints.

### Stage 2: Domain Incremental Pretraining

For Stage 2 pretraining, there are several configurations available:

- `stage2.sh`: Pretraining using interleaved molecule-text data from USPTO-Application.
- `stage2_rxn_nc.sh`: Pretraining using interleaved reaction data and name conversion tasks (g2s, s(g)2i, s(g)2f).
- `stage2_all.sh`: Pretraining using interleaved reaction data and all name conversion tasks (i2s, i2f).
- `stage2_skip_align.sh`: Skipping Stage 1 and directly starting with Stage 2 pretraining, only training the projector.
- `stage2_skip_align_fulltune.sh`: Skipping Stage 1 and directly starting with Stage 2 pretraining, finetuning the entire model.

To run a specific Stage 2 pretraining configuration, execute the corresponding script. For example:
```bash
bash scripts/pretrain_multi_molecule/stage2_rxn_nc.sh
```

## Stage 3 Downstream Tasks

For Stage 3 finetuning, we include finetuning scripts for various downstream tasks. Each task has its own directory under `scripts/build_dataset/` to build the dataset and `scripts/sft/` to run the finetuning. There are several configurations available:

- `stage3_freezeLLM.sh`: Finetuning the projector with a frozen LLM on Stage 3 downstream tasks.
- `stage3_lora.sh`: Finetuning the projector and apply LoRA to train the LLM on Stage 3 downstream tasks.
- `stage3_rxn_nc.sh`: Finetuning the LLM (pretrained using `stage2_rxn_nc.sh`) on Stage 3 downstream tasks.
- `stage3_skip_align_fulltune.sh`: Skipping Stage 1 and train with the full model on Stage 2 pretraining data and Stage 3 downstream tasks.
- `stage3_skip_stage2.sh`: Skipping Stage 2 and train with the full model on Stage 1 pretraining data and Stage 3 downstream tasks.
- `stage3_skip_stage12.sh`: Skipping Stage 1 and Stage 2 and train with the full model on Stage 3 downstream tasks.
- `stage3.sh`: Train with the full model on Stage 3 directly.

To run a specific Stage 3 finetuning configuration, execute the corresponding script. For example:
```bash
bash scripts/sft/sft_lora/stage3_rxn_nc.sh
```

## Evaluation

Here is a list of all the downstream tasks and the corresponding commands to run the evaluation:

## Reaction Prediction
### Forward Prediction

To evaluate the forward reaction prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_forward_reaction_prediction.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_forward_reaction_prediction.sh
```

### Retrosynthesis Prediction

To evaluate the retrosynthesis prediction task, use the following command:
```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_retrosynthesis.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_retrosynthesis.sh
```

## Reaction Condition Prediction
### Reagent Prediction
To evaluate the reagent prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reagent_prediction.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_reagent_prediction.sh
```

### Catalyst Prediction
To evaluate the catalyst prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_catalyst_prediction.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_catalyst_prediction.sh
```

### Solvent Prediction
To evaluate the solvent prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_solvent_prediction.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_solvent_prediction.sh
```

## Reaction Condition Recommendation  
### Reagent Selection
To evaluate the reagent selection task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reagent_selection.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_reagent_selection.sh
```

## Reaction Type Classification
To evaluate the reaction type classification task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reaction_classification.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_reaction_classification.sh
```

## Yield Prediction
To evaluate the yield prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_yields_regression.sh

# For full model
bash scripts/evaluate/sft_full/evaluate_yields_regression.sh
```

## Model Serving

To serve the trained model using a Flask server, run:

```
python scripts/serve_model.py --model_name_or_path <path_to_model> --model_lora_path <path_to_lora_model> --port <port_number>
```

This will start a Flask server that exposes a `/generate` endpoint for generating predictions using the trained model.

## Dataset Preparation

The `scripts/build_dataset` directory contains scripts for preparing datasets for different tasks. Follow the instructions within each task-specific directory to prepare the datasets.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- This project builds upon the work of various open-source libraries and frameworks. We would like to acknowledge their contributions.
    - [multi_token](https://github.com/sshh12/multi_token): We mostly built upon this implementation to support multi-token molecules.
    - [Hugging Face Transformers](https://github.com/huggingface/transformers)
    - [LLaVA](https://github.com/haotian-liu/LLaVA)
    - [VILA](https://github.com/Efficient-Large-Model/VILA)

- We also thank the researchers and developers whose ideas and implementations have inspired and guided this project.

For more details and advanced usage, please refer to the documentation and source code.
