# Bioagent
## Pretraining
### Data
The data used for pretraining is obtained using the PubChem API. Following the same process in MoleculeSTM, we obtain the SMILES strings for the molecules as well as the descriptions.

### Training
```bash
bash scripts/pretrain_molecule/pretrain_mol2d.sh
```

## Forward Reaction Prediction
### Data
We use the USPTO dataset for forward reaction prediction. It is turned into instruction format by Mol-Instructions.

### Training
```bash
bash scripts/forward_reaction_prediction/finetune_mol2d_forward_reaction_prediction.sh
```

### Evaluation
```bash
bash scripts/forward_reaction_prediction/evaluate_mol2d_forward_reaction_prediction.sh
```

## Retrosynthesis
### Data
We use the USPTO dataset for retrosynthesis. It is turned into instruction format by Mol-Instructions.

### Training
```bash
bash scripts/retrosynthesis/finetune_mol2d_retrosynthesis.sh
```

### Evaluation
```bash
# TODO: There might be a bug in the evaluation script
bash scripts/retrosynthesis/evaluate_mol2d_retrosynthesis.sh
```