from presto.chemistry_tools.evaluator import (
    MoleculeSMILESEvaluator, 
    MoleculeSELFIESEvaluator,
    MoleculeCaptionEvaluator,
    ClassificationEvaluator,
    RegressionEvaluator,
)
from presto.chemistry_tools.reaction import ReactionEquation, multicomponent_smiles_to_list, list_to_multicomponent_smiles
from presto.chemistry_tools.smiles import smiles_to_graph, graph_to_smiles, smiles_to_coords

EVALUATOR_BUILDERS = {
    "smiles": MoleculeSMILESEvaluator,
    "selfies": MoleculeSELFIESEvaluator,
    "caption": MoleculeCaptionEvaluator,
    "classification": ClassificationEvaluator,
    "regression": RegressionEvaluator,
}