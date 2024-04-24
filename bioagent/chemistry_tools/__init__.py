from bioagent.chemistry_tools.evaluator import MoleculeSMILESEvaluator, MoleculeCaptionEvaluator, ClassificationEvaluator, RegressionEvaluator
from bioagent.chemistry_tools.reaction import ReactionEquation, multicomponent_smiles_to_list, list_to_multicomponent_smiles
from bioagent.chemistry_tools.smiles import smiles_to_graph, graph_to_smiles, smiles_to_coords

EVALUATOR_BUILDERS = {
    "smiles": MoleculeSMILESEvaluator,
    "caption": MoleculeCaptionEvaluator,
    "regression": RegressionEvaluator,
    "classification": ClassificationEvaluator
}