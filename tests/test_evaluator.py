import pytest

from bioagent.chemistry_tools.evaluator import MoleculeSMILESEvaluator


MOLECULE_EVALUATOR_TEST_CASES = [
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], True, True),
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, True),
    (["Sorry, I don't know."], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False),
    ([""], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False),
]

@pytest.mark.parametrize("predictions, references, exact_match, validity", MOLECULE_EVALUATOR_TEST_CASES)
def test_molecule_smiles_evaluator(predictions, references, exact_match, validity):
    evaluator = MoleculeSMILESEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["exact_match", "validity"])
    assert score["exact_match"][0] == exact_match, f"Expected {exact_match}, but got {score['exact_match'][0]}"
    assert score["validity"][0] == validity, f"Expected {validity}, but got {score['validity'][0]}"
    