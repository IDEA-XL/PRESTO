import pytest
import numpy as np

from bioagent.chemistry_tools.evaluator import MoleculeSMILESEvaluator, ClassificationEvaluator, RegressionEvaluator, MoleculeCaptionEvaluator

CLASSIFICATION_EVALUATOR_TEST_CASES = [
    ([0, 1, 0, 1], [1, 1, 0, 0], 0.5, 0.5, 0.5),
    ([0, 1, 0, 1], [1, 0, 1, 0], 0.0, 0.0, 0.0),
]

REGRESSION_EVALUATOR_TEST_CASES = [
    ([1.0, 2.0, 3.0], [1.1, 1.9, 3.1], 0.01, 0.1, 0.985),
    ([4.0, 5.0, 6.0], [3.9, 5.1, 5.9], 0.01, 0.1, 0.985),
    ([7.0, 8.0, 9.0], [6.0, 7.0, 8.0], 1.0000000, 1.0000000, -0.5),
]

MOLECULE_EVALUATOR_TEST_CASES = [
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], True, True, True),
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, True, True),
    (["Sorry, I don't know."], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False, True),
    ([""], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False, True),
    (["CCOC(=O)c1ccccc1Br"], ["CCOC(=O)c1ccccc1Br"], True, True, False),
    ([""], ["CCOC(=O)c1ccccc1Br"], False, False, False),
    (["Sorry, I don't know."], ["CCOC(=O)c1ccccc1Br"], False, False, False),
    (["O=C(Cl)CCC(F)(F)F"], ["CCOC(=O)c1ccccc1Br"], False, True, False),
]

CAPTION_EVALUATOR_TEST_CASES = [
    (["This is a test caption."], ["This is a test caption."], 1.0, 1.0, 0.99768, 1.0),
    (["This is a test caption."], ["Daniel"], 0.0, 0.0, 0.0, 0.0),
]

@pytest.mark.parametrize("predictions, references, bleu_2, bleu_4, meteor, rouge_l", CAPTION_EVALUATOR_TEST_CASES)
def test_caption_evaluator(predictions, references, bleu_2, bleu_4, meteor, rouge_l):
    evaluator = MoleculeCaptionEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["bleu-2", "bleu-4", "meteor", "rouge-L"], verbose=True, full_results=True)
    assert np.isclose(score["bleu-2"][0], bleu_2), f"Expected {bleu_2}, but got {score['bleu-2'][0]} for bleu2"
    assert np.isclose(score["bleu-4"][0], bleu_4), f"Expected {bleu_4}, but got {score['bleu-4'][0]} for bleu4"
    assert np.isclose(score["meteor"][0], meteor), f"Expected {meteor}, but got {score['meteor'][0]} for meteor"
    assert np.isclose(score["rouge-L"][0], rouge_l), f"Expected {rouge_l}, but got {score['rouge-L'][0]} for rougeL"


@pytest.mark.parametrize("predictions, references, exact_match, validity, selfies", MOLECULE_EVALUATOR_TEST_CASES)
def test_molecule_smiles_evaluator(predictions, references, exact_match, validity, selfies):
    evaluator = MoleculeSMILESEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["exact_match", "validity", "bleu"], verbose=True, selfies=selfies, full_results=True)
    assert score["exact_match"][0] == exact_match, f"Expected {exact_match}, but got {score['exact_match'][0]} for exact match"
    assert score["validity"][0] == validity, f"Expected {validity}, but got {score['validity'][0]} for validity"


@pytest.mark.parametrize("predictions, references, accuracy, f1_micro, f1_macro", CLASSIFICATION_EVALUATOR_TEST_CASES)
def test_classification_evaluator(predictions, references, accuracy, f1_micro, f1_macro):
    evaluator = ClassificationEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["accuracy", "f1_micro", "f1_macro"], verbose=True)
    assert score["accuracy"] == accuracy, f"Expected {accuracy}, but got {score['accuracy']} for accuracy"
    assert score["f1_micro"] == f1_micro, f"Expected {f1_micro}, but got {score['f1_micro']} for f1_micro"
    assert score["f1_macro"] == f1_macro, f"Expected {f1_macro}, but got {score['f1_macro']} for f1_macro"


@pytest.mark.parametrize("predictions, references, mse, mae, r2", REGRESSION_EVALUATOR_TEST_CASES)
def test_regression_evaluator(predictions, references, mse, mae, r2):
    evaluator = RegressionEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["mse", "mae", "r2"], verbose=True)
    assert np.isclose(score["mse"], mse), f"Expected {mse}, but got {score['mse']} for mse"
    assert np.isclose(score["mae"], mae), f"Expected {mae}, but got {score['mae']} for mae"
    assert np.isclose(score["r2"], r2), f"Expected {r2}, but got {score['r2']} for r2"