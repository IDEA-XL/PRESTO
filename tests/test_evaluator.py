import pytest
import numpy as np

from bioagent.chemistry_tools.evaluator import MoleculeSMILESEvaluator, ClassificationEvaluator, RegressionEvaluator, MoleculeCaptionEvaluator

CLASSIFICATION_EVALUATOR_TEST_CASES = [
    ([0, 1, 0, 1], [0, 1, 1, 1], 0.75, 0.8),
    ([1, 0, 1, 0], [1, 1, 1, 0], 0.75, 0.8),
    ([0, 0, 0, 0], [1, 1, 1, 1], 0.0, 0.0),
]

REGRESSION_EVALUATOR_TEST_CASES = [
    ([1.0, 2.0, 3.0], [1.1, 1.9, 3.1], 0.01, 0.1, 0.985),
    ([4.0, 5.0, 6.0], [3.9, 5.1, 5.9], 0.01, 0.1, 0.985),
    ([7.0, 8.0, 9.0], [6.0, 7.0, 8.0], 1.0000000, 1.0000000, -0.5),
]

MOLECULE_EVALUATOR_TEST_CASES = [
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], True, True),
    (["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1]"], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, True),
    (["Sorry, I don't know."], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False),
    ([""], ["[C][N][C][=Branch1][C][=O][C][=C][C][=C][C][=N][Ring1][=Branch1]"], False, False),
]

CAPTION_EVALUATOR_TEST_CASES = [
    (["This is a test caption."], ["This is a test caption."], 1.0, 1.0, 0.99768, 1.0),
    (["This is a test caption."], ["Daniel"], 0.0, 0.0, 0.0, 0.0),
]

@pytest.mark.parametrize("predictions, references, bleu_2, bleu_4, meteor, rouge_l", CAPTION_EVALUATOR_TEST_CASES)
def test_caption_evaluator(predictions, references, bleu_2, bleu_4, meteor, rouge_l):
    evaluator = MoleculeCaptionEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["bleu-2", "bleu-4", "meteor", "rouge-L"], verbose=True)
    assert np.isclose(score["bleu-2"][0], bleu_2), f"Expected {bleu_2}, but got {score['bleu-2'][0]} for bleu2"
    assert np.isclose(score["bleu-4"][0], bleu_4), f"Expected {bleu_4}, but got {score['bleu-4'][0]} for bleu4"
    assert np.isclose(score["meteor"][0], meteor), f"Expected {meteor}, but got {score['meteor'][0]} for meteor"
    assert np.isclose(score["rouge-L"][0], rouge_l), f"Expected {rouge_l}, but got {score['rouge-L'][0]} for rougeL"


@pytest.mark.parametrize("predictions, references, exact_match, validity", MOLECULE_EVALUATOR_TEST_CASES)
def test_molecule_smiles_evaluator(predictions, references, exact_match, validity):
    evaluator = MoleculeSMILESEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["exact_match", "validity", "bleu"], verbose=True, selfies=True)
    assert score["exact_match"][0] == exact_match, f"Expected {exact_match}, but got {score['exact_match'][0]} for exact match"
    assert score["validity"][0] == validity, f"Expected {validity}, but got {score['validity'][0]} for validity"


@pytest.mark.parametrize("predictions, references, accuracy, f1", CLASSIFICATION_EVALUATOR_TEST_CASES)
def test_classification_evaluator(predictions, references, accuracy, f1):
    evaluator = ClassificationEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["accuracy", "f1_score"], verbose=True)
    assert score["accuracy"][0] == accuracy, f"Expected {accuracy}, but got {score['accuracy'][0]} for accuracy"
    assert score["f1_score"][0] == f1, f"Expected {f1}, but got {score['f1_score'][0]} for f1 score"


@pytest.mark.parametrize("predictions, references, mse, mae, r2", REGRESSION_EVALUATOR_TEST_CASES)
def test_regression_evaluator(predictions, references, mse, mae, r2):
    evaluator = RegressionEvaluator()
    score = evaluator.evaluate(predictions, references, metrics=["mse", "mae", "r2"], verbose=True)
    assert np.isclose(score["mse"][0], mse), f"Expected {mse}, but got {score['mse'][0]} for mse"
    assert np.isclose(score["mae"][0], mae), f"Expected {mae}, but got {score['mae'][0]} for mae"
    assert np.isclose(score["r2"][0], r2), f"Expected {r2}, but got {score['r2'][0]} for r2"