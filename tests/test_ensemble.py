import numpy as np
import pytest

from silksight.ensemble import (
    agreement_predictions,
    fit_temperature,
    fuse,
    oracle_correct,
    softmax,
)
from silksight.evaluation import evaluate_predictions

CLASSES = ["chor-minor", "registan"]


def test_agreement_answers_only_when_both_models_accept_the_same_landmark():
    knn = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.4, 0.1]])
    classifier = np.array([[0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]])

    answers = agreement_predictions(knn, classifier, CLASSES, 0.5, 0.6)

    # agree / disagree / classifier below its threshold / kNN below its threshold
    assert answers == ["chor-minor", "unknown", "unknown", "unknown"]


def test_fitted_temperature_sharpens_close_cosine_scores():
    # Right class wins by only 0.05, as with real cosine similarities.
    scores = np.array([[0.75, 0.70], [0.68, 0.73]] * 5)
    labels = ["chor-minor", "registan"] * 5

    temperature = fit_temperature(scores, CLASSES, labels)

    assert temperature < 0.05
    assert softmax(scores, temperature)[0, 0] > 0.9


def test_temperature_ignores_unknown_photos_and_needs_a_known_one():
    with pytest.raises(ValueError):
        fit_temperature(np.array([[0.5, 0.4]]), CLASSES, ["unknown"])


def test_fusion_averages_distributions_of_the_same_shape():
    fused = fuse(np.array([[0.7, 0.7]]), np.array([[1.0, 0.0]]), temperature=0.1)

    assert np.allclose(fused, [[0.75, 0.25]])
    with pytest.raises(ValueError):
        fuse(np.zeros((1, 2)), np.zeros((1, 3)), 0.1)


def test_oracle_counts_photos_either_model_gets_right():
    labels = ["registan", "chor-minor", "unknown"]

    assert (
        oracle_correct(
            ["registan", "unknown", "registan"], ["unknown", "chor-minor", "registan"], labels
        )
        == 2
    )


def test_rule_reports_have_no_threshold_and_still_score_open_set():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])

    report = evaluate_predictions(
        scores, CLASSES, ["chor-minor", "unknown"], ["chor-minor", "unknown"], None
    )

    assert report.threshold is None
    assert report.known_accepted_correct == 1.0
    assert report.unknown_rejection == 1.0


def _answer(model: str, scores: list[float], threshold: float):
    from silksight.recognizer import decide

    return decide(model, "test", CLASSES, np.array(scores), threshold)


@pytest.mark.parametrize(
    ("knn_scores", "classifier_scores", "landmark"),
    [
        ([0.9, 0.1], [0.8, 0.2], "chor-minor"),  # both accept the same landmark
        ([0.9, 0.1], [0.2, 0.8], None),  # both accept, but disagree
        ([0.9, 0.1], [0.55, 0.45], None),  # classifier below its threshold
    ],
)
def test_served_agreement_rule_keeps_both_answers(knn_scores, classifier_scores, landmark):
    from silksight.recognizer import decide_ensemble

    params = {"rule": "agreement", "knn_temperature": 0.1, "fused_threshold": 0.5}
    knn, classifier = np.array(knn_scores), np.array(classifier_scores)

    result = decide_ensemble(
        params,
        CLASSES,
        _answer("knn", knn_scores, 0.5),
        _answer("classifier", classifier_scores, 0.6),
        knn,
        classifier,
    )

    assert result.landmark_id == landmark
    assert result.recognized is (landmark is not None)
    assert result.threshold is None
    assert [m.model for m in result.members] == ["knn", "classifier"]
    assert len(result.candidates) == 2
