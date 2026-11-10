import numpy as np
import pytest

from silksight.embeddings import normalize_rows
from silksight.evaluation import evaluate, predict, tune_threshold
from silksight.knn import class_scores

CLASSES = ["a", "b"]


def test_class_scores_average_top_m_neighbours_per_class():
    gallery = normalize_rows(np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]))
    query = normalize_rows(np.array([[1.0, 0.0]]))

    best_only = class_scores(query, gallery, ["a", "a", "b"], CLASSES, top_m=1)
    averaged = class_scores(query, gallery, ["a", "a", "b"], CLASSES, top_m=2)

    assert best_only[0].tolist() == pytest.approx([1.0, 0.0])
    assert averaged[0].tolist() == pytest.approx([0.9, 0.0])  # mean of 1.0 and 0.8


def test_class_scores_reject_class_without_gallery_images():
    with pytest.raises(ValueError, match="no gallery"):
        class_scores(np.ones((1, 2)), np.ones((1, 2)), ["a"], CLASSES, top_m=1)


def test_low_top_score_becomes_unknown():
    scores = np.array([[0.9, 0.1], [0.3, 0.2]])
    assert predict(scores, CLASSES, threshold=0.5) == ["a", "unknown"]


def test_evaluate_separates_closed_set_accuracy_from_rejection():
    scores = np.array([[0.9, 0.1], [0.4, 0.3], [0.2, 0.8], [0.6, 0.1]])
    labels = ["a", "a", "a", "unknown"]

    report = evaluate(scores, CLASSES, labels, threshold=0.5)

    assert report.top1 == pytest.approx(2 / 3)  # threshold ignored
    assert report.known_accepted_correct == pytest.approx(1 / 3)
    assert report.known_false_rejection == pytest.approx(1 / 3)
    assert report.unknown_rejection == 0.0  # 0.6 passed the threshold
    assert report.confusion["unknown"] == {"a": 1}


def test_tuned_threshold_rejects_unknowns_scoring_below_known_photos():
    scores = np.array([[0.9, 0.1], [0.1, 0.8], [0.4, 0.3], [0.2, 0.35]])
    labels = ["a", "b", "unknown", "unknown"]

    threshold = tune_threshold(scores, CLASSES, labels)

    assert 0.4 < threshold <= 0.8
    assert evaluate(scores, CLASSES, labels, threshold).balanced_open_set == 1.0
