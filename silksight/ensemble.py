"""Combine Model A (kNN) and Model B (classifier) into one open-set answer.

Two rules are compared on validation data only:

- ``agreement``: answer a landmark only when both models, each with its own val-tuned
  threshold, accept the photo and name the same landmark; otherwise "unknown". No new
  parameters are fitted.
- ``fusion``: turn kNN cosine scores into a distribution with a softmax whose temperature is
  fitted on known val photos, average it with the classifier's softmax, and apply a threshold
  tuned on val. The two raw scores live on different scales, so they are never averaged as is.

Fused values are model scores, not calibrated probabilities of being right.
"""

import numpy as np

from silksight.dataset import UNKNOWN_LABEL
from silksight.evaluation import predict

AGREEMENT = "agreement"
FUSION = "fusion"
RULES = (AGREEMENT, FUSION)
# Cosine similarities of different landmarks differ by a few hundredths, so useful
# temperatures are small; the grid spans that range on a log scale.
TEMPERATURES = np.geomspace(0.002, 1.0, 120)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = scores / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, classes: list[str], labels: list[str]) -> float:
    """Temperature with the lowest negative log-likelihood on the known (non-unknown) photos."""
    index = {name: i for i, name in enumerate(classes)}
    rows = [i for i, label in enumerate(labels) if label != UNKNOWN_LABEL]
    if not rows:
        raise ValueError("fitting a temperature needs at least one known photo")
    targets = np.array([index[labels[i]] for i in rows])
    known = scores[rows]
    best, best_nll = float(TEMPERATURES[0]), np.inf
    for temperature in TEMPERATURES:
        probs = softmax(known, float(temperature))
        nll = -np.log(probs[np.arange(len(rows)), targets] + 1e-12).mean()
        if nll < best_nll:
            best, best_nll = float(temperature), nll
    return best


def fuse(knn_scores: np.ndarray, classifier_probs: np.ndarray, temperature: float) -> np.ndarray:
    if knn_scores.shape != classifier_probs.shape:
        raise ValueError("both models must score the same photos over the same classes")
    return (softmax(knn_scores, temperature) + classifier_probs) / 2


def agreement_predictions(
    knn_scores: np.ndarray,
    classifier_probs: np.ndarray,
    classes: list[str],
    knn_threshold: float,
    classifier_threshold: float,
) -> list[str]:
    knn = predict(knn_scores, classes, knn_threshold)
    classifier = predict(classifier_probs, classes, classifier_threshold)
    return [a if a == b else UNKNOWN_LABEL for a, b in zip(knn, classifier)]


def oracle_correct(knn: list[str], classifier: list[str], labels: list[str]) -> int:
    """Photos at least one model answers correctly: a ceiling no real rule can exceed."""
    return sum(label in (a, b) for a, b, label in zip(knn, classifier, labels))
