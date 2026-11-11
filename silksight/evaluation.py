"""Open-set evaluation shared by both models: closed-set accuracy on known landmarks, plus the
"I don't recognize this" decision made by a threshold on the top class score.

Scores are model scores (cosine similarity for kNN, softmax probability for the classifier),
not calibrated probabilities; a threshold only makes sense for the model it was tuned on.
"""

from dataclasses import dataclass

import numpy as np

from silksight.dataset import UNKNOWN_LABEL


@dataclass(frozen=True)
class OpenSetReport:
    threshold: float | None  # None for rules that decide "unknown" without one score cut
    known_count: int
    unknown_count: int
    top1: float  # closed-set: best class is right, threshold ignored
    top3: float
    known_accepted_correct: float  # known photo, accepted, right landmark
    known_false_rejection: float  # known photo answered "unknown"
    unknown_rejection: float  # unknown photo answered "unknown"
    balanced_open_set: float  # mean of known_accepted_correct and unknown_rejection
    macro_f1: float  # over the landmarks plus "unknown", after the threshold
    per_class_recall: dict[str, float]
    confusion: dict[str, dict[str, int]]  # true -> predicted -> count, non-zero cells only


def predict(scores: np.ndarray, classes: list[str], threshold: float) -> list[str]:
    best = scores.argmax(axis=1)
    top = scores.max(axis=1)
    return [classes[i] if s >= threshold else UNKNOWN_LABEL for i, s in zip(best, top)]


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _top_k(scores: np.ndarray, classes: list[str], labels: list[str], k: int) -> float:
    ranked = np.argsort(-scores, axis=1)[:, :k]
    index = {label: i for i, label in enumerate(classes)}
    hits = sum(index[label] in row for label, row in zip(labels, ranked))
    return _rate(hits, len(labels))


def _macro_f1(labels: list[str], predicted: list[str], names: list[str]) -> float:
    f1s = []
    for name in names:
        tp = sum(t == name and p == name for t, p in zip(labels, predicted))
        fp = sum(t != name and p == name for t, p in zip(labels, predicted))
        fn = sum(t == name and p != name for t, p in zip(labels, predicted))
        f1s.append(_rate(2 * tp, 2 * tp + fp + fn))
    return float(np.mean(f1s))


def evaluate(
    scores: np.ndarray, classes: list[str], labels: list[str], threshold: float
) -> OpenSetReport:
    predicted = predict(scores, classes, threshold)
    return evaluate_predictions(scores, classes, labels, predicted, float(threshold))


def evaluate_predictions(
    scores: np.ndarray,
    classes: list[str],
    labels: list[str],
    predicted: list[str],
    threshold: float | None,
) -> OpenSetReport:
    """Score open-set answers made by any rule; ``scores`` only feed closed-set top-1/top-3."""
    if UNKNOWN_LABEL in classes:
        raise ValueError("classes are the known landmarks; 'unknown' is decided by the rule")
    if len(predicted) != len(labels):
        raise ValueError("one prediction per photo is required")
    known = [i for i, label in enumerate(labels) if label != UNKNOWN_LABEL]
    unknown = [i for i, label in enumerate(labels) if label == UNKNOWN_LABEL]
    known_labels = [labels[i] for i in known]

    confusion: dict[str, dict[str, int]] = {}
    for label, guess in zip(labels, predicted):
        row = confusion.setdefault(label, {})
        row[guess] = row.get(guess, 0) + 1

    accepted_correct = sum(predicted[i] == labels[i] for i in known)
    false_rejection = sum(predicted[i] == UNKNOWN_LABEL for i in known)
    rejected_unknown = sum(predicted[i] == UNKNOWN_LABEL for i in unknown)
    known_rate = _rate(accepted_correct, len(known))
    unknown_rate = _rate(rejected_unknown, len(unknown))
    present = sorted(set(labels))
    return OpenSetReport(
        threshold=threshold,
        known_count=len(known),
        unknown_count=len(unknown),
        top1=_top_k(scores[known], classes, known_labels, 1),
        top3=_top_k(scores[known], classes, known_labels, 3),
        known_accepted_correct=known_rate,
        known_false_rejection=_rate(false_rejection, len(known)),
        unknown_rejection=unknown_rate,
        balanced_open_set=(known_rate + unknown_rate) / 2 if unknown else known_rate,
        macro_f1=_macro_f1(labels, predicted, present),
        per_class_recall={
            name: _rate(confusion[name].get(name, 0), sum(confusion[name].values()))
            for name in present
        },
        confusion=confusion,
    )


def tune_threshold(scores: np.ndarray, classes: list[str], labels: list[str]) -> float:
    """Pick the threshold with the best balanced open-set score on validation data.

    Candidates are the observed top scores, so the result is a real cut between two val
    photos; ties keep the lower threshold (fewer false "unknown" answers on known places).
    """
    candidates = np.unique(scores.max(axis=1))  # the lowest one accepts every photo
    best_threshold, best_value = float(candidates[0]), -1.0
    for threshold in candidates:
        value = evaluate(scores, classes, labels, float(threshold)).balanced_open_set
        if value > best_value:
            best_threshold, best_value = float(threshold), value
    return best_threshold
