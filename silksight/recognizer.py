"""Serve both trained models, and their val-selected combination, behind one interface:
photo in, landmark or "unknown" out.

Models are loaded on first use. One lock serializes loading and inference: a single local
process shares one GPU/MPS device, and concurrent forward passes would only compete for it.
"""

import json
import threading
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from PIL import Image

from silksight.settings import ENSEMBLE_DIR, MODEL_A_DIR, MODEL_B_DIR

MEMBER_NAMES = ("knn", "classifier")
ENSEMBLE = "ensemble"
MODEL_NAMES = (*MEMBER_NAMES, ENSEMBLE)
TOP_CANDIDATES = 3


class ModelUnavailableError(RuntimeError):
    """The trained artifacts for a model are missing (run the training scripts first)."""


@dataclass(frozen=True)
class Candidate:
    landmark_id: str
    score: float


@dataclass(frozen=True)
class Recognition:
    model: str
    score_kind: str  # what ``score`` means for this model; not a probability of being right
    threshold: float | None  # None when the decision is a rule, not one score cut
    recognized: bool
    landmark_id: str | None
    candidates: list[Candidate]
    members: list["Recognition"] = field(default_factory=list)  # ensemble: each model's answer


class _Scorer(Protocol):
    classes: list[str]
    threshold: float
    score_kind: str

    def scores(self, image: Image.Image) -> np.ndarray: ...


class _KnnScorer:
    score_kind = "cosine similarity"

    def __init__(self) -> None:
        from silksight.embeddings import BACKBONES, Embedder

        path = MODEL_A_DIR / "index.npz"
        if not path.exists():
            raise ModelUnavailableError("run python -m scripts.evaluate_knn first")
        with np.load(path) as index:
            self._gallery = index["vectors"]
            self._labels = index["labels"].tolist()
            self._top_m = int(index["top_m"])
            self.threshold = float(index["threshold"])
            backbone = str(index["backbone"])
        self.classes = sorted(set(self._labels))
        self._embedder = Embedder(BACKBONES[backbone])

    def scores(self, image: Image.Image) -> np.ndarray:
        from silksight.knn import class_scores

        query = self._embedder.embed([image])
        return class_scores(query, self._gallery, self._labels, self.classes, self._top_m)[0]


class _ClassifierScorer:
    score_kind = "softmax probability"

    def __init__(self) -> None:
        import torch

        from silksight.classifier import build_model, eval_transform
        from silksight.embeddings import default_device

        path = MODEL_B_DIR / "model.pt"
        if not path.exists():
            raise ModelUnavailableError("run python -m scripts.train_classifier first")
        self._device = default_device()
        checkpoint = torch.load(path, map_location=self._device)
        self.classes = checkpoint["classes"]
        self.threshold = float(checkpoint["threshold"])
        self._model = build_model(len(self.classes), pretrained=False)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self._device).eval()
        self._transform = eval_transform()
        self._torch = torch

    def scores(self, image: Image.Image) -> np.ndarray:
        batch = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            return self._torch.softmax(self._model(batch), dim=1)[0].cpu().numpy()


_SCORERS = {"knn": _KnnScorer, "classifier": _ClassifierScorer}


class Recognizers:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: dict[str, _Scorer] = {}
        self._ensemble: dict | None = None

    @property
    def loaded(self) -> list[str]:
        return sorted(self._loaded)

    def _scorer(self, model: str) -> _Scorer:
        if model not in self._loaded:
            self._loaded[model] = _SCORERS[model]()
        return self._loaded[model]

    def _ensemble_params(self, knn: _Scorer, classifier: _Scorer) -> dict:
        if self._ensemble is None:
            path = ENSEMBLE_DIR / "params.json"
            if not path.exists():
                raise ModelUnavailableError("run python -m scripts.evaluate_ensemble first")
            params = json.loads(path.read_text("utf-8"))
            # The rule was tuned on these exact thresholds and class order; a retrained
            # member model would silently change its meaning.
            if (
                knn.classes != classifier.classes
                or params["knn_threshold"] != knn.threshold
                or params["classifier_threshold"] != classifier.threshold
            ):
                raise ModelUnavailableError("models changed: re-run scripts.evaluate_ensemble")
            self._ensemble = params
        return self._ensemble

    def recognize(self, image: Image.Image, model: str) -> Recognition:
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {', '.join(MODEL_NAMES)}")
        if model != ENSEMBLE:
            with self._lock:
                scorer = self._scorer(model)
                scores = scorer.scores(image)
            return decide(model, scorer.score_kind, scorer.classes, scores, scorer.threshold)

        with self._lock:
            knn, classifier = self._scorer("knn"), self._scorer("classifier")
            params = self._ensemble_params(knn, classifier)
            knn_scores, classifier_scores = knn.scores(image), classifier.scores(image)
        return decide_ensemble(
            params,
            knn.classes,
            decide("knn", knn.score_kind, knn.classes, knn_scores, knn.threshold),
            decide(
                "classifier",
                classifier.score_kind,
                classifier.classes,
                classifier_scores,
                classifier.threshold,
            ),
            knn_scores,
            classifier_scores,
        )


def decide(
    model: str, score_kind: str, classes: list[str], scores: np.ndarray, threshold: float
) -> Recognition:
    order = np.argsort(-scores)[:TOP_CANDIDATES]
    candidates = [Candidate(classes[i], round(float(scores[i]), 4)) for i in order]
    recognized = candidates[0].score >= threshold
    return Recognition(
        model=model,
        score_kind=score_kind,
        threshold=round(threshold, 4),
        recognized=recognized,
        landmark_id=candidates[0].landmark_id if recognized else None,
        candidates=candidates,
    )


def decide_ensemble(
    params: dict,
    classes: list[str],
    knn: Recognition,
    classifier: Recognition,
    knn_scores: np.ndarray,
    classifier_scores: np.ndarray,
) -> Recognition:
    """Combine both answers with the rule selected on val (see silksight.ensemble)."""
    from silksight.ensemble import AGREEMENT, fuse

    fused = fuse(knn_scores[None, :], classifier_scores[None, :], params["knn_temperature"])[0]
    score_kind = "mean of kNN softmax and classifier probability"
    if params["rule"] == AGREEMENT:
        # Candidates still come from the fused scores so the "closest matches" stay useful.
        ranked = decide(ENSEMBLE, score_kind, classes, fused, threshold=0.0)
        agreed = (
            knn.recognized and classifier.recognized and knn.landmark_id == classifier.landmark_id
        )
        return replace(
            ranked,
            threshold=None,
            recognized=agreed,
            landmark_id=knn.landmark_id if agreed else None,
            members=[knn, classifier],
        )
    result = decide(ENSEMBLE, score_kind, classes, fused, params["fused_threshold"])
    return replace(result, members=[knn, classifier])
