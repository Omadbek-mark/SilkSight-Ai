"""Nearest-neighbour landmark scores over stored training embeddings (Model A)."""

import numpy as np


def class_scores(
    queries: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: list[str],
    classes: list[str],
    top_m: int,
) -> np.ndarray:
    """Return a (queries, classes) matrix: the mean cosine similarity of each query to its
    ``top_m`` most similar gallery photos of that class.

    Averaging a few neighbours instead of taking only the best one makes a single
    near-duplicate training photo less decisive. Rows must already be L2-normalized, so the
    dot product is the cosine similarity. Classes with fewer photos average what they have.
    """
    if top_m < 1:
        raise ValueError("top_m must be at least 1")
    similarity = queries @ gallery.T
    labels = np.asarray(gallery_labels)
    scores = np.empty((len(queries), len(classes)), dtype=np.float32)
    for column, label in enumerate(classes):
        members = similarity[:, labels == label]
        if members.shape[1] == 0:
            raise ValueError(f"class {label!r} has no gallery images")
        m = min(top_m, members.shape[1])
        best = -np.partition(-members, m - 1, axis=1)[:, :m]
        scores[:, column] = best.mean(axis=1)
    return scores
