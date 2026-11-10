"""Evaluate Model A (frozen embeddings + kNN) with an "unknown" threshold.

Usage: python -m scripts.evaluate_knn   (offline; needs scripts.embed_dataset output)

The gallery is the train split. ``top_m`` and the threshold are chosen on val only; test is
scored once with those choices. Writes artifacts/model_a/report.json and, for the best
backbone on val, artifacts/model_a/index.npz (gallery used by the API later).
"""

import json
from dataclasses import asdict

import numpy as np

from silksight.dataset import UNKNOWN_LABEL
from silksight.evaluation import OpenSetReport, evaluate, tune_threshold
from silksight.knn import class_scores
from silksight.settings import EMBEDDINGS_DIR, MODEL_A_DIR

TOP_M_CHOICES = (1, 3, 5)


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(EMBEDDINGS_DIR / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def split(data: dict[str, np.ndarray], name: str) -> tuple[np.ndarray, list[str]]:
    mask = data["splits"] == name
    return data["vectors"][mask], data["labels"][mask].tolist()


def summary(report: OpenSetReport) -> str:
    return (
        f"top1 {report.top1:.3f}  top3 {report.top3:.3f}  "
        f"known-ok {report.known_accepted_correct:.3f}  "
        f"false-rej {report.known_false_rejection:.3f}  "
        f"unknown-rej {report.unknown_rejection:.3f}  macroF1 {report.macro_f1:.3f}"
    )


def main() -> int:
    backbones = sorted(path.stem for path in EMBEDDINGS_DIR.glob("*.npz"))
    if not backbones:
        print("No embeddings yet: run python -m scripts.embed_dataset")
        return 1
    results = {}
    for backbone in backbones:
        data = load(backbone)
        gallery, gallery_labels = split(data, "train")
        classes = sorted(set(gallery_labels) - {UNKNOWN_LABEL})
        val, val_labels = split(data, "val")
        test, test_labels = split(data, "test")

        tuned = []
        for top_m in TOP_M_CHOICES:
            val_scores = class_scores(val, gallery, gallery_labels, classes, top_m)
            threshold = tune_threshold(val_scores, classes, val_labels)
            tuned.append((evaluate(val_scores, classes, val_labels, threshold), top_m))
        val_report, top_m = max(tuned, key=lambda item: item[0].balanced_open_set)

        test_scores = class_scores(test, gallery, gallery_labels, classes, top_m)
        test_report = evaluate(test_scores, classes, test_labels, val_report.threshold)
        results[backbone] = {
            "top_m": top_m,
            "val": asdict(val_report),
            "test": asdict(test_report),
            "val_by_top_m": {m: report.balanced_open_set for report, m in tuned},
        }
        print(f"\n{backbone}  (top_m={top_m}, threshold={val_report.threshold:.3f})")
        print(f"  val   {summary(val_report)}")
        print(f"  test  {summary(test_report)}")

    # Choose by validation, never by test, so the test numbers stay an honest estimate.
    best = max(results, key=lambda name: results[name]["val"]["balanced_open_set"])
    MODEL_A_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "frozen embeddings + kNN",
        "gallery_split": "train",
        "selected_backbone": best,
        "backbones": results,
    }
    (MODEL_A_DIR / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    data = load(best)
    gallery, gallery_labels = split(data, "train")
    np.savez_compressed(
        MODEL_A_DIR / "index.npz",
        vectors=gallery,
        labels=np.array(gallery_labels),
        backbone=np.array(best),
        top_m=np.array(results[best]["top_m"]),
        threshold=np.array(results[best]["val"]["threshold"]),
    )
    print(f"\nSelected on val: {best}. Wrote {MODEL_A_DIR.name}/report.json and index.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
