"""Compare ways to combine Model A and Model B (offline; needs both models trained).

Usage: python -m scripts.evaluate_ensemble

Every choice (rule, kNN temperature, fused threshold) is made on val; test is scored once
with those choices. The rule that is not selected is still reported on test for transparency,
but it is never picked by its test numbers. Writes artifacts/ensemble/report.json and
params.json (what the API serves).
"""

import json
from dataclasses import asdict

import numpy as np
import torch

from silksight.classifier import build_model, predict_proba
from silksight.dataset import UNKNOWN_LABEL, read_manifest
from silksight.embeddings import default_device
from silksight.ensemble import (
    AGREEMENT,
    FUSION,
    agreement_predictions,
    fit_temperature,
    fuse,
    oracle_correct,
)
from silksight.evaluation import (
    OpenSetReport,
    evaluate,
    evaluate_predictions,
    predict,
    tune_threshold,
)
from silksight.knn import class_scores
from silksight.settings import (
    DATA_DIR,
    EMBEDDINGS_DIR,
    ENSEMBLE_DIR,
    MANIFEST_PATH,
    MODEL_A_DIR,
    MODEL_B_DIR,
)


def knn_scores(backbone: str, top_m: int) -> tuple[dict[str, np.ndarray], list[str], dict]:
    with np.load(EMBEDDINGS_DIR / f"{backbone}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    train = arrays["splits"] == "train"
    gallery_labels = arrays["labels"][train].tolist()
    classes = sorted(set(gallery_labels) - {UNKNOWN_LABEL})
    scores, files = {}, {}
    for name in ("val", "test"):
        mask = arrays["splits"] == name
        scores[name] = class_scores(
            arrays["vectors"][mask], arrays["vectors"][train], gallery_labels, classes, top_m
        )
        files[name] = arrays["files"][mask].tolist()
    return scores, classes, files


def summary(report: OpenSetReport) -> str:
    return (
        f"known-ok {report.known_accepted_correct:.3f}  "
        f"false-rej {report.known_false_rejection:.3f}  "
        f"unknown-rej {report.unknown_rejection:.3f}  "
        f"balanced {report.balanced_open_set:.3f}  macroF1 {report.macro_f1:.3f}"
    )


def main() -> int:
    model_a = json.loads((MODEL_A_DIR / "report.json").read_text("utf-8"))
    backbone = model_a["selected_backbone"]
    top_m = model_a["backbones"][backbone]["top_m"]
    knn_threshold = model_a["backbones"][backbone]["val"]["threshold"]
    knn, classes, knn_files = knn_scores(backbone, top_m)

    device = default_device()
    checkpoint = torch.load(MODEL_B_DIR / "model.pt", map_location=device)
    if checkpoint["classes"] != classes:
        raise SystemExit("Model A and Model B were trained on different class lists")
    classifier_threshold = float(checkpoint["threshold"])
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    rows = read_manifest(MANIFEST_PATH)
    labels, classifier = {}, {}
    for name in ("val", "test"):
        split_rows = [row for row in rows if row.split == name]
        if [row.file for row in split_rows] != knn_files[name]:
            raise SystemExit("embeddings are out of date with the manifest: re-run embed_dataset")
        labels[name] = [row.label for row in split_rows]
        classifier[name] = predict_proba(model, [DATA_DIR / row.file for row in split_rows], device)

    temperature = fit_temperature(knn["val"], classes, labels["val"])
    fused = {name: fuse(knn[name], classifier[name], temperature) for name in ("val", "test")}
    fused_threshold = tune_threshold(fused["val"], classes, labels["val"])

    reports: dict[str, dict[str, OpenSetReport]] = {}
    answers: dict[str, dict[str, list[str]]] = {}
    for name in ("val", "test"):
        y = labels[name]
        answers[name] = {
            "knn": predict(knn[name], classes, knn_threshold),
            "classifier": predict(classifier[name], classes, classifier_threshold),
            AGREEMENT: agreement_predictions(
                knn[name], classifier[name], classes, knn_threshold, classifier_threshold
            ),
            FUSION: predict(fused[name], classes, fused_threshold),
        }
        reports[name] = {
            "knn": evaluate(knn[name], classes, y, knn_threshold),
            "classifier": evaluate(classifier[name], classes, y, classifier_threshold),
            AGREEMENT: evaluate_predictions(
                fused[name], classes, y, answers[name][AGREEMENT], None
            ),
            FUSION: evaluate(fused[name], classes, y, fused_threshold),
        }

    # Chosen by validation, never by test, so the test numbers stay an honest estimate.
    selected = max((AGREEMENT, FUSION), key=lambda rule: reports["val"][rule].balanced_open_set)

    overlap = {}
    for name in ("val", "test"):
        a, b, y = answers[name]["knn"], answers[name]["classifier"], labels[name]
        overlap[name] = {
            "photos": len(y),
            "both_correct": sum(p == t and q == t for p, q, t in zip(a, b, y)),
            "only_knn_correct": sum(p == t and q != t for p, q, t in zip(a, b, y)),
            "only_classifier_correct": sum(p != t and q == t for p, q, t in zip(a, b, y)),
            "neither_correct": sum(p != t and q != t for p, q, t in zip(a, b, y)),
            "oracle_correct": oracle_correct(a, b, y),
            "agree": sum(p == q for p, q in zip(a, b)),
        }

    for name in ("val", "test"):
        print(f"\n{name}")
        for rule, report in reports[name].items():
            mark = " *" if rule == selected else ""
            print(f"  {rule:11} {summary(report)}{mark}")
        o = overlap[name]
        print(
            f"  correct answers: both {o['both_correct']}, only kNN {o['only_knn_correct']}, "
            f"only classifier {o['only_classifier_correct']}, neither {o['neither_correct']}; "
            f"oracle ceiling {o['oracle_correct']}/{o['photos']}"
        )
    print(
        f"\nSelected on val: {selected} (temperature {temperature:.4f}, "
        f"fused threshold {fused_threshold:.3f})"
    )

    params = {
        "rule": selected,
        "knn_backbone": backbone,
        "knn_top_m": top_m,
        "knn_threshold": knn_threshold,
        "classifier_threshold": classifier_threshold,
        "knn_temperature": temperature,
        "fused_threshold": fused_threshold,
    }
    report = {
        "model": "ensemble of Model A (kNN) and Model B (classifier)",
        "selected_on": "val balanced_open_set",
        "params": params,
        "overlap": overlap,
        **{
            name: {rule: asdict(r) for rule, r in split_reports.items()}
            for name, split_reports in reports.items()
        },
    }
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    (ENSEMBLE_DIR / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    (ENSEMBLE_DIR / "params.json").write_text(json.dumps(params, indent=2), "utf-8")
    print(f"Wrote {ENSEMBLE_DIR.name}/report.json and params.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
