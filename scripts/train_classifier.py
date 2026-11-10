"""Fine-tune Model B (ConvNeXt-Tiny) and evaluate it on the same test set as Model A.

Usage: python -m scripts.train_classifier   (first run downloads ImageNet weights, ~110 MB)

Train split trains; val picks the best epoch (known-photo top-1) and then the "unknown"
threshold; test is scored once. Writes artifacts/model_b/model.pt and report.json.
"""

import json
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from silksight.classifier import (
    ImageDataset,
    TrainConfig,
    build_model,
    class_weights,
    predict_proba,
    train_transform,
)
from silksight.dataset import UNKNOWN_LABEL, read_manifest
from silksight.embeddings import default_device
from silksight.evaluation import evaluate, tune_threshold
from silksight.settings import DATA_DIR, MANIFEST_PATH, MODEL_B_DIR, MODELS_DIR


def main() -> int:
    config = TrainConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.hub.set_dir(str(MODELS_DIR / "torch"))
    device = default_device()

    rows = read_manifest(MANIFEST_PATH)
    by_split = {
        name: [row for row in rows if row.split == name] for name in ("train", "val", "test")
    }
    classes = sorted({row.label for row in by_split["train"]} - {UNKNOWN_LABEL})
    index = {name: i for i, name in enumerate(classes)}
    train = by_split["train"]
    val_known = [row for row in by_split["val"] if row.label != UNKNOWN_LABEL]

    loader = torch.utils.data.DataLoader(
        ImageDataset(
            [DATA_DIR / row.file for row in train],
            [index[row.label] for row in train],
            train_transform(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = build_model(len(classes), pretrained=True).to(device)
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": config.backbone_lr}, {"params": head, "lr": config.head_lr}],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(loader)
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights([row.label for row in train], classes).to(device),
        label_smoothing=config.label_smoothing,
    )

    MODEL_B_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_B_DIR / "model.pt"
    val_paths = [DATA_DIR / row.file for row in val_known]
    val_targets = np.array([index[row.label] for row in val_known])
    best_epoch, best_top1, history = 0, -1.0, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(targets)
        val_top1 = float((predict_proba(model, val_paths, device).argmax(1) == val_targets).mean())
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_top1": val_top1})
        print(
            f"epoch {epoch:2d}  loss {total / len(train):.3f}  val top1 {val_top1:.3f}  "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )
        if val_top1 > best_top1:
            best_epoch, best_top1 = epoch, val_top1
            torch.save({"state_dict": model.state_dict(), "classes": classes}, checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    reports = {}
    for name in ("val", "test"):
        split_rows = by_split[name]
        scores = predict_proba(model, [DATA_DIR / row.file for row in split_rows], device)
        labels = [row.label for row in split_rows]
        if name == "val":
            threshold = tune_threshold(scores, classes, labels)
        reports[name] = evaluate(scores, classes, labels, threshold)
        r = reports[name]
        print(
            f"{name:5} top1 {r.top1:.3f}  top3 {r.top3:.3f}  "
            f"known-ok {r.known_accepted_correct:.3f}  false-rej {r.known_false_rejection:.3f}  "
            f"unknown-rej {r.unknown_rejection:.3f}  macroF1 {r.macro_f1:.3f}"
        )

    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "threshold": threshold}, checkpoint
    )
    report = {
        "model": "ConvNeXt-Tiny fine-tuned (ImageNet-1k init)",
        "device": device,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "train_seconds": round(time.perf_counter() - started),
        "history": history,
        "val": asdict(reports["val"]),
        "test": asdict(reports["test"]),
    }
    (MODEL_B_DIR / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(f"best epoch {best_epoch}; wrote {MODEL_B_DIR.name}/model.pt and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
