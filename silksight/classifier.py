"""Model B: an ImageNet-pretrained ConvNeXt-Tiny fine-tuned on the landmark classes.

Only the known landmarks are classes; "unknown" is decided afterwards by a threshold on the top
softmax probability (see ``evaluation``), so unknown photos are never used for training.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    seed: int = 7


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    head = model.classifier[2]
    model.classifier[2] = nn.Linear(head.in_features, num_classes)
    return model


def train_transform() -> transforms.Compose:
    # Mild crops and colour changes: tourists frame the same facade differently and shoot at
    # different times of day. No vertical flips or big rotations; buildings stay upright.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


def class_weights(labels: list[str], classes: list[str]) -> torch.Tensor:
    """Inverse-frequency loss weights (mean 1) so 11-photo classes are not drowned out by
    40-photo ones."""
    counts = Counter(labels)
    missing = [name for name in classes if counts[name] == 0]
    if missing:
        raise ValueError(f"no training images for: {', '.join(missing)}")
    raw = np.array([1.0 / counts[name] for name in classes])
    return torch.tensor(raw / raw.mean(), dtype=torch.float32)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], targets: list[int], transform: transforms.Compose):
        self.paths, self.targets, self.transform = paths, targets, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.targets[index]


@torch.inference_mode()
def predict_proba(model: nn.Module, paths: list[Path], device: str, batch: int = 32) -> np.ndarray:
    model.eval()
    loader = torch.utils.data.DataLoader(
        ImageDataset(paths, [0] * len(paths), eval_transform()), batch_size=batch
    )
    chunks = [torch.softmax(model(x.to(device)), dim=1).cpu().numpy() for x, _ in loader]
    return np.concatenate(chunks).astype(np.float32)
