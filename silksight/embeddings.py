"""Frozen image backbones that turn a photo into an L2-normalized embedding.

Torch and transformers are imported only when a backbone is created, so importing this module
does not load weights or touch the network.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from silksight.settings import MODELS_DIR


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    kind: str  # "dinov2" | "clip"


BACKBONES = {
    spec.name: spec
    for spec in (
        BackboneSpec("dinov2-base", "facebook/dinov2-base", "dinov2"),
        BackboneSpec("clip-b16", "openai/clip-vit-base-patch16", "clip"),
    )
}


def default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """One loaded backbone; ``embed`` returns float32 rows with unit length."""

    def __init__(self, spec: BackboneSpec, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel

        self.spec = spec
        self.device = device or default_device()
        cache = str(MODELS_DIR / "huggingface")
        self._processor = AutoImageProcessor.from_pretrained(spec.hf_id, cache_dir=cache)
        model_class = CLIPModel if spec.kind == "clip" else AutoModel
        self._model = model_class.from_pretrained(spec.hf_id, cache_dir=cache).to(self.device)
        self._model.eval()
        self._torch = torch

    def _features(self, pixels: Any) -> Any:
        if self.spec.kind == "clip":
            output = self._model.get_image_features(pixel_values=pixels)
            return output if isinstance(output, self._torch.Tensor) else output.pooler_output
        # DINOv2: the layer-normed CLS token is the global image descriptor.
        return self._model(pixel_values=pixels).pooler_output

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )
        with self._torch.inference_mode():
            features = self._features(inputs["pixel_values"].to(self.device))
        vectors = features.float().cpu().numpy()
        return normalize_rows(vectors)

    def embed_files(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        batches = []
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            batches.append(self.embed(images))
        return np.concatenate(batches).astype(np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)
