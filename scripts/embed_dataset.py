"""Compute frozen-backbone embeddings for every manifest image (Model A input).

Usage: python -m scripts.embed_dataset [dinov2-base] [clip-b16]
       (default: all backbones; first run downloads weights into .models/)

Writes artifacts/embeddings/<backbone>.npz with the manifest files, labels, splits and vectors.
"""

import sys
import time

import numpy as np

from silksight.dataset import read_manifest
from silksight.embeddings import BACKBONES, Embedder
from silksight.settings import DATA_DIR, EMBEDDINGS_DIR, MANIFEST_PATH


def main() -> int:
    names = sys.argv[1:] or list(BACKBONES)
    unknown = [name for name in names if name not in BACKBONES]
    if unknown:
        print(f"Unknown backbone(s): {', '.join(unknown)}; choose from {', '.join(BACKBONES)}")
        return 2
    rows = read_manifest(MANIFEST_PATH)
    paths = [DATA_DIR / row.file for row in rows]
    missing = [path for path in paths if not path.exists()]
    if missing:
        print(f"{len(missing)} manifest images are missing, e.g. {missing[0]}")
        return 1
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        embedder = Embedder(BACKBONES[name])
        started = time.perf_counter()
        vectors = embedder.embed_files(paths)
        seconds = time.perf_counter() - started
        target = EMBEDDINGS_DIR / f"{name}.npz"
        np.savez_compressed(
            target,
            vectors=vectors,
            files=np.array([row.file for row in rows]),
            labels=np.array([row.label for row in rows]),
            splits=np.array([row.split for row in rows]),
        )
        print(
            f"{name}: {len(rows)} images, dim {vectors.shape[1]}, {seconds:.0f}s "
            f"on {embedder.device} -> {target.relative_to(DATA_DIR.parent)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
