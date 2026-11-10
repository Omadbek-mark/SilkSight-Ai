"""Dataset rules: license filter, author-diverse selection, author-separated splits, manifest."""

import csv
import hashlib
import io
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from silksight.commons import CommonsImage

UNKNOWN_LABEL = "unknown"
MIN_SIDE_PIXELS = 256
JPEG_QUALITY = 90

# Free licenses whose attribution we can record; NC/ND and unclear licenses are rejected.
_ALLOWED_LICENSE = re.compile(
    r"^(cc0|public domain|pd\b|cc[ -]by(-sa)?[ -]\d|attribution$)", re.IGNORECASE
)
_FORBIDDEN_LICENSE = re.compile(r"\b(nc|nd)\b", re.IGNORECASE)


class DatasetError(ValueError):
    """An image or manifest breaks a dataset invariant."""


@dataclass(frozen=True)
class ManifestRow:
    label: str  # landmark id, or "unknown" for the open-set evaluation images
    split: str  # train | val | test
    file: str  # path relative to the data directory
    sha256: str
    source_group: str  # landmark id, or the unknown-source id it came from
    title: str
    page_url: str
    author: str
    license: str
    license_url: str


MANIFEST_FIELDS = [field.name for field in fields(ManifestRow)]


def is_allowed_license(license_name: str) -> bool:
    name = license_name.strip()
    return bool(_ALLOWED_LICENSE.match(name)) and not _FORBIDDEN_LICENSE.search(name)


def select_diverse(images: list[CommonsImage], limit: int) -> list[CommonsImage]:
    """Round-robin over authors so one prolific uploader cannot dominate a class."""
    by_author: dict[str, list[CommonsImage]] = defaultdict(list)
    for image in images:
        by_author[image.author].append(image)
    queues = [by_author[author] for author in sorted(by_author)]
    selected: list[CommonsImage] = []
    while len(selected) < limit and any(queues):
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
    return selected


def _stable_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def assign_splits(
    authors: list[str], group: str, held_out: tuple[tuple[str, int], ...]
) -> list[str]:
    """Give every image of one author the same split, so test photos never share a
    photographer (same camera, day, angle) with training photos.

    ``held_out`` is e.g. (("test", 5), ("val", 3)): smallest author groups go to whichever
    held-out split is furthest from its minimum, which keeps those sets diverse and stops one
    split from taking every small group; the rest is train. Train always keeps at least half
    of the images, so a class with few photographers may get small held-out sets.
    """
    counts: dict[str, int] = defaultdict(int)
    for author in authors:
        counts[author] += 1
    order = sorted(counts, key=lambda author: (counts[author], _stable_key(group, author)))
    minimum = dict(held_out)
    taken = {split: 0 for split in minimum}

    split_of: dict[str, str] = {}
    train_size = len(authors)
    for author in order:
        needy = [split for split in minimum if taken[split] < minimum[split]]
        if not needy or train_size - counts[author] < len(authors) / 2:
            split_of[author] = "train"
            continue
        split = min(needy, key=lambda name: taken[name] / minimum[name])
        split_of[author] = split
        taken[split] += counts[author]
        train_size -= counts[author]
    return [split_of[author] for author in authors]


def normalize_image(data: bytes) -> bytes:
    """Decode, apply EXIF orientation, drop all metadata and re-encode as RGB JPEG."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise DatasetError(f"not a readable image: {error}") from error
    if min(upright.size) < MIN_SIDE_PIXELS:
        raise DatasetError(f"too small: {upright.size[0]}x{upright.size[1]}")
    output = io.BytesIO()
    upright.save(output, "JPEG", quality=JPEG_QUALITY)
    return output.getvalue()


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    ordered = sorted(rows, key=lambda row: (row.label, row.split, row.title))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in ordered)


def read_manifest(path: Path) -> list[ManifestRow]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return [ManifestRow(**row) for row in csv.DictReader(file)]


def read_exclusions(path: Path) -> dict[str, str]:
    """Manually reviewed exclusions: Commons title -> reason."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["title"]: row["reason"] for row in csv.DictReader(file)}


def representative_photos(rows: list[ManifestRow]) -> dict[str, ManifestRow]:
    """One training photo per landmark for display, chosen by title so it stays stable."""
    photos: dict[str, ManifestRow] = {}
    for row in sorted(rows, key=lambda row: row.title):
        if row.split == "train" and row.label != UNKNOWN_LABEL:
            photos.setdefault(row.label, row)
    return photos
