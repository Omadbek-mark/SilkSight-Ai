"""Download landmark and "unknown place" photos from Wikimedia Commons and write the manifest.

Usage: python -m scripts.build_dataset           (network; rerunning reuses downloaded files)
       python -m scripts.build_dataset --reuse   (offline; re-applies exclusions and splits)

Images go to data/images/ (git-ignored). data/manifest.csv records author, license and split
for every image; data/exclusions.csv holds manually reviewed removals and is applied first.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace

from silksight.catalog import LandmarkSeed, load_seed
from silksight.commons import (
    CommonsError,
    CommonsImage,
    category_images,
    download,
    search_images,
    subcategories,
)
from silksight.dataset import (
    UNKNOWN_LABEL,
    DatasetError,
    ManifestRow,
    assign_splits,
    is_allowed_license,
    normalize_image,
    read_exclusions,
    read_manifest,
    select_diverse,
    write_manifest,
)
from silksight.settings import (
    CATALOG_PATH,
    CATALOG_SEED_PATH,
    DATA_DIR,
    EXCLUSIONS_PATH,
    IMAGES_DIR,
    MANIFEST_PATH,
    UNKNOWN_SOURCES_PATH,
)

TARGET_PER_LANDMARK = 40
MAX_SUBCATEGORIES = 8
UNKNOWN_PER_SOURCE = 8
LANDMARK_HELD_OUT = (("test", 5), ("val", 3))


def landmark_candidates(seed: LandmarkSeed, category: str | None) -> list[CommonsImage]:
    pool = category_images(category, 50) if category else []
    if category and len(pool) < TARGET_PER_LANDMARK:
        for subcategory in subcategories(category)[:MAX_SUBCATEGORIES]:
            pool += category_images(subcategory, 20)
            if len(pool) >= 50:
                break
    if len(pool) < TARGET_PER_LANDMARK and seed.commons_search:
        pool += search_images(seed.commons_search, 30)
    return list({image.title: image for image in pool}.values())


def fetch(image: CommonsImage, label: str, cached: dict[str, ManifestRow]) -> tuple[str, str]:
    """Return (relative file, sha256 of the downloaded bytes); reuse files already on disk."""
    previous = cached.get(image.title)
    if previous and (DATA_DIR / previous.file).exists():
        return previous.file, previous.sha256
    data = download(image)
    sha256 = hashlib.sha256(data).hexdigest()
    relative = f"images/{label}/{sha256[:16]}.jpg"
    target = DATA_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(normalize_image(data))
    return relative, sha256


def collect(
    label: str,
    group: str,
    images: list[CommonsImage],
    cached: dict[str, ManifestRow],
    seen_sha: dict[str, str],
    problems: list[str],
) -> list[dict]:
    collected = []
    for image in images:
        try:
            file, sha256 = fetch(image, label, cached)
        except (DatasetError, CommonsError) as error:
            problems.append(f"{group}: skipped {image.title}: {error}")
            continue
        if sha256 in seen_sha:
            problems.append(f"{group}: duplicate of {seen_sha[sha256]}: {image.title}")
            continue
        seen_sha[sha256] = image.title
        collected.append({"image": image, "file": file, "sha256": sha256})
    return collected


def to_rows(label: str, group: str, items: list[dict]) -> list[ManifestRow]:
    return [
        ManifestRow(
            label=label,
            split="train",  # placeholder; resplit() assigns the real split by author
            file=item["file"],
            sha256=item["sha256"],
            source_group=group,
            title=item["image"].title,
            page_url=item["image"].page_url,
            author=item["image"].author,
            license=item["image"].license,
            license_url=item["image"].license_url,
        )
        for item in items
    ]


def usable(images: list[CommonsImage], exclusions: dict[str, str]) -> list[CommonsImage]:
    return [
        image
        for image in images
        if image.title not in exclusions and is_allowed_license(image.license)
    ]


def resplit(label: str, group: str, rows: list[ManifestRow]) -> list[ManifestRow]:
    authors = [row.author for row in rows]
    if label == UNKNOWN_LABEL:
        # Unknown photos never train a class: half by author to test, the rest to val
        # (where the "unknown" threshold is tuned).
        splits = assign_splits(authors, group, (("test", len(rows) // 2),))
        splits = ["val" if split == "train" else split for split in splits]
    else:
        splits = assign_splits(authors, group, LANDMARK_HELD_OUT)
    return [replace(row, split=split) for row, split in zip(rows, splits, strict=True)]


def reuse_manifest() -> int:
    """Apply exclusions, removed landmarks and split rules to the reviewed manifest offline,
    without refilling excluded slots with new unreviewed Commons images."""
    labels = {seed.id for seed in load_seed(CATALOG_SEED_PATH)} | {UNKNOWN_LABEL}
    exclusions = read_exclusions(EXCLUSIONS_PATH)
    groups: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in read_manifest(MANIFEST_PATH):
        if row.label in labels and row.title not in exclusions:
            groups[(row.label, row.source_group)].append(row)

    rows: list[ManifestRow] = []
    for (label, group), group_rows in sorted(groups.items()):
        rows += resplit(label, group, sorted(group_rows, key=lambda row: row.title))
        counts = Counter(row.split for row in rows if row.source_group == group)
        print(f"{group:28} {dict(sorted(counts.items()))}")
    write_manifest(MANIFEST_PATH, rows)
    print(f"Wrote {len(rows)} images to {MANIFEST_PATH.name}")
    return 0


def main() -> int:
    if "--reuse" in sys.argv[1:]:
        return reuse_manifest()
    seeds = load_seed(CATALOG_SEED_PATH)
    catalog = {
        item["id"]: item for item in json.loads(CATALOG_PATH.read_text("utf-8"))["landmarks"]
    }
    unknown_sources = json.loads(UNKNOWN_SOURCES_PATH.read_text("utf-8"))
    exclusions = read_exclusions(EXCLUSIONS_PATH)
    cached = {row.title: row for row in read_manifest(MANIFEST_PATH)}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    problems: list[str] = []
    seen_sha: dict[str, str] = {}

    for seed in seeds:
        candidates = usable(
            landmark_candidates(seed, catalog[seed.id]["commons_category"]), exclusions
        )
        selected = select_diverse(candidates, TARGET_PER_LANDMARK)
        items = collect(seed.id, seed.id, selected, cached, seen_sha, problems)
        landmark_rows = resplit(seed.id, seed.id, to_rows(seed.id, seed.id, items))
        rows += landmark_rows
        print(f"{seed.id:28} {dict(Counter(row.split for row in landmark_rows))}", flush=True)

    for source in unknown_sources:
        selected = select_diverse(
            usable(category_images(source["category"], 30), exclusions), UNKNOWN_PER_SOURCE
        )
        items = collect(UNKNOWN_LABEL, source["id"], selected, cached, seen_sha, problems)
        unknown_rows = resplit(
            UNKNOWN_LABEL, source["id"], to_rows(UNKNOWN_LABEL, source["id"], items)
        )
        rows += unknown_rows
        print(f"unknown/{source['id']:20} {dict(Counter(r.split for r in unknown_rows))}")

    write_manifest(MANIFEST_PATH, rows)
    print(f"Wrote {len(rows)} images to {MANIFEST_PATH.name}")
    for problem in problems:
        print(f"NOTE {problem}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
