"""Landmark catalog: the curated seed list and validated Hotelia links."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from silksight.settings import (
    CATEGORIES,
    COUNTRY_CODES,
    HOTELIA_BASE_URL,
    HOTELIA_DESTINATIONS,
)

LANDMARK_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")


class CatalogError(ValueError):
    """The seed catalog breaks an invariant the rest of the project relies on."""


@dataclass(frozen=True)
class LandmarkSeed:
    id: str
    qid: str
    name_en: str
    country: str
    city: str | None
    category: str
    hotelia_destination: str | None
    # Curated Uzbek display name when the Wikidata label is missing or too long.
    name_uz: str | None = None
    # Extra Commons file search when the category alone has too few photos.
    commons_search: str | None = None


def parse_seed(records: list[dict]) -> list[LandmarkSeed]:
    """Validate raw seed records; every problem is reported, not just the first."""
    errors: list[str] = []
    seeds: list[LandmarkSeed] = []
    seen_ids: set[str] = set()
    seen_qids: set[str] = set()

    for index, record in enumerate(records):
        label = f"#{index} ({record.get('id', '?')})"
        try:
            seed = LandmarkSeed(**record)
        except TypeError as error:
            errors.append(f"{label}: {error}")
            continue

        if not LANDMARK_ID_PATTERN.match(seed.id):
            errors.append(f"{label}: id must be lowercase kebab-case")
        if not QID_PATTERN.match(seed.qid):
            errors.append(f"{label}: qid must look like Q12345")
        if seed.country not in COUNTRY_CODES:
            errors.append(f"{label}: unknown country {seed.country!r}")
        if seed.category not in CATEGORIES:
            errors.append(f"{label}: unknown category {seed.category!r}")
        if seed.hotelia_destination is not None:
            if seed.hotelia_destination not in HOTELIA_DESTINATIONS:
                errors.append(f"{label}: Hotelia has no city {seed.hotelia_destination!r}")
            elif seed.hotelia_destination != seed.city:
                # A hotel link must point to the landmark's own city, never a "nearest" guess.
                errors.append(f"{label}: hotelia_destination must equal city")
        if seed.id in seen_ids:
            errors.append(f"{label}: duplicate id")
        if seed.qid in seen_qids:
            errors.append(f"{label}: duplicate qid")

        seen_ids.add(seed.id)
        seen_qids.add(seed.qid)
        seeds.append(seed)

    if errors:
        raise CatalogError("Invalid landmark seed:\n" + "\n".join(errors))
    return seeds


def load_seed(path: Path) -> list[LandmarkSeed]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise CatalogError("Landmark seed must be a JSON list")
    return parse_seed(records)


def hotelia_url(
    destination: str | None,
    check_in: str | None = None,
    check_out: str | None = None,
) -> str | None:
    """Build a Hotelia search link; landmarks outside Hotelia's cities get no link."""
    if destination is None:
        return None
    if destination not in HOTELIA_DESTINATIONS:
        raise CatalogError(f"Hotelia has no city {destination!r}")
    params = {"destination": destination}
    if check_in and check_out:
        params |= {"checkIn": check_in, "checkOut": check_out}
    return f"{HOTELIA_BASE_URL}/hotel?{urlencode(params)}"


def load_catalog(path: Path) -> dict[str, dict]:
    """Built landmark facts (``scripts.build_catalog`` output) keyed by landmark id."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {landmark["id"]: landmark for landmark in data["landmarks"]}


def load_summaries(path: Path) -> dict[str, dict]:
    """Wikipedia lead texts (``scripts.build_summaries`` output); empty if not built yet."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)["summaries"]
