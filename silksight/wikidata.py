"""Turn Wikidata entities into catalog facts; facts are copied, never generated."""

from dataclasses import asdict, dataclass

import requests

from silksight.catalog import LandmarkSeed

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "SilkSight/0.1 (local portfolio project; catalog builder)"
LANGUAGES = ("en", "uz", "ru")
WORLD_HERITAGE_SITE = "Q9259"
MAX_IDS_PER_REQUEST = 50  # wbgetentities limit for anonymous clients
TIMEOUT_SECONDS = 30


class WikidataError(RuntimeError):
    """Wikidata did not return a usable entity."""


@dataclass(frozen=True)
class LandmarkFacts:
    id: str
    qid: str
    country: str
    city: str | None
    category: str
    hotelia_destination: str | None
    names: dict[str, str]
    display_names: dict[str, str]
    descriptions: dict[str, str]
    latitude: float
    longitude: float
    # QID of the World Heritage Site this place is (part of), when Wikidata links it.
    # None means "not linked in Wikidata", not "not a World Heritage Site".
    unesco_site_qid: str | None
    wikipedia: dict[str, str]
    wikidata_url: str
    commons_category: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_entities(
    qids: list[str], props: str = "labels|descriptions|claims|sitelinks/urls"
) -> dict:
    if len(qids) > MAX_IDS_PER_REQUEST:
        raise WikidataError(f"At most {MAX_IDS_PER_REQUEST} entities per request")
    if not qids:
        return {}
    response = requests.get(
        WIKIDATA_API_URL,
        params={
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": props,
            "languages": "|".join(LANGUAGES),
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_SECONDS,
        allow_redirects=False,
    )
    if response.status_code == 429:
        raise WikidataError("Wikidata rate limit (HTTP 429); wait a few minutes and rerun")
    response.raise_for_status()
    return response.json().get("entities", {})


def _claim_values(entity: dict, prop: str) -> list:
    values = []
    for claim in entity.get("claims", {}).get(prop, []):
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") == "value":
            values.append(snak["datavalue"]["value"])
    return values


def _texts(entity: dict, key: str) -> dict[str, str]:
    return {
        language: entity[key][language]["value"]
        for language in LANGUAGES
        if language in entity.get(key, {})
    }


def _is_world_heritage_site(entity: dict) -> bool:
    return WORLD_HERITAGE_SITE in {value.get("id") for value in _claim_values(entity, "P1435")}


def parent_qids(entity: dict) -> list[str]:
    """``part of`` (P361) targets; a building is often listed only through its ensemble."""
    return [value["id"] for value in _claim_values(entity, "P361") if "id" in value]


def unesco_site_qid(qid: str, entity: dict, parents: dict[str, dict]) -> str | None:
    if _is_world_heritage_site(entity):
        return qid
    for parent_qid in parent_qids(entity):
        if _is_world_heritage_site(parents.get(parent_qid, {})):
            return parent_qid
    return None


def facts_from_entity(
    seed: LandmarkSeed, entity: dict, parents: dict[str, dict] | None = None
) -> LandmarkFacts:
    if "missing" in entity:
        raise WikidataError(f"{seed.id}: {seed.qid} does not exist")

    coordinates = _claim_values(entity, "P625")
    if not coordinates:
        raise WikidataError(f"{seed.id}: {seed.qid} has no coordinates (P625)")

    names = _texts(entity, "labels")
    sitelinks = entity.get("sitelinks", {})
    wikipedia = {
        language: sitelinks[f"{language}wiki"]["url"]
        for language in LANGUAGES
        if f"{language}wiki" in sitelinks and "url" in sitelinks[f"{language}wiki"]
    }

    return LandmarkFacts(
        id=seed.id,
        qid=seed.qid,
        country=seed.country,
        city=seed.city,
        category=seed.category,
        hotelia_destination=seed.hotelia_destination,
        names=names,
        display_names={
            "en": names.get("en", seed.name_en),
            "uz": seed.name_uz or names.get("uz") or names.get("en", seed.name_en),
        },
        descriptions=_texts(entity, "descriptions"),
        latitude=coordinates[0]["latitude"],
        longitude=coordinates[0]["longitude"],
        unesco_site_qid=unesco_site_qid(seed.qid, entity, parents or {}),
        wikipedia=wikipedia,
        wikidata_url=f"https://www.wikidata.org/wiki/{seed.qid}",
        commons_category=next(iter(_claim_values(entity, "P373")), None),
    )


def label_mismatch(seed: LandmarkSeed, facts: LandmarkFacts) -> str | None:
    """Flag a seed whose QID may point to a different entity than intended."""
    label = facts.names.get("en")
    if label is None or label.casefold() != seed.name_en.casefold():
        return f"{seed.id}: expected {seed.name_en!r}, Wikidata label is {label!r}"
    return None
