import pytest

from silksight.catalog import CatalogError, hotelia_url, load_seed, parse_seed
from silksight.settings import CATALOG_SEED_PATH, COUNTRY_CODES

VALID = {
    "id": "registan",
    "qid": "Q1373583",
    "name_en": "Registan",
    "country": "UZ",
    "city": "Samarkand",
    "category": "architecture",
    "hotelia_destination": "Samarkand",
}


def test_committed_seed_is_valid_and_covers_every_country():
    seeds = load_seed(CATALOG_SEED_PATH)
    assert {seed.country for seed in seeds} == set(COUNTRY_CODES)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hotelia_destination": "Samarqand"}, "Hotelia has no city"),
        ({"city": "Khiva"}, "must equal city"),
        ({"qid": "1373583"}, "qid must look like"),
        ({"id": "Registan Square"}, "kebab-case"),
    ],
)
def test_invalid_seed_is_rejected(override, message):
    with pytest.raises(CatalogError, match=message):
        parse_seed([VALID | override])


def test_duplicates_are_rejected():
    with pytest.raises(CatalogError, match="duplicate id"):
        parse_seed([VALID, VALID | {"qid": "Q1"}])


def test_hotelia_url_uses_exact_city_and_optional_dates():
    assert hotelia_url(None) is None
    assert hotelia_url("Samarkand") == "https://hotelia.uz/hotel?destination=Samarkand"
    assert hotelia_url("Bukhara", "2026-12-01", "2026-12-03") == (
        "https://hotelia.uz/hotel?destination=Bukhara&checkIn=2026-12-01&checkOut=2026-12-03"
    )
