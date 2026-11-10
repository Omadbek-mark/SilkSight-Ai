import pytest

from silksight.catalog import LandmarkSeed
from silksight.wikidata import WikidataError, facts_from_entity, label_mismatch

SEED = LandmarkSeed(
    id="registan",
    qid="Q1373583",
    name_en="Registan",
    country="UZ",
    city="Samarkand",
    category="architecture",
    hotelia_destination="Samarkand",
)


def _value(value):
    return {"mainsnak": {"snaktype": "value", "datavalue": {"value": value}}}


ENTITY = {
    "labels": {"en": {"value": "Registan"}, "uz": {"value": "Registon"}},
    "descriptions": {"en": {"value": "historical center of Samarkand"}},
    "claims": {
        "P625": [_value({"latitude": 39.6547, "longitude": 66.9758})],
        "P1435": [_value({"id": "Q9259"})],
    },
    "sitelinks": {"enwiki": {"url": "https://en.wikipedia.org/wiki/Registan"}},
}


def test_facts_are_copied_from_the_entity():
    facts = facts_from_entity(SEED, ENTITY)
    assert facts.names == {"en": "Registan", "uz": "Registon"}
    assert facts.display_names == {"en": "Registan", "uz": "Registon"}
    assert (facts.latitude, facts.longitude) == (39.6547, 66.9758)
    assert facts.unesco_site_qid == "Q1373583"
    assert facts.wikipedia == {"en": "https://en.wikipedia.org/wiki/Registan"}
    assert label_mismatch(SEED, facts) is None


def test_unesco_site_is_found_through_the_parent_ensemble():
    entity = ENTITY | {"claims": {"P625": ENTITY["claims"]["P625"], "P361": [_value({"id": "Q2"})]}}
    parents = {"Q2": {"claims": {"P1435": [_value({"id": "Q9259"})]}}}
    assert facts_from_entity(SEED, entity, parents).unesco_site_qid == "Q2"
    assert facts_from_entity(SEED, entity, {}).unesco_site_qid is None


def test_curated_uzbek_name_overrides_the_wikidata_label():
    seed = LandmarkSeed(**(SEED.__dict__ | {"name_uz": "Registon maydoni"}))
    assert facts_from_entity(seed, ENTITY).display_names["uz"] == "Registon maydoni"


def test_entity_without_coordinates_is_rejected():
    with pytest.raises(WikidataError, match="no coordinates"):
        facts_from_entity(SEED, ENTITY | {"claims": {}})


def test_wrong_entity_label_is_flagged():
    facts = facts_from_entity(SEED, ENTITY | {"labels": {"en": {"value": "Registan Desert"}}})
    assert "Registan Desert" in label_mismatch(SEED, facts)
