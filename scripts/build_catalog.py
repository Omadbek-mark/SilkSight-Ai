"""Build catalog/landmarks.json from the seed list and Wikidata (explicit network action).

Usage: python -m scripts.build_catalog
"""

import json
import sys
from datetime import date

from silksight.catalog import load_seed
from silksight.settings import CATALOG_PATH, CATALOG_SEED_PATH
from silksight.wikidata import facts_from_entity, fetch_entities, label_mismatch, parent_qids


def main() -> int:
    seeds = load_seed(CATALOG_SEED_PATH)
    entities = fetch_entities([seed.qid for seed in seeds])
    parent_ids = sorted({qid for entity in entities.values() for qid in parent_qids(entity)})
    parents = fetch_entities(parent_ids, props="claims")

    landmarks = []
    warnings = []
    for seed in seeds:
        facts = facts_from_entity(seed, entities.get(seed.qid, {"missing": ""}), parents)
        if mismatch := label_mismatch(seed, facts):
            warnings.append(mismatch)
        landmarks.append(facts.to_dict())

    catalog = {
        "source": "Wikidata",
        "license": "CC0 1.0",
        "fetched_on": date.today().isoformat(),
        "landmarks": landmarks,
    }
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", "utf-8")

    print(f"Wrote {len(landmarks)} landmarks to {CATALOG_PATH.name}")
    for warning in warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
