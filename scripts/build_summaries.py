"""Copy Wikipedia lead sections (en, uz) for every catalog landmark (explicit network action).

Usage: python -m scripts.build_summaries

Writes catalog/summaries.json: landmark id -> language -> text with title, URL, revision id
and CC BY-SA 4.0 license. Languages without an article are left out, never machine-translated.
"""

import json
from collections import defaultdict
from datetime import date

from silksight.catalog import load_catalog
from silksight.settings import CATALOG_PATH, SUMMARIES_PATH, SUMMARY_LANGUAGES
from silksight.wikipedia import WIKIPEDIA_LICENSE, article_title, fetch_intros


def main() -> int:
    catalog = load_catalog(CATALOG_PATH)
    wanted: dict[str, dict[str, str]] = defaultdict(dict)  # language -> title -> landmark id
    for landmark_id, item in catalog.items():
        for language in SUMMARY_LANGUAGES:
            if url := item["wikipedia"].get(language):
                lang, title = article_title(url)
                wanted[lang][title] = landmark_id

    summaries: dict[str, dict] = {landmark_id: {} for landmark_id in catalog}
    for language, titles in wanted.items():
        intros = fetch_intros(language, sorted(titles))
        for title, landmark_id in titles.items():
            if title in intros:
                summaries[landmark_id][language] = intros[title].to_dict()
        print(f"{language}: {sum(t in intros for t in titles)}/{len(titles)} articles copied")

    output = {
        "source": "Wikipedia",
        "license": WIKIPEDIA_LICENSE,
        "fetched_on": date.today().isoformat(),
        "summaries": summaries,
    }
    SUMMARIES_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", "utf-8")
    for landmark_id, texts in summaries.items():
        missing = [lang for lang in SUMMARY_LANGUAGES if lang not in texts]
        if missing:
            print(f"  {landmark_id}: no {', '.join(missing)} text")
    print(f"Wrote {SUMMARIES_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
