"""Copy the lead section of each landmark's Wikipedia article, with attribution.

Wikipedia text is CC BY-SA 4.0: every stored text keeps its article title, URL and revision id
so the UI can credit it and link back. The text is copied, not rewritten or summarized by a
model; only whitespace and empty brackets left by removed markup are cleaned up.
"""

import re
from dataclasses import asdict, dataclass
from urllib.parse import unquote, urlparse

from silksight.commons import CommonsError, polite_get

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
BATCH_SIZE = 20  # the extracts API returns at most 20 intro extracts per request

_EMPTY_BRACKETS = re.compile(r"\s*\(\s*[,;]?\s*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_SPACES = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class ArticleIntro:
    language: str
    title: str
    url: str
    revision_id: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self) | {"license": WIKIPEDIA_LICENSE, "license_url": WIKIPEDIA_LICENSE_URL}


def article_title(url: str) -> tuple[str, str]:
    """``https://uz.wikipedia.org/wiki/Registon_maydoni`` -> ("uz", "Registon maydoni")."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host.endswith(".wikipedia.org") or not parsed.path.startswith("/wiki/"):
        raise ValueError(f"not a Wikipedia article URL: {url}")
    return host.split(".")[0], unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")


def clean_extract(text: str) -> str:
    """Tidy what the plain-text extract leaves behind (e.g. "Registan ( ) is" after
    pronunciation markup is stripped), keeping paragraphs."""
    paragraphs = []
    for paragraph in text.split("\n"):
        paragraph = _EMPTY_BRACKETS.sub("", paragraph)
        paragraph = _SPACE_BEFORE_PUNCT.sub(r"\1", paragraph)
        paragraph = _SPACES.sub(" ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def parse_intros(language: str, data: dict) -> dict[str, ArticleIntro]:
    """Map requested titles (before redirects/normalization) to their intros."""
    query = data.get("query", {})
    aliases = {}
    for step in query.get("normalized", []) + query.get("redirects", []):
        aliases[step["to"]] = aliases.get(step["from"], step["from"])
    intros = {}
    for page in query.get("pages", []):
        text = clean_extract(page.get("extract", ""))
        if page.get("missing") or not text:
            continue
        intro = ArticleIntro(
            language=language,
            title=page["title"],
            url=page["fullurl"],
            revision_id=int(page["lastrevid"]),
            text=text,
        )
        requested = page["title"]
        while requested in aliases:
            requested = aliases[requested]
        intros[requested] = intro
        intros[page["title"]] = intro
    return intros


def fetch_intros(language: str, titles: list[str]) -> dict[str, ArticleIntro]:
    api = f"https://{language}.wikipedia.org/w/api.php"
    intros: dict[str, ArticleIntro] = {}
    for start in range(0, len(titles), BATCH_SIZE):
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "inprop": "url",
            "redirects": "1",
            "titles": "|".join(titles[start : start + BATCH_SIZE]),
        }
        response = polite_get(api, params)
        if not response.headers.get("content-type", "").startswith("application/json"):
            raise CommonsError(f"{language}.wikipedia.org did not return JSON")
        intros |= parse_intros(language, response.json())
    return intros
