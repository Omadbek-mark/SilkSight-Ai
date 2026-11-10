"""Wikimedia Commons client: list candidate images with author and license metadata."""

import html
import re
import time
from dataclasses import dataclass

import requests

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SilkSight/0.1 (local portfolio project; dataset builder)"
# Wikimedia throttles bulk downloads of originals and non-standard sizes; 960 is one of the
# pre-rendered standard thumbnail widths (https://w.wiki/GHai).
THUMB_WIDTH = 960
TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 1.0
MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 30
MAX_BACKOFF_SECONDS = 120

IMAGE_HOSTS = ("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")
ALLOWED_MIME_TYPES = ("image/jpeg", "image/png")
# Subcategories that rarely show the place as a visitor would photograph it.
SKIPPED_SUBCATEGORY = re.compile(
    r"\b(in art|interior|inside|maps?|plans?|drawings?|paintings?|postcards?|stamps?|coins?|"
    r"banknotes?|models?|historical|old photographs|videos?|audio|documents?|ceilings?)\b",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")


class CommonsError(RuntimeError):
    """Commons did not answer with usable data."""


@dataclass(frozen=True)
class CommonsImage:
    title: str
    page_url: str
    thumb_url: str
    width: int
    height: int
    author: str
    license: str
    license_url: str


def polite_get(url: str, params: dict | None = None) -> requests.Response:
    """GET politely: one request at a time, backing off on HTTP 429."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(REQUEST_PAUSE_SECONDS)
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt < MAX_ATTEMPTS:
            retry_after = response.headers.get("retry-after", "")
            wait = int(retry_after) if retry_after.isdigit() else DEFAULT_BACKOFF_SECONDS
            wait = min(wait, MAX_BACKOFF_SECONDS)
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
    raise CommonsError("Wikimedia rate limit (HTTP 429); wait a few minutes and rerun")


def _get(params: dict) -> dict:
    query = {"action": "query", "format": "json", "formatversion": "2"}
    response = polite_get(COMMONS_API_URL, params | query)
    if not response.headers.get("content-type", "").startswith("application/json"):
        raise CommonsError("Commons API did not return JSON")
    return response.json()


def download(image: "CommonsImage") -> bytes:
    if not image.thumb_url.startswith(IMAGE_HOSTS) or "/thumb/" not in image.thumb_url:
        raise CommonsError(f"not a standard Wikimedia thumbnail: {image.thumb_url}")
    return polite_get(image.thumb_url).content


def plain_text(value: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", value)).split())


def _image_from_page(page: dict) -> CommonsImage | None:
    info = (page.get("imageinfo") or [None])[0]
    if not info or info.get("mime") not in ALLOWED_MIME_TYPES:
        return None
    meta = info.get("extmetadata", {})
    author = plain_text(meta.get("Artist", {}).get("value", ""))
    license_name = plain_text(meta.get("LicenseShortName", {}).get("value", ""))
    if not author or not license_name:
        return None  # attribution is mandatory, so unattributable files are unusable
    thumb_url = info.get("thumburl", "")
    if "/thumb/" not in thumb_url:
        return None  # smaller than the thumbnail width: only the throttled original exists
    return CommonsImage(
        title=page["title"],
        page_url=info["descriptionurl"],
        thumb_url=thumb_url,
        width=info["width"],
        height=info["height"],
        author=author,
        license=license_name,
        license_url=meta.get("LicenseUrl", {}).get("value", ""),
    )


_IMAGEINFO = {
    "prop": "imageinfo",
    "iiprop": "url|size|mime|extmetadata",
    "iiurlwidth": THUMB_WIDTH,
    "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
}


def _images(params: dict, limit: int) -> list[CommonsImage]:
    data = _get(params | _IMAGEINFO)
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda page: page["title"])
    images = [image for page in pages if (image := _image_from_page(page))]
    return images[:limit]


def category_images(category: str, limit: int) -> list[CommonsImage]:
    return _images(
        {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": min(limit, 50),
        },
        limit,
    )


def subcategories(category: str) -> list[str]:
    data = _get(
        {
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "subcat",
            "cmlimit": 50,
        }
    )
    names = [item["title"].removeprefix("Category:") for item in data["query"]["categorymembers"]]
    return [name for name in names if not SKIPPED_SUBCATEGORY.search(name)]


def search_images(term: str, limit: int) -> list[CommonsImage]:
    return _images(
        {
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {term}",
            "gsrnamespace": 6,
            "gsrlimit": min(limit, 50),
        },
        limit,
    )
