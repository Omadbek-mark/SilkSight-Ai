"""SilkSight HTTP API (local). Run with:

    uvicorn silksight.api:create_app --factory --port 8000

Uploaded photos are decoded in memory, never stored and never sent to another service.
"""

from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from silksight.catalog import hotelia_url, load_catalog, load_summaries
from silksight.dataset import ManifestRow, read_manifest, representative_photos
from silksight.recognizer import MODEL_NAMES, ModelUnavailableError, Recognizers
from silksight.settings import (
    APP_NAME,
    CATALOG_PATH,
    IMAGES_DIR,
    MANIFEST_PATH,
    SUMMARIES_PATH,
)
from silksight.uploads import MAX_UPLOAD_BYTES, UploadError, read_upload

DEFAULT_MODEL = "knn"
WEB_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _photo(row: ManifestRow | None) -> dict | None:
    if row is None:
        return None
    return {
        # manifest paths are "images/<label>/<hash>.jpg"; /images serves data/images
        "url": "/" + row.file,
        "title": row.title,
        "author": row.author,
        "license": row.license,
        "license_url": row.license_url,
        "source_url": row.page_url,
    }


def create_app(recognizers: Recognizers | None = None) -> FastAPI:
    catalog = load_catalog(CATALOG_PATH)
    summaries = load_summaries(SUMMARIES_PATH)
    photos = representative_photos(read_manifest(MANIFEST_PATH))
    models = recognizers or Recognizers()

    app = FastAPI(title=f"{APP_NAME} API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=WEB_ORIGINS, allow_methods=["GET", "POST"])
    if IMAGES_DIR.exists():
        app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

    def summary(landmark_id: str) -> dict:
        item = catalog[landmark_id]
        return {
            "id": landmark_id,
            "name": item["display_names"],
            "country": item["country"],
            "city": item["city"],
            "category": item["category"],
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "photo": _photo(photos.get(landmark_id)),
        }

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "landmarks": len(catalog), "models_loaded": models.loaded}

    @app.get("/api/landmarks")
    def list_landmarks() -> list[dict]:
        return [summary(landmark_id) for landmark_id in catalog]

    @app.get("/api/landmarks/{landmark_id}")
    def get_landmark(landmark_id: str) -> dict:
        if landmark_id not in catalog:
            raise HTTPException(404, "landmark not found")
        item = catalog[landmark_id]
        return summary(landmark_id) | {
            "names": item["names"],
            "descriptions": item["descriptions"],
            "unesco_site_qid": item["unesco_site_qid"],
            "wikipedia": item["wikipedia"],
            # Wikipedia lead text per language (CC BY-SA, with title/URL/revision for credit);
            # a missing language is absent, not machine-translated.
            "summary": summaries.get(landmark_id, {}),
            "wikidata_url": item["wikidata_url"],
            "hotelia_url": hotelia_url(item["hotelia_destination"]),
            "facts_source": "Wikidata (CC0)",
        }

    @app.post("/api/recognize")
    async def recognize(
        file: UploadFile = File(...),
        model: str = Query(DEFAULT_MODEL, pattern="^(" + "|".join(MODEL_NAMES) + ")$"),
    ) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            image = read_upload(data)
        except UploadError as error:
            raise HTTPException(413 if error.too_large else 422, str(error)) from error
        try:
            result = await run_in_threadpool(models.recognize, image, model)
        except ModelUnavailableError as error:
            raise HTTPException(503, f"model {model!r} is not trained: {error}") from error
        response = asdict(result)
        response["landmark"] = summary(result.landmark_id) if result.landmark_id else None
        for candidate in response["candidates"]:
            candidate["name"] = catalog[candidate["landmark_id"]]["display_names"]
        return response

    return app
