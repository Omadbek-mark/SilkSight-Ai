import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from silksight.api import create_app
from silksight.recognizer import Recognition, decide
from silksight.uploads import MAX_UPLOAD_BYTES


class FakeRecognizers:
    """Stands in for the real models: no weights, no device."""

    loaded: list[str] = []

    def __init__(self, recognized: bool = True) -> None:
        self.recognized = recognized
        self.calls: list[tuple[tuple[int, int], str]] = []

    def recognize(self, image: Image.Image, model: str) -> Recognition:
        self.calls.append((image.size, model))
        top = 0.9 if self.recognized else 0.3
        scores = np.array([top, 0.2, 0.1])
        return decide(model, "test score", ["registan", "gur-e-amir", "chor-minor"], scores, 0.5)


def jpeg_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "blue").save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def fake() -> FakeRecognizers:
    return FakeRecognizers()


@pytest.fixture
def client(fake: FakeRecognizers) -> TestClient:
    return TestClient(create_app(fake))


def test_landmark_detail_has_facts_and_hotelia_link_only_for_hotelia_cities(client):
    registan = client.get("/api/landmarks/registan").json()
    charyn = client.get("/api/landmarks/charyn-canyon").json()

    assert registan["hotelia_url"] == "https://hotelia.uz/hotel?destination=Samarkand"
    assert registan["latitude"] and registan["wikidata_url"].endswith("Q1373583")
    assert charyn["hotelia_url"] is None


def test_unknown_landmark_is_404(client):
    assert client.get("/api/landmarks/eiffel-tower").status_code == 404


def test_recognize_returns_landmark_and_named_candidates(client, fake):
    response = client.post(
        "/api/recognize?model=classifier", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["recognized"] is True
    assert body["landmark"]["id"] == "registan"
    assert [c["landmark_id"] for c in body["candidates"]] == [
        "registan",
        "gur-e-amir",
        "chor-minor",
    ]
    assert body["candidates"][0]["name"]["en"] == "Registan"
    assert fake.calls == [((64, 48), "classifier")]


def test_low_score_answers_unknown_but_keeps_candidates():
    client = TestClient(create_app(FakeRecognizers(recognized=False)))

    body = client.post("/api/recognize", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})

    assert body.json()["recognized"] is False
    assert body.json()["landmark"] is None
    assert len(body.json()["candidates"]) == 3


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        (b"not an image", 422),
        (b"", 422),
        (b"\xff" * (MAX_UPLOAD_BYTES + 10), 413),
    ],
)
def test_bad_uploads_are_rejected_before_any_model_runs(client, fake, payload, status):
    response = client.post("/api/recognize", files={"file": ("a.jpg", payload, "image/jpeg")})

    assert response.status_code == status
    assert fake.calls == []


def test_unsupported_format_and_unknown_model_are_rejected(client, fake):
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, "GIF")

    assert (
        client.post("/api/recognize", files={"file": ("a.gif", gif.getvalue())}).status_code == 422
    )
    assert (
        client.post(
            "/api/recognize?model=resnet", files={"file": ("a.jpg", jpeg_bytes())}
        ).status_code
        == 422
    )
    assert fake.calls == []


def test_exif_orientation_is_applied_before_recognition(client, fake):
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90°: a 64x48 sensor image is shown as 48x64
    Image.new("RGB", (64, 48)).save(buffer, "JPEG", exif=exif)

    client.post("/api/recognize", files={"file": ("a.jpg", buffer.getvalue(), "image/jpeg")})

    assert fake.calls == [((48, 64), "knn")]


def test_landmark_detail_carries_attributed_wikipedia_text_without_translation(client):
    registan = client.get("/api/landmarks/registan").json()["summary"]
    almaty_lake = client.get("/api/landmarks/big-almaty-lake").json()["summary"]

    assert registan["uz"]["text"].startswith("Registon")
    assert registan["en"]["license"] == "CC BY-SA 4.0"
    assert registan["en"]["url"].startswith("https://en.wikipedia.org/")
    assert registan["en"]["revision_id"] > 0
    assert "uz" not in almaty_lake  # no Uzbek article: left out, not translated
