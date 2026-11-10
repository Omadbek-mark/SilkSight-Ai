# SilkSight

Recognize famous historical and natural places of Central Asia (Uzbekistan, Kazakhstan,
Kyrgyzstan, Tajikistan, Turkmenistan) from a photo, then show facts, a map and — where the city is
covered — a link to hotels on [Hotelia](https://hotelia.uz).

Status: **both models and the FastAPI service are ready.** The Next.js web UI comes next. Local
only.

## Setup

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

## Catalog

`catalog/landmarks.seed.json` lists 23 landmarks with their Wikidata IDs. Build the fact file
(network request to Wikidata):

```sh
python -m scripts.build_catalog
```

This writes `catalog/landmarks.json` with names (en/uz/ru), coordinates, Wikipedia links and the
World Heritage Site the place belongs to when Wikidata links it (`unesco_site_qid`; `null` means
"not linked in Wikidata", not "not listed"). Facts come from Wikidata (CC0); nothing is generated
by a language model. `name_uz` in the seed is a curated display name where the Wikidata label is
missing or too long.

## Dataset

```sh
python -m scripts.build_dataset    # network: Wikimedia Commons, ~30 min, resumable
python -m scripts.contact_sheets   # numbered review grids in outputs/contact_sheets/
```

- Up to 40 photos per landmark from its Commons category (then subcategories, then a file search
  for the few places with little coverage), picked round-robin across photographers.
- Only free licenses with a named author (CC0, public domain, CC BY, CC BY-SA); NC/ND rejected.
- Images are re-encoded as JPEG without EXIF and stored in `data/images/` (not in Git).
  `data/manifest.csv` records title, author, license and split for every image.
- Splits are by photographer: one author's photos are all in train, val or test. Test gets at
  least 5 and val 3 images per landmark where enough distinct authors exist.
- `unknown` images (similar mosques in Iran/Afghanistan/Turkey/India, other towers, canyons,
  lakes, streets) are only in val/test — they evaluate the "not a known place" answer.
- `data/exclusions.csv` lists manually rejected images (wrong place, map, artwork, interior
  close-up…). After a review, run `--reuse`: it applies exclusions and splits offline, without
  refilling the removed slots with new unreviewed images.

Current dataset (built and reviewed 2026-11-09): 23 landmarks, 650 landmark images
(train 470 / val 69 / test 111) and 76 unknown images (val 40 / test 36) from 10 sources.
Classes are uneven: Tashkent TV tower and Iskanderkul have 11 images, Sulaiman-Too, Ark and
Sultan Sanjar 14–15, while most have 26–40. Several test sets have only 2–4 images, so per-class
test numbers are noisy. Dropped for too few usable Commons photos after review: Hisor fortress
(search returned unrelated places), Tash Rabat (4), Bayterek (7, mostly views from inside).

## Model A: frozen embeddings + kNN

```sh
python -m scripts.embed_dataset   # first run downloads weights into .models/ (~1 GB)
python -m scripts.evaluate_knn    # offline; writes artifacts/model_a/report.json, index.npz
```

- A frozen backbone turns each photo into a unit vector: DINOv2-base (CLS token) or CLIP
  ViT-B/16 (image tower). No training; the train split is the search gallery.
- A landmark's score is the mean cosine similarity to its `top_m` closest gallery photos. If the
  best score is under a threshold, the answer is "unknown".
- `top_m` (1/3/5), the threshold and the backbone are chosen on **val** only (best mean of "known
  photo accepted and right" and "unknown photo rejected"). Test is scored once afterwards.

Results (run 2026-11-10, Mac M-series MPS; test = 111 landmark photos + 36 unknown):

| Backbone | top_m | Threshold | Top-1 | Top-3 | Known accepted & right | Known wrongly rejected | Unknown rejected | Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **DINOv2-base** (selected) | 1 | 0.711 | **0.802** | **0.937** | **0.667** | 0.234 | 0.694 | **0.683** |
| CLIP ViT-B/16 | 3 | 0.877 | 0.703 | 0.901 | 0.468 | 0.369 | 0.889 | 0.535 |

Top-1/Top-3 ignore the threshold (closed set). The other columns apply it. Embedding all 726
images took 15 s (DINOv2) and 12 s (CLIP) on MPS; a rerun gave identical numbers.

What the numbers do and do not say:

- The test set is small (2–9 photos per landmark), so per-landmark numbers are noisy. These
  Commons photos are not phone snapshots, so this is not a real-world accuracy figure. The
  threshold is a cosine-similarity cut, not a probability.
- The main error is Shah-i-Zinda → Gur-e Amir (7 of 9). Shah-i-Zinda's test photos are mostly
  one photographer's interior dome and tile shots, and these look like Gur-e Amir's interior.
  The data was not changed to fix this, because that would tune on the test set.
- Unknown photos the selected model accepted: Kalta Minor, Shah-i-Zinda and Charyn Canyon
  look-alikes (2 each). The threshold trades these against wrongly rejected known photos (23%).
- CLIP rejects unknowns better (0.889) but rejects far more real landmarks too. DINOv2 is
  stronger at telling similar Timurid buildings apart.

## Model B: fine-tuned ConvNeXt-Tiny

```sh
python -m scripts.train_classifier   # first run downloads ImageNet weights (~110 MB)
```

- ConvNeXt-Tiny from ImageNet-1k weights, all layers trained: 15 epochs, AdamW, backbone LR
  1e-4 and head LR 1e-3, cosine schedule, label smoothing 0.1, inverse-frequency class weights,
  mild crop/flip/colour augmentation, seed 7.
- Only the 23 landmarks are classes. "Unknown" comes from a threshold on the top softmax
  probability, tuned on val as for Model A. The checkpoint with the best val top-1 is kept
  (epoch 3; val top-1 stayed at 0.80–0.81 afterwards).
- Training took 5 min on MPS. Output goes to `artifacts/model_b/` (model.pt, report.json).

## Comparison on the same test set

Run 2026-11-10; test = 111 landmark photos + 36 unknown; one training run (one seed).

| Model | Threshold | Top-1 | Top-3 | Known accepted & right | Known wrongly rejected | Unknown rejected | Macro-F1 |
|---|---|---|---|---|---|---|---|
| A: DINOv2-base + kNN | 0.711 (cosine) | 0.802 | 0.937 | 0.667 | 0.234 | 0.694 | 0.683 |
| B: ConvNeXt-Tiny fine-tune | 0.409 (softmax) | 0.802 | 0.955 | 0.658 | 0.279 | 0.722 | 0.721 |

- **No clear winner on this data.** Both give the right final answer (landmark or "unknown")
  for 99 of 147 test photos, but not the same ones: 23 are right only with A and 23 only with
  B. Differences of 1–3 photos are within the noise of a test set this small.
- They fail differently on Shah-i-Zinda's interior shots. A names them Gur-e Amir (7 of 9).
  B mostly answers "unknown" (6 of 9). For a travel app, "unknown" is the safer mistake.
- Trade-offs: A needs no training, and a new landmark is added by putting its photos in the
  gallery. B needs retraining for new classes but is a single forward pass with no gallery.
- Not tried: averaging the two models' answers. They disagree on many photos, so it might help,
  but the combination would also have to be tuned on val only.

## API (FastAPI)

```sh
uvicorn silksight.api:create_app --factory --host 127.0.0.1 --port 8000
# docs: http://127.0.0.1:8000/docs
```

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | status, landmark count, models loaded so far |
| GET | `/api/landmarks` | all landmarks: names (en/uz), country, city, coordinates, photo |
| GET | `/api/landmarks/{id}` | plus Wikidata names/descriptions, Wikipedia links, UNESCO site, Hotelia link |
| POST | `/api/recognize?model=knn\|classifier` | `recognized`, `landmark`, top-3 `candidates` with scores, `threshold`, `score_kind` |
| GET | `/images/...` | dataset photos; each `photo` object carries author, license and source page |

- Upload: one JPEG/PNG/WebP, at most 10 MB and 40 MP. The format and size are checked from the
  header before decoding, and EXIF orientation is applied. Errors are 422 (bad file or model
  name) and 413 (too large). The photo stays in memory: it is not saved, its metadata is not
  kept, and it is not sent anywhere.
- `model` defaults to `knn` (Model A). The two models tied on test; kNN can take new landmarks
  without retraining. Both models load on first use. One lock serializes model loading and
  inference in the single local process.
- Below the threshold the answer is `recognized: false`, but the top-3 candidates are still
  returned, so the UI can say "not sure — maybe…". Scores are model scores (`score_kind`), not
  accuracy.
- `hotelia_url` is `null` when the landmark's city is not a Hotelia destination.

Live check (2026-11-10, MPS): the first kNN request took 4.4 s (loading DINOv2) and later
requests about 0.04 s. The first classifier request took 1.0 s. A Registan test photo was right
with the classifier but named Gur-e Amir by kNN. A Taj Mahal photo was "unknown" for both.
Path traversal under `/images` (`..` and `%2e%2e`, sent unnormalized) returns 404. CORS allows
only the local Next.js origin (`localhost:3000`).

Known gap: Wikidata descriptions are short and sometimes wrong. The Uzbek description of
Registan is literally "best". This is not enough for the "detailed info" goal yet.

## Plan

1. Catalog
2. Image dataset (done) from Wikimedia Commons with a license manifest, author-separated test split and
   an "unknown places" set
3. Model A: DINOv2/CLIP embeddings + kNN with an "unknown" threshold (done)
4. Model B: fine-tuned classifier, compared on the same test set (done)
5. FastAPI service (done)
6. Next.js web app (EN/UZ): upload → result card → map → Hotelia link
