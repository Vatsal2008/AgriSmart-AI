# AgriSmart India: multi-crop model

A second, much larger model covering **37 crops widely grown in India** (179 draft classes), built
alongside the hackathon's core model but kept fully separate so it can never affect the core score.

- **Dataset build:** [`model/training/india/01_data_prep`](../training/india/01_data_prep) — pulls
  PlantVillage plus about 25 other sources (mostly clean, per-class-labelled Hugging Face mirrors of
  published research datasets), and applies a **universal leakage guard**: every crop+class is clustered
  for near-duplicates (perceptual hash + DINOv2 embedding) before an 85/15 split, so an augmented copy of
  a training photo can never land in the test set. See its output's `README.md` and `dedup_report.json`
  once it has run.
- **Training:** [`model/training/india/02_train`](../training/india/02_train) — one flat DINOv2 ViT-B/14
  classifier over every crop's classes, trained the same two-stage way as the core model (linear probe,
  then fine-tune). Target: **macro-F1 ≥ 0.95** on the held-out split.
- **Crop-conditioned inference:** at prediction time, giving a crop name restricts the answer to that
  crop's own classes — no retraining needed for that; see `model/india/predict.py` (written by the
  training notebook) and its `crop_classes` mapping in `model.pt`.
- **App:** [`server/live_camera_india/live_camera_india.py`](../../server/live_camera_india/live_camera_india.py) —
  a live camera / photo-upload page with a crop picker, a full statistics panel of every class considered,
  and spoken results in 12 Indian languages plus English (see
  [`server/live_camera_india/india_translations.json`](../../server/live_camera_india/india_translations.json)).
  Until `model/india/model.pt` exists, it runs in a clearly labelled **mock mode** over the same taxonomy,
  so the whole app can be built and tested before training finishes.

Weights are not stored in git; see `model/india/weights.json` (added once the model is trained) for the
GitHub Release download, the same pattern as `model/weights.json` for the core model.

## Translations

`app/india_translations.json` is a first draft, not reviewed by native speakers. Crop names and
interface text are hand-translated with normal confidence. Disease/condition names are fully translated
for Hindi, Bengali, Marathi, Tamil, Telugu and Gujarati; for the other six languages (Urdu, Kannada,
Odia, Malayalam, Punjabi, Assamese) they are phonetic renderings of the English term in the local
script — a defensible starting point, not vernacular coinages. Treat both as a draft for a native-speaker
review, the same standard set for the Android app spec (`docs/android_app_prompt.md`, section 8.5).
