# AgriSmart India: multi-crop model

A second, much larger model covering **59 crops grown in India, with 387 disease/condition classes**.
It is built alongside the hackathon's core model but kept fully separate, so it can never affect the
core score.

- **Dataset build:** [`model/training/india/01_data_prep`](../training/india/01_data_prep) pulls **80
  sources**:
  - PlantVillage (lab photos)
  - PlantWild v1 + v2 (field photos, about 25 crops)
  - 64 Hugging Face `Project-AgML` mirrors of published research datasets
  - 14 Kaggle sets

  Many of the sources were photographed in Indian fields: Tomato-Village, Indian onion, Assam tea
  (TeaLeafNet), okra yellow vein mosaic, Maharashtra soybean, Kashmir apple, Karnataka coconut,
  pomegranate, rice and black gram. A **universal leakage guard** then runs over every photo:
  - Exact copies are merged, and copies that carry conflicting labels are dropped.
  - Near-copies are clustered, and each cluster is kept whole on one side of the 85/15 split.
  - Both checks use a perceptual hash plus a flip-invariant DINOv2 embedding.
- **Training:** [`model/training/india/02_train`](../training/india/02_train) trains one flat DINOv2
  ViT-B/14 classifier over every crop's classes, the same two-stage way as the core model (linear probe,
  then fine-tune). Target: **macro-F1 ≥ 0.95**, counted on the *clean* test set: the test photos with
  no training photo at DINOv2 cosine ≥ 0.95, as re-checked by the training notebook's leakage audit.
  Results are also reported for the whole test set, with the crop given, and per photo type
  (lab / field / mixed).

## Dataset (build of 12 Sep 2026)

| | train | test |
|---|---:|---:|
| **All photos** | 238,028 | 28,328 |
| Lab (PlantVillage) | 35,474 | 4,853 |
| Field (PlantWild) | 17,317 | 2,983 |
| Mixed (research datasets, mostly field) | 185,237 | 20,492 |

How the build got there:
- 351,802 photos were loaded.
- 65,473 exact copies were merged. Many sources ship pre-augmented duplicates.
- 3,008 copies carrying conflicting labels were dropped.
- Each class was capped at 3,000 photos, shared fairly between sources.
- 4 classes with fewer than 30 photos were dropped.

The build takes 65 minutes on a Kaggle T4. Photos are stored at 384 px, 6.7 GB in total.

Licences vary. Most sources are CC BY 4.0 or CC0; some are non-commercial (PlantVillage, PlantWild,
the Ripon cotton set, one tea set, MoringaLeafNet and the mango set). This model is for
non-commercial research and education, and the photos themselves are not redistributed.
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

## Results: model v1 (12 Sep 2026)

This is DINOv2 ViT-B/14 at 224 px, trained with 2 linear-probe epochs and then 14 fine-tune epochs
of 80k class-balanced draws each. The run took 4.8 h on a Kaggle T4. The full scores are in
[`metrics.json`](metrics.json), [`report_per_crop.csv`](report_per_crop.csv) and
[`report_per_class.csv`](report_per_class.csv).

| Test set | photos | macro-F1 | accuracy |
|---|---:|---:|---:|
| **Clean test (headline)** | 27,900 | **0.906** | 0.929 |
| Clean test, crop chosen in the app | 27,900 | **0.926** | 0.942 |
| Lab photos (PlantVillage) | 4,853 | 0.997 | 0.996 |
| Research-dataset photos, clean part | 20,144 | 0.926 | 0.940 |
| Field photos (PlantWild) | 2,983 | 0.744 (0.857 with the crop chosen) | 0.747 |

- **Target:** not met yet. The target is macro-F1 ≥ 0.95 on the clean test. A v2 run is training:
  336 px, layer-wise learning-rate decay and weight EMA, continued from v1.
- **Leakage audit:** 428 of the 28,328 test photos (1.5%) have a training photo at DINOv2 cosine
  ≥ 0.95. Those photos are left out of the clean number.
- **Weakest crops by macro-F1:** ginger 0.61, ash gourd 0.69, okra 0.75, spinach 0.76,
  cauliflower 0.78. Most of these are field-photo classes with few examples. The model tells field
  photos apart much less reliably than lab photos. Choosing the crop in the app helps most exactly
  there.

**Weights.** They are stored in GitHub Release `india-model-v1` (344 MB). See
[`weights.json`](weights.json) for the URL and SHA-256. To switch the live-scan app from mock mode to
the real model, download them into this folder:

```bash
gh release download india-model-v1 -R wpzvqrs8/SIH_2026 -p model.pt -D model/india
```

## Translations

`server/live_camera_india/india_translations.json` covers every crop and class above. It is a first
draft, not reviewed by native speakers. Crop names and
interface text are hand-translated with normal confidence. Disease/condition names are fully translated
for Hindi, Bengali, Marathi, Tamil, Telugu and Gujarati; for the other six languages (Urdu, Kannada,
Odia, Malayalam, Punjabi, Assamese) they are phonetic renderings of the English term in the local
script — a defensible starting point, not vernacular coinages. Treat both as a draft for a native-speaker
review, the same standard set for the Android app spec (`docs/android_app_prompt.md`, section 8.5).
