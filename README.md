# AgriSmart AI

**SIH 2026 internal hackathon · L. J. Institute of Engineering and Technology · Problem Statement 1: AgriSmart AI**

A crop-disease detector for farmers: show it a leaf and it names the disease, or says the leaf looks healthy.
It learns from clean lab photos (PlantVillage) but is built to work on real field photos, where plain lab-trained
models fall apart.

| Module | Status |
|---|---|
| **Core: crop-disease detection** | Trained and tested; `predict.py` interface ready |
| **Add-on: live camera scan** | Point the laptop camera at a leaf, or at a phone photo of one; the page shows and says the disease |
| **Extension: India multi-crop model** | 59 crops, 387 classes, trained and released; see [`model/india/README.md`](model/india/README.md) |
| **Offline app: Android APK + Windows `run.bat`** | No internet or server: photo in, disease, cause and what to do out, spoken in 13 languages; three models to pick from. See [`offline/README.md`](offline/README.md) |
| **Android app** | Native Kotlin client scaffolded from [`docs/android_app_prompt.md`](docs/android_app_prompt.md); see `DECISIONS.md` |
| Bonus C: weather intelligence | Planned |
| Bonus D: sustainability score | Planned |
| Bonus E: farmer assistant (Gujarati / Hindi) | Planned |

## Results (model v2)

| Test set | Images | Macro-F1 | Accuracy |
|---|---|---|---|
| PlantVillage lab photos of leaves not seen in training | 7,520 | 0.995 | 0.995 |
| **PlantDoc field photos, never trained on** | 2,806 | **0.714** | **0.743** |

For comparison, a PlantVillage-only ResNet-50 scores 0.285 macro-F1 on PlantDoc
([PMC13236948](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)). The organizers' hidden field set is the real test;
its baseline and the final class list are published at kickoff.

![Field confusion matrix](report/figures/confusion_field.png)

Full details: [report/model_report.md](report/model_report.md) · every run: [report/experiments.md](report/experiments.md).

## Quick start (about 5 minutes)

Needs Python 3.11–3.13. Runs on CPU; no GPU needed.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python model/predict.py --image samples/tomato_late_blight_1.jpg
```

It prints `Tomato___Late_blight`. The first run downloads the weights (87 MB) from the
[`model-v2` release](https://github.com/wpzvqrs8/SIH_2026/releases/tag/model-v2) and checks their SHA-256.
Add `--top 3` to see the three likeliest classes with probabilities.

On Linux or macOS, activate with `source .venv/bin/activate`. On Linux, install the CPU build of PyTorch first to skip
a 2 GB CUDA download: `pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu`.

From Python:

```python
import sys
sys.path.insert(0, "model")
from predict import predict

predict("samples/potato_early_blight_1.jpg")   # -> "Potato___Early_blight"
```

### Live camera scan

```bash
python server/live_camera/live_camera.py
```

A browser tab opens. Press **Start camera**, allow camera access, and hold a leaf (or a photo of one on your phone)
inside the square. About three scans a second run on the laptop's CPU; once most recent scans agree, the page shows
the disease and says it out loud. When it isn't sure, it asks you to move closer instead of guessing. Everything stays
on the laptop, works offline, and by default listens on Wi-Fi too (`--host 127.0.0.1` to keep it laptop-only), so the
Android app (see [`docs/android_app_prompt.md`](docs/android_app_prompt.md)) can reach it from a phone on the same
network by pointing `API_BASE_URL` at the laptop's Wi-Fi address.

### AgriSmart India: 59 crops, 13 languages

```bash
python server/live_camera_india/live_camera_india.py
```

The same idea, generalised to every crop in [`model/india/README.md`](model/india/README.md). You can choose a crop or
let the model guess. The page asks which crop it is when two look-alike plants are close. It shows what to do for the
disease, and says the result in 12 Indian languages plus English.

### Offline app (Android APK and Windows `run.bat`)

The same checker with no server and no internet. The model runs inside the app. How it works:

- Take one photo (auto capture when the picture is steady) or upload one.
- The photo freezes, then gets checked, so the phone doesn't have to stay pointed at the leaf.
- The app shows the disease, **why it happens** and **what to do**, and reads all of it out in 13 languages.
  The voice keeps going when you move the phone away.
- You can switch between the India v1, India v2 and core models.
- If the photo doesn't look like the crop you picked, it asks "Is this sugarcane? Yes / No" rather than guessing.
- There is an optional **online check** (your own free keys). Pl@ntNet's picture search and a vision AI confirm or
  correct the offline answer, and they also name plants the offline model doesn't know, such as neem or tulsi.

Downloads and details: [`offline/README.md`](offline/README.md).

## How it works

1. **Dataset** ([model/training/01_data_prep](model/training/01_data_prep)): PlantVillage lab photos with the official
   leaf-grouped split (no leaf appears in both training and validation), plus real field photos from PlantWild v2
   (diseases) and PlantWild v1 (healthy leaves). PlantDoc is kept for validation only. Any training photo that looks
   like a PlantDoc photo is removed (perceptual hash + DINOv2 similarity): 660 were caught.
2. **Model** ([model/training/02_train](model/training/02_train)): DINOv2 ViT-S/14 from timm at 224 px, trained in two
   stages (head only, then everything at a low learning rate). Half the lab photos get their background swapped for
   field scenery; augmentation stays natural (crops, flips, rotation, colour jitter, blur); sampling is class-balanced
   with field photos drawn three times as often. The checkpoint is chosen by field macro-F1, never lab accuracy.
3. **Inference** ([model/predict.py](model/predict.py)): about 0.2 s per photo on a laptop CPU after a one-time
   load of a few seconds.

Training runs on Kaggle's free T4 GPUs; see [model/README.md](model/README.md) to reproduce it.

## Repository layout

```
app/        Android client (Kotlin)
server/     live camera scan: local web page + server (core and India models)
offline/    offline app: web page with the models inside, Windows run.bat, Android APK project, build tools
model/      predict.py interface, labels, weight download info, Kaggle training notebooks
report/     one-page model report, experiment log, results, figures, dataset card
samples/    4 PlantDoc field photos for a quick test (CC BY 4.0)
docs/       team guide: fine-tuning basics, dataset choices, plan
```

## Datasets and licences

| Dataset | Used for | Licence |
|---|---|---|
| PlantVillage (Mohanty et al., 2016) via Kaggle `abdallahalidev/plantvillage-dataset`; leaf map and split from Hugging Face `mohanty/PlantVillage` | Training + lab validation (the core dataset) | CC BY-NC-SA 4.0 (Kaggle copy); CC BY-SA 3.0 (Hugging Face copy) |
| PlantDoc (Singh et al., 2020, [doi:10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196)) via Kaggle `nirmalsankalana/plantdoc-dataset` | Field validation only; 4 sample photos in `samples/` | CC BY 4.0 |
| PlantWild v1 / v2 (Wei et al., ACM MM 2024, [arXiv:2408.03120](https://arxiv.org/abs/2408.03120)) via Hugging Face `uqtwei2/PlantWild` | Extra field training photos | CC BY-NC-ND 4.0; images are not redistributed here |
| PlantSeg (Wei et al., Scientific Data 2025) via Kaggle `weitianqi/plantseg` / Zenodo 14935094 | Checked; every usable photo duplicated PlantWild, so none were kept | CC BY 4.0 |

No training images are stored in this repository. The dataset build is described in
[report/dataset/README.md](report/dataset/README.md).

## Known limitations

- **Draft class list.** The 28 classes are the PlantVillage classes that also exist in PlantDoc. The organizers'
  official list (about 15–20 classes) replaces it at kickoff.
- **Hardest classes on field photos:** tomato bacterial spot (F1 0.25), tomato mosaic virus (0.49), tomato and potato
  early blight (0.51). Small-spot tomato diseases and potato early vs late blight are often confused.
- **Optimistic field score.** PlantDoc was used to choose the checkpoint. The last epoch won, so the effect is small,
  but the hidden set is the real test.
- **Label noise in PlantDoc**, e.g. a raspberry leaf labelled as soybean; some "errors" are the dataset's own mistakes.
- **No "not a leaf" answer.** The model always picks one of its classes. The live scan shows "Can't tell yet" when
  confidence is low, but a confident wrong answer on a non-leaf is still possible.
- **Photos of a phone screen** add glare and moiré; hold the phone steady with its brightness up.

## Demo video

Coming soon (3–5 minutes).

## Originality declaration

All code in this repository was written by our team during the hackathon window (10–15 September 2026), with help
from an AI coding assistant (Claude Code), which the rules allow. No public notebook or solution was copied.
We reuse these open-source libraries and pretrained weights: PyTorch, torchvision, timm, DINOv2 (Meta AI, Apache-2.0),
scikit-learn, pandas, imagehash and the Hugging Face Hub client. Datasets are credited above.

## Team

_Add team members here._
