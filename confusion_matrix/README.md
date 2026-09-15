# Confusion matrices

Every model's confusion matrices, one folder per model. In each matrix:
- a row is the true class, and each row sums to 100%;
- the dark diagonal is right answers, and any colour off it is a mix-up;
- the labelled cells are mix-ups of 5% or more, and classes recognised less than 90% of the time.

| Folder | Model | Test photos | Files |
|---|---|---|---|
| [`india_v2/`](india_v2) | **India v2** (latest, 59 crops, 387 classes) | 27,900 clean test photos: right crop 98.3%, exact class 92.7%; field photos: right crop 86.6% | [crops](india_v2/india_v2_confusion_crops.png) · [field photos](india_v2/india_v2_confusion_crops_field.png) · [all 387 classes](india_v2/india_v2_confusion_full.png) · [crop counts](india_v2/india_v2_confusion_crops.csv) · [top 30 mix-ups](india_v2/india_v2_top_confusions.csv) |
| [`india_v1/`](india_v1) | **India v1** (default in the apps) | 27,900 clean test photos: right crop 98.3%, exact class 92.9%; field photos: right crop 86.5% | [crops](india_v1/india_v1_confusion_crops.png) · [field photos](india_v1/india_v1_confusion_crops_field.png) · [all 387 classes](india_v1/india_v1_confusion_full.png) · [crop counts](india_v1/india_v1_confusion_crops.csv) · [top 30 mix-ups](india_v1/india_v1_top_confusions.csv) |
| [`core_v2/`](core_v2) | **Core model v2** (13 crops, 28 classes) | 2,806 PlantDoc field photos (macro-F1 0.714); 7,520 PlantVillage lab photos (0.995) | [field photos](core_v2/confusion_field.png) · [lab photos](core_v2/confusion_lab.png) |

![India v2: which crop does the model see?](india_v2/india_v2_confusion_crops.png)

## Redrawing them

- **India models:**
  - `python model/training/india/03_eval/india_confusion.py --model v2` redraws India v2.
  - `--model v1` redraws India v1.
  - The script reads each run's saved test predictions (`model/india/test_predictions_v2.csv` and
    `model/india/test_predictions.csv`) and writes into this folder.
  - "Clean" means test photos with a near-identical copy in the training set are left out (DINOv2 cosine
    ≥ 0.95).
- **Core model:** its matrices come from its own training run
  ([`model/training/02_train`](../model/training/02_train)).

The models themselves are in [`../models`](../models).
