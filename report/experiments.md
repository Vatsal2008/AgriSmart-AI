# AgriSmart experiment log

One row per training run. **Field F1** is macro-F1 on PlantDoc field photos (2,806 images, never trained on), our stand-in
for the organizers' hidden field test. **Lab F1** is macro-F1 on PlantVillage photos of leaves not in training.
Checkpoints are chosen by field F1, so field numbers are slightly optimistic; the hidden set is the real test.

## Runs

| Run | Date | Classes | Backbone | Training data | Augmentation | Epochs (probe + fine-tune) | Lab F1 | Field F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| smoke-test v1 | 10 Sep | 4 | ConvNeXt-T | PlantVillage only, random split | timm default | 0 + 1 | 0.992 | 0.390 | Pipeline check only; shows the lab-to-field gap |
| agrismart-train v1 (`e3_dinov2s_bg_field`) | 10 Sep | 28 | DINOv2 ViT-S/14 @ 224 | PlantVillage 31,022 + field 4,235 (PlantWild v1/v2) | timm RandAugment m9 inc1, background replacement 0.5, blur 0.2 | 2 + 5 of 6 | 0.992 | 0.698 (acc 0.719, no TTA) | Cancelled on Kaggle during fine-tune epoch 6, before testing. Field F1 still rising. RandAugment inverted/solarised colours |
| agrismart-train v2 (`e3b_dinov2s_natural_aug`) | 10 Sep | 28 | DINOv2 ViT-S/14 @ 224 | same as v1 | natural only (crop, flip, rotate, perspective, colour jitter, blur, erasing), background replacement 0.5 | 1 + 8 | 0.995 | **0.714** (acc 0.743, with flip TTA; 0.720 without) | Best checkpoint = last epoch, so little selection bias. 43.6 min on T4. `predict.py` self-test passed (fresh process; 3/6 random field photos correct) |

Reference point: a PlantVillage-only ResNet-50 scores 0.285 macro-F1 on PlantDoc (21 classes; PMC13236948).

## Findings so far (after v2)

- **Training longer helped most.** Field F1 was still rising at fine-tune epochs 6–8 (0.720, 0.714, 0.720).
- **The augmentation change is inconclusive.** At fine-tune epoch 5, v1 (RandAugment) scored 0.698 and v2 (natural augmentations) 0.689; v1 was cancelled before it could be compared further.
- **Flip TTA did not help:** 0.720 without it, 0.714 with it.
- **Weakest field classes (F1):** tomato bacterial spot 0.25, tomato mosaic virus 0.49, tomato early blight 0.51, potato early blight 0.51, corn gray leaf spot 0.61, tomato septoria 0.61, tomato healthy 0.61.
- **Main confusions:** tomato bacterial spot ↔ septoria (30 and 33 photos); potato early → late blight (48 of 155); corn northern leaf blight → gray leaf spot (53 of 180); tomato yellow leaf curl → mosaic (35) and → healthy (25); peach healthy → apple healthy (19).
- **PlantDoc label noise shows up among the "errors":** e.g. a photo labelled Soybean healthy is a five-leaflet raspberry leaf (the model said raspberry); watermarked stock photos and collage figures also appear. Worth stating in the report's limitations.

## Local inference check (11 Sep, laptop CPU)

- Machine: i7-13700H, no GPU; PyTorch 2.14.0 (CPU), timm 1.0.26, in `agrismart-ai\.venv`.
- v2 `model.pt` (86.6 MB): loads in 7.1 s; about 0.15–0.2 s per photo after the first (0.8 s warm-up), with flip TTA.
- `python predict.py --image <photo>` in a fresh process: 8.6 s including startup (rules allow about 10 minutes to reproduce).
- Sanity check on 4 PlantDoc test photos (tomato late blight, potato early blight, corn common rust, tomato healthy): 4/4 correct. Not a score: these are from the validation set.

## Ideas to try next (one change per run)

1. Higher resolution (336 px): the weak classes are small-spot diseases.
2. Bigger backbone (DINOv2 ViT-B/14).
3. More field photos for weak classes (PlantWild v1 disease folders, de-duplicated against v2 and PlantDoc).
4. Longer fine-tuning (10–12 epochs).

## Dataset v1 (`agrismart-data-prep` v1, built 10 Sep 2026 14:35 UTC)

- 28 classes (draft list: PlantVillage classes that also exist in PlantDoc).
- train 35,257 = PlantVillage 31,022 (official leaf-grouped split, 52 photos moved to keep leaves together) + PlantWild v2 1,975 (diseases) + PlantWild v1 2,260 (healthy leaves).
- val_lab 7,520 (PlantVillage) · val_field 2,806 (PlantDoc train + test folders).
- Leakage guard removed 660 training candidates that looked like PlantDoc photos (PlantWild v2 260, PlantSeg 259, PlantWild v1 140).
- PlantSeg contributed nothing new: all 2,335 mapped photos were duplicates of PlantWild v2 or PlantDoc look-alikes.
- PlantDoc: 58 duplicates and 58 label-conflict photos dropped from validation.
- Weak spot: Tomato spider mites has only 2 PlantDoc photos, so its field F1 is noise.
