# AgriSmart AI: Beginner's Guide to Fine-Tuning, Datasets and the 5-Day Plan

**SIH-2026 Internal Hackathon · L. J. Institute of Engineering and Technology · Problem Statement 1**
Prepared on kickoff day, Thursday 10 Sep 2026. **Commit window: Thu 10 → Tue 15 Sep.**

---

## 0. Read this first (TL;DR)

- **The core task is an image classifier:** it takes a leaf photo and outputs a disease (or "healthy"). You train it on **PlantVillage** (lab photos). The organizers score it on **hidden field photos** using **macro-F1**.
- **The trap:** a model trained on lab photos scores about **99%** on more lab photos and about **30%** on real field photos. The challenge is about **generalisation** (working on photos unlike the training ones), not fitting.
- **The recipe:** a strong pretrained backbone (DINOv2), plus removing the background shortcut, plus extra field photos (de-duplicated), plus choosing models by their score on field photos.
- **Where to train:** Kaggle Notebooks give you free GPUs (30 h/week per account). Your laptop has no NVIDIA GPU, so use it only for the app and for single-image prediction.
- **Datasets:**
  - Use PlantVillage, ideally the official Hugging Face copy, whose split already keeps each leaf on one side (no leakage).
  - Use PlantDoc **only to check your model, never to train it**.
  - Add PlantWild and PlantSeg for real field training photos.
- **Bonus modules:**
  - **C:** weather advice from the free Open-Meteo API.
  - **D:** a sustainability score with a published formula.
  - **E:** a farmer assistant in Gujarati/Hindi that answers from trusted sources (retrieval, "RAG"), not a fine-tuned LLM.
- **At kickoff, ask the organizers:** "May we train on the public PlantDoc dataset?"

---

## 1. The challenge on one page

| Item | What the PDF says |
|---|---|
| Mandatory core | Classify a leaf/crop image into a disease class (or "healthy") from the provided label set. Show a clear result plus basic precautionary guidance. |
| Minimum bar | A trained model (transfer learning encouraged), an honest train/validation/test split, macro-F1 and a confusion matrix on the held-out test set, and a minimal interface (web app, notebook UI, or CLI + screenshots) that predicts on a new image. |
| Training data | PlantVillage: about 54,000 lab images with a uniform background, released at kickoff. You may **add other public data if you cite it**. |
| Test data | A separate, unseen **field-condition** set "derived from PlantDoc-style real-world images" (natural light, clutter, occlusion). Used only by the judges; a small public sample is shared for format checks. **Never train on it.** |
| Classes | A fixed list of about 15–20 crop–disease classes (plus "healthy") that exist in **both** datasets. The official list ships with the kickoff data. |
| Interface | `predict(image_path) -> class_label` in Python, **or** `python predict.py --image <path>` that prints the class. It must load your weights and run with no manual steps. The exact signature and label format are published with the data. |
| Primary metric | **Macro-F1** on the held-out field set. You must also report a confusion matrix and per-class precision/recall. |
| Repo | Public GitHub: `/README.md`, `/src` or `/app`, `/model` (training + inference + predict), `/report` (one-page model report), `requirements.txt`. A judge must reproduce a prediction in **under ~10 minutes**. |
| Timeframe | All substantive work committed **10–15 Sep**. Pre-built repos are disqualified. Cite reused code and data, and include an originality declaration. AI coding assistants are allowed. |

**Scoring (100 points):**

| Axis | Points | What drives it |
|---|---|---|
| AI/ML implementation | 25 | Mostly held-out macro-F1 against baseline bands, plus a sound method (honest split, no leakage, correct metrics) |
| Technical implementation | 20 | Reproducibility (capped if the core can't run), code quality, integration, deployment |
| Innovation & creativity | 15 | A novel approach scores 12–15; a tutorial-level solution scores 0–6 |
| Sustainability & social impact | 15 | A quantified benefit **with a stated method** scores 12–15 |
| User experience | 10 | Farmer-friendly, clear, regional-language effort |
| Problem understanding | 10 | Clear framing and honest limitations |
| Presentation & demo | 5 | A 3–5 minute video |

**Tie-break order:** core macro-F1, then reproducibility, then depth of bonus modules, then innovation.

---

## 2. Fine-tuning 101 (image models, in plain language)

### 2.1 Training vs fine-tuning
- **Training from scratch** starts from random weights and needs millions of images. It's not for you.
- **Transfer learning / fine-tuning** starts from a model already trained on a huge image collection, so it already "sees" edges, textures and shapes. You then teach it your classes with a few thousand images. Think of a qualified doctor learning a new specialty rather than teaching a newborn.
- A classification model has two parts:
  - the **backbone**, which turns an image into a feature vector;
  - the **head**, a small final layer that maps those features to your N classes.
- When fine-tuning, you **replace the head** with a new one that has N outputs.

### 2.2 The training loop
```
images ──► DataLoader (batches) ──► model ──► scores (logits) ──► loss (cross-entropy)
                                   ▲                                        │
                                   └──── optimizer step (AdamW) ◄── gradients (backward)
```
- **Epoch:** one full pass over the training data.
- **Batch size:** images per step (32–64 at 224 px on a Kaggle T4).
- **Learning rate (LR):** how big each update is. It's the most important setting. Too high diverges; too low learns nothing.
- **Scheduler:** changes the LR over time (a short warm-up, then cosine decay).
- **Loss:** cross-entropy. Label smoothing 0.1 helps the model stay less over-confident.

```python
import timm, torch

model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=NUM_CLASSES)
cfg = timm.data.resolve_model_data_config(model)
train_tf = timm.data.create_transform(**cfg, is_training=True, auto_augment="rand-m9-mstd0.5-inc1")
eval_tf = timm.data.create_transform(**cfg)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # after every epoch: evaluate on the FIELD validation set, keep the checkpoint with the best macro-F1
```

### 2.3 Two-stage fine-tuning: linear probe, then fine-tune (LP-FT)
1. **Linear probe:** freeze the backbone and train only the new head (3–5 epochs, LR about 1e-3).
2. **Fine-tune:** unfreeze everything and continue with a **small** LR for the backbone (1e-5 to 3e-5) and a larger one for the head.

**Why:** a freshly initialised head sends large, noisy gradients into the backbone and damages the good pretrained features. That hurts most on data unlike the training set (field photos). LP-FT avoids the damage ([Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054)).

```python
# stage 1 — linear probe
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
# ... train 3–5 epochs with lr=1e-3 ...

# stage 2 — full fine-tune with two learning rates
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not n.startswith("head")], "lr": 2e-5},
    {"params": model.get_classifier().parameters(), "lr": 2e-4},
], weight_decay=0.05)
```

### 2.4 Overfitting and early stopping
- If training loss keeps falling while the validation metric stalls or drops, the model is **overfitting** (memorising). Keep the checkpoint with the **best field-validation macro-F1**, and stop after about 3 epochs without improvement.
- You will see about 99% on PlantVillage validation within an epoch or two. **That is not success.** It only shows the model learned the lab domain.

### 2.5 Data augmentation
- Each time an image is loaded it gets a random transformation, so the model never sees exactly the same picture twice and learns what actually matters.
- Useful transformations here: `RandomResizedCrop` (scale 0.3–1), flips, rotation, colour jitter (brightness, contrast, saturation, hue), Gaussian blur, RandAugment, and **background replacement** (section 5.3).
- Never augment validation or test images. The one exception is test-time augmentation (TTA) at the very end.

### 2.6 Splits and leakage (judges score this)
- **Train:** the model learns from it. **Validation:** you pick settings and checkpoints with it. **Test:** the final number, touched once.
- **Leakage** means information from test images reaches training, which inflates your score. The traps in *this* challenge:
  1. PlantVillage has **several photos of the same physical leaf**. Split by `leaf_id`, never by image.
  2. PlantDoc has **the same images in its train and test folders** (GitHub issue #4).
  3. PlantDoc, PlantWild and PlantSeg were all **scraped from the web**, so they share images. De-duplicate (section 4.3).
  4. "Pre-augmented" datasets put copies of one photo in both train and validation.

### 2.7 Metrics: why macro-F1
- **Accuracy** = correct / total. It's misleading when some classes are rare.
- **Precision** for class X: of everything predicted X, how much really is X. **Recall** for class X: of all real X, how many were found. **F1** combines precision and recall into one number (their harmonic mean).
- **Macro-F1** is the plain average of per-class F1. Every class counts equally, so ignoring a rare class hurts a lot.
- **Confusion matrix:** rows are true classes, columns are predicted classes. It shows exactly which diseases get mixed up (for example tomato early vs late blight).

```python
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

print("macro-F1:", f1_score(y_true, y_pred, average="macro"))
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASSES, xticks_rotation=90)
```

### 2.8 Domain shift
- **Source domain (lab):** one leaf, plain grey or black background, even lighting.
- **Target domain (field):** clutter, sun and shadow, several leaves, blur, other plants.
- Models grab the easiest clue that works in training. In PlantVillage that is often **the background**, which doesn't exist in field photos. Section 3 has the numbers.

### 2.9 LLM fine-tuning vs RAG (for bonus E)
- **Fine-tuning an LLM** (LoRA/QLoRA) changes its weights to learn a style or format. It needs GPUs and curated data, and it doesn't reliably add facts.
- **RAG (retrieval-augmented generation)** keeps the LLM as it is. You retrieve relevant passages from **your own trusted knowledge base** and tell the LLM to answer only from them, with citations.
- The rubric says *"Grounded answers score higher than free-form generation"*, so **use RAG. You don't need to fine-tune an LLM in this hackathon.**

---

## 3. Why this challenge is hard: the numbers

| Finding | Number | Source |
|---|---|---|
| ResNet-50 fine-tuned on PlantVillage, tested on PlantVillage | 99.73% accuracy, 0.996 macro-F1 | [PMC13236948](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/) |
| Same model tested on PlantDoc field photos (21 shared classes) | **32.05% accuracy, 0.285 macro-F1** (a 67.7-point drop), yet average confidence stays around 80%, so it is confidently wrong | same |
| A model given only **8 background pixels** per PlantVillage image | **49.0% accuracy** (random guessing is 2.6%) | [arXiv 2206.04374](https://arxiv.org/abs/2206.04374) |
| Frozen backbones compared on field photos | ResNet-50 28.0% · CLIP ViT-B/16 41.4% · ViT-S/16 41.9% · **DINOv2-S/14 43.2%** · ensemble of 5 DINOv2 heads 43.8% (and far better calibrated) | PMC13236948 |
| Light adaptation tricks | AdaBN 32.1 → 34.3% · feature moment matching 36.6% · adversarial DANN **25.2% (worse)** | same |
| Adding real field photos to training (PlantDoc paper) | up to **+31% accuracy** | [PlantDoc paper](https://arxiv.org/abs/1911.10317) |

**Takeaways:**
1. **Backbone quality matters most.** Start from DINOv2, not a plain ImageNet ResNet.
2. **No single trick fixes it.** Combine a strong backbone, background removal or replacement, real field photos and strong augmentation.
3. **Measure on field photos.** PlantVillage validation accuracy tells you almost nothing about your final score.

---

## 4. Datasets

### 4.1 Your four links

| Link | What it is | Ready to use? | Verdict |
|---|---|---|---|
| [Kaggle: abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | PlantVillage in **3 versions** (`color`, `grayscale`, `segmented`), 38 classes, 14 crops. Licence CC BY-NC-SA 4.0, usability 8.75, 124K downloads, not updated in 7 years. | **Yes on Kaggle.** Attach it to a notebook, no download needed. It has **no split**. | ✅ **Use it**, especially `segmented/` for background replacement. Keep only the shared classes and split by leaf. |
| [Kaggle: mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) | A re-upload of the same ~54,303 **color** images, 38 classes. Labelled GPL-2, which is not the original licence. Usability 6.25. | Yes, but any split that comes with it (or that you make at random) is **not grouped by leaf**, so photos of the same leaf land in both train and val. | ❌ **Skip.** Redundant with #1, and its licence is unclear. |
| [GitHub: pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | 2,598 **web-scraped field photos**, 13 species, 28 class folders. CC BY 4.0, 436 stars. | **Partly.** Open issues: [#4 same images in train and test](https://github.com/pratikkayal/PlantDoc-Dataset/issues), #5 "the dataset is incorrect" (labels), #3 **filenames are illegal on Windows**, so `git clone` breaks on your PC. Use it on Kaggle/Colab (Linux) or through a Hugging Face copy. | ⚠️ **Validation only, never training.** The hidden test is "derived from PlantDoc-style" images and may contain these exact photos. Training on them could be judged leakage. Ask the organizers. |
| [DOI 10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196) | The **PlantDoc paper** (CoDS-COMAD 2020), not a dataset. | n/a | 📄 **Cite it** in your README whenever you use PlantDoc. |

### 4.2 Better and additional datasets

| Dataset | Size and licence | Why it helps | Caveats |
|---|---|---|---|
| ⭐ [**mohanty/PlantVillage**](https://huggingface.co/datasets/mohanty/PlantVillage) (Hugging Face, official, updated Feb 2026) | 54,306 images, 14 crops, 26 diseases, in 3 configs (`color`, `grayscale`, `segmented`), about 2 GB. CC BY-SA 3.0. | **Official, with a ready leaf-grouped 80/20 split** (color: 43,596 train / 10,709 test) and a `leaf_id` column, so no leakage. `load_dataset("mohanty/PlantVillage", "segmented")` | Switch to the organizers' copy for the final submission (the README must say "core = provided dataset"). |
| ⭐ [**PlantWild v1/v2**](https://huggingface.co/datasets/uqtwei2/PlantWild) (ACM MM 2024) | v1: **18,542 field photos, 89 classes** (33 healthy + 56 diseased), `plantwild.zip` 2.7 GB. v2: expert-refined, **115 classes**, `plantwild_v2.zip` 1.6 GB. A text description for every disease. CC BY-NC-ND 4.0. | The best source of extra **field** training photos. The disease descriptions can feed the bonus-E knowledge base. | Scraped from Google Images, Ecosia and Baidu, like PlantDoc, so **de-duplicate against PlantDoc**. The licence is non-commercial and no-derivatives: cite it, and don't redistribute the images in your repo. The Hugging Face preview is broken, so download the zip. |
| ⭐ [**PlantSeg**](https://zenodo.org/records/14935094) (Scientific Data 2025, [GitHub](https://github.com/tqwei05/PlantSeg)) | **11,458 field photos, 115 diseases**, with segmentation masks (LabelMe JSON and COCO). CC BY 4.0. `plantsegv3.zip` 1.62 GB. | Field photos plus **lesion masks**, useful for lesion-focused crops and augmentation. The most permissive licence of the three. | Same authors and web sources as PlantWild, so the two **overlap**. De-duplicate. |
| [Potato Leaf Disease, uncontrolled environment](https://data.mendeley.com/datasets/ptz377bwb8/1) (Mendeley, 2023) | 3,076 potato field photos from Central Java, 7 classes (virus, bacteria, fungi, pest, nematode, phytophthora, healthy). | Real potato field photos. | Labels are pathogen groups ("Phytophthora" ≈ late blight), so only some of them map to the shared classes. |
| Tomato real-world ([Mendeley rnbsw72zb5](https://data.mendeley.com/datasets/rnbsw72zb5/1)) and Multi-Crop ([Mendeley z6jp232g5j](https://data.mendeley.com/datasets/z6jp232g5j/1)) | About 1,000 and 6,895 images. CC BY 4.0. | Some extra field photos of *healthy* leaves. | Labels are only **healthy vs diseased**, and the Multi-Crop images are of unclear origin. Low value here. |
| 🚫 **Avoid** | "New Plant Diseases Dataset" (Kaggle, 87,900 files) and Kaggle "master" or "merged" plant-disease sets. | — | The first is PlantVillage **pre-augmented**, so augmented copies of the same photo sit in both train and valid. Merged sets often contain PlantDoc images, which would contaminate the field test. |

### 4.3 De-duplication recipe (do this before adding any extra data)
```python
import imagehash
from PIL import Image

def phash(path):
    return imagehash.phash(Image.open(path).convert("RGB"))

# two images are near-duplicates if the Hamming distance is small
is_dup = (phash(a) - phash(b)) <= 8
```
- Also compare **backbone embeddings** (for example DINOv2 features), using a cosine similarity above about 0.95. This catches resized, cropped or watermarked copies that the hash misses.
- Remove from **training** anything that is near-duplicate to PlantDoc (train *and* test folders) or to the organizers' public sample.
- Also remove duplicates between PlantDoc's own train and test folders before using them as your validation set.

### 4.4 Class mapping, PlantVillage ↔ PlantDoc (draft until the official list arrives)
Checked against the `mohanty/PlantVillage` label list and PlantDoc's `train/` folders. **28 pairs exist**; the organizers will pick about 15–20 of them. Keep the final list in an editable `classes.yaml`.

| # | Suggested label | PlantVillage folder | PlantDoc folder |
|---|---|---|---|
| 1 | Apple scab | `Apple___Apple_scab` | `Apple Scab Leaf` |
| 2 | Apple cedar rust | `Apple___Cedar_apple_rust` | `Apple rust leaf` |
| 3 | Apple healthy | `Apple___healthy` | `Apple leaf` |
| 4 | Bell pepper bacterial spot | `Pepper,_bell___Bacterial_spot` | `Bell_pepper leaf spot` |
| 5 | Bell pepper healthy | `Pepper,_bell___healthy` | `Bell_pepper leaf` |
| 6 | Blueberry healthy | `Blueberry___healthy` | `Blueberry leaf` |
| 7 | Cherry healthy | `Cherry_(including_sour)___healthy` | `Cherry leaf` |
| 8 | Corn gray leaf spot | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn Gray leaf spot` |
| 9 | Corn common rust | `Corn_(maize)___Common_rust_` | `Corn rust leaf` |
| 10 | Corn northern leaf blight | `Corn_(maize)___Northern_Leaf_Blight` | `Corn leaf blight` |
| 11 | Grape black rot | `Grape___Black_rot` | `grape leaf black rot` |
| 12 | Grape healthy | `Grape___healthy` | `grape leaf` |
| 13 | Peach healthy | `Peach___healthy` | `Peach leaf` |
| 14 | Potato early blight | `Potato___Early_blight` | `Potato leaf early blight` |
| 15 | Potato late blight | `Potato___Late_blight` | `Potato leaf late blight` |
| 16 | Raspberry healthy | `Raspberry___healthy` | `Raspberry leaf` |
| 17 | Soybean healthy | `Soybean___healthy` | `Soyabean leaf` |
| 18 | Squash powdery mildew | `Squash___Powdery_mildew` | `Squash Powdery mildew leaf` |
| 19 | Strawberry healthy | `Strawberry___healthy` | `Strawberry leaf` |
| 20 | Tomato bacterial spot | `Tomato___Bacterial_spot` | `Tomato leaf bacterial spot` |
| 21 | Tomato early blight | `Tomato___Early_blight` | `Tomato Early blight leaf` |
| 22 | Tomato late blight | `Tomato___Late_blight` | `Tomato leaf late blight` |
| 23 | Tomato leaf mold | `Tomato___Leaf_Mold` | `Tomato mold leaf` |
| 24 | Tomato septoria leaf spot | `Tomato___Septoria_leaf_spot` | `Tomato Septoria leaf spot` |
| 25 | Tomato spider mites | `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato two spotted spider mites leaf` |
| 26 | Tomato mosaic virus | `Tomato___Tomato_mosaic_virus` | `Tomato leaf mosaic virus` |
| 27 | Tomato yellow leaf curl virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato leaf yellow virus` |
| 28 | Tomato healthy | `Tomato___healthy` | `Tomato leaf` |

**PlantVillage-only classes (no field counterpart, drop them):** Apple black rot, Cherry powdery mildew, Corn healthy, Grape esca, Grape leaf blight, Orange citrus greening, Peach bacterial spot, Potato healthy, Strawberry leaf scorch, Tomato target spot. The PDF's example list mentions "Apple Black Rot", which PlantDoc does not have, so **always use the official list** once it ships.

---

## 5. Recommended model recipe and experiment plan

### 5.1 Set up validation before any modelling
| Set | Built from | Use |
|---|---|---|
| **V-lab** | PlantVillage validation split, grouped by leaf | Sanity check only (expect 97–99%) |
| **V-field** ⭐ | PlantDoc images of the shared classes (train and test folders, de-duplicated; you never train on them) plus the organizers' public sample | **The number you optimise.** Report it as "field-proxy macro-F1". |

Keep an **experiment log** with one row per run: experiment ID, backbone, training data, augmentation, epochs, V-lab F1, V-field F1, notes. It becomes the "method" section of your report and shows the judges a sound process.

### 5.2 Experiment ladder (each step builds on the best so far)
| ID | Change | Why |
|---|---|---|
| E0 | `convnext_tiny.fb_in22k_ft_in1k` or `efficientnet_b3.ra2_in1k`, PlantVillage `color`, basic augmentation, about 10 epochs | Your **honest baseline**. Expect a low V-field score, and write it down. |
| E1 | + strong augmentation (RandomResizedCrop scale 0.3–1, colour jitter, blur, rotation, RandAugment) | Stop the model relying on exact colours and framing |
| E2 | + **background replacement** using `segmented` leaves on random backgrounds (5.3) | Remove the background shortcut |
| E3 | Backbone → **DINOv2** `vit_small_patch14_dinov2.lvd142m` or `vit_base_patch14_dinov2.lvd142m` with **LP-FT** | Gave the biggest single gain in the research |
| E4 | + **extra field photos** from PlantWild v2 and PlantSeg, mapped to the shared classes and de-duplicated | Real field photos are the most direct fix |
| E5 | + class-balanced sampling, label smoothing, TTA (flips and crops), ensemble of the 2–3 best models | Small, reliable gains for macro-F1 |
| E6 (optional) | Crop the leaf or remove the background before classifying (for example with `rembg`) | Makes field photos look more like training photos. Worth innovation points, but keep it **only if V-field improves** |

### 5.3 Background replacement (sketch)
```python
import numpy as np
from PIL import Image

def composite(seg_leaf: Image.Image, background: Image.Image) -> Image.Image:
    """Paste a PlantVillage 'segmented' leaf (black background) onto a new background."""
    leaf = seg_leaf.convert("RGB")
    arr = np.asarray(leaf).astype(np.int16)
    mask = ((arr.sum(axis=2) > 30) * 255).astype(np.uint8)   # non-black pixels = leaf
    bg = background.convert("RGB").resize(leaf.size)
    return Image.composite(leaf, bg, Image.fromarray(mask))
```
- For backgrounds, use random crops of soil, grass and foliage photos, or of de-duplicated PlantWild/PlantSeg photos from classes you don't use. **Never use PlantDoc or public-sample images as backgrounds**, because that is leakage again.
- Apply the replacement to about 50% of training images, so the model also keeps seeing some original-looking images.

### 5.4 Starting hyperparameters (Kaggle T4 / P100)
| Setting | Value |
|---|---|
| Image size | 224 px. For DINOv2 pass `img_size=224` to `timm.create_model`; the default is 518, which is much slower. |
| Batch size | 32–64 |
| Optimizer | AdamW, weight decay 0.05 |
| Learning rate | Linear probe: 1e-3 for the head. Fine-tune: 2e-5 for the backbone, 2e-4 for the head. Cosine schedule with a warm-up of about 1 epoch. |
| Epochs | Linear probe 3–5, fine-tune 5–10. Time the first epoch and plan from that. |
| Loss | Cross-entropy with `label_smoothing=0.1` |
| Speed | Mixed precision: `torch.autocast("cuda", dtype=torch.float16)` plus `torch.amp.GradScaler("cuda")`, and `num_workers=4` |
| Reproducibility | Fix random seeds, save the config next to each checkpoint, log every run |

### 5.5 The `predict` contract
- Put `predict(image_path) -> str` in `/model/predict.py`. Load the model **once** (cache it in a global), run it on CPU, and return the exact label string the organizers publish.
- Host large weights on a **GitHub Release** or the **Hugging Face Hub** and download them automatically on first run. Then test a **fresh clone** on a clean machine and time it; it must take under 10 minutes.

---

## 6. Compute setup

| Where | Use it for | How |
|---|---|---|
| **Kaggle Notebooks** ⭐ | All training | 1. Create an account and **verify your phone** (needed for GPU and internet). 2. New Notebook → Settings → Accelerator **GPU T4 ×2** or **P100**, Internet **on**. 3. *Add Input* → search "PlantVillage Dataset" (abdallahalidev); it appears under `/kaggle/input/…`. 4. Train, saving weights to `/kaggle/working`. 5. **Save Version → Save & Run All (Commit)** runs in the background for up to 12 h, even with the tab closed. |
| Quota | — | **30 GPU-hours per week per account.** Every teammate has their own quota, so run different experiments on different accounts at the same time. |
| Google Colab | Fallback | Free T4 with variable limits. Keep data on Google Drive. |
| Your laptop (i7-13700H, 16 GB, Intel Iris Xe, **no CUDA**) | `predict.py`, the app, the demo | CPU inference on one image at 224 px is fast enough for a demo. **Don't train here.** |

---

## 7. Bonus modules: C, D and E

### C. Weather intelligence
- **Data source:** the [Open-Meteo](https://open-meteo.com/en/docs) forecast API, free for non-commercial use with **no API key**. Example for Ahmedabad:
  `https://api.open-meteo.com/v1/forecast?latitude=23.02&longitude=72.57&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m&daily=precipitation_sum,et0_fao_evapotranspiration&timezone=Asia%2FKolkata&forecast_days=3`
- **Transparent rules.** Publish them in the README; the thresholds below are illustrative, so tune and cite them:

| Condition (next 24–48 h) | Advice |
|---|---|
| Rain probability ≥ 60% | "Delay irrigation, rain likely" |
| 2 consecutive days with minimum temperature ≥ 10 °C and ≥ 6 h at relative humidity ≥ 90% | "High late-blight risk for tomato and potato: inspect and avoid overhead watering." This follows the UK **Hutton criteria** (AHDB); verify the source before citing it. |
| Temperature ≥ 35 °C and no rain | "Heat stress: irrigate early morning or evening" |
| Wind ≥ 20 km/h | "Avoid spraying today (drift)" |
| A fungal disease was detected and humid weather is forecast | Raise the alert level in the app |

### D. Sustainability score (publish the exact formula)
- **Water efficiency W** (0–100) = `100 × min(1, crop_water_need / water_applied)`. Here `crop_water_need = Kc × ET₀`: ET₀ comes from Open-Meteo's `et0_fao_evapotranspiration`, and Kc from the FAO-56 crop-coefficient table.
- **Input use I** (0–100): starts at 100, with penalties for fertilizer or pesticide use above the recommended dose, and a bonus for spot-treating only when a disease is detected.
- **Crop health H** (0–100): healthy = 100; diseased = `100 × (1 − severity)` (or confidence-weighted).
- **Score** = `0.4·W + 0.3·I + 0.3·H`. Bands: ≥ 80 good, 60–79 fair, < 60 needs action. Show suggestions for the weakest component.
- For the 15 impact points, **quantify with a stated method**, for example "In a simulated week, skipping irrigation on days with ≥ 60% rain probability saved X litres per acre (method: …)".

### E. Farmer assistant (grounded, Gujarati/Hindi)
```
disease result + weather + farm inputs
        │
        ▼
retrieve top-k passages from YOUR knowledge base  ──►  LLM prompt: "Answer ONLY from the context,
(one card per class: symptoms, cause, precautions,       cite the source, reply in {Gujarati|Hindi|English}"
 organic and chemical options, sources)                           │
                                                                  ▼
                                                    answer on screen (+ optional voice)
```
- **Knowledge base:** one Markdown/JSON card per class, with sources (state agricultural university or ICAR extension pages, PlantWild disease descriptions, and so on). Keep the precautions conservative, and **don't give pesticide doses** unless they come from an official source.
- **Retrieval:** with only about 20 cards, simple keyword matching works. For something better, use multilingual embeddings (for example `intfloat/multilingual-e5-small`) with cosine similarity.
- **LLM:** any hosted API with reasonable Indian-language support (for example Groq-hosted open models or Gemini).
  - Put the key in `.env`, add `.env` to `.gitignore`, and commit a `.env.example` instead. **The repo is public**, and leaked keys get abused within minutes.
  - Don't copy key files into the project folder.
- **Regional language:** ask the LLM to answer directly in Gujarati or Hindi, and have a native speaker on the team check about 10 sample answers (UX points).
- **Voice (optional):** Whisper for speech-to-text and gTTS for text-to-speech both support Gujarati and Hindi.
- **Offline fallback:** with no internet or LLM, show the static precaution card. "Offline / low-connectivity mode" is on the organizers' innovation list.

---

## 8. Day-by-day plan (Thu 10 → Tue 15 Sep)

| Day | Model track | App / bonus track | Done when |
|---|---|---|---|
| **Thu 10** (today) | Everyone reads sections 1–3. Create Kaggle accounts and verify phones. Run a first notebook: load PlantVillage and train ConvNeXt-T for 1–2 epochs on a few classes to learn the loop. | **Create the GitHub repo today** (inside the window) with the required folders and a README skeleton. | Everyone has run one training loop |
| **Fri 11** (or kickoff day) | Kickoff data → `classes.yaml` → data pipeline (filter classes, split by leaf, PlantDoc V-field with mapping and de-duplication) → **E0 baseline** | Streamlit or Gradio skeleton: upload an image and show a dummy prediction. Start writing knowledge-base cards. | Baseline V-lab and V-field numbers logged |
| **Sat 12** | E1–E4 in parallel on different teammates' Kaggle accounts. Download PlantWild v2 and PlantSeg, map classes, de-duplicate. | Weather module (Open-Meteo) and its rules | Best V-field model identified |
| **Sun 13** | E5 (TTA, ensemble) if time allows. Train the final model, publish the weights, write `predict.py`, test it on the public sample, produce the report (confusion matrix, per-class P/R). | Sustainability score. Assistant (RAG plus Gujarati/Hindi). | `python predict.py --image x.jpg` works from a fresh clone |
| **Mon 14** | Freeze the model. Write limitations and failure cases for the report. | Put the model into the app, run the full demo flow, deploy (optional: Hugging Face Spaces or Streamlit Community Cloud, both free on CPU), polish the UX | Demo flow runs start to finish |
| **Tue 15** | **No new training.** | README, citations and licences, originality declaration, reproducibility test on a clean machine (under 10 min), record the 3–5 min video, submit | Submitted |

**Suggested roles (typical 6-person SIH team; adjust to yours):** 2 on model experiments · 1 on data (mapping, de-duplication, V-field) · 1 on the app and integration · 1 on bonus C/D/E · 1 on docs, report, video and reproducibility testing.
**Commit small and often.** The judges check that the commit history spans 10–15 Sep.

---

## 9. Submission checklist

- [ ] Public repo with `/README.md`, `/src` or `/app`, `/model` (training + `predict.py`), `/report`, `requirements.txt`
- [ ] `predict(image_path)` / `python predict.py --image …` works from a **fresh clone**; weights download automatically
- [ ] Report fields: task (N classes), dataset and **exact split sizes**, model and hyperparameters, **macro-F1 + accuracy + confusion matrix + per-class P/R**, baseline comparison, **honest limitations** (lab vs field, classes with few field photos)
- [ ] README: modules built, setup under 10 min, datasets with sources and licences, metrics, architecture overview, known limitations, video link, app link
- [ ] Citations: PlantVillage (Mohanty et al. 2016), the PlantDoc paper, PlantWild, PlantSeg, timm/DINOv2, and any code or notebooks referenced (originality declaration)
- [ ] No secrets in the repo (`.env` ignored, `.env.example` provided)
- [ ] Commit history spread across 10–15 Sep
- [ ] 3–5 min video showing: new image → disease + confidence + precautions → weather advice → sustainability score → assistant answering in Gujarati/Hindi

---

## 10. Questions to ask the organizers at kickoff

1. **May we train on the public PlantDoc dataset, or is the held-out set drawn from it?** If you're unsure, don't train on it.
2. What is the **baseline macro-F1**, and what are the scoring band thresholds?
3. The exact **label strings** and **`predict()` signature**. Is "healthy" one class or one per crop?
4. What are the hidden test images like: format and resolution? A single leaf or the whole plant?
5. Are there limits for `predict`: CPU only? Maximum seconds per image? Maximum weight-file size?
6. May we train on web-scraped research datasets with non-commercial licences (PlantWild, CC BY-NC-ND)?

---

## 11. Learning resources

- PyTorch transfer-learning tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Hugging Face image-classification guide: https://huggingface.co/docs/transformers/tasks/image_classification
- timm documentation: https://huggingface.co/docs/timm/index
- fast.ai *Practical Deep Learning* (lessons 1–2): https://course.fast.ai/
- scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- LP-FT paper (why the two-stage fine-tune): https://arxiv.org/abs/2202.10054
- DINOv2 paper: https://arxiv.org/abs/2304.07193
- Open-Meteo API docs: https://open-meteo.com/en/docs

---

## Sources (checked 10 Sep 2026)

- Problem statement: `oxpusiikphgm6s6uv2xs.pdf` (SIH-2026 PS-1, 8 pages)
- Kaggle: [PlantVillage Dataset (abdallahalidev)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) · [PlantVillage (mohitsingh1804)](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- PlantDoc: [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) · [issues](https://github.com/pratikkayal/PlantDoc-Dataset/issues) · [paper (arXiv 1911.10317)](https://arxiv.org/abs/1911.10317) · [DOI](https://doi.org/10.1145/3371158.3371196)
- [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)
- PlantWild: [Hugging Face](https://huggingface.co/datasets/uqtwei2/PlantWild) · [paper (arXiv 2408.03120)](https://arxiv.org/abs/2408.03120) · [GitHub (MVPDR)](https://github.com/tqwei05/MVPDR)
- PlantSeg: [Zenodo 14935094](https://zenodo.org/records/14935094) · [GitHub](https://github.com/tqwei05/PlantSeg) · [Scientific Data article](https://www.nature.com/articles/s41597-025-06513-4)
- [Potato Leaf Disease Dataset in Uncontrolled Environment (Mendeley)](https://data.mendeley.com/datasets/ptz377bwb8/1)
- [Quantifying the reliability gap in cross-domain plant disease classification (PMC13236948)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)
- [Uncovering bias in the PlantVillage dataset (arXiv 2206.04374)](https://arxiv.org/abs/2206.04374)
- [Kaggle free GPU quota, 2026 guide](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/)
- timm model cards: [DINOv2 ViT-S/14](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) · [DINOv2 ViT-B/14](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m) · [ConvNeXt-T](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k) · [EfficientNet-B3](https://huggingface.co/timm/efficientnet_b3.ra2_in1k) (all Apache-2.0; DINOv3 has a custom licence)
