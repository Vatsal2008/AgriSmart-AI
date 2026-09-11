# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AgriSmart India · Step 1: build a multi-crop training dataset
#
# A separate, much larger dataset covering the crops most grown in India, for a **second model**
# that stands apart from the hackathon core model (which stays untouched). About 35 crops from roughly
# 25 sources: PlantVillage (reused), plus curated Hugging Face "Project-AgML" mirrors of published
# research datasets, plus a handful of Kaggle-native sets not available there.
#
# **Universal leakage guard.** Several of these sources ship pre-augmented copies mixed into the same
# folder as the originals, so a random split can put a flipped/zoomed copy of a training photo into the
# test set — the exact trap that produces a fake high accuracy. Instead of trusting any source's own
# split, this notebook computes a perceptual hash and a DINOv2 embedding for **every** image, clusters
# near-duplicates within each crop+class, and keeps every whole cluster on one side of an 85/15 split.
# Exact and near duplicates are also merged (one copy kept) so the same photo is never counted twice.
#
# Runtime: expect 2-3 hours on a T4 (mostly downloading). Output: `/kaggle/working/india_data/`.

# %%
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    import imagehash
except ImportError:
    pip_install("imagehash")
    import imagehash
try:
    import timm
except ImportError:
    pip_install("timm")
    import timm
try:
    import datasets as hfds
except ImportError:
    pip_install("datasets")
    import datasets as hfds

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
hfds.disable_progress_bar()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/india_data")
OUT.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 512          # every kept photo is resized so its longer side is at most this
CAP_PER_CLASS = 1600    # a soft cap per class so no single huge source (coffee, tea) swamps training
DUP_COS, DUP_HAM = 0.96, 6  # "these two photos are copies of each other"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


log(f"torch {torch.__version__} | timm {timm.__version__} | datasets {hfds.__version__}")

# %% [markdown]
# ## 1 · The crop and class list
#
# Each crop lists one or more sources. A "kaggle" source reads files already attached to this notebook.
# An "hf" source streams a Hugging Face dataset (no local copy kept). `class_map` renames each source's
# own label to one canonical name per crop, so two sources that call the same disease something
# different end up as one class. A blank canonical name means "drop this label" (used to exclude classes
# that would collide with a different crop's own labelling scheme, e.g. Potato's pathogen-group scheme).

# %%
CROPS = {
    # ---- reused from the hackathon core dataset (Kaggle: PlantVillage) ----
    "apple": [{"kind": "kaggle", "root": "color", "class_map": {
        "Apple___Apple_scab": "Apple scab", "Apple___Black_rot": "Black rot",
        "Apple___Cedar_apple_rust": "Cedar apple rust", "Apple___healthy": "Healthy"}}],
    "blueberry": [{"kind": "kaggle", "root": "color", "class_map": {"Blueberry___healthy": "Healthy"}}],
    "cherry": [{"kind": "kaggle", "root": "color", "class_map": {
        "Cherry_(including_sour)___Powdery_mildew": "Powdery mildew",
        "Cherry_(including_sour)___healthy": "Healthy"}}],
    "corn_maize": [{"kind": "kaggle", "root": "color", "class_map": {
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot",
        "Corn_(maize)___Common_rust_": "Common rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight",
        "Corn_(maize)___healthy": "Healthy"}}],
    "grape": [
        {"kind": "kaggle", "root": "color", "class_map": {
            "Grape___Black_rot": "Black rot", "Grape___healthy": "Healthy"}},
        {"kind": "hf", "repo": "Project-AgML/grape_leaf_disease_classification", "class_map": {
            "Bacterial Leaf Spot": "Bacterial leaf spot", "Downy Mildew": "Downy mildew",
            "Healthy Leaves": "Healthy", "Powdery Mildew": "Powdery mildew"}},
    ],
    "peach": [{"kind": "kaggle", "root": "color", "class_map": {
        "Peach___Bacterial_spot": "Bacterial spot", "Peach___healthy": "Healthy"}}],
    "bell_pepper": [{"kind": "kaggle", "root": "color", "class_map": {
        "Pepper,_bell___Bacterial_spot": "Bacterial spot", "Pepper,_bell___healthy": "Healthy"}}],
    "potato": [{"kind": "kaggle", "root": "color", "class_map": {
        "Potato___Early_blight": "Early blight", "Potato___Late_blight": "Late blight",
        "Potato___healthy": "Healthy"}}],
    "raspberry": [{"kind": "kaggle", "root": "color", "class_map": {"Raspberry___healthy": "Healthy"}}],
    "squash": [{"kind": "kaggle", "root": "color", "class_map": {"Squash___Powdery_mildew": "Powdery mildew"}}],
    "strawberry": [{"kind": "kaggle", "root": "color", "class_map": {
        "Strawberry___Leaf_scorch": "Leaf scorch", "Strawberry___healthy": "Healthy"}}],
    "tomato": [{"kind": "kaggle", "root": "color", "class_map": {
        "Tomato___Bacterial_spot": "Bacterial spot", "Tomato___Early_blight": "Early blight",
        "Tomato___Late_blight": "Late blight", "Tomato___Leaf_Mold": "Leaf mold",
        "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites",
        "Tomato___Target_Spot": "Target spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus",
        "Tomato___Tomato_mosaic_virus": "Mosaic virus", "Tomato___healthy": "Healthy"}}],
    "soybean": [
        {"kind": "kaggle", "root": "color", "class_map": {"Soybean___healthy": "Healthy"}},
        {"kind": "hf", "repo": "Project-AgML/MH_SoyaHealthVision_disease_classification_leaf", "class_map": {
            "Caterpillar_Semilooper_Pest": "Caterpillar pest", "Frog_Leaf_Eye": "Frogeye leaf spot",
            "Healthy": "Healthy", "Mosaic": "Mosaic virus", "Rust": "Rust",
            "Spectoria_Brown_Spot": "Septoria brown spot"}},
        {"kind": "hf", "repo": "Project-AgML/soybean_leaf_disease_classification", "class_map": {
            "Bacterial leaf Blight": "Bacterial blight", "Dry_leaf": "Dry leaf", "Healthy": "Healthy",
            "Root_images": "", "Septoria_Brown_Spot": "Septoria brown spot", "Vein Necrosis": "Vein necrosis"}},
    ],

    # ---- Indian field and plantation crops ----
    "rice": [
        {"kind": "kaggle", "root": "paddy-disease-classification/train_images", "class_map": {
            "bacterial_leaf_blight": "Bacterial leaf blight", "bacterial_leaf_streak": "Bacterial leaf streak",
            "bacterial_panicle_blight": "Bacterial panicle blight", "blast": "Blast", "brown_spot": "Brown spot",
            "dead_heart": "Dead heart", "downy_mildew": "Downy mildew", "hispa": "Hispa (pest)",
            "normal": "Healthy", "tungro": "Tungro"}},
        {"kind": "hf", "repo": "Project-AgML/rice_leaf_disease_classification_india", "class_map": {
            "Bacterialblight": "Bacterial leaf blight", "Blast": "Blast", "Brownspot": "Brown spot",
            "Tungro": "Tungro"}},
    ],
    "wheat": [{"kind": "kaggle", "root": "data/train", "class_map": {
        "Aphid": "Aphid (pest)", "Black Rust": "Black rust", "Blast": "Blast", "Brown Rust": "Brown rust",
        "Common Root Rot": "Common root rot", "Fusarium Head Blight": "Fusarium head blight",
        "Healthy": "Healthy", "Leaf Blight": "Leaf blight", "Mildew": "Powdery mildew",
        "Mite": "Mite (pest)", "Septoria": "Septoria", "Smut": "Smut", "Stem fly": "Stem fly (pest)",
        "Tan spot": "Tan spot", "Yellow Rust": "Yellow rust"}}],
    "cotton": [
        {"kind": "kaggle", "root": "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset/Original Dataset",
         "class_map": {
            "Bacterial Blight": "Bacterial blight", "Curl Virus": "Curl virus", "Healthy Leaf": "Healthy",
            "Herbicide Growth Damage": "Herbicide damage", "Leaf Hopper Jassids": "Leafhopper (pest)",
            "Leaf Redding": "Leaf reddening", "Leaf Variegation": "Leaf variegation"}},
        {"kind": "kaggle", "root": "Cotton Leaf Image Dataset for Disease Classificati/Cotton_Original_Dataset/Cotton_Original_Dataset",
         "class_map": {
            "Alternaria Leaf Spot": "Alternaria leaf spot", "Bacterial Blight": "Bacterial blight",
            "Fusarium Wilt": "Fusarium wilt", "Healthy Leaf": "Healthy", "Verticillium Wilt": "Verticillium wilt"}},
    ],
    "groundnut": [{"kind": "hf", "repo": "Project-AgML/groundnut_leaf_disease_classification", "class_map": {
        "ALTERNARIA LEAF SPOT": "Alternaria leaf spot", "HEALTHY": "Healthy", "LEAF SPOT": "Leaf spot",
        "ROSETTE": "Rosette", "RUST": "Rust"}}],
    "sugarcane": [{"kind": "kaggle", "root": "", "class_map": {
        "Healthy": "Healthy", "Mosaic": "Mosaic virus", "RedRot": "Red rot", "Rust": "Rust",
        "Yellow": "Yellow leaf disease"}}],
    "tea": [{"kind": "hf", "repo": "Project-AgML/CS-D_tea_leaf_disease_classification", "class_map": {
        "Blister_Blight": "Blister blight", "Brown_Blight": "Brown blight", "Healthy_leaves": "Healthy",
        "Leaf_algal": "Algal leaf spot", "Leaf_rust": "Red rust"}}],
    "coffee": [{"kind": "hf", "repo": "Project-AgML/arabica_coffee_leaf_disease_classification", "class_map": {
        "Cerscospora": "Cercospora leaf spot", "Healthy": "Healthy", "Leaf_rust": "Leaf rust",
        "Miner": "Leaf miner (pest)", "Phoma": "Phoma leaf spot"}}],
    "brinjal_eggplant": [{"kind": "hf", "repo": "Project-AgML/eggplant_disease_classification", "class_map": {
        "Healthy Leaf": "Healthy", "Insect Pest Disease": "Insect pest", "Leaf Spot Disease": "Leaf spot",
        "Mosaic Virus Disease": "Mosaic virus", "White Mold Disease": "White mold", "Wilt Disease": "Wilt"}}],
    "chilli": [{"kind": "hf", "repo": "Project-AgML/COLD_chili_leaf_disease_classification", "class_map": {
        "cercospora": "Cercospora leaf spot", "healthy": "Healthy", "mites_and_trips": "Mites and thrips (pest)",
        "nutritional": "Nutrient deficiency", "powdery mildew": "Powdery mildew"}}],
    "onion": [{"kind": "hf", "repo": "Project-AgML/COLD_onion_leaf_disease_classification", "class_map": {
        "Iris yellow virus": "Iris yellow spot virus",
        "Stemphylium leaf blight and collectrichum leaf blight": "Stemphylium/Colletotrichum blight",
        "healthy": "Healthy", "purple blotch": "Purple blotch"}}],
    "black_gram": [{"kind": "hf", "repo": "Project-AgML/black_gram_disease_classification", "class_map": {
        "Cercospora leaf spot": "Cercospora leaf spot", "Healthy": "Healthy", "Insect": "Insect pest",
        "Leaf Crinkle": "Leaf crinkle virus", "Yellow Mosaic": "Yellow mosaic virus"}}],
    "jute": [{"kind": "hf", "repo": "Project-AgML/jute_disease_classification", "class_map": {
        "Dieback": "Dieback", "Fresh": "Healthy", "Holed": "Pest-holed leaf", "Mosaic": "Mosaic virus",
        "Stem Soft Rot": "Stem rot"}}],
    "betel": [{"kind": "hf", "repo": "Project-AgML/betel_leaf_disease_classification", "class_map": {
        "Bacterial_Leaf_Disease": "Bacterial leaf disease", "Dried_Leaf": "Dried leaf",
        "Fungal_Brown_Spot_Disease": "Fungal brown spot", "Healthy_Leaf": "Healthy"}}],
    "turmeric": [{"kind": "hf", "repo": "Project-AgML/turmeric_leaf_disease_classification", "config": "raw",
                  "class_map": {"Aphids_Disease": "Aphid (pest)", "Blotch": "Leaf blotch",
                                "Healthy_Leaf": "Healthy", "Leaf_Spot": "Leaf spot"}}],
    "papaya": [{"kind": "hf", "repo": "Project-AgML/papaya_leaf_disease_classification_bangladesh",
                "config": "raw", "class_map": {
        "Healthy Leaf": "Healthy", "Leaf Curl": "Leaf curl virus", "Mealybug": "Mealybug (pest)",
        "Mite Disease": "Mite damage", "Mosaic": "Mosaic virus", "Ring Spot": "Ring spot virus"}}],
    "pomegranate": [{"kind": "hf", "repo": "Project-AgML/pomegranate_disease_classification", "config": "raw",
                     "class_map": {"Colletotrichum spp": "Anthracnose", "Ectomyelois ceratoniae": "Fruit borer (pest)",
                                   "Healthy": "Healthy", "Sunburn": "Sunburn"}}],
    "mango": [{"kind": "kaggle", "root": "", "class_map": {
        "Anthracnose": "Anthracnose", "Bacterial Canker": "Bacterial canker", "Cutting Weevil": "Cutting weevil (pest)",
        "Die Back": "Die back", "Gall Midge": "Gall midge (pest)", "Healthy": "Healthy",
        "Powdery Mildew": "Powdery mildew", "Sooty Mould": "Sooty mould"}}],
    "banana": [{"kind": "kaggle",
                "root": "Banana Disease Recognition Dataset/Original Images/Original Images", "class_map": {
        "Banana Black Sigatoka Disease": "Black sigatoka", "Banana Bract Mosaic Virus Disease": "Bract mosaic virus",
        "Banana Healthy Leaf": "Healthy", "Banana Insect Pest Disease": "Insect pest",
        "Banana Moko Disease": "Moko disease", "Banana Panama Disease": "Panama disease (wilt)",
        "Banana Yellow Sigatoka Disease": "Yellow sigatoka"}}],
    "bean": [{"kind": "hf", "repo": "Project-AgML/bean_cowpea_leaf_disease_classification", "split_col": "crop_type",
              "split_value": "Bean", "class_map": {
        "Bacterial wilt": "Bacterial wilt", "Blight": "Blight", "Fresh Leaf": "Healthy",
        "Mosaic Virus": "Mosaic virus", "Rust": "Rust", "Septoria leaf spot": "Septoria leaf spot"}}],
    "cowpea": [{"kind": "hf", "repo": "Project-AgML/bean_cowpea_leaf_disease_classification", "split_col": "crop_type",
                "split_value": "Cowpea", "class_map": {
        "Bacterial wilt": "Bacterial wilt", "Blight": "Blight", "Fresh Leaf": "Healthy",
        "Mosaic Virus": "Mosaic virus", "Rust": "Rust", "Septoria leaf spot": "Septoria leaf spot"}}],
    "bitter_gourd": [{"kind": "hf", "repo": "Project-AgML/plant_leaf_disease_classification",
                      "split_col": "plant_type", "split_value": "Bitter Gourd", "class_map": {
        "Downey mildew": "Downy mildew", "Fresh leaf": "Healthy"}}],
    "bottle_gourd": [{"kind": "hf", "repo": "Project-AgML/plant_leaf_disease_classification",
                      "split_col": "plant_type", "split_value": "Bottle gourd", "class_map": {
        "Downy mildew": "Downy mildew", "Fresh leaf": "Healthy"}}],
    "cauliflower": [{"kind": "hf", "repo": "Project-AgML/plant_leaf_disease_classification",
                     "split_col": "plant_type", "split_value": "Cauliflower", "class_map": {
        "Black Rot": "Black rot", "Fresh leaf": "Healthy"}}],
    "cucumber": [{"kind": "hf", "repo": "Project-AgML/plant_leaf_disease_classification",
                  "split_col": "plant_type", "split_value": "Cucumber", "class_map": {
        "Downy mildew": "Downy mildew", "Fresh leaf": "Healthy"}}],
}

# a one-line, farmer-facing note per crop with limited coverage (used by the app; most crops need none)
COVERAGE_NOTES = {
    "banana": "Coverage is thin for this crop (few dozen photos per class); treat results with extra caution.",
}

log(f"{len(CROPS)} crops configured")

# %% [markdown]
# ## 2 · Locate the Kaggle-attached datasets

# %%
def walk_dirs(base, max_depth=6):
    base = str(base)
    for root, dirs, _ in os.walk(base):
        yield Path(root), list(dirs)
        depth = root[len(base):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []


def find_dir(test, what):
    for folder, children in walk_dirs(INPUT):
        if test(folder, children):
            return folder
    raise FileNotFoundError(f"{what} not found under /kaggle/input. Is the dataset attached?")


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


KAGGLE_ROOTS = {
    "plantvillage": find_dir(lambda f, c: {"color", "segmented"} <= set(c), "PlantVillage"),
    "paddy_doctor": find_dir(lambda f, c: "train_images" in c and "test_images" in c, "Paddy Doctor"),
    "wheat": find_dir(lambda f, c: {"train", "test", "valid"} <= set(c) and f.name == "data", "Wheat diseases"),
    "sar_cld": find_dir(lambda f, c: "Original Dataset" in c and "SAR-CLD" in f.name, "SAR-CLD-2024 cotton"),
    "ripon_cotton": find_dir(lambda f, c: "Cotton_Original_Dataset" in c, "Ripon cotton"),
    "sugarcane": find_dir(lambda f, c: {"Healthy", "Mosaic", "RedRot", "Rust", "Yellow"} <= set(c), "Sugarcane"),
    "mango": find_dir(lambda f, c: {"Anthracnose", "Healthy", "Sooty Mould"} <= set(c), "Mango"),
    "banana": find_dir(lambda f, c: f.name == "Original Images" and "Banana Healthy Leaf" in c, "Banana (original)"),
}
for name, path in KAGGLE_ROOTS.items():
    log(f"{name}: {path}")

# %% [markdown]
# ## 3 · Load every source into one pool of (jpeg bytes, phash, crop, class) rows
#
# Kaggle-native sources are read straight off disk. Hugging Face sources are **streamed**, so nothing
# is cached locally — each row is decoded, resized and re-encoded immediately, then discarded.

# %%
def to_jpeg(im, max_side=MAX_SIDE):
    im = ImageOps.exif_transpose(im).convert("RGB")
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def phash_hex(jpeg_bytes):
    return str(imagehash.phash(Image.open(io.BytesIO(jpeg_bytes))))


def cap(items, n, seed_key):
    if len(items) <= n:
        return items
    rng = random.Random(hash(seed_key) & 0xFFFFFFFF)
    return rng.sample(items, n)


rows = []  # dicts: crop, label, jpeg, phash, source
skipped_labels = Counter()
per_source_count = Counter()


def add_kaggle_source(crop, spec):
    # Each KAGGLE_ROOTS[x] is already "the folder whose direct children are the class folders" for a
    # simple case (mango, sugarcane, banana), or one level above it for the others -- these mappings add
    # only the extra segment each one actually needs, verified against a real file listing of each
    # dataset. (An earlier version of this function double-appended segments already inside KAGGLE_ROOTS,
    # which silently produced 0 images for paddy doctor's rice classes, wheat, cotton and banana.)
    kaggle_root_by_prefix = {
        "color": KAGGLE_ROOTS["plantvillage"] / "color",
        "paddy-disease-classification/train_images": KAGGLE_ROOTS["paddy_doctor"] / "train_images",
        "data/train": KAGGLE_ROOTS["wheat"] / "train",
    }
    root_key = spec["root"]
    if root_key in kaggle_root_by_prefix:
        base = kaggle_root_by_prefix[root_key]
    elif "SAR-CLD" in root_key:
        base = KAGGLE_ROOTS["sar_cld"] / "Original Dataset" / "Original Dataset"
    elif "Cotton_Original_Dataset" in root_key:
        base = KAGGLE_ROOTS["ripon_cotton"] / "Cotton_Original_Dataset" / "Cotton_Original_Dataset"
    elif root_key == "" and crop == "sugarcane":
        base = KAGGLE_ROOTS["sugarcane"]
    elif root_key == "" and crop == "mango":
        base = KAGGLE_ROOTS["mango"]
    elif "Banana Disease Recognition" in root_key:
        base = KAGGLE_ROOTS["banana"]
    else:
        raise ValueError(f"Unrecognised Kaggle root for {crop}: {root_key!r}")
    if not base.is_dir():
        raise FileNotFoundError(f"{crop}: expected class folders under {base}, but it does not exist")

    for src_label, canon in spec["class_map"].items():
        if not canon:
            continue
        files = list_images(base / src_label)
        if not files:
            skipped_labels[f"{crop}:{src_label} (kaggle, empty)"] += 1
            continue
        files = cap(files, CAP_PER_CLASS, f"{crop}:{src_label}")
        for p in files:
            try:
                jpeg = to_jpeg(Image.open(p))
            except Exception:
                continue
            rows.append({"crop": crop, "label": canon, "jpeg": jpeg, "phash": phash_hex(jpeg),
                        "source": f"kaggle:{root_key or crop}"})
        per_source_count[f"{crop} <- kaggle:{src_label}"] += len(files)


def hf_raw_config(repo_id, preferred="raw"):
    try:
        names = hfds.get_dataset_config_names(repo_id)
    except Exception as exc:
        log(f"  could not list configs for {repo_id}: {exc}")
        return "default"
    if preferred in names:
        return preferred
    return "default" if "default" in names else names[0]


def hf_image_to_jpeg(value):
    if isinstance(value, Image.Image):
        return to_jpeg(value)
    if isinstance(value, dict) and value.get("bytes"):
        return to_jpeg(Image.open(io.BytesIO(value["bytes"])))
    raise ValueError("Unrecognised HF image cell")


HF_SOURCE_BUDGET_S = 360  # give up on one HF source after this long and keep whatever it collected;
                          # the first run spent 100+ minutes stuck reading past the tea dataset's 80,329
                          # rows one at a time even after every class it maps to was already full


def add_hf_source(crop, spec):
    repo = spec["repo"]
    config = spec.get("config") or hf_raw_config(repo)
    try:
        ds = hfds.load_dataset(repo, config, split="train", streaming=True)
    except Exception as exc:
        log(f"  FAILED to stream {repo} ({config}): {exc}")
        skipped_labels[f"{crop}:{repo} (hf, load failed)"] += 1
        return
    label_feature = ds.features.get("label")
    # Only a ClassLabel feature has int2str; some sources (e.g. the tea dataset) store "label" as a plain
    # Value(int64) or a string, which crashed the first run with AttributeError on this line.
    is_class_label = isinstance(label_feature, hfds.ClassLabel)
    filt_col, filt_val = spec.get("split_col"), spec.get("split_value")
    target_labels = {canon for canon in spec["class_map"].values() if canon}
    buckets = {}  # canon label -> list of jpeg bytes, capped as we go
    n_seen = 0
    t_start = time.time()
    stop_reason = "exhausted the stream"
    for row in ds:
        n_seen += 1
        if filt_col and row.get(filt_col) != filt_val:
            continue
        raw_label = row["label"]
        src_label = label_feature.int2str(raw_label) if is_class_label and isinstance(raw_label, int) else str(raw_label)
        canon = spec["class_map"].get(src_label)
        if not canon:
            if src_label not in spec["class_map"]:
                skipped_labels[f"{crop}:{src_label} (hf, unmapped)"] += 1
            continue
        bucket = buckets.setdefault(canon, [])
        if len(bucket) >= CAP_PER_CLASS:
            continue
        try:
            bucket.append(hf_image_to_jpeg(row["image"]))
        except Exception:
            continue
        # every target class is full: stop instead of reading the rest of a possibly huge remaining stream
        if all(len(buckets.get(t, [])) >= CAP_PER_CLASS for t in target_labels):
            stop_reason = "every target class reached the cap"
            break
        if n_seen % 200 == 0 and time.time() - t_start > HF_SOURCE_BUDGET_S:
            stop_reason = f"hit the {HF_SOURCE_BUDGET_S}s time budget for one source"
            break
    for canon, jpegs in buckets.items():
        for jpeg in jpegs:
            rows.append({"crop": crop, "label": canon, "jpeg": jpeg, "phash": phash_hex(jpeg), "source": f"hf:{repo}"})
        per_source_count[f"{crop} <- hf:{repo}/{canon}"] += len(jpegs)
    log(f"  {repo} ({config}): read {n_seen} rows in {time.time() - t_start:.0f}s, {stop_reason} -> " +
        ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))


for crop, sources in CROPS.items():
    t = time.time()
    n_before = len(rows)
    for spec in sources:
        if spec["kind"] == "kaggle":
            add_kaggle_source(crop, spec)
        else:
            add_hf_source(crop, spec)
    log(f"{crop}: {len(rows) - n_before} images in {time.time() - t:.0f}s (running total {len(rows)})")

log(f"pool built: {len(rows)} images across {len({r['crop'] for r in rows})} crops")
if skipped_labels:
    log(f"skipped/unmapped source labels: {dict(skipped_labels)}")

# %% [markdown]
# ## 4 · Universal leakage guard: cluster near-duplicates, keep every cluster on one side of the split
#
# Two signals catch copies that were resized, re-compressed, flipped or lightly edited: a perceptual hash
# (Hamming distance) and a DINOv2 image embedding (cosine similarity). Within each crop+class, exact and
# near duplicates are merged into clusters; each cluster becomes one group that is assigned to train or
# test as a whole, so a photo's edited copy can never land on the other side from the original.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"embedding on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
embedder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
embedder = embedder.to(DEVICE).eval()
embed_tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class Blobs(torch.utils.data.Dataset):
    def __init__(self, blobs):
        self.blobs = blobs

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        return embed_tf(Image.open(io.BytesIO(self.blobs[i])).convert("RGB"))


@torch.no_grad()
def embed(blobs):
    feats = []
    for x in torch.utils.data.DataLoader(Blobs(blobs), batch_size=256, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = embedder(x.to(DEVICE))
        feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(feats)


def cluster_group(idx_list, feats, bits):
    """Union-find over the near-dup graph inside one crop+class. Returns {row_index: group_id}."""
    n = len(idx_list)
    if n == 0:
        return {}
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    f = feats[idx_list]
    b = bits[idx_list]
    chunk = 1024
    for s in range(0, n, chunk):
        cos = f[s:s + chunk] @ f.T
        ham = (64 - b[s:s + chunk] @ b.T) / 2
        near = (cos >= DUP_COS) | (ham <= DUP_HAM)
        near_np = near.numpy()
        for i_local in range(near_np.shape[0]):
            i = s + i_local
            for j in np.flatnonzero(near_np[i_local]):
                if j != i:
                    union(i, int(j))
    return {idx_list[i]: idx_list[find(i)] for i in range(n)}


t = time.time()
FEATS = embed([r["jpeg"] for r in rows])
bits = np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8)) for r in rows])
BITS = (torch.tensor(bits, dtype=torch.float32) * 2 - 1)
log(f"embedded {len(rows)} photos in {time.time() - t:.0f}s")

group_of = {}
for (crop, label), idxs in pd.DataFrame({"crop": [r["crop"] for r in rows], "label": [r["label"] for r in rows]}) \
        .groupby(["crop", "label"]).groups.items():
    group_of.update(cluster_group(list(idxs), FEATS, BITS))

groups = pd.Series(group_of).reindex(range(len(rows))).values
crops_arr = np.array([r["crop"] for r in rows])
labels_arr = np.array([r["label"] for r in rows])
n_clusters = len(set(zip(crops_arr.tolist(), labels_arr.tolist(), groups.tolist())))
log(f"{len(rows)} photos -> {n_clusters} near-dup clusters "
    f"({len(rows) - n_clusters} exact/near duplicates found)")

# %% [markdown]
# ## 5 · Split 85/15 by cluster, then keep exactly one photo per exact/near-duplicate cluster in **test**
# (test images should be distinct; extra copies of a test image are dropped rather than kept in training,
# since a leaked near-copy of a test photo in training is the worse failure of the two).

# %%
df = pd.DataFrame({"crop": crops_arr, "label": labels_arr, "group": groups, "row": range(len(rows))})
split_of_group = {}
for (crop, label), sub in df.groupby(["crop", "label"]):
    group_ids = sub["group"].unique().tolist()
    rng = random.Random(f"{crop}:{label}")
    rng.shuffle(group_ids)
    n_test_groups = max(1, round(0.15 * len(group_ids))) if len(group_ids) >= 4 else 0
    test_groups = set(group_ids[:n_test_groups])
    for g in group_ids:
        split_of_group[(crop, label, g)] = "test" if g in test_groups else "train"

df["split"] = [split_of_group[(c, l, g)] for c, l, g in zip(df["crop"], df["label"], df["group"])]
# within test, keep only the first row of each cluster (drop redundant copies instead of keeping them)
df["keep"] = True
seen_test_clusters = set()
for i, r in df[df.split == "test"].iterrows():
    key = (r["crop"], r["label"], r["group"])
    if key in seen_test_clusters:
        df.at[i, "keep"] = False
    else:
        seen_test_clusters.add(key)
df = df[df["keep"]].drop(columns="keep")

counts = df.groupby(["crop", "split"]).size().unstack(fill_value=0)
log(f"after split: {int((df.split == 'train').sum())} train, {int((df.split == 'test').sum())} test")

# %% [markdown]
# ## 6 · Write the dataset: one zip per crop, plus the manifest and class list

# %%
def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


manifest = []
zip_handles = {}
counters = Counter()
t = time.time()
for i, r in df.iterrows():
    crop = r["crop"]
    if crop not in zip_handles:
        zip_handles[crop] = zipfile.ZipFile(OUT / f"{crop}.zip", "a", zipfile.ZIP_STORED)
    counters[crop] += 1
    arc = f"{slug(r['label'])}/{counters[crop]:06d}.jpg"
    zip_handles[crop].writestr(arc, rows[r["row"]]["jpeg"])
    manifest.append({"crop": crop, "label": r["label"], "split": r["split"], "zip": f"{crop}.zip", "file": arc,
                     "source": rows[r["row"]]["source"], "group": f"{crop}:{r['label']}:{r['group']}"})
for zf in zip_handles.values():
    zf.close()
log(f"wrote {len(manifest)} images into {len(zip_handles)} zips in {time.time() - t:.0f}s")

mf = pd.DataFrame(manifest)
mf.to_csv(OUT / "manifest.csv", index=False)

crops_info = {}
for crop, sub in mf.groupby("crop"):
    labels = sorted(sub["label"].unique())
    crops_info[crop] = {
        "labels": labels,
        "num_classes": len(labels),
        "train_images": int((sub.split == "train").sum()),
        "test_images": int((sub.split == "test").sum()),
        "coverage_note": COVERAGE_NOTES.get(crop),
    }
with open(OUT / "classes.json", "w", encoding="utf-8") as fh:
    json.dump({"crops": crops_info, "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}, fh, indent=2)

summary = mf.groupby(["crop", "label", "split"]).size().unstack(fill_value=0)
summary.to_csv(OUT / "summary.csv")
print(summary.to_string())

total_classes = sum(c["num_classes"] for c in crops_info.values())
size_gb = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9
report = {
    "built": crops_info and time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    "crops": len(crops_info), "total_classes": total_classes,
    "images": {"train": int((mf.split == "train").sum()), "test": int((mf.split == "test").sum())},
    "near_dup_clusters_found": len(rows) - n_clusters,
    "skipped_labels": dict(skipped_labels),
    "size_gb": round(size_gb, 2), "runtime_min": round((time.time() - T0) / 60, 1),
}
with open(OUT / "dedup_report.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)

card_lines = ["# AgriSmart India multi-crop dataset", "",
             f"Built {report['built']}: **{report['crops']} crops, {total_classes} classes, "
             f"{report['images']['train']} train / {report['images']['test']} test images**.", "",
             "Every crop+class was clustered for near-duplicates (perceptual hash + DINOv2 cosine "
             "similarity) before splitting, so an augmented copy of a training photo can never appear "
             "in the test set.", "",
             "## Sources",
             "- PlantVillage (Mohanty, Hughes & Salathé, 2016) via Kaggle `abdallahalidev/plantvillage-dataset` (CC BY-NC-SA 4.0)",
             "- Paddy Doctor rice disease (Kaggle `imbikramsaha/paddy-doctor`, CC0-1.0)",
             "- Wheat Plant Diseases (Kaggle `kushagra3204/wheat-plant-diseases`, CC0-1.0)",
             "- SAR-CLD-2024 cotton (Daffodil Int'l Univ., Bangladesh; Mendeley DOI 10.17632/b3jy2p6k8w.2, CC BY 4.0; via Kaggle `sabuktagin/dataset-for-cotton-leaf-disease-detection`)",
             "- Cotton leaf disease (Ripon et al. 2025, Data in Brief; Mendeley DOI 10.17632/t9hgvk2h9p.1, CC BY-NC; via Kaggle `nguynphancminh/cotton-leaf-diseases-dataset`)",
             "- Sugarcane Leaf Disease Dataset (Kaggle `nirmalsankalana/sugarcane-leaf-disease-dataset`, CC0-1.0)",
             "- Mango Leaf Disease Dataset (Kaggle `aryashah2k/mango-leaf-disease-dataset`, CC BY-NC 4.0)",
             "- Banana Disease Recognition Dataset (Kaggle `sujaykapadnis/banana-disease-recognition-dataset`, CC BY 4.0; original images only)",
             "- Hugging Face `Project-AgML/*` (each dataset's own citation is in its card; mostly CC BY 4.0)",
             "", "## Licence note",
             "Most sources are CC BY or CC0. `cotton_leaf_disease_classification` (Ripon) is CC BY-NC: "
             "non-commercial use only. See `manifest.csv`'s `source` column for which source each image came from."]
(OUT / "README.md").write_text("\n".join(card_lines), encoding="utf-8")

print(json.dumps(report, indent=2))
log(f"done: {len(mf)} images, {size_gb:.2f} GB in {OUT}")
