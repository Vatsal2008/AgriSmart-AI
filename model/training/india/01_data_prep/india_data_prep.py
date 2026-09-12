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
# that stands apart from the hackathon core model (which stays untouched). About 55 crops from
# roughly 75 sources:
# - **PlantVillage**, reused from the core model (lab photos).
# - **PlantWild** v1 and v2, web-collected field photos.
# - About 60 curated Hugging Face "Project-AgML" mirrors of published research datasets. Many of
#   them were collected in Indian fields: Maharashtra soybean, Karnataka coconut, Kashmir apple,
#   Assam banana, Andhra black gram, Indian rice and more.
# - A handful of Kaggle-native sets that are not available on Hugging Face.
#
# **How loading works.**
# 1. Every Hugging Face source is downloaded as parquet, one repo at a time, with the next repo
#    prefetched in the background. It is *not* streamed: streaming was ~4 rows/s and silently lost
#    whole classes when a time budget cut it short.
# 2. The loader reads only the label columns first and picks a seeded random sample per class.
# 3. Only those rows are decoded, in parallel on every CPU, then the repo is deleted from disk.
#
# **Universal leakage guard.**
# 1. Every kept photo gets a perceptual hash and a flip-invariant DINOv2 embedding.
# 2. Exact copies are merged, and a copy carrying two different labels is dropped.
# 3. Near-copies are clustered, for example augmented/cropped variants or burst shots of one leaf.
# 4. Every whole cluster stays on one side of the 85/15 split, so an edited copy of a training photo
#    can never be in the test set.
#
# Output: `/kaggle/working/india_data/`. Set `QUICK = True` for a 15-minute smoke test of every
# loader type.

# %%
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QUICK = False  # the smoke-test kernel flips this to True: a few crops, tiny caps, small downloads


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


for _mod in ("imagehash", "timm"):
    try:
        __import__(_mod)
    except ImportError:
        pip_install(_mod)
pip_install("-U", "huggingface_hub", "hf_xet")  # hf_xet: several-times-faster downloads from the Hub

import imagehash
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import timm
import torch
import torchvision.transforms as T
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/india_data")
STAGE = Path("/tmp/india_stage")      # resized JPEGs, before they are zipped into OUT
HF_CACHE = Path("/tmp/india_hf")      # one sub-folder per source, deleted as soon as it is processed
for d in (OUT, STAGE, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 384                 # longer side of every kept photo (training crops 224 from this)
JPEG_Q = 88
CAP_PER_SOURCE_CLASS = 2000    # at most this many photos of one class from any one source
CAP_PER_CLASS = 3000           # at most this many per crop+class, shared fairly between sources
MIN_PER_CLASS = 30             # classes smaller than this after dedupe are dropped (too few to learn/test)
EXACT_COS, EXACT_HAM = 0.985, 2   # "the same photo" (re-encoded / resized / flipped)
GROUP_COS, GROUP_HAM = 0.95, 6    # "a near-copy" (augmented crop, colour-jittered, burst shot)
MAX_CLUSTER = 64               # a near-copy cluster may not grow past this (stops chain reactions)
TEST_FRAC = 0.15
MAX_REPO_GB = 20               # skip any single Hugging Face source bigger than this (the largest is ~13 GB)
LOAD_BUDGET_MIN = 330          # stop starting new sources after this long; keep what is loaded
PARQUET_REV = "refs/convert/parquet"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
N_CPU = os.cpu_count() or 4
SEED = 42

if QUICK:
    CAP_PER_SOURCE_CLASS, CAP_PER_CLASS, MAX_REPO_GB, MIN_PER_CLASS = 40, 80, 3, 10

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


random.seed(SEED)
np.random.seed(SEED)
log(f"torch {torch.__version__} | timm {timm.__version__} | {N_CPU} CPUs | QUICK={QUICK}")

# %% [markdown]
# ## 1 · The crop and class list
#
# Each crop lists its sources. `class_map` maps a source's own label (matched case- and
# punctuation-insensitively) to one canonical class name per crop. That way two sources that name
# the same disease differently become one class, and look-alike classes that sources split
# inconsistently are merged. For example, black and yellow Sigatoka are merged, and groundnut
# early and late leaf spot become "Tikka leaf spot". Labels left out of a map are skipped on
# purpose: pathogen-group labels ("Fungi", "Bacteria"), classes from other countries' pests,
# and ambiguous classes such as "Insect" or "Diseased".
#
# Source kinds:
# - `kaggle`: files attached to this notebook.
# - `hf`: a Hugging Face dataset. `filter` keeps only the rows of one crop in a multi-crop repo.
# - `zip`: a zip file inside a Hugging Face repo (PlantWild).

# %%
PV_DS = "abdallahalidev/plantvillage-dataset"


def pv(class_map):
    """PlantVillage (lab photos, plain backgrounds)."""
    return {"kind": "kaggle", "ds": PV_DS, "find": {"has": ["color", "segmented"]}, "subdirs": ["color"],
            "domain": "lab", "class_map": class_map}


def hf(repo, class_map, config=None, filt=None, domain="mixed"):
    """A Project-AgML (or other) Hugging Face image-classification dataset."""
    return {"kind": "hf", "repo": repo, "config": config, "filter": filt, "domain": domain, "class_map": class_map}


def kg(ds, find, class_map, subdirs=("",), domain="mixed"):
    return {"kind": "kaggle", "ds": ds, "find": find, "subdirs": list(subdirs), "domain": domain,
            "class_map": class_map}


def pw(class_map):
    """PlantWild v1 + v2 field photos (web-collected). v2 is a re-curated superset of v1's disease
    classes; v1 alone has the healthy "<crop> leaf" folders. Photos in both are merged by the dedupe."""
    return [{"kind": "zip", "repo": "uqtwei2/PlantWild", "file": f, "domain": "field", "class_map": class_map}
            for f in ("plantwild.zip", "plantwild_v2.zip")]


AGML = "Project-AgML/"
GOURD_MAP = {"Anthracnose": "Anthracnose", "Anthracnose lesions": "Anthracnose", "Black Rot": "Black rot",
             "Downey mildew": "Downy mildew", "Downy mildew": "Downy mildew", "Fresh leaf": "Healthy",
             "Fusarium wilt": "Fusarium wilt", "Mosaic virus": "Mosaic virus"}

CROPS = {
    # ------------------------------------------------------------ cereals
    "rice": [
        kg("imbikramsaha/paddy-doctor", {"has": ["train_images", "test_images"]}, subdirs=["train_images"], class_map={
            "bacterial_leaf_blight": "Bacterial leaf blight", "bacterial_leaf_streak": "Bacterial leaf streak",
            "bacterial_panicle_blight": "Bacterial panicle blight", "blast": "Blast", "brown_spot": "Brown spot",
            "dead_heart": "Dead heart (stem borer)", "downy_mildew": "Downy mildew", "hispa": "Hispa (pest)",
            "normal": "Healthy", "tungro": "Tungro"}),
        hf(AGML + "rice_leaf_disease_classification_india", {
            "Bacterialblight": "Bacterial leaf blight", "Blast": "Blast", "Brownspot": "Brown spot", "Tungro": "Tungro"}),
        hf(AGML + "rice_leaf_disease_classification", {
            "Bacterial_Leaf_Blight": "Bacterial leaf blight", "Brown_Spot": "Brown spot", "Healthy_Rice_Leaf": "Healthy",
            "Leaf_Blast": "Blast", "Leaf_Scald": "Leaf scald", "Sheath_Blight": "Sheath blight"}),
        hf(AGML + "rice_disease_classification_bangladesh", config="raw", class_map={
            "Healthy": "Healthy", "Leaf Scald": "Leaf scald", "Rice Blast": "Blast",
            "Rice Leaffolder": "Leaf folder (pest)", "Rice Tungro": "Tungro"}),
        *pw({"rice blast": "Blast", "rice sheath blight": "Sheath blight", "rice leaf": "Healthy"}),
    ],
    "wheat": [
        kg("kushagra3204/wheat-plant-diseases", {"has": ["train", "test", "valid"], "name": "data"},
           subdirs=["train", "valid", "test"], class_map={
            "Aphid": "Aphid (pest)", "Black Rust": "Black (stem) rust", "Blast": "Blast", "Brown Rust": "Brown (leaf) rust",
            "Common Root Rot": "Common root rot", "Fusarium Head Blight": "Fusarium head blight", "Healthy": "Healthy",
            "Leaf Blight": "Leaf blight", "Mildew": "Powdery mildew", "Mite": "Mite (pest)", "Septoria": "Septoria blotch",
            "Smut": "Smut", "Stem fly": "Stem fly (pest)", "Tan spot": "Tan spot", "Yellow Rust": "Yellow (stripe) rust"}),
        *pw({"wheat bacterial leaf streak (black chaff)": "Bacterial leaf streak", "wheat head scab": "Fusarium head blight",
             "wheat leaf rust": "Brown (leaf) rust", "wheat loose smut": "Smut", "wheat powdery mildew": "Powdery mildew",
             "wheat septoria blotch": "Septoria blotch", "wheat stem rust": "Black (stem) rust",
             "wheat stripe rust": "Yellow (stripe) rust"}),
    ],
    "finger_millet_ragi": [kg("prajwalbax/finger-millet-ragi-dataset", {"has": ["downy", "healthy", "smut"]},
                              class_map={"downy": "Downy mildew", "healthy": "Healthy", "mottle": "Mottle streak virus",
                                         "smut": "Smut", "wilt": "Wilt"})],
    "corn_maize": [
        pv({"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot",
            "Corn_(maize)___Common_rust_": "Common rust", "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight",
            "Corn_(maize)___healthy": "Healthy"}),
        hf(AGML + "TOM2024_disease_classification", config="raw", filt={"col": "crop_type", "values": ["maize", "corn"]},
           class_map={"curvulariosis-d": "Curvularia leaf spot", "helminthosporiosis-d": "Northern leaf blight",
                      "rust-d": "Common rust", "spodoptera-frugiperda-a": "Fall armyworm (pest)",
                      "spodoptera-frugiperda-p": "Fall armyworm (pest)", "healthy-leaf": "Healthy"}),
        hf(AGML + "crop_pest_disease_classification", config="raw", filt={"col": None, "values": ["maize"]},
           class_map={"fall armyworm": "Fall armyworm (pest)", "grasshoper": "Grasshopper (pest)", "healthy": "Healthy",
                      "leaf beetle": "Leaf beetle (pest)", "leaf blight": "Northern leaf blight"}),
        *pw({"corn gray leaf spot": "Gray leaf spot", "corn northern leaf blight": "Northern leaf blight",
             "corn rust": "Common rust", "corn smut": "Common smut", "corn leaf": "Healthy"}),
    ],

    # ------------------------------------------------------------ pulses and oilseeds
    "soybean": [
        pv({"Soybean___healthy": "Healthy"}),
        hf(AGML + "MH_SoyaHealthVision_disease_classification_leaf", {
            "Caterpillar_Semilooper_Pest": "Semilooper caterpillar (pest)", "Frog_Leaf_Eye": "Frogeye leaf spot",
            "Healthy": "Healthy", "Mosaic": "Mosaic virus", "Rust": "Rust", "Spectoria_Brown_Spot": "Septoria brown spot"}),
        hf(AGML + "soybean_leaf_disease_classification", {
            "Bacterial leaf Blight": "Bacterial blight", "Dry_leaf": "Dry leaf", "Healthy": "Healthy",
            "Septoria_Brown_Spot": "Septoria brown spot", "Vein Necrosis": "Vein necrosis"}),
        hf(AGML + "soybean_leaf_disease_classification_brazil", {
            "Caterpillar": "Semilooper caterpillar (pest)", "Healthy": "Healthy"}),
        *pw({"soybean bacterial blight": "Bacterial blight", "soybean brown spot": "Septoria brown spot",
             "soybean downy mildew": "Downy mildew", "soybean frog eye leaf spot": "Frogeye leaf spot",
             "soybean mosaic": "Mosaic virus", "soybean rust": "Rust", "soybean leaf": "Healthy"}),
    ],
    "groundnut": [
        # early and late leaf spot are the two "tikka" diseases; sources split them inconsistently -> one class
        hf(AGML + "groundnut_leaf_disease_classification", {
            "ALTERNARIA LEAF SPOT": "Alternaria leaf spot", "HEALTHY": "Healthy", "LEAF SPOT": "Tikka leaf spot",
            "ROSETTE": "Rosette virus", "RUST": "Rust"}),
        hf(AGML + "groundnut_leaf_disease_classification_2", {
            "early_leaf_spot": "Tikka leaf spot", "healthy leaf": "Healthy", "late leaf spot": "Tikka leaf spot",
            "nutrition deficiency": "Nutrient deficiency", "rust": "Rust"}),
    ],
    "black_gram": [
        hf(AGML + "black_gram_disease_classification", {
            "Cercospora leaf spot": "Cercospora leaf spot", "Healthy": "Healthy", "Insect": "Insect pest damage",
            "Leaf Crinkle": "Leaf crinkle virus", "Yellow Mosaic": "Yellow mosaic virus"}),
        hf(AGML + "blackgram_plant_leaf_disease_classification", {
            "anthracnose": "Anthracnose", "healthy": "Healthy", "leaf_crinckle": "Leaf crinkle virus",
            "powdery_mildew": "Powdery mildew", "yellow_mosaic": "Yellow mosaic virus"}),
    ],
    "lentil": [hf(AGML + "lentil_disease_classification", {
        "Ascochyta blight": "Ascochyta blight", "Lentil Rust": "Rust", "Normal": "Healthy", "Powdery Mildew": "Powdery mildew"})],
    "bean": [
        hf(AGML + "bean_cowpea_leaf_disease_classification", filt={"col": "crop_type", "values": ["bean"]}, class_map={
            "Bacterial wilt": "Bacterial wilt", "Blight": "Blight", "Fresh Leaf": "Healthy", "Mosaic Virus": "Mosaic virus",
            "Rust": "Rust", "Septoria leaf spot": "Septoria leaf spot"}),
        hf(AGML + "bean_disease_classification_tanzania", {
            "anthracnose": "Anthracnose", "healthy": "Healthy", "rust": "Rust"}),
        *pw({"bean halo blight": "Halo blight", "bean mosaic virus": "Mosaic virus", "bean rust": "Rust",
             "bean leaf": "Healthy"}),
    ],
    "cowpea": [hf(AGML + "bean_cowpea_leaf_disease_classification", filt={"col": "crop_type", "values": ["cowpea"]},
                  class_map={"Bacterial wilt": "Bacterial wilt", "Blight": "Blight", "Fresh Leaf": "Healthy",
                             "Mosaic Virus": "Mosaic virus", "Rust": "Rust", "Septoria leaf spot": "Septoria leaf spot"})],
    "sunflower": [hf(AGML + "sunflower_disease_classification", {
        "Downy_mildew": "Downy mildew", "Fresh_leaf": "Healthy", "Gray_mold": "Gray mold", "Leaf_scars": "Leaf scars"})],

    # ------------------------------------------------------------ cash and plantation crops
    "cotton": [
        kg("sabuktagin/dataset-for-cotton-leaf-disease-detection", {"has": ["Original Dataset"], "name_contains": "SAR-CLD"},
           subdirs=["Original Dataset/Original Dataset"], class_map={
            "Bacterial Blight": "Bacterial blight", "Curl Virus": "Leaf curl virus", "Healthy Leaf": "Healthy",
            "Herbicide Growth Damage": "Herbicide damage", "Leaf Hopper Jassids": "Leafhopper / jassid (pest)",
            "Leaf Redding": "Leaf reddening", "Leaf Variegation": "Leaf variegation"}),
        kg("nguynphancminh/cotton-leaf-diseases-dataset", {"has": ["Cotton_Original_Dataset"]},
           subdirs=["Cotton_Original_Dataset/Cotton_Original_Dataset"], class_map={
            "Alternaria Leaf Spot": "Alternaria leaf spot", "Bacterial Blight": "Bacterial blight",
            "Fusarium Wilt": "Fusarium wilt", "Healthy Leaf": "Healthy", "Verticillium Wilt": "Verticillium wilt"}),
    ],
    "sugarcane": [
        kg("nirmalsankalana/sugarcane-leaf-disease-dataset", {"has": ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]},
           class_map={"Healthy": "Healthy", "Mosaic": "Mosaic virus", "RedRot": "Red rot", "Rust": "Rust",
                      "Yellow": "Yellow leaf disease"}),
        hf(AGML + "sugarcane_leaf_disease_classification", {
            "Banded Chlorosis": "Banded chlorosis", "Brown Spot": "Brown spot", "BrownRust": "Rust", "Dried": "Dried leaf",
            "Grassy shoot": "Grassy shoot", "Healthy": "Healthy", "Pokkah Boeng": "Pokkah boeng", "Sett Rot": "Sett rot",
            "Smut": "Smut", "Viral Disease": "Mosaic virus", "Yellow Leaf": "Yellow leaf disease"}),
    ],
    "tea": [
        hf(AGML + "tea_leaf_disease_classification", {
            "algal_spot": "Algal leaf spot", "brown_blight": "Brown blight", "gray_blight": "Gray blight",
            "healthy": "Healthy", "helopeltis": "Tea mosquito bug (helopeltis)", "red_spot": "Red leaf spot"}),
        hf(AGML + "teaLeafBD_disease_classification", {
            "Brown Blight": "Brown blight", "Gray Blight": "Gray blight", "Green mirid bug": "Green mirid bug (pest)",
            "Healthy leaf": "Healthy", "Helopeltis": "Tea mosquito bug (helopeltis)", "Red spider": "Red spider mite (pest)",
            "Tea algal leaf spot": "Algal leaf spot"}),
        hf(AGML + "tea_leaf_disease_classification_bangladesh", {
            "Healthy": "Healthy", "Tea red leaf spot": "Red leaf spot"}),
        # TeaLeafNet: Assam tea gardens, leaves photographed on a black background
        kg("harjindersinghdibru/tealeafnet", {"has": ["BB", "GL", "RR", "RSM"]}, class_map={
            "BB": "Blister blight", "GL": "Healthy", "RR": "Red rust (algal)", "RSM": "Red spider mite (pest)"}),
    ],
    "coffee": [
        hf(AGML + "arabica_coffee_leaf_disease_classification", {
            "Cerscospora": "Cercospora leaf spot", "Healthy": "Healthy", "Leaf_rust": "Leaf rust",
            "Miner": "Leaf miner (pest)", "Phoma": "Phoma leaf spot"}),
        *pw({"coffee leaf rust": "Leaf rust", "coffee brown eye spot": "Cercospora leaf spot", "coffee leaf": "Healthy"}),
    ],
    "jute": [hf(AGML + "jute_disease_classification", {
        "Dieback": "Dieback", "Fresh": "Healthy", "Holed": "Pest-holed leaf", "Mosaic": "Mosaic virus",
        "Stem Soft Rot": "Stem rot"})],
    "tobacco": [*pw({"tobacco brown spot": "Brown spot", "tobacco frogeye leaf spot": "Frogeye leaf spot",
                     "tobacco mosaic virus": "Mosaic virus", "tobacco leaf": "Healthy"})],
    "coconut": [hf(AGML + "coconut_tree_disease_classification", {
        "Bud_Root_Dropping": "Bud root dropping", "Bud_Rot": "Bud rot", "Gray_Leaf_Spot": "Grey leaf spot",
        "Leaf_Rot": "Leaf rot", "Stem_Bleeding": "Stem bleeding"})],
    "cashew": [hf(AGML + "crop_pest_disease_classification", config="raw", filt={"col": None, "values": ["cashew"]},
                  class_map={"anthracnose": "Anthracnose", "gumosis": "Gummosis", "healthy": "Healthy",
                             "leaf miner": "Leaf miner (pest)", "red rust": "Red rust (algal)"})],
    "betel": [
        hf(AGML + "betel_leaf_disease_classification", {
            "Bacterial_Leaf_Disease": "Bacterial leaf disease", "Dried_Leaf": "Dried leaf",
            "Fungal_Brown_Spot_Disease": "Fungal brown spot", "Healthy_Leaf": "Healthy"}),
        hf(AGML + "betel_leaf_disease_classification_2", config="raw", class_map={
            "Healthy_Leaf": "Healthy", "Leaf_Rot": "Leaf rot", "Leaf_Spot": "Leaf spot"}),
    ],

    # ------------------------------------------------------------ spices
    "turmeric": [
        hf(AGML + "turmeric_leaf_disease_classification", config="raw", class_map={
            "Aphids_Disease": "Aphid (pest)", "Blotch": "Leaf blotch", "Healthy_Leaf": "Healthy", "Leaf_Spot": "Leaf spot"}),
        hf(AGML + "turmeric_disease_classification", config="raw", class_map={
            "Dry Leaf": "Dry leaf", "Healthy Leaf": "Healthy", "Leaf Blotch": "Leaf blotch",
            "Rhizome Disease Root": "Rhizome rot", "Rhizome Healthy Root": "Healthy rhizome"}),
    ],
    "chilli": [hf(AGML + "COLD_chili_leaf_disease_classification", config="raw", class_map={
        "cercospora": "Cercospora leaf spot", "healthy": "Healthy", "mites_and_trips": "Mites and thrips (pest)",
        "nutritional": "Nutrient deficiency", "powdery mildew": "Powdery mildew"})],
    "ginger": [*pw({"ginger leaf spot": "Leaf spot", "ginger sheath blight": "Sheath blight", "ginger leaf": "Healthy"})],
    "garlic": [*pw({"garlic leaf blight": "Leaf blight", "garlic rust": "Rust", "garlic leaf": "Healthy"})],

    # ------------------------------------------------------------ vegetables
    "tomato": [
        pv({"Tomato___Bacterial_spot": "Bacterial spot", "Tomato___Early_blight": "Early blight",
            "Tomato___Late_blight": "Late blight", "Tomato___Leaf_Mold": "Leaf mold",
            "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites (pest)", "Tomato___Target_Spot": "Target spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus",
            "Tomato___Tomato_mosaic_virus": "Mosaic virus", "Tomato___healthy": "Healthy"}),
        hf(AGML + "TOM2024_disease_classification", config="raw", filt={"col": "crop_type", "values": ["tomato"]},
           class_map={"alternaria-d": "Early blight", "blossom-end-rot-d": "Blossom end rot", "fusarium-d": "Fusarium wilt",
                      "healthy-leaf": "Healthy", "healthy-fruit": "Healthy", "tomato-late-blight-d": "Late blight",
                      "tuta-absoluta-p": "Tuta absoluta leaf miner (pest)",
                      "helicoverpa-armigera-p": "Fruit borer (pest)", "mite-d": "Spider mites (pest)",
                      "sunburn-d": "Sunscald"}),
        hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["tomato"]}, class_map={
            "Tomato Bacterial spot": "Bacterial spot", "Tomato Fresh leaf": "Healthy",
            "Tomato leaf curl virus": "Yellow leaf curl virus", "Tomato spotted wilt": "Spotted wilt virus"}),
        hf(AGML + "crop_pest_disease_classification", config="raw", filt={"col": None, "values": ["tomato"]},
           class_map={"healthy": "Healthy", "leaf curl": "Yellow leaf curl virus",
                      "septoria leaf spot": "Septoria leaf spot", "verticillium wilt": "Verticillium wilt"}),
        # Tomato-Village: photos from Indian tomato fields (Variant-a = its single-label classification split)
        kg("mamtag/tomato-village", {"has": ["train", "val", "test"], "name": "Variant-a(Multiclass Classification)"},
           subdirs=["train", "val", "test"], domain="field", class_map={
            "Early_blight": "Early blight", "Healthy": "Healthy", "Late_blight": "Late blight",
            "Leaf Miner": "Leaf miner (pest)", "Magnesium Deficiency": "Magnesium deficiency",
            "Nitrogen Deficiency": "Nitrogen deficiency", "Pottassium Deficiency": "Potassium deficiency",
            "Spotted Wilt Virus": "Spotted wilt virus"}),
        *pw({"tomato bacterial leaf spot": "Bacterial spot", "tomato early blight": "Early blight",
             "tomato late blight": "Late blight", "tomato leaf mold": "Leaf mold", "tomato mosaic virus": "Mosaic virus",
             "tomato septoria leaf spot": "Septoria leaf spot", "tomato yellow leaf curl virus": "Yellow leaf curl virus",
             "tomato leaf": "Healthy"}),
    ],
    "potato": [
        pv({"Potato___Early_blight": "Early blight", "Potato___Late_blight": "Late blight", "Potato___healthy": "Healthy"}),
        hf(AGML + "potato_leaf_disease_classification", {
            "Phytopthora": "Late blight", "Healthy": "Healthy", "Virus": "Virus disease", "Pest": "Insect pest damage"}),
        *pw({"potato early blight": "Early blight", "potato late blight": "Late blight", "potato leaf": "Healthy"}),
    ],
    "onion": [
        hf(AGML + "COLD_onion_leaf_disease_classification", config="raw", class_map={
            "Iris yellow virus": "Iris yellow spot virus",
            "Stemphylium leaf blight and collectrichum leaf blight": "Stemphylium / Colletotrichum blight",
            "healthy": "Healthy", "purple blotch": "Purple blotch"}),
        hf(AGML + "TOM2024_disease_classification", config="raw", filt={"col": "crop_type", "values": ["onion"]},
           class_map={"alternaria-d": "Purple blotch", "fusarium-d": "Fusarium basal rot", "healthy-leaf": "Healthy",
                      "caterpillar-p": "Caterpillar (pest)"}),
        # Indian onion set (partly overlaps TOM2024's onion photos -- the dedupe merges those); its
        # "Iris yellow virus_augment" folder is pre-augmented, so it is left out
        kg("tejasbargujepatil/onion-diseases", {"has": ["Healthy leaves", "Purple blotch"]}, class_map={
            "Alternaria_D": "Purple blotch", "Botrytis Leaf Blight": "Botrytis leaf blight",
            "Caterpillar-P": "Caterpillar (pest)", "Downy mildew": "Downy mildew", "Fusarium-D": "Fusarium basal rot",
            "Healthy leaves": "Healthy", "Purple blotch": "Purple blotch", "Rust": "Rust",
            "stemphylium Leaf Blight": "Stemphylium / Colletotrichum blight",
            "Xanthomonas Leaf Blight": "Xanthomonas leaf blight"}),
    ],
    "brinjal_eggplant": [
        hf(AGML + "eggplant_disease_classification", config="raw", class_map={
            "Healthy Leaf": "Healthy", "Insect Pest Disease": "Insect pest damage", "Leaf Spot Disease": "Cercospora leaf spot",
            "Mosaic Virus Disease": "Mosaic virus", "White Mold Disease": "White mold", "Wilt Disease": "Wilt"}),
        hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["eggplant"]}, class_map={
            "Eggplant Cercopora leaf spot": "Cercospora leaf spot", "Eggplant begomovirus": "Begomovirus leaf curl",
            "Eggplant fresh leaf": "Healthy", "Eggplant verticillium wilt": "Wilt"}),
        hf(AGML + "BrinjalFruitX_disease_classification", {
            "Brinjal Fruit Creaking": "Fruit cracking", "Healty Brinjal": "Healthy", "Phomopsis Bright": "Phomopsis blight",
            "Shoot and Fruit Borer": "Shoot and fruit borer (pest)", "Wet Rot": "Fruit rot"}),
        *pw({"eggplant cercospora leaf spot": "Cercospora leaf spot", "eggplant phomopsis fruit rot": "Phomopsis blight",
             "eggplant leaf": "Healthy"}),
    ],
    "okra": [
        hf(AGML + "OkraDiseaseNet_disease_classification", {
            "Alternaria Leaf Spot": "Alternaria leaf spot", "Cercospora Leaf Spot": "Cercospora leaf spot",
            "Downy Mildew": "Downy mildew", "Healthy": "Healthy", "Leaf curly virus": "Leaf curl virus",
            "Phyllosticta leaf spot": "Phyllosticta leaf spot"}),
        # Indian field photos of okra's most damaging disease; "diseased" in this set is yellow vein mosaic
        kg("manojgadde/yellow-vein-mosaic-disease", {"has": ["train", "val", "test"], "name": "data"},
           subdirs=["train", "val", "test"], class_map={
            "diseased okra leaf": "Yellow vein mosaic virus", "fresh okra leaf": "Healthy"}),
    ],
    "cauliflower": [
        hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["cauliflower"]},
           class_map=GOURD_MAP),
        hf(AGML + "cauliflower_leaf_disease_classification", {
            "Black Rot": "Black rot", "Healthy": "Healthy", "Insect Hole": "Insect pest damage"}),
        hf(AGML + "VegNet_cauliflower_disease_classification", config="raw", class_map={
            "Bacterial spot rot": "Bacterial soft rot", "Black Rot": "Black rot", "No disease": "Healthy",
            "Disease Free": "Healthy", "Downy Mildew": "Downy mildew"}),
        *pw({"cauliflower alternaria leaf spot": "Alternaria leaf spot", "cauliflower bacterial soft rot": "Bacterial soft rot",
             "cauliflower leaf": "Healthy"}),
    ],
    "cabbage": [*pw({"cabbage alternaria leaf spot": "Alternaria leaf spot", "cabbage black rot": "Black rot",
                     "cabbage downy mildew": "Downy mildew", "cabbage leaf": "Healthy"})],
    "radish": [hf(AGML + "radish_leaf_disease_classification", {
        "Black leaf spot": "Black leaf spot", "Downey mildew": "Downy mildew", "Fresh leaf": "Healthy",
        "Mosaic virus": "Mosaic virus", "flea beetle": "Flea beetle (pest)"})],
    "spinach": [hf(AGML + "IDDMSLD_spinach_leaf_disease_classification", {
        "Anthracnose": "Anthracnose", "Bacterial-Spot": "Bacterial spot", "Downy-Mildew": "Downy mildew",
        "Healthy-Leaf": "Healthy", "Pest-Damage": "Insect pest damage"})],
    "malabar_spinach": [hf(AGML + "malabar_spinach_disease_classification", config="raw", class_map={
        "anthracnose_leaf_spot": "Anthracnose leaf spot", "healthy": "Healthy", "straw_mite": "Mite damage"})],
    "cucumber": [
        hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["cucumber"]},
           class_map=GOURD_MAP),
        hf(AGML + "cucumber_disease_classification", {
            "Anthracnose": "Anthracnose", "Bacterial_Wilt": "Bacterial wilt", "Belly_Rot": "Belly rot (fruit)",
            "Downy_Mildew": "Downy mildew", "Fresh_Cucumber": "Healthy", "Fresh_Leaf": "Healthy",
            "Gummy_Stem_Blight": "Gummy stem blight", "Pythium_Fruit_Rot": "Pythium fruit rot"}),
        *pw({"cucumber angular leaf spot": "Angular leaf spot", "cucumber bacterial wilt": "Bacterial wilt",
             "cucumber powdery mildew": "Powdery mildew", "cucumber leaf": "Healthy"}),
    ],
    "bitter_gourd": [hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["bitter gourd"]},
                        class_map=GOURD_MAP)],
    "bottle_gourd": [
        hf(AGML + "plant_leaf_disease_classification", filt={"col": "plant_type", "values": ["bottle gourd"]},
           class_map=GOURD_MAP),
        hf(AGML + "AgriVision4_disease_classification", config="raw", filt={"col": "crop", "values": ["bottle gourd"]},
           class_map={"Alternaria_Leaf_Blight": "Alternaria leaf blight", "Angular_Leaf_Spot": "Angular leaf spot",
                      "Anthracnose": "Anthracnose", "Downy_Mildew": "Downy mildew", "Downy": "Downy mildew",
                      "Healthy": "Healthy", "Healthy_leaf": "Healthy", "Mosaic": "Mosaic virus",
                      "Mosaic_Virus": "Mosaic virus", "Yellow_Mosaic_Virus": "Mosaic virus"}),
    ],
    "pumpkin": [kg("tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh",
                   {"has": ["Healthy Leaf", "Mosaic Disease"], "name": "Original Dataset"}, class_map={
        "Bacterial Leaf Spot": "Bacterial leaf spot", "Downy Mildew": "Downy mildew", "Healthy Leaf": "Healthy",
        "Mosaic Disease": "Mosaic virus", "Powdery_Mildew": "Powdery mildew"})],
    "ash_gourd": [hf(AGML + "ash_gourd_disease_classification", config="raw", class_map={
        "Aphid": "Aphid (pest)", "Downy mildew": "Downy mildew", "Healthy": "Healthy", "Leaf curl": "Leaf curl virus",
        "Leaf miner": "Leaf miner (pest)"})],
    "squash": [
        pv({"Squash___Powdery_mildew": "Powdery mildew"}),
        *pw({"squash powdery mildew": "Powdery mildew", "squash leaf": "Healthy"}),
    ],
    "watermelon": [hf(AGML + "watermelon_disease_classification", config="raw", class_map={
        "Anthracnose": "Anthracnose", "Downy_Mildew": "Downy mildew", "Healthy": "Healthy", "Mosaic_Virus": "Mosaic virus"})],
    "bell_pepper": [
        pv({"Pepper,_bell___Bacterial_spot": "Bacterial spot", "Pepper,_bell___healthy": "Healthy"}),
        *pw({"bell pepper bacterial spot": "Bacterial spot", "bell pepper blossom end rot": "Blossom end rot",
             "bell pepper leaf": "Healthy"}),
    ],

    # ------------------------------------------------------------ fruits
    "mango": [kg("aryashah2k/mango-leaf-disease-dataset", {"has": ["Anthracnose", "Healthy", "Sooty Mould"]}, class_map={
        "Anthracnose": "Anthracnose", "Bacterial Canker": "Bacterial canker", "Cutting Weevil": "Cutting weevil (pest)",
        "Die Back": "Die back", "Gall Midge": "Gall midge (pest)", "Healthy": "Healthy",
        "Powdery Mildew": "Powdery mildew", "Sooty Mould": "Sooty mould"})],
    "banana": [
        # black and yellow Sigatoka are merged: several sources don't separate them, and the advice is the same
        kg("sujaykapadnis/banana-disease-recognition-dataset", {"has": ["Banana Healthy Leaf"], "name": "Original Images"},
           class_map={"Banana Black Sigatoka Disease": "Sigatoka leaf spot",
                      "Banana Bract Mosaic Virus Disease": "Bract mosaic virus", "Banana Healthy Leaf": "Healthy",
                      "Banana Insect Pest Disease": "Insect pest damage", "Banana Moko Disease": "Moko (bacterial wilt)",
                      "Banana Panama Disease": "Panama disease (fusarium wilt)",
                      "Banana Yellow Sigatoka Disease": "Sigatoka leaf spot"}),
        hf(AGML + "BananaLSD_leaf_disease_classification", config="raw", class_map={
            "cordana": "Cordana leaf spot", "healthy": "Healthy", "pestalotiopsis": "Pestalotiopsis leaf spot",
            "sigatoka": "Sigatoka leaf spot"}),
        hf(AGML + "PFSD_Musa_banana_disease_classification", {
            "BACTERIAL SOFT ROT": "Bacterial soft rot", "BANANA APHIDS": "Aphid (pest)",
            "BANANA FRUIT- SCARRING BEETLE": "Fruit scarring beetle (pest)", "BLACK SIGATOKA": "Sigatoka leaf spot",
            "PANAMA DISEASE": "Panama disease (fusarium wilt)", "POTASSIUM DEFICIENCY": "Potassium deficiency",
            "PSEUDOSTEM WEEVIL": "Pseudostem weevil (pest)", "YELLOW SIGATOKA": "Sigatoka leaf spot"}),
        hf(AGML + "banana_disease_classification_tanzania", {
            "black_sigatoka": "Sigatoka leaf spot", "fusarium_wilt": "Panama disease (fusarium wilt)", "healthy": "Healthy"}),
        hf(AGML + "banana_leaf_disease_classification", {"healthy": "Healthy", "segatoka": "Sigatoka leaf spot"}),
        *pw({"banana anthracnose": "Anthracnose", "banana black leaf streak": "Sigatoka leaf spot",
             "banana bunchy top": "Bunchy top virus", "banana cigar end rot": "Cigar end rot",
             "banana cordana leaf spot": "Cordana leaf spot", "banana panama disease": "Panama disease (fusarium wilt)",
             "banana leaf": "Healthy"}),
    ],
    "citrus_orange": [
        pv({"Orange___Haunglongbing_(Citrus_greening)": "Citrus greening (HLB)"}),
        hf(AGML + "orange_leaf_disease_classification", {
            "citrus_canker": "Citrus canker", "citrus_greening": "Citrus greening (HLB)",
            "citrus_mealybugs": "Mealybug (pest)", "die_back": "Dieback", "healthy_leaf": "Healthy",
            "powdery_mildew": "Powdery mildew", "shot_hole": "Shot hole", "spiny_whitefly": "Spiny whitefly (pest)",
            "yellow_dragon": "Citrus greening (HLB)"}),
        hf(AGML + "citrus_fruit_leaf_disease_classification", {
            "black_spot": "Black spot", "canker": "Citrus canker", "greening": "Citrus greening (HLB)", "healthy": "Healthy"}),
        hf(AGML + "citrusuat_disease_classification", {
            "Citrus_leafminer": "Leaf miner (pest)", "Greasy_spot": "Greasy spot", "HLB": "Citrus greening (HLB)",
            "Healthy": "Healthy", "Texas_mite": "Mite damage", "Red_scale": "Scale insect (pest)",
            "Red_scale_sequelae": "Scale insect (pest)", "Fe": "Nutrient deficiency", "Mg": "Nutrient deficiency",
            "Mn": "Nutrient deficiency", "N": "Nutrient deficiency", "Zn": "Nutrient deficiency"}),
        *pw({"citrus canker": "Citrus canker", "citrus greening disease": "Citrus greening (HLB)"}),
    ],
    "lemon": [
        hf(AGML + "lemon_leaf_disease_classification", config="raw", class_map={
            "Anthracnose": "Anthracnose", "Bacterial Blight": "Bacterial blight", "Citrus Canker": "Citrus canker",
            "Curl Virus": "Leaf curl virus", "Deficiency Leaf": "Nutrient deficiency", "Dry Leaf": "Dry leaf",
            "Healthy Leaf": "Healthy", "Sooty Mould": "Sooty mould", "Spider Mites": "Spider mites (pest)"}),
        hf(AGML + "three_plant_leaf_disease_classification", config="raw", filt={"col": "crop_type", "values": ["lemon"]},
           class_map={"Anthracnose": "Anthracnose", "Healthy": "Healthy", "Healthy_Leaf": "Healthy",
                      "Heathy_Leaf": "Healthy", "Leaf_Curl": "Leaf curl virus", "Sooty_Mold": "Sooty mould"}),
    ],
    "guava": [
        hf(AGML + "guava_disease_classification", config="raw", class_map={
            "Anthracnose": "Anthracnose", "Canker": "Canker", "Dot": "Dot (algal spot)", "Healthy": "Healthy",
            "Rust": "Red rust", "Scab": "Scab", "Styler end root": "Styler end rot"}),
        hf(AGML + "guava_disease_pakistan", {
            "Canker": "Canker", "Dot": "Dot (algal spot)", "Mummification": "Fruit mummification", "Rust": "Red rust"}),
        hf(AGML + "guava_disease_classification_bangladesh", config="raw", class_map={
            "Disease Free": "Healthy", "Phytopthora": "Phytophthora fruit rot", "Red rust": "Red rust", "Scab": "Scab",
            "Styler and Root": "Styler end rot"}),
    ],
    "papaya": [
        hf(AGML + "papaya_leaf_disease_classification_bangladesh", config="raw", class_map={
            "Healthy Leaf": "Healthy", "Leaf Curl": "Leaf curl virus", "Mealybug": "Mealybug (pest)",
            "Mite Disease": "Mite damage", "Mosaic": "Mosaic virus", "Ring Spot": "Ring spot virus"}),
        hf(AGML + "papaya_leaf_disease_classification_bd", {
            "Antracnose": "Anthracnose", "Bacterial Spot": "Bacterial spot", "Healthy Leaf": "Healthy",
            "Leaf Curl": "Leaf curl virus", "Mealybug": "Mealybug (pest)", "Mite Disease": "Mite damage",
            "Mosaic": "Mosaic virus", "Ring Spot": "Ring spot virus"}),
        hf(AGML + "AgriVision4_disease_classification", config="raw", filt={"col": "crop", "values": ["papaya"]},
           class_map={"Anthracnose": "Anthracnose", "Healthy": "Healthy", "Healthy_leaf": "Healthy",
                      "Mosaic": "Mosaic virus", "Mosaic_Virus": "Mosaic virus"}),
    ],
    "pomegranate": [
        hf(AGML + "pomegranate_disease_classification", config="raw", class_map={
            "Colletotrichum spp": "Anthracnose", "Ectomyelois ceratoniae": "Fruit borer (pest)", "Healthy": "Healthy",
            "Sunburn": "Sunburn"}),
        hf(AGML + "pomegranate_disease_classification_india", {
            "Alternaria": "Alternaria fruit spot", "Anthracnose": "Anthracnose",
            "Bacterial_Blight": "Bacterial blight (oily spot)", "Cercospora": "Cercospora fruit spot", "Healthy": "Healthy"}),
    ],
    "grape": [
        pv({"Grape___Black_rot": "Black rot", "Grape___Esca_(Black_Measles)": "Esca (black measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Isariopsis leaf spot", "Grape___healthy": "Healthy"}),
        hf(AGML + "grape_leaf_disease_classification", {
            "Bacterial Leaf Spot": "Bacterial leaf spot", "Downy Mildew": "Downy mildew", "Healthy Leaves": "Healthy",
            "Powdery Mildew": "Powdery mildew"}),
        *pw({"grape black rot": "Black rot", "grape downy mildew": "Downy mildew",
             "grapevine leafroll disease": "Leafroll virus", "grape leaf": "Healthy"}),
    ],
    "apple": [
        pv({"Apple___Apple_scab": "Apple scab", "Apple___Black_rot": "Black rot",
            "Apple___Cedar_apple_rust": "Cedar apple rust", "Apple___healthy": "Healthy"}),
        hf(AGML + "apple_leaf_disease_classification", {
            "Alternaria": "Alternaria leaf blotch", "Apple_Mosaic": "Mosaic virus", "Healthy": "Healthy"}),
        *pw({"apple black rot": "Black rot", "apple mosaic virus": "Mosaic virus", "apple rust": "Cedar apple rust",
             "apple scab": "Apple scab", "apple leaf": "Healthy"}),
    ],
    "custard_apple": [hf(AGML + "custard_apple_disease_classification", {
        "Athracnose": "Anthracnose", "Blank Canker": "Black canker", "Diplodia Rot": "Diplodia fruit rot",
        "Leaf spot on Leaves": "Leaf spot", "Leaf spot on fruit": "Fruit spot", "Mealy Bug": "Mealybug (pest)"})],
    "jamun": [hf(AGML + "java_plum_leaf_disease_classification", {
        "Bacterial_Spot": "Bacterial spot", "Brown_Blight": "Brown blight", "Dry": "Dry leaf", "Healthy": "Healthy",
        "Powdery_Mildew": "Powdery mildew", "Sooty_Mold": "Sooty mould"})],
    "moringa": [hf(AGML + "MoringaLeafNet_disease_classification", config="raw", class_map={
        "Bacterial Leaf Spot": "Bacterial leaf spot", "Cercospora Leaf Spot": "Cercospora leaf spot",
        "Healthy": "Healthy", "Yellow": "Yellowing leaf"})],
    "cherry": [
        pv({"Cherry_(including_sour)___Powdery_mildew": "Powdery mildew", "Cherry_(including_sour)___healthy": "Healthy"}),
        *pw({"cherry leaf spot": "Leaf spot", "cherry powdery mildew": "Powdery mildew", "cherry leaf": "Healthy"}),
    ],
    "peach": [
        pv({"Peach___Bacterial_spot": "Bacterial spot", "Peach___healthy": "Healthy"}),
        *pw({"peach brown rot": "Brown rot", "peach leaf curl": "Leaf curl", "peach scab": "Scab", "peach leaf": "Healthy"}),
    ],
    "strawberry": [
        pv({"Strawberry___Leaf_scorch": "Leaf scorch", "Strawberry___healthy": "Healthy"}),
        *pw({"strawberry anthracnose": "Anthracnose", "strawberry leaf scorch": "Leaf scorch",
             "strawberry leaf": "Healthy"}),
    ],
    "blueberry": [pv({"Blueberry___healthy": "Healthy"})],
    "raspberry": [pv({"Raspberry___healthy": "Healthy"})],
}

QUICK_CROPS = {"rice", "cotton", "banana", "onion", "cashew", "cabbage", "cucumber"}
if QUICK:
    CROPS = {k: v for k, v in CROPS.items() if k in QUICK_CROPS}
log(f"{len(CROPS)} crops configured")

# %% [markdown]
# ## 2 · Helpers: stable sampling, the parallel resize worker, and the source groups
#
# Sources shared by several crops (PlantVillage, PlantWild, the multi-crop Hugging Face repos) are
# downloaded and indexed **once** per repo, then split between crops.

# %%
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def stable_rng(key):
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:12], 16))


def sample(items, n, key):
    if len(items) <= n:
        return list(items)
    return stable_rng(key).sample(list(items), n)


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _prep_one(job):
    """Decode one photo (path or bytes), downscale while decoding, save as JPEG, return its phash."""
    src, out = job
    try:
        im = Image.open(src if isinstance(src, str) else io.BytesIO(src))
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode straight at 1/2-1/8 size (fast)
        im = ImageOps.exif_transpose(im).convert("RGB")
        if min(im.size) < 48:
            return None
        s = MAX_SIDE / max(im.size)
        if s < 1:
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
        im.save(out, "JPEG", quality=JPEG_Q)
        return str(imagehash.phash(im))
    except Exception:
        return None


POOL = mp.get_context("fork").Pool(N_CPU)  # forked before CUDA is ever touched
META = []                  # one dict per kept photo: crop, label, source, domain, path, phash
DECODE_FAILED = Counter()  # source -> photos that could not be decoded
UNMAPPED = Counter()       # "crop | source | label" -> rows skipped because the label has no mapping
SOURCE_ERRORS = []
_next_id = 0


def run_jobs(items):
    """items: list of (src, crop, label, source, domain) -> resized into STAGE, appended to META."""
    global _next_id
    jobs, metas = [], []
    for src, crop, label, source, domain in items:
        p = str(STAGE / f"{_next_id:07d}.jpg")
        _next_id += 1
        jobs.append((src, p))
        metas.append({"crop": crop, "label": label, "source": source, "domain": domain, "path": p})
    for m, ph in zip(metas, POOL.imap(_prep_one, jobs, chunksize=4)):
        if ph:
            m["phash"] = ph
            META.append(m)
        else:
            DECODE_FAILED[m["source"]] += 1


def source_key(spec):
    if spec["kind"] == "kaggle":
        return f"kaggle:{spec['ds']}"
    if spec["kind"] == "hf":
        return f"hf:{spec['repo']}"
    return f"zip:{spec['repo']}/{spec['file']}"


GROUPS = {}
for crop, specs in CROPS.items():
    for spec in specs:
        k = source_key(spec)
        GROUPS.setdefault(k, {"key": k, "kind": spec["kind"], "spec0": spec, "members": []})["members"].append((crop, spec))
# Kaggle first (no download), so the first Hugging Face download is prefetched while they are read
GROUPS = sorted(GROUPS.values(), key=lambda g: (g["kind"] != "kaggle", g["key"]))
log(f"{len(GROUPS)} distinct sources: " + ", ".join(f"{k}={v}" for k, v in Counter(g['kind'] for g in GROUPS).items()))

# %% [markdown]
# ## 3 · Loaders for the three source kinds

# %%
HF = HfApi()


def dataset_root(ref):
    owner, name = ref.split("/")
    for p in (INPUT / "datasets" / owner / name, INPUT / name):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Kaggle dataset {ref} is not attached to this notebook")


_find_cache = {}


def find_in(root, cond, max_depth=4):
    """Breadth-first search of one dataset (not all of /kaggle/input) for the folder matching cond."""
    key = (str(root), json.dumps(cond, sort_keys=True))
    if key in _find_cache:
        return _find_cache[key]
    frontier = [str(root)]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            try:
                kids = [e for e in os.scandir(d) if e.is_dir()]
            except OSError:
                continue
            names = {e.name for e in kids}
            base = os.path.basename(d)
            if (set(cond.get("has", [])) <= names and (not cond.get("name") or base == cond["name"])
                    and (not cond.get("name_contains") or cond["name_contains"] in base)):
                _find_cache[key] = Path(d)
                return Path(d)
            nxt.extend(e.path for e in kids)
        frontier = sorted(nxt)
    raise FileNotFoundError(f"no folder matching {cond} under {root}")


def list_images(folder):
    try:
        return sorted(e.path for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(IMG_EXT))
    except OSError:
        return []


def process_kaggle(group, fetched):
    for crop, spec in group["members"]:
        top = find_in(dataset_root(spec["ds"]), spec["find"])
        cmap = {norm(k): v for k, v in spec["class_map"].items() if v}
        cands = defaultdict(list)
        for sub in spec["subdirs"]:
            base = top / sub if sub else top
            if not base.is_dir():
                raise FileNotFoundError(f"{crop}: expected class folders under {base}")
            for e in os.scandir(base):
                if not e.is_dir():
                    continue
                canon = cmap.get(norm(e.name))
                if canon:
                    cands[canon].extend(list_images(e.path))
                elif spec["ds"] != PV_DS:  # PlantVillage holds every crop; other crops' folders are expected
                    UNMAPPED[f"{crop} | {spec['ds']} | {e.name}"] += 1
        items = []
        for canon, files in cands.items():
            for p in sample(files, CAP_PER_SOURCE_CLASS, f"{group['key']}|{crop}|{canon}"):
                items.append((p, crop, canon, group["key"], spec["domain"]))
        run_jobs(items)


def hf_parquet_files(repo, config):
    tree = HF.list_repo_tree(repo, repo_type="dataset", revision=PARQUET_REV, recursive=True)
    files = [(t.path, getattr(t, "size", 0) or 0) for t in tree if t.path.endswith(".parquet")]
    configs = sorted({p.split("/")[0] for p, _ in files})
    if not configs:
        raise FileNotFoundError(f"{repo} has no parquet export")
    if config is None:
        plain = [c for c in configs if "augment" not in c.lower()]
        config = "raw" if "raw" in configs else "default" if "default" in configs else (plain or configs)[0]
    return config, [(p, s) for p, s in files if p.split("/")[0] == config]


def fetch(group):
    """Runs in a background thread: download one source while the previous one is being decoded."""
    cache = HF_CACHE / slug(group["key"])
    if group["kind"] == "hf":
        repo = group["spec0"]["repo"]
        configs = {spec.get("config") for _, spec in group["members"]}
        config, files = hf_parquet_files(repo, next(iter(configs)))
        gb = sum(s for _, s in files) / 1e9
        free_gb = shutil.disk_usage("/tmp").free / 1e9
        if gb > MAX_REPO_GB or gb > free_gb - 8:
            raise RuntimeError(f"skipped: {gb:.1f} GB (limit {MAX_REPO_GB} GB, {free_gb:.0f} GB free)")
        t = time.time()

        def one(path):
            for attempt in range(5):
                try:
                    return hf_hub_download(repo, path, repo_type="dataset", revision=PARQUET_REV, cache_dir=str(cache))
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(15 * (attempt + 1))

        with ThreadPoolExecutor(8) as ex:
            local = list(ex.map(one, [p for p, _ in files]))
        return {"config": config, "paths": local, "gb": gb, "dl_s": time.time() - t, "cache": cache}
    if group["kind"] == "zip":
        t = time.time()
        p = hf_hub_download(group["spec0"]["repo"], group["spec0"]["file"], repo_type="dataset", cache_dir=str(cache))
        return {"paths": [p], "gb": os.path.getsize(p) / 1e9, "dl_s": time.time() - t, "cache": cache}
    return {}


def hf_features(pf):
    raw = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if not raw:
        return {}
    return json.loads(raw).get("info", {}).get("features", {}) or {}


_viewer_cache = {}


def viewer_features(repo, config):
    """Column features (with ClassLabel names) from the Dataset Viewer API. Some parquet exports carry no
    label names of their own -- the first smoke test read cucumber's labels as bare 0..7 and kept 0 photos."""
    if (repo, config) not in _viewer_cache:
        url = f"https://datasets-server.huggingface.co/info?dataset={repo}&config={config}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                _viewer_cache[(repo, config)] = json.load(r).get("dataset_info", {}).get("features", {}) or {}
        except Exception as exc:
            log(f"  could not read label names for {repo} ({config}) from the Dataset Viewer: {exc!r}")
            _viewer_cache[(repo, config)] = {}
    return _viewer_cache[(repo, config)]


def decoded_column(tbl, col, feats):
    vals = tbl.column(col).to_pylist()
    names = (feats.get(col) or {}).get("names")
    if names:
        return [names[v] if isinstance(v, int) and 0 <= v < len(names) else v for v in vals]
    return vals


def process_hf(group, fetched):
    files = []  # per parquet file: (ParquetFile, image column, {column: decoded values})
    for path in fetched["paths"]:
        pf = pq.ParquetFile(path)
        feats = hf_features(pf)
        cols = pf.schema_arrow.names
        for c in cols:  # integer columns with no names in the file: take the names from the Dataset Viewer
            if pa.types.is_integer(pf.schema_arrow.field(c).type) and not (feats.get(c) or {}).get("names"):
                vf = viewer_features(group["spec0"]["repo"], fetched["config"]).get(c) or {}
                if vf.get("names"):
                    feats[c] = vf
        img_col = next((c for c in cols if (feats.get(c) or {}).get("_type") == "Image"), "image")
        small = [c for c in cols if c != img_col]
        tbl = pf.read(columns=small)
        files.append((pf, img_col, {c: decoded_column(tbl, c, feats) for c in small}))
    if not files:
        return
    cols0 = files[0][2]
    lab_col = "label" if "label" in cols0 else next(iter(cols0))

    chosen = defaultdict(list)  # file index -> [(row, crop, canon, domain)]
    for crop, spec in group["members"]:
        cmap = {norm(k): v for k, v in spec["class_map"].items() if v}
        filt = spec.get("filter")
        fcol, allowed = None, None
        if filt:
            allowed = {norm(v) for v in filt["values"]}
            # the column is named per repo ("crop_type", "crop", "plant_type"); when not given, find the one
            # holding the wanted value, searching every file (a multi-file repo may be sorted by crop)
            fcol = filt["col"] if filt["col"] in cols0 else next(
                (c for c in cols0 if c != lab_col
                 and any(norm(v) in allowed for _, _, cols in files for v in cols.get(c, []))), None)
            if fcol is None:
                raise RuntimeError(f"{crop}: no column in {group['key']} contains {filt['values']}")
        cands = defaultdict(list)
        for fi, (_, _, cols) in enumerate(files):
            fvals = cols[fcol] if fcol else None
            for r, lab in enumerate(cols[lab_col]):
                if fvals is not None and norm(fvals[r]) not in allowed:
                    continue
                canon = cmap.get(norm(lab))
                if canon:
                    cands[canon].append((fi, r))
                else:
                    UNMAPPED[f"{crop} | {group['key']} | {lab}"] += 1
        for canon, refs in cands.items():
            for fi, r in sample(refs, CAP_PER_SOURCE_CLASS, f"{group['key']}|{crop}|{canon}"):
                chosen[fi].append((r, crop, canon, spec["domain"]))

    for fi, picks in chosen.items():
        pf, img_col, _ = files[fi]
        by_row = {r: (crop, canon, dom) for r, crop, canon, dom in picks}
        start = 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            local = [r - start for r in range(start, start + n) if r in by_row]
            if local:
                col = pf.read_row_group(g, columns=[img_col]).column(0).combine_chunks()
                blobs = col.field("bytes") if pa.types.is_struct(col.type) else col
                vals = blobs.take(pa.array(local)).to_pylist()
                items = [(b, *by_row[start + lr][:2], group["key"], by_row[start + lr][2])
                         for lr, b in zip(local, vals) if b]
                run_jobs(items)
            start += n


def process_zip(group, fetched):
    zf = zipfile.ZipFile(fetched["paths"][0])
    by_folder = defaultdict(list)
    for name in zf.namelist():
        if name.lower().endswith(IMG_EXT) and "/" in name:
            by_folder[norm(name.rsplit("/", 2)[-2])].append(name)
    items = []
    for crop, spec in group["members"]:
        for folder, canon in spec["class_map"].items():
            members = by_folder.get(norm(folder), [])
            for name in sample(members, CAP_PER_SOURCE_CLASS, f"{group['key']}|{crop}|{canon}|{folder}"):
                items.append((name, crop, canon, spec["domain"]))
    for s in range(0, len(items), 256):
        batch = items[s:s + 256]
        run_jobs([(zf.read(name), crop, canon, group["key"], dom) for name, crop, canon, dom in batch])
    zf.close()


PROCESS = {"kaggle": process_kaggle, "hf": process_hf, "zip": process_zip}

# %% [markdown]
# ## 4 · Load every source (downloads are prefetched one ahead; each is deleted once decoded)

# %%
DL = ThreadPoolExecutor(1)
futures = {}


def ensure_prefetch(i):
    if i < len(GROUPS) and i not in futures and GROUPS[i]["kind"] != "kaggle":
        futures[i] = DL.submit(fetch, GROUPS[i])


for i, group in enumerate(GROUPS):
    if (time.time() - T0) / 60 > LOAD_BUDGET_MIN:
        SOURCE_ERRORS.append((group["key"], f"not started: {LOAD_BUDGET_MIN}-minute loading budget used up"))
        continue
    ensure_prefetch(i)
    j = i + 1
    while j < len(GROUPS) and GROUPS[j]["kind"] == "kaggle":
        j += 1
    ensure_prefetch(j)
    n0, t = len(META), time.time()
    fetched = {}
    try:
        fetched = futures.pop(i).result() if i in futures else {}
        PROCESS[group["kind"]](group, fetched)
        dl = f", {fetched['gb']:.2f} GB downloaded in {fetched['dl_s']:.0f}s" if fetched.get("gb") is not None else ""
        log(f"[{i + 1}/{len(GROUPS)}] {group['key']}: +{len(META) - n0} photos in {time.time() - t:.0f}s{dl} "
            f"(total {len(META)})")
    except Exception as exc:
        SOURCE_ERRORS.append((group["key"], repr(exc)[:300]))
        log(f"[{i + 1}/{len(GROUPS)}] {group['key']}: FAILED {exc!r}"[:400])
    finally:
        if fetched.get("cache"):
            shutil.rmtree(fetched["cache"], ignore_errors=True)
        gc.collect()

DL.shutdown(wait=False, cancel_futures=True)
POOL.close()
POOL.join()
log(f"loaded {len(META)} photos; decode failures {sum(DECODE_FAILED.values())}; failed sources {len(SOURCE_ERRORS)}")
for k, err in SOURCE_ERRORS:
    log(f"  source problem: {k}: {err}")
if UNMAPPED:
    log("labels present in a source but deliberately not mapped (rows skipped):")
    for k, n in sorted(UNMAPPED.items()):
        print(f"    {n:6d}  {k}")

df = pd.DataFrame(META)
print(df.groupby(["crop", "label"]).size().to_string())

# %% [markdown]
# ## 5 · Universal leakage guard
#
# 1. **Embed** every photo with DINOv2-S, averaging the photo and its mirror image, so a flipped
#    copy embeds almost identically. Also take a 64-bit perceptual hash.
# 2. **Exact copies** (cos ≥ 0.985 or hash distance ≤ 2) inside one crop are merged into one photo.
#    If the copies carry *different* labels, both are dropped, because at least one label is wrong.
# 3. **Cap** each class at `CAP_PER_CLASS`, shared fairly between sources, so no single huge source
#    swamps a class.
# 4. **Near copies** (cos ≥ 0.95 or hash distance ≤ 6) are joined into clusters of at most 64
#    photos. Each cluster goes entirely to train or entirely to test, and test keeps only one photo
#    per cluster.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"embedding {len(df)} photos on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
embedder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
embedder = embedder.to(DEVICE).eval()
embed_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class Staged(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return embed_tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed(paths):
    out = []
    for x in torch.utils.data.DataLoader(Staged(paths), batch_size=256, num_workers=N_CPU, pin_memory=True):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = embedder(x).float() + embedder(torch.flip(x, dims=[3])).float()
        out.append(torch.nn.functional.normalize(f, dim=1).half().cpu())
    return torch.cat(out)


t = time.time()
FEATS = embed(df["path"].tolist())
BITS = torch.tensor(np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in df["phash"]]),
                    dtype=torch.int8) * 2 - 1
log(f"embedded in {time.time() - t:.0f}s")
embedder = embedder.cpu()  # free the GPU for the similarity matrices below
torch.cuda.empty_cache()


def neighbour_pairs(idx, cos_thr, ham_thr):
    """All pairs (i<j, as positions in idx) whose cosine >= cos_thr or phash Hamming <= ham_thr."""
    F = FEATS[idx].to(DEVICE).float()
    B = BITS[idx].to(DEVICE).float()
    n, out_i, out_j = len(idx), [], []
    for s in range(0, n, 2048):
        cos = F[s:s + 2048] @ F.T
        ham = (64 - B[s:s + 2048] @ B.T) / 2
        hit = (cos >= cos_thr) | (ham <= ham_thr)
        ii, jj = torch.nonzero(hit, as_tuple=True)
        ii = ii + s
        keep = jj > ii
        out_i.append(ii[keep].cpu().numpy())
        out_j.append(jj[keep].cpu().numpy())
    return np.concatenate(out_i), np.concatenate(out_j)


# ---- step 2: merge exact copies inside each crop
drop = np.zeros(len(df), bool)
n_conflicts = 0
labels = df["label"].values
for crop, sub in df.groupby("crop"):
    idx = sub.index.values
    pi, pj = neighbour_pairs(idx, EXACT_COS, EXACT_HAM)
    for a, b in zip(idx[pi], idx[pj]):
        if labels[a] != labels[b]:
            if not (drop[a] and drop[b]):
                n_conflicts += 1
            drop[a] = drop[b] = True
        elif not drop[a]:
            drop[b] = True
n_exact = int(drop.sum())
df = df[~drop].reset_index(drop=True)
log(f"exact copies merged: {n_exact} photos removed ({n_conflicts} copies had conflicting labels and were dropped)")

# ---- step 3: cap each class, sharing the cap fairly between its sources


def fair_pick(sub, cap, key):
    by_src = {s: list(g.index) for s, g in sub.groupby("source")}
    rng = stable_rng(key)
    for v in by_src.values():
        rng.shuffle(v)
    order = sorted(by_src, key=lambda s: len(by_src[s]))
    left, picked = cap, []
    for k, s in enumerate(order):
        take = min(len(by_src[s]), left // (len(order) - k))
        picked += by_src[s][:take]
        left -= take
    return picked


keep_idx = []
for (crop, label), sub in df.groupby(["crop", "label"]):
    keep_idx += fair_pick(sub, CAP_PER_CLASS, f"{crop}|{label}") if len(sub) > CAP_PER_CLASS else list(sub.index)
n_capped = len(df) - len(keep_idx)
df = df.loc[sorted(keep_idx)].reset_index(drop=True)
log(f"class cap {CAP_PER_CLASS}: {n_capped} surplus photos left out")

# re-align the embedding tensors with the surviving rows
orig_row = {p: i for i, p in enumerate(pd.DataFrame(META)["path"])}
rows = torch.tensor([orig_row[p] for p in df["path"]])
FEATS, BITS = FEATS[rows], BITS[rows]

# ---- step 4: near-copy clusters (union-find with a size cap), inside each crop
parent = np.arange(len(df))
size = np.ones(len(df), int)


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


n_blocked = 0
for crop, sub in df.groupby("crop"):
    idx = sub.index.values
    pi, pj = neighbour_pairs(idx, GROUP_COS, GROUP_HAM)
    for a, b in zip(idx[pi], idx[pj]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > MAX_CLUSTER:
            n_blocked += 1
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
df["cluster"] = [find(i) for i in range(len(df))]
n_clusters = df["cluster"].nunique()
log(f"{len(df)} photos in {n_clusters} near-copy clusters ({n_blocked} merges blocked by the size cap)")

# drop classes too small to learn and test
counts = df.groupby(["crop", "label"])["cluster"].transform("size")
too_small = df[counts < MIN_PER_CLASS].groupby(["crop", "label"]).size()
if len(too_small):
    log(f"dropping {len(too_small)} classes with < {MIN_PER_CLASS} photos: " +
        ", ".join(f"{c}/{l} ({n})" for (c, l), n in too_small.items()))
df = df[counts >= MIN_PER_CLASS].reset_index(drop=True)

# %% [markdown]
# ## 6 · Split 85/15 by cluster, stratified by class (a cluster counts toward its majority label)

# %%
maj = df.groupby("cluster")["label"].agg(lambda s: s.value_counts().index[0])
csize = df.groupby("cluster").size()
test_clusters = set()
for (crop, label), sub in df.groupby(["crop", "label"]):
    cl = sorted(c for c in sub["cluster"].unique() if maj[c] == label)
    stable_rng(f"split|{crop}|{label}").shuffle(cl)
    target, got = TEST_FRAC * len(sub), 0
    for c in cl:
        if got >= target:
            break
        test_clusters.add(c)
        got += csize[c]
df["split"] = np.where(df["cluster"].isin(test_clusters), "test", "train")
# test keeps one photo per cluster and label, so a burst of near-identical shots is scored once
dup_in_test = (df["split"] == "test") & df.duplicated(["cluster", "label"])
df = df[~dup_in_test].reset_index(drop=True)
log(f"split: {int((df.split == 'train').sum())} train, {int((df.split == 'test').sum())} test "
    f"({int(dup_in_test.sum())} redundant near-copies dropped from test)")

# %% [markdown]
# ## 7 · Write the dataset: one zip per crop, the manifest, class list, summary and a card

# %%
t = time.time()
manifest = []
for crop, sub in df.groupby("crop"):
    with zipfile.ZipFile(OUT / f"{crop}.zip", "w", zipfile.ZIP_STORED) as zf:
        for n, r in enumerate(sub.itertuples(), 1):
            arc = f"{slug(r.label)}/{n:06d}.jpg"
            zf.write(r.path, arc)
            manifest.append({"crop": crop, "label": r.label, "split": r.split, "zip": f"{crop}.zip", "file": arc,
                             "source": r.source, "domain": r.domain, "group": f"{crop}:{r.cluster}"})
shutil.rmtree(STAGE, ignore_errors=True)
mf = pd.DataFrame(manifest)
mf.to_csv(OUT / "manifest.csv", index=False)
log(f"wrote {len(mf)} photos into {mf['zip'].nunique()} zips in {time.time() - t:.0f}s")

crops_info = {}
for crop, sub in mf.groupby("crop"):
    per_class = sub[sub.split == "train"].groupby("label").size()
    labels_sorted = sorted(sub["label"].unique())
    thin = per_class.reindex(labels_sorted, fill_value=0).min() < 100
    crops_info[crop] = {
        "labels": labels_sorted, "num_classes": len(labels_sorted),
        "train_images": int((sub.split == "train").sum()), "test_images": int((sub.split == "test").sum()),
        "sources": sorted(sub["source"].unique()),
        "coverage_note": ("Some diseases of this crop have fewer than 100 training photos; "
                          "treat those results with extra caution.") if thin else None,
    }
built = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
with open(OUT / "classes.json", "w", encoding="utf-8") as fh:
    json.dump({"crops": crops_info, "built": built}, fh, indent=2)

summary = mf.groupby(["crop", "label", "split"]).size().unstack(fill_value=0)
summary.to_csv(OUT / "summary.csv")
print(summary.to_string())

total_classes = sum(c["num_classes"] for c in crops_info.values())
size_gb = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9
report = {
    "built": built, "quick_mode": QUICK, "crops": len(crops_info), "total_classes": total_classes,
    "images": {"train": int((mf.split == "train").sum()), "test": int((mf.split == "test").sum())},
    "images_by_domain": mf.groupby(["domain", "split"]).size().unstack(fill_value=0).to_dict(),
    "exact_copies_removed": n_exact, "label_conflicts_dropped": n_conflicts, "class_cap_surplus": n_capped,
    "near_copy_clusters": int(n_clusters), "classes_dropped_too_small": {f"{c}/{l}": int(n) for (c, l), n in too_small.items()},
    "decode_failures": dict(DECODE_FAILED), "source_errors": dict(SOURCE_ERRORS),
    "unmapped_labels": dict(UNMAPPED), "size_gb": round(size_gb, 2), "runtime_min": round((time.time() - T0) / 60, 1),
}
with open(OUT / "dedup_report.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)

src_counts = mf.groupby("source").size().sort_values(ascending=False)
card = ["# AgriSmart India multi-crop dataset", "",
        f"Built {built}: **{len(crops_info)} crops, {total_classes} classes, "
        f"{report['images']['train']} train / {report['images']['test']} test photos**.", "",
        "Every crop was checked for exact and near-duplicate photos (perceptual hash + flip-invariant DINOv2 "
        "embedding) before splitting; each near-copy cluster sits entirely in train or entirely in test.", "",
        "The `domain` column of `manifest.csv` marks `lab` (PlantVillage, plain background), `field` "
        "(PlantWild, photos taken in real fields/gardens) and `mixed` (research datasets, mostly field photos "
        "taken by the dataset authors), so accuracy can be reported per kind of photo.", "",
        "## Sources (photos kept)", ""]
card += [f"- `{s}`: {n}" for s, n in src_counts.items()]
card += ["", "Each Hugging Face `Project-AgML/*` repo's card carries the original paper's citation. PlantVillage: "
         "Mohanty, Hughes & Salathé (2016). PlantWild: Wei et al., ACM MM 2024.", "",
         "## Licence note",
         "Most sources are CC BY 4.0 or CC0. Some are non-commercial: PlantVillage (CC BY-NC-SA), PlantWild "
         "(CC BY-NC-ND), the Ripon cotton set, `tea_leaf_disease_classification`, `MoringaLeafNet` and the mango "
         "set (CC BY-NC). This dataset and any model trained on it are for non-commercial research and "
         "education, and the photos themselves are not redistributed."]
(OUT / "README.md").write_text("\n".join(card), encoding="utf-8")

print(json.dumps({k: v for k, v in report.items() if k != "unmapped_labels"}, indent=2, default=str))
log(f"done: {len(mf)} photos, {size_gb:.2f} GB in {OUT}")
