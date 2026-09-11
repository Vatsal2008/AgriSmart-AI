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
# # AgriSmart AI · Step 1: build the training dataset
#
# This notebook gathers every image we train and validate on into **one clean, leakage-checked dataset**.
# The training notebook (`agrismart-train`) attaches this notebook's output as its input.
#
# | Source | Used for | Notes |
# |---|---|---|
# | PlantVillage `color` (lab) | training + lab validation | the data the organizers provide; split **by leaf** with the official leaf map |
# | PlantVillage `segmented` | background replacement | leaves with the background removed; training pastes them onto new backgrounds |
# | PlantWild v2 (field, diseases) | training | expert-refined in-the-wild photos |
# | PlantWild v1 (field, healthy leaves) | training | v2 has no healthy classes, so healthy field photos come from v1 |
# | PlantSeg (field) | training | same authors as PlantWild, so overlapping photos are merged as duplicates |
# | PlantDoc (field) | **field validation only** | the closest thing to the hidden test set, so it is never trained on |
#
# **Leakage guard:** any training photo that is a near-copy of a PlantDoc photo is dropped
# (perceptual hash + DINOv2 similarity). Duplicates inside the training pool are merged, and
# duplicates that disagree on the label are dropped entirely.
#
# Runtime: about 15–25 minutes on a T4. Output: `/kaggle/working/agrismart_data/`.

# %%
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

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/agrismart_data")
TMP = Path("/tmp/agrismart_prep")
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

FIELD_MAX_SIDE = 512            # field photos are resized so the longer side is at most this
N_BACKGROUNDS = 2000            # background crops used for background replacement
LEAK_COS, LEAK_HAM = 0.93, 10   # "near-copy of a PlantDoc photo" (strict on purpose)
DUP_COS, DUP_HAM = 0.96, 6      # "duplicate inside one pool"
EXTRA_NEVER_TRAIN = []          # later: folders that must never leak into training, e.g. the organizers' public sample

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


log(f"torch {torch.__version__} | timm {timm.__version__}")

# %% [markdown]
# ## 1 · Class list
# The draft list is the 28 PlantVillage classes that also exist in PlantDoc.
# **At kickoff, set `KEEP` to the organizers' official list and re-run.** Nothing else needs to change.
# Labels use PlantVillage folder names, the naming of the data the organizers provide.

# %%
# label -> (PlantDoc folder, PlantWild v2 / PlantSeg disease name, PlantWild v1 healthy-leaf folder)
CLASS_MAP = {
    "Apple___Apple_scab": ("apple scab leaf", "apple scab", None),
    "Apple___Cedar_apple_rust": ("apple rust leaf", "apple rust", None),
    "Apple___healthy": ("apple leaf", None, "apple leaf"),
    "Blueberry___healthy": ("blueberry leaf", None, "blueberry leaf"),
    "Cherry_(including_sour)___healthy": ("cherry leaf", None, "cherry leaf"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn gray leaf spot", "corn gray leaf spot", None),
    "Corn_(maize)___Common_rust_": ("corn rust leaf", "corn rust", None),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn leaf blight", "corn northern leaf blight", None),
    "Grape___Black_rot": ("grape leaf black rot", "grape black rot", None),
    "Grape___healthy": ("grape leaf", None, "grape leaf"),
    "Peach___healthy": ("peach leaf", None, "peach leaf"),
    "Pepper,_bell___Bacterial_spot": ("bell pepper leaf spot", "bell pepper bacterial spot", None),
    "Pepper,_bell___healthy": ("bell pepper leaf", None, "bell pepper leaf"),
    "Potato___Early_blight": ("potato leaf early blight", "potato early blight", None),
    "Potato___Late_blight": ("potato leaf late blight", "potato late blight", None),
    "Raspberry___healthy": ("raspberry leaf", None, "raspberry leaf"),
    "Soybean___healthy": ("soyabean leaf", None, "soybean leaf"),
    "Squash___Powdery_mildew": ("squash powdery mildew leaf", "squash powdery mildew", None),
    "Strawberry___healthy": ("strawberry leaf", None, "strawberry leaf"),
    "Tomato___Bacterial_spot": ("tomato leaf bacterial spot", "tomato bacterial leaf spot", None),
    "Tomato___Early_blight": ("tomato early blight leaf", "tomato early blight", None),
    "Tomato___Late_blight": ("tomato leaf late blight", "tomato late blight", None),
    "Tomato___Leaf_Mold": ("tomato mold leaf", "tomato leaf mold", None),
    "Tomato___Septoria_leaf_spot": ("tomato septoria leaf spot", "tomato septoria leaf spot", None),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato two spotted spider mites leaf", None, None),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato leaf yellow virus", "tomato yellow leaf curl virus", None),
    "Tomato___Tomato_mosaic_virus": ("tomato leaf mosaic virus", "tomato mosaic virus", None),
    "Tomato___healthy": ("tomato leaf", None, "tomato leaf"),
}
KEEP = list(CLASS_MAP)  # <- replace with the official list at kickoff

LABELS = sorted(KEEP)
LABEL_IDX = {label: i for i, label in enumerate(LABELS)}


def norm(name):
    """'Tomato_leaf late-blight' -> 'tomato leaf late blight'."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


PD_TO_LABEL = {CLASS_MAP[l][0]: l for l in KEEP}
DISEASE_TO_LABEL = {CLASS_MAP[l][1]: l for l in KEEP if CLASS_MAP[l][1]}
HEALTHY_TO_LABEL = {CLASS_MAP[l][2]: l for l in KEEP if CLASS_MAP[l][2]}
# crops that appear in our classes; field photos of *other* crops become background material
SHARED_CROPS = {"apple", "bell", "blueberry", "cherry", "corn", "grape", "peach", "potato",
                "raspberry", "soybean", "squash", "strawberry", "tomato"}
print(f"{len(LABELS)} classes")

# %% [markdown]
# ## 2 · Locate the attached datasets and download the rest
# Kaggle mounts attached datasets under `/kaggle/input/`. The exact path has changed over time, so we search for them.

# %%
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def walk_dirs(base, max_depth=6):
    """Yield (folder, child folder names), never descending into PlantVillage class folders."""
    base = str(base)
    for root, dirs, _ in os.walk(base):
        yield Path(root), list(dirs)
        depth = root[len(base):].count(os.sep)
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if "___" not in d]


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


PV_ROOT = find_dir(lambda f, c: {"color", "segmented"} <= set(c), "PlantVillage (color + segmented)")
PS_ROOT = find_dir(lambda f, c: {"images", "annotations"} <= set(c) and (f / "Metadatav2.csv").exists(), "PlantSeg")
PD_SPLITS = [f for f, c in walk_dirs(INPUT)
             if f.name.lower() in {"train", "test"} and any(norm(x) == "tomato leaf" for x in c)]
assert PD_SPLITS, "PlantDoc train/test folders not found. Is nirmalsankalana/plantdoc-dataset attached?"
log(f"PlantVillage: {PV_ROOT}")
log(f"PlantSeg:     {PS_ROOT}")
log(f"PlantDoc:     {[str(p) for p in PD_SPLITS]}")


def hf_file(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=TMP)


t = time.time()
PW2_ZIP = hf_file("uqtwei2/PlantWild", "plantwild_v2.zip")
PW1_ZIP = hf_file("uqtwei2/PlantWild", "plantwild.zip")
PV_SPLIT_TRAIN = hf_file("mohanty/PlantVillage", "splits/color_train.txt")
PV_SPLIT_TEST = hf_file("mohanty/PlantVillage", "splits/color_test.txt")
with open(hf_file("mohanty/PlantVillage", "leaf_grouping/leaf-map.json"), encoding="utf-8") as fh:
    LEAF_MAP = json.load(fh)
log(f"downloads finished in {time.time() - t:.0f}s")

# %% [markdown]
# ## 3 · PlantVillage: split by leaf
# PlantVillage has several photos of the same physical leaf. We follow the official leaf-grouped split
# (`mohanty/PlantVillage`), then check that no leaf ends up in both train and validation.

# %%
def pv_leaf_id(label, filename):
    """Same rule as the official loader: the photo id after '___', looked up in the leaf map."""
    ident = filename.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    ident = re.sub(r"\.(jpe?g|png)$", "", ident, flags=re.I).strip()
    options = LEAF_MAP.get(ident.lower().strip(), [])
    for option in options:
        if len(options) == 1 or label in option:
            return option
    return f"{label}:::solo_{ident.lower()}"


def read_split(path):
    keys = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("/")
            if len(parts) >= 2:
                keys.add((parts[-2], parts[-1]))
    return keys


def stable_bucket(text, n=5):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % n


pv_train_keys, pv_test_keys = read_split(PV_SPLIT_TRAIN), read_split(PV_SPLIT_TEST)
pv_rows, seg_missing = [], 0
for label in KEEP:
    seg_by_stem = {p.name.replace("_final_masked", "").rsplit(".", 1)[0]: p
                   for p in list_images(PV_ROOT / "segmented" / label)}
    color_files = list_images(PV_ROOT / "color" / label)
    if not color_files:
        print(f"WARNING: no PlantVillage images for {label}")
    for p in color_files:
        key = (label, p.name)
        split = "train" if key in pv_train_keys else "val_lab" if key in pv_test_keys else None
        seg = seg_by_stem.get(p.stem)
        seg_missing += seg is None
        pv_rows.append({"label": label, "path": p, "seg_path": seg, "group": pv_leaf_id(label, p.name), "split": split})
pv = pd.DataFrame(pv_rows)

# photos missing from the official lists follow their leaf group, or a stable hash of it
listed = pv["split"].apply(lambda s: isinstance(s, str))
group_split = pv[listed].groupby("group")["split"].agg(lambda s: s.mode().iloc[0])
pv.loc[~listed, "split"] = [group_split.get(g, "val_lab" if stable_bucket(g) == 0 else "train")
                            for g in pv.loc[~listed, "group"]]
# every leaf lives on exactly one side (majority wins)
majority = pv.groupby("group")["split"].agg(lambda s: s.mode().iloc[0])
moved = int((pv["split"] != pv["group"].map(majority)).sum())
pv["split"] = pv["group"].map(majority)
assert pv.groupby("group")["split"].nunique().max() == 1
log(f"PlantVillage: {len(pv)} photos | train {int((pv.split == 'train').sum())} / val_lab {int((pv.split == 'val_lab').sum())} | "
    f"{int((~listed).sum())} not in the official lists | {moved} moved to keep leaves together | "
    f"{seg_missing} without a segmented version")

# %% [markdown]
# ## 4 · Field photos: PlantDoc (validation) and PlantWild / PlantSeg (training)

# %%
pd_rows, pd_unmapped = [], Counter()
for split_dir in PD_SPLITS:
    for folder in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        label = PD_TO_LABEL.get(norm(folder.name))
        files = list_images(folder)
        if label is None:
            pd_unmapped[folder.name] += len(files)
        for p in files:
            pd_rows.append({"source": "plantdoc", "label": label, "spec": ("file", str(p), ""),
                            "orig": f"plantdoc/{split_dir.name}/{folder.name}/{p.name}"})
pd_found = {norm(d.name) for s in PD_SPLITS for d in s.iterdir() if d.is_dir()}
log(f"PlantDoc: {len(pd_rows)} photos | folders outside KEEP (still used for the leakage check): {dict(pd_unmapped)} | "
    f"KEEP classes with no PlantDoc folder: {sorted(set(PD_TO_LABEL) - pd_found)}")


def zip_images(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return [i.filename for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(IMG_EXT)]


field_rows, bg_candidates = [], []
pw2_classes, pw1_classes = set(), set()
for name in zip_images(PW2_ZIP):  # plantwild_v2/<disease>/<file>
    cls = norm(name.split("/")[-2])
    pw2_classes.add(cls)
    if cls in DISEASE_TO_LABEL:
        field_rows.append({"source": "plantwild_v2", "label": DISEASE_TO_LABEL[cls], "spec": ("zip", PW2_ZIP, name), "orig": name})
for name in zip_images(PW1_ZIP):  # plantwild/images/<class>/<file>
    cls = norm(name.split("/")[-2])
    pw1_classes.add(cls)
    if cls in HEALTHY_TO_LABEL:
        field_rows.append({"source": "plantwild_v1", "label": HEALTHY_TO_LABEL[cls], "spec": ("zip", PW1_ZIP, name), "orig": name})
    elif cls.split()[0] not in SHARED_CROPS:
        bg_candidates.append(("zip", PW1_ZIP, name))

with zipfile.ZipFile(PW1_ZIP) as zf:  # disease descriptions: material for the bonus-E assistant
    prompts = [n for n in zf.namelist() if n.endswith("plantwild_prompts.json")]
    if prompts:
        (OUT / "plantwild_prompts.json").write_bytes(zf.read(prompts[0]))

meta = pd.read_csv(PS_ROOT / "Metadatav2.csv")
ps_files = {p.name: p for sub in ("train", "val", "test") for p in list_images(PS_ROOT / "images" / sub)}
ps_diseases = {norm(d) for d in meta["Disease"].unique()}
ps_missing = 0
for name, disease in zip(meta["Name"], meta["Disease"]):
    label = DISEASE_TO_LABEL.get(norm(disease))
    if label is None:
        continue
    p = ps_files.get(name)
    if p is None:
        ps_missing += 1
        continue
    field_rows.append({"source": "plantseg", "label": label, "spec": ("file", str(p), ""), "orig": f"plantseg/{p.parent.name}/{p.name}"})

never_rows = [{"source": "never_train", "label": None, "spec": ("file", str(p), ""), "orig": str(p)}
              for folder in EXTRA_NEVER_TRAIN for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXT]

log(f"field training candidates: {dict(Counter(r['source'] for r in field_rows))} | backgrounds pool: {len(bg_candidates)}")
log(f"disease names not found in PlantWild v2: {sorted(set(DISEASE_TO_LABEL) - pw2_classes)} | in PlantSeg: "
    f"{sorted(set(DISEASE_TO_LABEL) - ps_diseases)} | healthy names not in PlantWild v1: {sorted(set(HEALTHY_TO_LABEL) - pw1_classes)} | "
    f"PlantSeg rows without an image: {ps_missing}")

# %% [markdown]
# ## 5 · Load, resize and fingerprint the field photos
# Each field photo is decoded once, resized (longer side at most 512 px), re-encoded as JPEG and given a perceptual hash.
# Four CPU processes do this in parallel.

# %%
_ZIPS = {}


def _read(spec):
    kind, container, name = spec
    if kind == "zip":
        zf = _ZIPS.get(container)
        if zf is None:
            zf = _ZIPS[container] = zipfile.ZipFile(container)
        return zf.read(name)
    return Path(container).read_bytes()


def load_field(spec):
    """-> (jpeg bytes, phash hex, width, height), or None if the file can't be read."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(_read(spec)))).convert("RGB")
        if min(im.size) < 48:
            return None
        scale = FIELD_MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        return buf.getvalue(), str(imagehash.phash(im)), im.width, im.height
    except Exception:
        return None


def make_background(spec):
    """A random square crop of a field photo of some other crop, as a 256 px JPEG."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(_read(spec)))).convert("RGB")
        rng = random.Random(spec[2])
        side = int(min(im.size) * rng.uniform(0.5, 1.0))
        if side < 64:
            return None
        x, y = rng.randint(0, im.width - side), rng.randint(0, im.height - side)
        crop = im.crop((x, y, x + side, y + side)).resize((256, 256), Image.BICUBIC)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return None


def parallel(fn, items, workers=4):
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(fn, items, chunksize=32))


pool_rows = pd_rows + never_rows + field_rows
t = time.time()
for row, res in zip(pool_rows, parallel(load_field, [r["spec"] for r in pool_rows])):
    row["ok"] = res is not None
    if res is not None:
        row["jpeg"], row["phash"], row["width"], row["height"] = res
unreadable = Counter(r["source"] for r in pool_rows if not r["ok"])
pool_rows = [r for r in pool_rows if r["ok"]]
log(f"decoded {len(pool_rows)} field photos in {time.time() - t:.0f}s | unreadable skipped: {dict(unreadable)}")

random.shuffle(bg_candidates)
bg_blobs = [b for b in parallel(make_background, bg_candidates[: int(N_BACKGROUNDS * 1.2)]) if b][:N_BACKGROUNDS]
log(f"background crops: {len(bg_blobs)}")

# %% [markdown]
# ## 6 · Remove near-duplicates and PlantDoc look-alikes
# Two checks catch copies that were resized, re-compressed or cropped: a perceptual hash (Hamming distance)
# and DINOv2 image embeddings (cosine similarity).
# 1. **PlantDoc itself:** its train and test folders share photos, so keep one copy (and drop copies with conflicting labels).
# 2. **Leakage guard:** drop every training candidate that looks like *any* PlantDoc photo.
# 3. **Training pool:** merge duplicates (PlantWild v2 first, then PlantSeg, then PlantWild v1); drop clusters whose labels disagree.

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEVICE == "cuda" else torch.float32
log(f"embedding on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
embedder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
embedder = embedder.to(DEVICE).eval()
embed_tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
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
        feats.append(torch.nn.functional.normalize(f.float(), dim=1).to(DT))
    return torch.cat(feats)


t = time.time()
FEATS = embed([r["jpeg"] for r in pool_rows])
bits = np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8)) for r in pool_rows])
BITS = torch.tensor(bits, device=DEVICE).to(DT) * 2 - 1  # +-1 per hash bit, so Hamming = (64 - a.b) / 2
log(f"embedded {len(pool_rows)} photos in {time.time() - t:.0f}s")


def near_dup(ia, ib, cos_thr, ham_thr, chunk=2048):
    """Boolean matrix [len(ia), len(ib)]: True where two photos look like copies of each other."""
    ia, ib = torch.as_tensor(ia, device=DEVICE), torch.as_tensor(ib, device=DEVICE)
    if len(ia) == 0 or len(ib) == 0:
        return np.zeros((len(ia), len(ib)), dtype=bool)
    fb, bb = FEATS[ib], BITS[ib]
    rows = []
    for s in range(0, len(ia), chunk):
        a = ia[s:s + chunk]
        cos = FEATS[a] @ fb.T
        ham = (64 - BITS[a] @ bb.T) / 2
        rows.append(((cos >= cos_thr) | (ham <= ham_thr)).cpu())
    return torch.cat(rows).numpy()


def greedy_dedup(idx, labels):
    """Keep one photo per cluster of copies; drop whole clusters whose labels disagree.
    Returns a status per photo: 1 kept, 2 duplicate, 3 label conflict."""
    dup = near_dup(idx, idx, DUP_COS, DUP_HAM)
    np.fill_diagonal(dup, False)
    labels = np.asarray(labels, dtype=object)
    status = np.zeros(len(idx), dtype=np.int8)
    for i in range(len(idx)):
        if status[i]:
            continue
        js = np.flatnonzero(dup[i])
        js = js[status[js] == 0]
        if len(js) and (labels[js] != labels[i]).any():
            status[i] = 3
            status[js] = 3
        else:
            status[i] = 1
            status[js] = 2
    return status


STATUS = {1: "kept", 2: "duplicate", 3: "label_conflict"}
src = np.array([r["source"] for r in pool_rows])
labels_all = [r["label"] for r in pool_rows]

pd_idx = np.flatnonzero(src == "plantdoc")
for k, s in zip(pd_idx, greedy_dedup(pd_idx, [labels_all[i] for i in pd_idx])):
    pool_rows[k]["dedup"] = STATUS[int(s)]

never_idx = np.flatnonzero(np.isin(src, ["plantdoc", "never_train"]))
field_idx = np.flatnonzero(~np.isin(src, ["plantdoc", "never_train"]))
leak = near_dup(field_idx, never_idx, LEAK_COS, LEAK_HAM).any(axis=1)
for k in field_idx[leak]:
    pool_rows[k]["dedup"] = "plantdoc_lookalike"

priority = {"plantwild_v2": 0, "plantseg": 1, "plantwild_v1": 2}
cand = field_idx[~leak]
cand = cand[np.argsort([priority[src[i]] for i in cand], kind="stable")]
for k, s in zip(cand, greedy_dedup(cand, [labels_all[i] for i in cand])):
    pool_rows[k]["dedup"] = STATUS[int(s)]
for k in np.flatnonzero(src == "never_train"):
    pool_rows[k]["dedup"] = "never_train"

dedup_table = pd.crosstab(pd.Series(src, name="source"), pd.Series([r["dedup"] for r in pool_rows], name="status"))
print(dedup_table.to_string())

# %% [markdown]
# ## 7 · Write the dataset
# Images go into a few uncompressed zip files (a notebook output with tens of thousands of loose files is slow to attach);
# `manifest.csv` lists every image with its label, source and split.

# %%
def slug(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


manifest = []
t = time.time()
with zipfile.ZipFile(OUT / "pv_color.zip", "w", zipfile.ZIP_STORED) as zc, \
        zipfile.ZipFile(OUT / "pv_segmented.zip", "w", zipfile.ZIP_STORED) as zs:
    for row in pv.itertuples():
        arc = f"pv/color/{slug(row.label)}/{row.path.name}"
        zc.write(row.path, arc)
        seg_arc = ""
        if row.split == "train" and isinstance(row.seg_path, Path):
            seg_arc = f"pv/segmented/{slug(row.label)}/{row.seg_path.name}"
            zs.write(row.seg_path, seg_arc)
        manifest.append({"file": arc, "zip": "pv_color.zip", "label": row.label, "source": "plantvillage",
                         "split": row.split, "group": row.group, "seg_file": seg_arc,
                         "orig": f"plantvillage/color/{row.label}/{row.path.name}", "phash": ""})
log(f"PlantVillage zips written in {time.time() - t:.0f}s")

counters = Counter()
with zipfile.ZipFile(OUT / "field_train.zip", "w", zipfile.ZIP_STORED) as zt, \
        zipfile.ZipFile(OUT / "field_val.zip", "w", zipfile.ZIP_STORED) as zv:
    for r in pool_rows:
        if r["label"] not in LABEL_IDX or r.get("dedup") != "kept":
            continue
        if r["source"] == "plantdoc":
            zf, zip_name, split = zv, "field_val.zip", "val_field"
        else:
            zf, zip_name, split = zt, "field_train.zip", "train"
        counters[r["source"]] += 1
        arc = f"{r['source']}/{slug(r['label'])}/{counters[r['source']]:06d}.jpg"
        zf.writestr(arc, r["jpeg"])
        manifest.append({"file": arc, "zip": zip_name, "label": r["label"], "source": r["source"], "split": split,
                         "group": f"{r['source']}:{r['orig']}", "seg_file": "", "orig": r["orig"], "phash": r["phash"]})
with zipfile.ZipFile(OUT / "backgrounds.zip", "w", zipfile.ZIP_STORED) as zb:
    for k, blob in enumerate(bg_blobs):
        zb.writestr(f"backgrounds/{k:05d}.jpg", blob)

mf = pd.DataFrame(manifest)
mf["label_idx"] = mf["label"].map(LABEL_IDX)
mf.to_csv(OUT / "manifest.csv", index=False)

with open(OUT / "classes.json", "w", encoding="utf-8") as fh:
    json.dump({
        "labels": LABELS,
        "class_map": {l: {"plantdoc": CLASS_MAP[l][0], "field_disease": CLASS_MAP[l][1], "field_healthy": CLASS_MAP[l][2]}
                      for l in LABELS},
        "backgrounds": "backgrounds.zip",
        "segmented": "pv_segmented.zip",
    }, fh, indent=2)

summary = pd.crosstab(mf["label"], mf["source"] + " / " + mf["split"]).reindex(LABELS, fill_value=0)
summary["train total"] = mf[mf.split == "train"].groupby("label").size().reindex(LABELS, fill_value=0)
summary.to_csv(OUT / "summary.csv")
print(summary.to_string())

# %% [markdown]
# ## 8 · Sanity checks, report and dataset card

# %%
train, val_lab = mf[mf.split == "train"], mf[mf.split == "val_lab"]
shared_leaves = set(train.loc[train.source == "plantvillage", "group"]) & set(val_lab["group"])
assert not shared_leaves, f"{len(shared_leaves)} PlantVillage leaves are in both train and val_lab"
assert not (train["source"] == "plantdoc").any(), "PlantDoc must never be in training"
missing_train = sorted(set(LABELS) - set(train["label"]))
missing_field = sorted(set(LABELS) - set(mf.loc[mf.split == "val_field", "label"]))
size_gb = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9

report = {
    "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    "classes": len(LABELS),
    "images": {split: int(n) for split, n in mf["split"].value_counts().items()},
    "images_by_source": {s: int(n) for s, n in mf["source"].value_counts().items()},
    "plantvillage": {"not_in_official_lists": int((~listed).sum()), "moved_to_keep_leaves_together": moved,
                     "without_segmented": seg_missing},
    "dedup": {src_: {st: int(n) for st, n in row.items() if n} for src_, row in dedup_table.iterrows()},
    "thresholds": {"leak_cosine": LEAK_COS, "leak_hamming": LEAK_HAM, "dup_cosine": DUP_COS, "dup_hamming": DUP_HAM},
    "unreadable": dict(unreadable),
    "classes_without_training_images": missing_train,
    "classes_without_field_validation": missing_field,
    "size_gb": round(size_gb, 2),
    "runtime_min": round((time.time() - T0) / 60, 1),
}
with open(OUT / "dedup_report.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)

counts = report["images"]
card = f"""# AgriSmart training dataset

Built {report['built']} by the `agrismart-data-prep` notebook: {len(LABELS)} classes, {len(mf)} images.

| Split | Images | What it is |
|---|---|---|
| train | {counts.get('train', 0)} | PlantVillage lab photos (split by leaf) + field photos from PlantWild v1/v2 and PlantSeg |
| val_lab | {counts.get('val_lab', 0)} | PlantVillage photos of leaves that are not in train |
| val_field | {counts.get('val_field', 0)} | PlantDoc field photos, never trained on |

Training field photos that look like any PlantDoc photo were removed (DINOv2 cosine >= {LEAK_COS} or pHash distance <= {LEAK_HAM}).

## Sources and licences
- **PlantVillage**: Mohanty, Hughes & Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science. Kaggle `abdallahalidev/plantvillage-dataset` (CC BY-NC-SA 4.0); leaf map and split from Hugging Face `mohanty/PlantVillage` (CC BY-SA 3.0).
- **PlantDoc**: Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD, doi:10.1145/3371158.3371196. Kaggle `nirmalsankalana/plantdoc-dataset` (CC BY 4.0). Validation only.
- **PlantWild v1/v2**: Wei et al. (2024), *Benchmarking In-the-wild Multimodal Plant Disease Recognition and A Versatile Baseline*, ACM Multimedia, arXiv:2408.03120. Hugging Face `uqtwei2/PlantWild` (CC BY-NC-ND 4.0: non-commercial, no redistribution).
- **PlantSeg**: Wei et al. (2025), *A Large-Scale In-the-wild Dataset for Plant Disease Segmentation*, Scientific Data. Kaggle `weitianqi/plantseg`, Zenodo 14935094 (CC BY 4.0).

Keep this output private: it contains PlantWild images, whose licence forbids redistribution.
"""
(OUT / "README.md").write_text(card, encoding="utf-8")

print(json.dumps(report, indent=2))
log(f"done: {len(mf)} images, {size_gb:.2f} GB in {OUT}")
