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
# # AgriSmart AI · Step 2: train and test the disease classifier
#
# Trains on the dataset built by `agrismart-data-prep` and reports **macro-F1 on field photos** (PlantDoc):
# the number that tracks the hidden test set. It also writes a ready-to-use `predict.py`.
#
# **How it trains (two stages):**
# 1. *Linear probe*: freeze the pretrained backbone and train only the new classifier head.
# 2. *Fine-tune*: unfreeze everything, with a small learning rate for the backbone.
#
# **What fights the lab-to-field gap:** a self-supervised DINOv2 backbone · PlantVillage leaves pasted onto new
# backgrounds · real field photos (PlantWild, PlantSeg) sampled more often · strong augmentation ·
# choosing the checkpoint by field macro-F1, never by lab accuracy.
#
# All settings live in the next cell. For the E0 baseline, set `backbone` to `convnext_tiny.fb_in22k_ft_in1k`,
# `use_field_train` to `False` and `bg_replace_prob` to `0`.

# %%
CFG = {
    "run_name": "e3b_dinov2s_natural_aug",
    "backbone": "vit_small_patch14_dinov2.lvd142m",  # E0 baseline: "convnext_tiny.fb_in22k_ft_in1k"
    "img_size": 224,
    "batch_size": 64,
    "lp_epochs": 1,            # stage 1: train only the new head (run v1 plateaued after one epoch)
    "ft_epochs": 8,            # stage 2: fine-tune everything (run v1 was still improving at epoch 5)
    "lr_head_lp": 1e-3,
    "lr_backbone": 2e-5,
    "lr_head": 2e-4,
    "weight_decay": 0.05,
    "warmup_frac": 0.05,
    "label_smoothing": 0.1,
    "use_field_train": True,   # add PlantWild / PlantSeg field photos to training
    "field_weight": 3.0,       # within a class, a field photo is drawn 3x as often as a lab photo
    "bg_replace_prob": 0.5,    # chance a PlantVillage training photo gets a new background
    "blur_prob": 0.2,
    "tta": True,               # final test: average the original and mirrored image
    "num_workers": 4,
    "seed": 42,
}

# %%
import json
import math
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFile, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
OUT = Path("/kaggle/working")
LOCAL = Path("/tmp/agrismart")
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


log(f"{torch.cuda.get_device_name(0) if USE_AMP else 'CPU'} | torch {torch.__version__} | timm {timm.__version__}")

# %% [markdown]
# ## 1 · Load the dataset
# The prep notebook's output is attached as input. Its zips are unpacked to local disk once.

# %%
def find_data():
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if "manifest.csv" in files and "classes.json" in files:
            return Path(root)
        if root[len(base):].count(os.sep) >= 6:
            dirs[:] = []
    raise FileNotFoundError("manifest.csv not found: attach the agrismart-data-prep notebook output as input.")


DATA = find_data()
mf = pd.read_csv(DATA / "manifest.csv", keep_default_na=False)
with open(DATA / "classes.json", encoding="utf-8") as fh:
    CLASSES = json.load(fh)
LABELS = CLASSES["labels"]
NUM_CLASSES = len(LABELS)
LABEL_IDX = {label: i for i, label in enumerate(LABELS)}

LOCAL.mkdir(parents=True, exist_ok=True)
ROOT = {}
for zip_name in sorted(set(mf["zip"]) | {"pv_segmented.zip", "backgrounds.zip"}):
    if zip_name == "pv_segmented.zip":
        sample = next((s for s in mf["seg_file"] if s), "")
    elif zip_name == "backgrounds.zip":
        sample = "backgrounds/00000.jpg"
    else:
        sample = mf.loc[mf["zip"] == zip_name, "file"].iloc[0]
    if (DATA / zip_name).exists():
        marker = LOCAL / f".{zip_name}.done"
        if not marker.exists():
            with zipfile.ZipFile(DATA / zip_name) as zf:
                zf.extractall(LOCAL)
            marker.touch()
        ROOT[zip_name] = LOCAL
    elif sample and (DATA / sample).exists():  # Kaggle already unpacked it
        ROOT[zip_name] = DATA
    else:
        raise FileNotFoundError(f"{zip_name} not found in {DATA}")

mf["path"] = [str(ROOT[z] / f) for z, f in zip(mf["zip"], mf["file"])]
mf["seg_path"] = [str(ROOT["pv_segmented.zip"] / s) if s else "" for s in mf["seg_file"]]
BACKGROUNDS = sorted(str(p) for p in (ROOT["backgrounds.zip"] / "backgrounds").glob("*.jpg"))

train_df = mf[mf["split"] == "train"]
if not CFG["use_field_train"]:
    train_df = train_df[train_df["source"] == "plantvillage"]
val_lab_df = mf[mf["split"] == "val_lab"]
val_field_df = mf[mf["split"] == "val_field"]

is_field = (train_df["source"] != "plantvillage").to_numpy()
counts = pd.DataFrame({
    "lab_train": train_df.loc[~is_field, "label"].value_counts(),
    "field_train": train_df.loc[is_field, "label"].value_counts(),
    "val_lab": val_lab_df["label"].value_counts(),
    "val_field": val_field_df["label"].value_counts(),
}).reindex(LABELS).fillna(0).astype(int)
print(counts.to_string())
log(f"train {len(train_df)} ({int(is_field.sum())} field) | val_lab {len(val_lab_df)} | val_field {len(val_field_df)} | "
    f"backgrounds {len(BACKGROUNDS)}")

# %% [markdown]
# ## 2 · Model, augmentation and data loaders
# Sampling is **class-balanced** (every class gets the same share of each epoch), and inside a class a field photo
# is drawn `field_weight` times as often as a lab photo.

# %%
model_kwargs = {"img_size": CFG["img_size"]} if "vit" in CFG["backbone"] else {}
model = timm.create_model(CFG["backbone"], pretrained=True, num_classes=NUM_CLASSES, **model_kwargs)
data_cfg = timm.data.resolve_model_data_config(model)
MEAN, STD = tuple(data_cfg["mean"]), tuple(data_cfg["std"])

# Natural augmentations only: crops, flips, rotation, perspective, lighting and colour shifts (blur is added in TrainSet).
# Run v1 used timm RandAugment, which also inverts and solarises colours: leaves turned purple, and the
# yellowing and browning that tell diseases apart were destroyed.
import torchvision.transforms as T

train_tf = T.Compose([
    T.RandomResizedCrop(CFG["img_size"], scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(0.2),
    T.RandomApply([T.RandomRotation(25, interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
    T.RandomPerspective(distortion_scale=0.2, p=0.2),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.04),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
eval_tf = timm.data.create_transform(
    input_size=CFG["img_size"], is_training=False, interpolation="bicubic", mean=MEAN, std=STD, crop_pct=0.9,
)


def composite(seg_path, bg_path, size=256):
    """Paste a segmented PlantVillage leaf (black background) onto a field background at a random scale and spot."""
    leaf = Image.open(seg_path).convert("RGB")
    arr = np.asarray(leaf, dtype=np.int16)
    mask = Image.fromarray(((arr.sum(axis=2) > 40) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    canvas = Image.open(bg_path).convert("RGB").resize((size, size), Image.BILINEAR)
    if random.random() < 0.5:
        canvas = ImageOps.mirror(canvas)
    canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.6, 1.3))
    scale = random.uniform(0.55, 1.0) * size / max(leaf.size)
    w, h = max(8, int(leaf.width * scale)), max(8, int(leaf.height * scale))
    leaf, mask = leaf.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.BILINEAR)
    canvas.paste(leaf, (random.randint(0, size - w), random.randint(0, size - h)), mask)
    return canvas


class TrainSet(Dataset):
    def __init__(self, df):
        self.paths = df["path"].tolist()
        self.segs = df["seg_path"].tolist()
        self.labels = [LABEL_IDX[l] for l in df["label"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.segs[i] and BACKGROUNDS and random.random() < CFG["bg_replace_prob"]:
            img = composite(self.segs[i], random.choice(BACKGROUNDS))
        else:
            img = Image.open(self.paths[i]).convert("RGB")
        if random.random() < CFG["blur_prob"]:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        return train_tf(img), self.labels[i]


class EvalSet(Dataset):
    def __init__(self, df):
        self.paths = df["path"].tolist()
        self.labels = [LABEL_IDX[l] for l in df["label"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


lab_n = train_df.loc[~is_field, "label"].value_counts()
field_n = train_df.loc[is_field, "label"].value_counts()
weights = [(CFG["field_weight"] if f else 1.0) / (lab_n.get(l, 0) + CFG["field_weight"] * field_n.get(l, 0))
           for l, f in zip(train_df["label"], is_field)]
sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_df), replacement=True)

loader_args = {"num_workers": CFG["num_workers"], "pin_memory": USE_AMP, "persistent_workers": CFG["num_workers"] > 0}
train_ds, val_lab_ds, val_field_ds = TrainSet(train_df), EvalSet(val_lab_df), EvalSet(val_field_df)
train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=True, **loader_args)
val_lab_dl = DataLoader(val_lab_ds, batch_size=128, **loader_args)
val_field_dl = DataLoader(val_field_ds, batch_size=128, **loader_args)

# Look at a few training images exactly as the model sees them (after augmentation).
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
for ax, k in zip(axes.flat, np.random.default_rng(1).choice(len(train_ds), 12, replace=False)):
    x, y = train_ds[int(k)]
    img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
    ax.imshow(img)
    ax.set_title(LABELS[y].replace("___", "\n").replace("_", " "), fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "train_batch_examples.png", dpi=100)
plt.show()

# %% [markdown]
# ## 3 · Train: linear probe, then fine-tune
# After every epoch we score both validation sets and keep the checkpoint with the **best field macro-F1**.

# %%
model.to(DEVICE)
head_params = list(model.get_classifier().parameters())
head_ids = {id(p) for p in head_params}
backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


def cosine_schedule(opt, total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


@torch.no_grad()
def predict_probs(loader, tta=False):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def score(probs, labels):
    y_true, y_pred = np.asarray(labels), probs.argmax(1)
    present = np.unique(y_true)
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred))}


history, best = [], {"field_f1": -1.0}


def run_stage(stage, epochs, param_groups):
    if epochs <= 0:
        return
    opt = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"])
    total = epochs * len(train_dl)
    sched = cosine_schedule(opt, total, max(1, int(CFG["warmup_frac"] * total)))
    trainable = [p for g in param_groups for p in g["params"]]
    for epoch in range(1, epochs + 1):
        model.train()
        t, loss_sum, seen = time.time(), torch.zeros((), device=DEVICE), 0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(y)
            seen += len(y)
        lab = score(predict_probs(val_lab_dl), val_lab_ds.labels)
        field = score(predict_probs(val_field_dl), val_field_ds.labels)
        row = {"stage": stage, "epoch": epoch, "train_loss": float(loss_sum) / seen,
               "lab_f1": lab["macro_f1"], "lab_acc": lab["accuracy"],
               "field_f1": field["macro_f1"], "field_acc": field["accuracy"], "minutes": (time.time() - t) / 60}
        history.append(row)
        marker = ""
        if field["macro_f1"] > best["field_f1"]:
            best.update(field_f1=field["macro_f1"], stage=stage, epoch=epoch)
            torch.save(model.state_dict(), OUT / "best_state.pt")
            marker = "  <- best so far"
        log(f"{stage} {epoch}/{epochs} | loss {row['train_loss']:.3f} | lab F1 {row['lab_f1']:.3f} | "
            f"field F1 {row['field_f1']:.3f} (acc {row['field_acc']:.3f}) | {row['minutes']:.1f} min{marker}")


for p in backbone_params:
    p.requires_grad = False
run_stage("linear-probe", CFG["lp_epochs"], [{"params": head_params, "lr": CFG["lr_head_lp"]}])
for p in backbone_params:
    p.requires_grad = True
run_stage("fine-tune", CFG["ft_epochs"], [{"params": backbone_params, "lr": CFG["lr_backbone"]},
                                          {"params": head_params, "lr": CFG["lr_head"]}])
log(f"best field macro-F1 {best['field_f1']:.3f} at {best['stage']} epoch {best['epoch']}")

hist = pd.DataFrame(history)
hist.to_csv(OUT / "history.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, len(hist) + 1)
ax.plot(x, hist["lab_f1"], marker="o", label="lab macro-F1 (PlantVillage)")
ax.plot(x, hist["field_f1"], marker="o", label="field macro-F1 (PlantDoc)")
if CFG["lp_epochs"] and CFG["ft_epochs"]:
    ax.axvline(CFG["lp_epochs"] + 0.5, color="grey", ls="--", lw=1)
    ax.text(CFG["lp_epochs"] + 0.6, 0.03, "fine-tune starts", color="grey", fontsize=8)
ax.set_xlabel("epoch")
ax.set_ylabel("macro-F1")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "training_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4 · Test the best checkpoint
# Reload the checkpoint with the best field macro-F1 and score it with test-time augmentation (original + mirrored image).
#
# **For the report:** this checkpoint was *chosen* using PlantDoc, so the PlantDoc number is slightly optimistic.
# The organizers' hidden field set is the real test.

# %%
model.load_state_dict(torch.load(OUT / "best_state.pt", map_location=DEVICE))
SHORT = [l.replace("___", ": ").replace("_", " ").replace(" (maize)", "").replace(" (including sour)", "")
         .replace("Pepper, bell", "Bell pepper") for l in LABELS]


def plot_confusion(y_true, y_pred, name, res):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), SHORT, rotation=90, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), SHORT, fontsize=8)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] < 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name}: macro-F1 {res['macro_f1']:.3f}, accuracy {res['accuracy']:.3f} "
                 "(colour = row share, numbers = counts)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=130)
    plt.show()


results = {}
for name, ds, dl in [("lab", val_lab_ds, val_lab_dl), ("field", val_field_ds, val_field_dl)]:
    probs = predict_probs(dl, tta=CFG["tta"])
    y_true, y_pred = np.asarray(ds.labels), probs.argmax(1)
    present = sorted(np.unique(y_true))
    results[name] = {**score(probs, ds.labels), "images": int(len(y_true))}
    rep = classification_report(y_true, y_pred, labels=present, target_names=[LABELS[i] for i in present],
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUT / f"report_{name}.csv")
    plot_confusion(y_true, y_pred, name, results[name])
    if name == "field":
        field_probs, field_true = probs, y_true
        per_class = pd.DataFrame(rep).T.loc[[LABELS[i] for i in present], ["precision", "recall", "f1-score", "support"]]
        print(per_class.sort_values("f1-score").round(3).to_string())
log(f"TEST  lab macro-F1 {results['lab']['macro_f1']:.3f} | field macro-F1 {results['field']['macro_f1']:.3f} "
    f"(accuracy {results['field']['accuracy']:.3f}, TTA={CFG['tta']})")

# Where does it go wrong on field photos? 16 random mistakes.
wrong = np.flatnonzero(field_probs.argmax(1) != field_true)
pick = np.random.default_rng(0).choice(wrong, size=min(16, len(wrong)), replace=False)
fig, axes = plt.subplots(4, 4, figsize=(14, 15))
for ax, k in zip(axes.flat, pick):
    ax.imshow(Image.open(val_field_ds.paths[k]).convert("RGB"))
    ax.set_title(f"true: {SHORT[field_true[k]]}\npred: {SHORT[field_probs[k].argmax()]} ({field_probs[k].max():.2f})",
                 fontsize=8)
for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "field_errors.png", dpi=100)
plt.show()

# %% [markdown]
# ## 5 · Package the model and `predict.py`
# `model.pt` holds the weights plus everything needed to rebuild the model. `predict.py` is the interface the
# organizers ask for: `predict(image_path) -> label`, or `python predict.py --image <path>`.

# %%
torch.save({"state_dict": model.state_dict(), "backbone": CFG["backbone"], "img_size": CFG["img_size"],
            "labels": LABELS, "mean": list(MEAN), "std": list(STD)}, OUT / "model.pt")
(OUT / "best_state.pt").unlink()
with open(OUT / "labels.json", "w", encoding="utf-8") as fh:
    json.dump(LABELS, fh, indent=1)

PREDICT_PY = '''"""AgriSmart crop-disease classifier: predict(image_path) -> class label.

Usage:
    python predict.py --image leaf.jpg [--weights model.pt] [--top 3]
"""
import argparse
from functools import lru_cache
from pathlib import Path

import timm
import torch
from PIL import Image

WEIGHTS = str(Path(__file__).resolve().parent / "model.pt")


@lru_cache(maxsize=2)
def load(weights=WEIGHTS):
    pkg = torch.load(weights, map_location="cpu", weights_only=False)
    kwargs = {"img_size": pkg["img_size"]} if "vit" in pkg["backbone"] else {}
    model = timm.create_model(pkg["backbone"], pretrained=False, num_classes=len(pkg["labels"]), **kwargs)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    tf = timm.data.create_transform(input_size=pkg["img_size"], interpolation="bicubic",
                                    mean=pkg["mean"], std=pkg["std"], crop_pct=0.9)
    return model, tf, pkg["labels"]


@torch.no_grad()
def predict_proba(image_path, weights=WEIGHTS):
    """Probability per class, averaged over the image and its mirror."""
    model, tf, labels = load(weights)
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    p = (model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    return dict(zip(labels, p[0].tolist()))


def predict(image_path, weights=WEIGHTS):
    probs = predict_proba(image_path, weights)
    return max(probs, key=probs.get)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict the crop disease in a leaf photo.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--top", type=int, default=1, help="show the top-N classes with probabilities")
    args = ap.parse_args()
    ranked = sorted(predict_proba(args.image, args.weights).items(), key=lambda kv: -kv[1])
    if args.top == 1:
        print(ranked[0][0])
    else:
        for label, p in ranked[: args.top]:
            print(f"{label}\\t{p:.3f}")
'''
(OUT / "predict.py").write_text(PREDICT_PY, encoding="utf-8")

# Test predict.py exactly as a judge would run it: a fresh process, one image at a time.
sample = val_field_df.sample(6, random_state=1)
agree = 0
for path, label in zip(sample["path"], sample["label"]):
    run = subprocess.run([sys.executable, str(OUT / "predict.py"), "--image", path],
                         capture_output=True, text=True, cwd=OUT)
    pred = run.stdout.strip() or f"ERROR: {run.stderr.strip()[-300:]}"
    agree += pred == label
    print(f"true {label:48s} predict.py -> {pred}")
print(f"predict.py matched the true label on {agree}/{len(sample)} field photos")

# %% [markdown]
# ## 6 · Summary

# %%
metrics = {
    "run": CFG["run_name"],
    "config": CFG,
    "best_checkpoint": best,
    "data": {"train": len(train_df), "train_field": int(is_field.sum()), "val_lab": len(val_lab_df),
             "val_field": len(val_field_df), "classes": NUM_CLASSES},
    "test": results,
    "minutes": round((time.time() - T0) / 60, 1),
}
with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)
print(json.dumps(metrics["test"], indent=2))
log("done. Outputs: model.pt, predict.py, labels.json, metrics.json, history.csv, report_*.csv, *.png")
