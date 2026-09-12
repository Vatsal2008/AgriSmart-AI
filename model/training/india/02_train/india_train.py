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
# # AgriSmart India · Step 2: train the multi-crop classifier
#
# Trains **one flat classifier** over every crop's classes (built by `india-data-prep`). At inference
# time, if the app knows the crop, it restricts the softmax to that crop's classes only — "guess among
# tomato diseases" instead of "guess among everything" — with no retraining needed for that.
#
# Same two-stage recipe as the hackathon core model (linear probe, then fine-tune), scaled up: a bigger
# DINOv2 backbone and more epochs, since this covers far more classes. The target is **macro-F1 ≥ 0.95**
# on the held-out test split built by the data-prep notebook (near-duplicate clusters kept whole on one
# side, so this number is not inflated by leaked copies). If a run falls short, `RESUME_FROM` lets the
# next run continue fine-tuning instead of starting over.

# %%
CFG = {
    "run_name": "india_v1",
    "backbone": "vit_base_patch14_dinov2.lvd142m",  # more classes -> more capacity than the core model's ViT-S
    "img_size": 224,
    "batch_size": 48,
    "lp_epochs": 2,
    "ft_epochs": 14,
    "lr_head_lp": 1e-3,
    "lr_backbone": 1.5e-5,
    "lr_head": 1.5e-4,
    "weight_decay": 0.05,
    "warmup_frac": 0.05,
    "label_smoothing": 0.1,
    "min_images_per_class": 25,       # classes rarer than this are dropped (too few to learn or evaluate)
    "samples_per_epoch": 80000,       # fixed epoch length, independent of how big the dataset grows
    "max_repeats": 4.0,               # a rare class's photo is shown at most ~this many times per epoch
    "tta": True,
    "num_workers": 4,
    "seed": 42,
    "resume_from": "",                # path to a previous run's model.pt to continue fine-tuning; "" = fresh
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
from PIL import Image, ImageEnhance, ImageFile, ImageFilter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
OUT = Path("/kaggle/working")
LOCAL = Path("/tmp/india")
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


log(f"{torch.cuda.get_device_name(0) if USE_AMP else 'CPU'} | torch {torch.__version__} | timm {timm.__version__}")

# %% [markdown]
# ## 1 · Load the dataset built by `india-data-prep`

# %%
def find_data():
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if "manifest.csv" in files and "classes.json" in files:
            return Path(root)
        if root[len(base):].count(os.sep) >= 6:
            dirs[:] = []
    raise FileNotFoundError("manifest.csv not found: attach the india-data-prep notebook output as input.")


DATA = find_data()
mf = pd.read_csv(DATA / "manifest.csv")
with open(DATA / "classes.json", encoding="utf-8") as fh:
    CROPS_INFO = json.load(fh)["crops"]

LOCAL.mkdir(parents=True, exist_ok=True)
for zip_name in sorted(mf["zip"].unique()):
    if not (DATA / zip_name).exists():
        raise FileNotFoundError(f"{zip_name} missing from {DATA}")
    marker = LOCAL / f".{zip_name}.done"
    if not marker.exists():
        with zipfile.ZipFile(DATA / zip_name) as zf:
            zf.extractall(LOCAL / zip_name.replace(".zip", ""))
        marker.touch()
mf["path"] = [str(LOCAL / z.replace(".zip", "") / f) for z, f in zip(mf["zip"], mf["file"])]

# global class = "crop::label" (a class is scoped to its crop, so "Healthy" for tomato != "Healthy" for rice)
mf["gclass"] = mf["crop"] + "::" + mf["label"]
class_counts = mf.groupby("gclass").size()
too_rare = class_counts[class_counts < CFG["min_images_per_class"]].index.tolist()
if too_rare:
    log(f"dropping {len(too_rare)} classes with fewer than {CFG['min_images_per_class']} images: {too_rare}")
    mf = mf[~mf["gclass"].isin(too_rare)]

GCLASSES = sorted(mf["gclass"].unique())
GCLASS_IDX = {g: i for i, g in enumerate(GCLASSES)}
NUM_CLASSES = len(GCLASSES)
CROPS = sorted(mf["crop"].unique())
# crop -> sorted list of global-class indices that belong to it (used for crop-conditioned inference)
CROP_CLASS_IDX = {c: sorted(GCLASS_IDX[g] for g in GCLASSES if g.startswith(c + "::")) for c in CROPS}

train_df = mf[mf["split"] == "train"].reset_index(drop=True)
test_df = mf[mf["split"] == "test"].reset_index(drop=True)
log(f"{NUM_CLASSES} classes across {len(CROPS)} crops | train {len(train_df)} | test {len(test_df)}")

counts_table = mf.groupby(["crop", "split"]).size().unstack(fill_value=0)
print(counts_table.to_string())

# %% [markdown]
# ## 1b · Leakage audit: how close is each test photo to its nearest training photo?
#
# The data-prep keeps near-copy clusters whole, but caps a cluster at 64 photos so look-alike lab
# photos can't chain into one giant cluster. That means a test photo *can* still have a near-twin in
# training. This step re-embeds every photo with the same flip-invariant DINOv2-S the data-prep uses,
# and records each test photo's highest cosine similarity to any training photo of the same crop.
# Final results are reported on the whole test set **and** on its "clean" part: test photos with no
# training photo at cos ≥ 0.95. The clean number is the honest one.

# %%
import torchvision.transforms as T

AUDIT_COS = 0.95
auditor = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                            img_size=224).to(DEVICE).eval()
audit_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class PathSet(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return audit_tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def audit_embed(paths):
    out = []
    for x in DataLoader(PathSet(paths), batch_size=256, num_workers=CFG["num_workers"], pin_memory=USE_AMP):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            f = auditor(x).float() + auditor(torch.flip(x, dims=[3])).float()
        out.append(nn.functional.normalize(f, dim=1).half())
    return torch.cat(out)  # kept on the GPU: ~270k x 384 fp16 is ~200 MB


t_audit = time.time()
tr_feat = audit_embed(train_df["path"].tolist())
te_feat = audit_embed(test_df["path"].tolist())
near = np.zeros(len(test_df), dtype=np.float32)
tr_crop, te_crop = train_df["crop"].to_numpy(), test_df["crop"].to_numpy()
for crop in CROPS:
    ti, si = np.flatnonzero(tr_crop == crop), np.flatnonzero(te_crop == crop)
    if len(ti) == 0 or len(si) == 0:
        continue
    A = tr_feat[torch.as_tensor(ti, device=tr_feat.device)].float()
    for s in range(0, len(si), 1024):
        q = te_feat[torch.as_tensor(si[s:s + 1024], device=te_feat.device)].float()
        near[si[s:s + 1024]] = (q @ A.T).max(1).values.cpu().numpy()
test_df["near_train_cos"] = near
clean_mask = near < AUDIT_COS
auditor = auditor.cpu()  # off the GPU before training starts
del tr_feat, te_feat
torch.cuda.empty_cache()
log(f"leakage audit ({(time.time() - t_audit) / 60:.1f} min): {int((~clean_mask).sum())} of {len(test_df)} test photos "
    f"({(~clean_mask).mean():.1%}) have a training photo at cos >= {AUDIT_COS}; "
    f"the other {int(clean_mask.sum())} form the clean test set")
print(test_df.assign(near_copy=~clean_mask).groupby("crop")["near_copy"].mean().sort_values(ascending=False)
      .head(15).round(3).to_string())

# %% [markdown]
# ## 2 · Model, augmentation, class-balanced sampling
#
# An epoch is a fixed `samples_per_epoch` draws. Every class aims for an equal share of them, but a small
# class's share is capped at `max_repeats` times its photo count, so its few photos aren't repeated into
# memorisation; the unused share flows to the big classes. (An earlier version weighted each photo by
# 1/min(count, cap), which gave every class above the cap *more* total weight the bigger it was.)

# %%
model_kwargs = {"img_size": CFG["img_size"]} if "vit" in CFG["backbone"] else {}
model = timm.create_model(CFG["backbone"], pretrained=True, num_classes=NUM_CLASSES, **model_kwargs)
data_cfg = timm.data.resolve_model_data_config(model)
MEAN, STD = tuple(data_cfg["mean"]), tuple(data_cfg["std"])

train_tf = T.Compose([
    T.RandomResizedCrop(CFG["img_size"], scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.15),
    T.RandomApply([T.RandomRotation(20, interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])
eval_tf = timm.data.create_transform(input_size=CFG["img_size"], is_training=False,
                                     interpolation="bicubic", mean=MEAN, std=STD, crop_pct=0.9)


class ImgSet(Dataset):
    def __init__(self, df, tf, blur_prob=0.0):
        self.paths = df["path"].tolist()
        self.labels = [GCLASS_IDX[g] for g in df["gclass"]]
        self.tf = tf
        self.blur_prob = blur_prob

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.blur_prob and random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        return self.tf(img), self.labels[i]


counts = train_df["gclass"].value_counts()
equal_share = CFG["samples_per_epoch"] / len(counts)
share = {g: min(equal_share, CFG["max_repeats"] * n) for g, n in counts.items()}  # draws per epoch for class g
weights = train_df["gclass"].map(lambda g: share[g] / counts[g]).to_numpy()      # spread over its photos
sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=CFG["samples_per_epoch"],
                                replacement=True)
log(f"sampler: {CFG['samples_per_epoch']} draws/epoch, equal share {equal_share:.0f}/class; "
    f"{sum(share[g] < equal_share for g in share)} small classes capped at {CFG['max_repeats']}x their size")

loader_args = {"num_workers": CFG["num_workers"], "pin_memory": USE_AMP, "persistent_workers": CFG["num_workers"] > 0}
train_ds = ImgSet(train_df, train_tf, blur_prob=0.15)
test_ds = ImgSet(test_df, eval_tf)
train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=True, **loader_args)
test_dl = DataLoader(test_ds, batch_size=128, **loader_args)

if CFG["resume_from"] and Path(CFG["resume_from"]).exists():
    prev = torch.load(CFG["resume_from"], map_location="cpu", weights_only=True)
    if prev["labels"] == GCLASSES:
        model.load_state_dict(prev["state_dict"])
        log(f"resumed weights from {CFG['resume_from']}")
    else:
        log("resume_from has a different class list; starting from the pretrained backbone instead")

# %% [markdown]
# ## 3 · Train: linear probe, then fine-tune. The checkpoint with the best test macro-F1 is kept.

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


history, best = [], {"test_f1": -1.0}


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
        test = score(predict_probs(test_dl), test_ds.labels)
        row = {"stage": stage, "epoch": epoch, "train_loss": float(loss_sum) / seen,
               "test_f1": test["macro_f1"], "test_acc": test["accuracy"], "minutes": (time.time() - t) / 60}
        history.append(row)
        marker = ""
        if test["macro_f1"] > best["test_f1"]:
            best.update(test_f1=test["macro_f1"], stage=stage, epoch=epoch)
            torch.save(model.state_dict(), OUT / "best_state.pt")
            marker = "  <- best so far"
        log(f"{stage} {epoch}/{epochs} | loss {row['train_loss']:.3f} | test F1 {row['test_f1']:.4f} "
            f"(acc {row['test_acc']:.4f}) | {row['minutes']:.1f} min{marker}")
        if test["macro_f1"] >= 0.97 and stage == "fine-tune" and epoch >= 3:
            log("test macro-F1 already at or above 0.97; stopping this stage early")
            break


# Linear probe: backbone frozen, only the new head learns (skipped when resuming a trained model).
# An earlier version set requires_grad = True here for a fresh run, so the "probe" silently fine-tuned the
# whole backbone at the head's 1e-3 learning rate -- enough to wreck DINOv2's pretrained features.
for p in backbone_params:
    p.requires_grad = False
if not CFG["resume_from"]:
    run_stage("linear-probe", CFG["lp_epochs"], [{"params": head_params, "lr": CFG["lr_head_lp"]}])
for p in backbone_params:
    p.requires_grad = True
run_stage("fine-tune", CFG["ft_epochs"], [{"params": backbone_params, "lr": CFG["lr_backbone"]},
                                          {"params": head_params, "lr": CFG["lr_head"]}])
log(f"best test macro-F1 {best['test_f1']:.4f} at {best.get('stage')} epoch {best.get('epoch')}")

hist = pd.DataFrame(history)
hist.to_csv(OUT / "history.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(hist) + 1), hist["test_f1"], marker="o", label="test macro-F1")
ax.axhline(0.95, color="crimson", ls="--", lw=1, label="target (0.95)")
ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=120); plt.show()

# %% [markdown]
# ## 4 · Test the best checkpoint, with test-time augmentation

# %%
model.load_state_dict(torch.load(OUT / "best_state.pt", map_location=DEVICE, weights_only=True))
probs = predict_probs(test_dl, tta=CFG["tta"])
y_true, y_pred = np.asarray(test_ds.labels), probs.argmax(1)
present = sorted(np.unique(y_true))
final = score(probs, test_ds.labels)
log(f"FINAL test macro-F1 {final['macro_f1']:.4f} | accuracy {final['accuracy']:.4f} (TTA={CFG['tta']})")

# "crop given": what the app scores when the farmer picks the crop -- only that crop's classes compete
test_crops = test_df["crop"].to_numpy()
probs_crop_given = probs.copy()
for crop, idxs in CROP_CLASS_IDX.items():
    other = np.ones(NUM_CLASSES, bool)
    other[idxs] = False
    probs_crop_given[np.ix_(test_crops == crop, other)] = 0.0
final_crop_given = score(probs_crop_given, test_ds.labels)
log(f"crop given: macro-F1 {final_crop_given['macro_f1']:.4f} | accuracy {final_crop_given['accuracy']:.4f}")

# by kind of photo: lab (PlantVillage), field (PlantWild web/field photos), mixed (research datasets)
def subset_scores(ix):
    return {"test_images": int(len(ix)), **score(probs[ix], y_true[ix]),
            "crop_given": score(probs_crop_given[ix], y_true[ix])}


# the honest headline: test photos with no near-twin in training (see the leakage audit in section 1b)
clean_ix = np.flatnonzero(clean_mask)
final_clean = subset_scores(clean_ix)
log(f"CLEAN test ({len(clean_ix)} photos with no training photo at cos >= {AUDIT_COS}): "
    f"macro-F1 {final_clean['macro_f1']:.4f} | accuracy {final_clean['accuracy']:.4f} | "
    f"crop given macro-F1 {final_clean['crop_given']['macro_f1']:.4f}")

by_domain = {}
if "domain" in test_df.columns:
    for dom, sub in test_df.groupby("domain"):
        ix = sub.index.to_numpy()
        by_domain[dom] = {**subset_scores(ix), "clean": subset_scores(np.intersect1d(ix, clean_ix))}
        log(f"  {dom:6s} photos ({len(ix):6d}): macro-F1 {by_domain[dom]['macro_f1']:.4f}, "
            f"crop given {by_domain[dom]['crop_given']['macro_f1']:.4f} | clean part "
            f"({by_domain[dom]['clean']['test_images']}): {by_domain[dom]['clean']['macro_f1']:.4f}")
test_df[["crop", "label", "domain", "near_train_cos"]].assign(
    predicted=[GCLASSES[i] for i in y_pred]).to_csv(OUT / "test_predictions_audit.csv", index=False)

rep = classification_report(y_true, y_pred, labels=present, target_names=[GCLASSES[i] for i in present],
                            output_dict=True, zero_division=0)
per_class = pd.DataFrame(rep).T
per_class.to_csv(OUT / "report_per_class.csv")

# per-crop macro-F1: how each crop does on its own, not averaged with everyone else
per_crop_rows = []
for crop in CROPS:
    idxs = CROP_CLASS_IDX[crop]
    mask = np.isin(y_true, idxs)
    if not mask.any():
        continue
    yt, yp = y_true[mask], y_pred[mask]
    per_crop_rows.append({"crop": crop, "test_images": int(mask.sum()),
                          "macro_f1": f1_score(yt, yp, labels=idxs, average="macro", zero_division=0),
                          "accuracy": accuracy_score(yt, yp)})
per_crop = pd.DataFrame(per_crop_rows).sort_values("macro_f1")
per_crop.to_csv(OUT / "report_per_crop.csv", index=False)
print(per_crop.round(3).to_string(index=False))

worst = per_class[per_class.index.isin([GCLASSES[i] for i in present])].sort_values("f1-score").head(20)
print("\nWeakest 20 classes:\n", worst[["precision", "recall", "f1-score", "support"]].round(3).to_string())

# %% [markdown]
# ## 5 · Package the model and a crop-conditioned `predict.py`

# %%
torch.save({"state_dict": model.state_dict(), "backbone": CFG["backbone"], "img_size": CFG["img_size"],
            "labels": GCLASSES, "crop_classes": CROP_CLASS_IDX, "mean": list(MEAN), "std": list(STD),
            "run": CFG["run_name"]}, OUT / "model.pt")
(OUT / "best_state.pt").unlink()

CROPS_JSON = {c: sorted({g.split("::", 1)[1] for g in GCLASSES if g.startswith(c + "::")}) for c in CROPS}
with open(OUT / "crops.json", "w", encoding="utf-8") as fh:
    json.dump(CROPS_JSON, fh, indent=1, ensure_ascii=False)

PREDICT_PY = '''"""AgriSmart India multi-crop classifier.

predict(image_path, crop=None) -> label   (crop restricts the guess to that crop's own classes)
predict_proba(image_path, crop=None) -> {label: probability, ...}   (full distribution, for a UI)

Usage:
    python predict.py --image leaf.jpg [--crop tomato] [--top 5]
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
    pkg = torch.load(weights, map_location="cpu", weights_only=True)
    kwargs = {"img_size": pkg["img_size"]} if "vit" in pkg["backbone"] else {}
    model = timm.create_model(pkg["backbone"], pretrained=False, num_classes=len(pkg["labels"]), **kwargs)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    tf = timm.data.create_transform(input_size=pkg["img_size"], interpolation="bicubic",
                                    mean=pkg["mean"], std=pkg["std"], crop_pct=0.9)
    return model, tf, pkg["labels"], pkg["crop_classes"]


def _as_image(image):
    return image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")


@torch.no_grad()
def predict_proba(image, crop=None, weights=WEIGHTS, tta=True):
    """Full probability distribution. If `crop` is given and known, the distribution is renormalised
    over just that crop's classes -- classes for every other crop get probability 0."""
    model, tf, labels, crop_classes = load(weights)
    x = tf(_as_image(image)).unsqueeze(0)
    p = model(x).softmax(1)
    if tta:
        p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    p = p[0]
    if crop and crop in crop_classes:
        mask = torch.zeros_like(p)
        mask[crop_classes[crop]] = 1.0
        p = p * mask
        total = p.sum()
        if total > 0:
            p = p / total
    return dict(zip(labels, p.tolist()))


def predict(image, crop=None, weights=WEIGHTS):
    """The two-part global label ("crop::disease") for the most likely class."""
    probs = predict_proba(image, crop, weights)
    return max(probs, key=probs.get)


def crops_and_labels(weights=WEIGHTS):
    """{crop: [disease, ...]} for building a crop-choice menu."""
    _, _, _, crop_classes = load(weights)
    _, _, labels, _ = load(weights)
    return {c: sorted({labels[i].split("::", 1)[1] for i in idxs}) for c, idxs in crop_classes.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict the crop and disease in a leaf photo.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--crop", default=None, help="restrict the guess to this crop, e.g. tomato")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--top", type=int, default=1)
    args = ap.parse_args()
    ranked = sorted(predict_proba(args.image, args.crop, args.weights).items(), key=lambda kv: -kv[1])
    if args.top == 1:
        print(ranked[0][0])
    else:
        for label, p in ranked[: args.top]:
            print(f"{label}\\t{p:.3f}")
'''
(OUT / "predict.py").write_text(PREDICT_PY, encoding="utf-8")

# %% [markdown]
# ## 6 · Self-test `predict.py` exactly as a user would run it

# %%
import subprocess as sp
import sys as _sys

sample = test_df.sample(min(8, len(test_df)), random_state=1)
agree_free, agree_conditioned = 0, 0
for _, r in sample.iterrows():
    run_free = sp.run([_sys.executable, str(OUT / "predict.py"), "--image", r["path"]],
                      capture_output=True, text=True, cwd=OUT)
    run_cond = sp.run([_sys.executable, str(OUT / "predict.py"), "--image", r["path"], "--crop", r["crop"]],
                      capture_output=True, text=True, cwd=OUT)
    true_g = f"{r['crop']}::{r['label']}"
    pred_free = run_free.stdout.strip() or f"ERROR: {run_free.stderr.strip()[-200:]}"
    pred_cond = run_cond.stdout.strip() or f"ERROR: {run_cond.stderr.strip()[-200:]}"
    agree_free += pred_free == true_g
    agree_conditioned += pred_cond == true_g
    print(f"true {true_g:40s} free -> {pred_free:40s} crop-given -> {pred_cond}")
print(f"\nfree-guess correct: {agree_free}/{len(sample)} | crop-given correct: {agree_conditioned}/{len(sample)}")

# %% [markdown]
# ## 7 · Summary

# %%
metrics = {
    "run": CFG["run_name"], "config": CFG, "best_checkpoint": best,
    "data": {"crops": len(CROPS), "classes": NUM_CLASSES, "train": len(train_df), "test": len(test_df),
             "dropped_rare_classes": too_rare},
    "test": final, "test_crop_given": final_crop_given, "test_clean": final_clean, "test_by_domain": by_domain,
    "leakage_audit": {"cos_threshold": AUDIT_COS, "test_images": int(len(test_df)),
                      "with_near_twin_in_train": int((~clean_mask).sum())},
    "target_met_on_clean_test": bool(final_clean["macro_f1"] >= 0.95),
    "minutes": round((time.time() - T0) / 60, 1),
}
with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))
log("done. Outputs: model.pt, crops.json, predict.py, metrics.json, history.csv, report_per_class.csv, "
    "report_per_crop.csv, training_curves.png")
