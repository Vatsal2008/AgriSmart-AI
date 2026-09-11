"""AgriSmart smoke test.

Fine-tunes ConvNeXt-T for one epoch on four PlantVillage classes, then scores it on
PlantDoc field photos of the same classes. It checks the GPU, the dataset mounts and the
push -> run -> download loop. It is not a real experiment: the lab split here is random,
whereas real runs split by leaf.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

SEED = 42
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
IMG_EXT = {".jpg", ".jpeg", ".png"}

# Display label -> (PlantVillage folder, PlantDoc folder name after norm())
CLASSES = {
    "Tomato early blight": ("Tomato___Early_blight", "tomato early blight leaf"),
    "Tomato late blight": ("Tomato___Late_blight", "tomato leaf late blight"),
    "Tomato healthy": ("Tomato___healthy", "tomato leaf"),
    "Potato early blight": ("Potato___Early_blight", "potato leaf early blight"),
}
LABELS = list(CLASSES)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def norm(name):
    """'Tomato_leaf late-blight' -> 'tomato leaf late blight'."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def walk_dirs(base, max_depth=5):
    """Yield (folder, child folder names), never descending into PlantVillage class folders."""
    base = str(base)
    for root, dirs, _ in os.walk(base):
        yield Path(root), list(dirs)
        depth = root[len(base):].count(os.sep)
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if "___" not in d]


def find_plantvillage_color():
    for folder, children in walk_dirs(INPUT):
        if folder.name == "color" and "Tomato___healthy" in children:
            return folder
    raise FileNotFoundError("PlantVillage 'color' folder not found under /kaggle/input")


def find_plantdoc_splits():
    splits = [
        folder
        for folder, children in walk_dirs(INPUT)
        if folder.name.lower() in {"train", "test"} and any(norm(c) == "tomato leaf" for c in children)
    ]
    if not splits:
        raise FileNotFoundError("PlantDoc train/test folders not found under /kaggle/input")
    return splits


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class ImageList(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def main():
    started = time.time()
    use_amp = torch.cuda.is_available()
    device = "cuda" if use_amp else "cpu"
    gpu = torch.cuda.get_device_name(0) if use_amp else "none (CPU)"
    print(f"Device: {gpu} | torch {torch.__version__} | timm {timm.__version__}", flush=True)

    pv_root = find_plantvillage_color()
    lab = []
    for idx, label in enumerate(LABELS):
        lab += [(str(p), idx) for p in list_images(pv_root / CLASSES[label][0])]
    random.shuffle(lab)
    cut = int(0.8 * len(lab))
    train_items, val_items = lab[:cut], lab[cut:]

    field, unreadable = [], 0
    plantdoc_splits = find_plantdoc_splits()
    for split in plantdoc_splits:
        by_name = {norm(d.name): d for d in split.iterdir() if d.is_dir()}
        for idx, label in enumerate(LABELS):
            folder = by_name.get(CLASSES[label][1])
            if folder is None:
                print(f"PlantDoc {split.name}: no folder for {label}", flush=True)
                continue
            for p in list_images(folder):
                if readable(p):
                    field.append((str(p), idx))
                else:
                    unreadable += 1
    print(
        f"PlantVillage: {len(train_items)} train / {len(val_items)} val | "
        f"PlantDoc field: {len(field)} images ({unreadable} unreadable skipped)",
        flush=True,
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(LABELS)).to(device)
    cfg = timm.data.resolve_model_data_config(model)
    train_tf = timm.data.create_transform(**cfg, is_training=True)
    eval_tf = timm.data.create_transform(**cfg)

    loader_args = dict(num_workers=4, pin_memory=use_amp)
    train_dl = torch.utils.data.DataLoader(
        ImageList(train_items, train_tf), batch_size=64, shuffle=True, drop_last=True, **loader_args
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    t0 = time.time()
    for step, (x, y) in enumerate(train_dl):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % 20 == 0:
            print(f"step {step:>3}/{len(train_dl)}  loss {loss.item():.3f}", flush=True)
    train_seconds = time.time() - t0

    @torch.no_grad()
    def predict(items):
        dl = torch.utils.data.DataLoader(ImageList(items, eval_tf), batch_size=128, **loader_args)
        model.eval()
        out = []
        for x, _ in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        return torch.cat(out).numpy()

    def score(items, name):
        y_true = np.array([y for _, y in items])
        y_pred = predict(items)
        labels = list(range(len(LABELS)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        result = {
            "images": int(len(items)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
            "images_per_class": {LABELS[i]: int((y_true == i).sum()) for i in labels},
            "confusion_matrix": cm.tolist(),
        }
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=30, colorbar=False)
        ax.set_title(f"{name}: macro-F1 {result['macro_f1']:.3f}, accuracy {result['accuracy']:.3f}")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_{name}.png", dpi=120)
        plt.close(fig)
        return result

    lab_result = score(val_items, "lab")
    field_result = score(field, "field") if field else None

    metrics = {
        "run": "agrismart-smoke-test",
        "model": MODEL_NAME,
        "epochs": 1,
        "device": gpu,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "classes": LABELS,
        "plantvillage_root": str(pv_root),
        "plantdoc_splits": [str(s) for s in plantdoc_splits],
        "train_images": len(train_items),
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "lab_validation": lab_result,
        "field_plantdoc": field_result,
        "note": "Smoke test only: random lab split (not leaf-grouped), one epoch, four classes.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Train time {metrics['train_seconds']}s, total {metrics['total_seconds']}s on {gpu}", flush=True)
    print(f"LAB   macro-F1 {lab_result['macro_f1']:.3f}  accuracy {lab_result['accuracy']:.3f}", flush=True)
    if field_result:
        print(f"FIELD macro-F1 {field_result['macro_f1']:.3f}  accuracy {field_result['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
