"""AgriSmart India multi-crop classifier.

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
            print(f"{label}\t{p:.3f}")
