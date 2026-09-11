"""AgriSmart crop-disease classifier: predict(image_path) -> class label.

Usage:
    python model/predict.py --image leaf.jpg [--top 3]

The weights (model/model.pt, 87 MB) are not stored in git. On first use they are downloaded
from the GitHub release named in model/weights.json and checked against its SHA-256.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import timm
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
WEIGHTS = str(HERE / "model.pt")
WEIGHTS_INFO = HERE / "weights.json"


def ensure_weights(path=WEIGHTS):
    """Download model.pt from the GitHub release if it isn't here yet, and verify it."""
    path = Path(path)
    if path.exists():
        return str(path)
    info = json.loads(WEIGHTS_INFO.read_text(encoding="utf-8"))
    print(f"Downloading model weights ({info['size_mb']} MB) from {info['url']} ...", file=sys.stderr, flush=True)
    part = path.with_suffix(".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(info["url"]) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != info["sha256"]:
        part.unlink()
        raise RuntimeError("The downloaded weights failed the SHA-256 check. Run the command again.")
    part.replace(path)
    return str(path)


@lru_cache(maxsize=2)
def load(weights=WEIGHTS):
    """Build the network once and cache it. Returns (model, transform, labels)."""
    if weights == WEIGHTS:
        ensure_weights(weights)
    # weights_only=True: the file may contain only tensors, numbers and strings, never code.
    pkg = torch.load(weights, map_location="cpu", weights_only=True)
    kwargs = {"img_size": pkg["img_size"]} if "vit" in pkg["backbone"] else {}
    model = timm.create_model(pkg["backbone"], pretrained=False, num_classes=len(pkg["labels"]), **kwargs)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    tf = timm.data.create_transform(input_size=pkg["img_size"], interpolation="bicubic",
                                    mean=pkg["mean"], std=pkg["std"], crop_pct=0.9)
    return model, tf, pkg["labels"]


def _as_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


@torch.no_grad()
def predict_proba(image, weights=WEIGHTS, tta=True):
    """Probability per class for an image path or PIL image.

    With tta=True the image and its mirror are averaged (what the reported scores use).
    """
    model, tf, labels = load(weights)
    x = tf(_as_image(image)).unsqueeze(0)
    p = model(x).softmax(1)
    if tta:
        p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
    return dict(zip(labels, p[0].tolist()))


def predict(image_path, weights=WEIGHTS):
    """The organizers' interface: one image in, one class label out."""
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
            print(f"{label}\t{p:.3f}")
