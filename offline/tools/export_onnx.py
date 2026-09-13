"""Export the three AgriSmart models to small ONNX files for the offline app (browser / Android WebView).

    python offline/tools/export_onnx.py

For each model: PyTorch -> ONNX (fp32) -> weight-only quantization (4-bit, or 8-bit if 4-bit changes answers)
-> offline/web/models/<id>.onnx, then a check that the quantized model picks the same class as the original
PyTorch model on a few hundred real leaf photos (both fed the web page's preprocessing). Weight-only keeps the
activations in float: plain dynamic int8 changed the answer on ~7% of photos for these ViTs, weight-only on
none. Writes offline/web/models/models.json, which the page reads for its model picker, crops and labels.
"""
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import timm
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "offline" / "web" / "models"
WORK = ROOT / "offline" / "build" / "onnx_fp32"
RUNS = Path(r"D:\Claude\Heckathon\kaggle\runs")
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"id": "india_v1", "name": "India v1 - 59 crops, 387 diseases (recommended)", "pt": ROOT / "model/india/model.pt",
     "metrics": RUNS / "india-train-v2/output/metrics.json", "bits": (4, 8)},
    {"id": "india_v2", "name": "India v2 - 59 crops, 387 diseases, 336 px", "pt": RUNS / "india-train-v2-run2/output/model.pt",
     "metrics": RUNS / "india-train-v2-run2/output/metrics.json", "bits": (4, 8)},
    {"id": "core", "name": "Core model - 13 crops, 28 classes (hackathon)", "pt": ROOT / "model/model.pt", "metrics": None,
     "bits": (8,)},  # 4-bit changed ~13% of this smaller model's answers
]
MIN_AGREEMENT = 0.99

# core model: PlantVillage folder names -> the same crop ids and disease names the India models use
PV = {
    "Apple___Apple_scab": ("apple", "Apple scab"), "Apple___Cedar_apple_rust": ("apple", "Cedar apple rust"),
    "Apple___healthy": ("apple", "Healthy"), "Blueberry___healthy": ("blueberry", "Healthy"),
    "Cherry_(including_sour)___healthy": ("cherry", "Healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn_maize", "Gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("corn_maize", "Common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn_maize", "Northern leaf blight"),
    "Grape___Black_rot": ("grape", "Black rot"), "Grape___healthy": ("grape", "Healthy"),
    "Peach___healthy": ("peach", "Healthy"), "Pepper,_bell___Bacterial_spot": ("bell_pepper", "Bacterial spot"),
    "Pepper,_bell___healthy": ("bell_pepper", "Healthy"), "Potato___Early_blight": ("potato", "Early blight"),
    "Potato___Late_blight": ("potato", "Late blight"), "Raspberry___healthy": ("raspberry", "Healthy"),
    "Soybean___healthy": ("soybean", "Healthy"), "Squash___Powdery_mildew": ("squash", "Powdery mildew"),
    "Strawberry___healthy": ("strawberry", "Healthy"), "Tomato___Bacterial_spot": ("tomato", "Bacterial spot"),
    "Tomato___Early_blight": ("tomato", "Early blight"), "Tomato___Late_blight": ("tomato", "Late blight"),
    "Tomato___Leaf_Mold": ("tomato", "Leaf mold"), "Tomato___Septoria_leaf_spot": ("tomato", "Septoria leaf spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "Spider mites (pest)"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "Yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "Mosaic virus"), "Tomato___healthy": ("tomato", "Healthy"),
}
CROP_PCT = 0.9


def split_label(g):
    if "::" in g:
        return tuple(g.split("::", 1))
    return PV[g]


def web_preprocess(img, size, mean, std):
    """What the web page does: shorter side -> floor(size / 0.9), centre crop `size`, scale to 0..1, normalise."""
    img = img.convert("RGB")
    short = int(size / CROP_PCT)
    w, h = img.size
    s = short / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


def test_images(n_per_source=70):
    """A few hundred real leaf photos from small public Hugging Face datasets, for the agreement check."""
    imgs = []
    for repo in ["Project-AgML/grape_leaf_disease_classification", "Project-AgML/apple_leaf_disease_classification",
                 "Project-AgML/tomato_leaf_disease", "Project-AgML/tea_leaf_disease_classification"]:
        path = hf_hub_download(repo, "default/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
        tbl = pq.read_table(path, columns=["image"]).column(0).combine_chunks()
        blobs = tbl.field("bytes").to_pylist()
        step = max(1, len(blobs) // n_per_source)
        imgs += [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs[::step][:n_per_source]]
    return imgs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    photos = test_images()
    print(f"{len(photos)} test photos for the agreement check")
    manifest = []
    for spec in MODELS:
        t0 = time.time()
        pkg = torch.load(spec["pt"], map_location="cpu", weights_only=True)
        size, mean, std = pkg["img_size"], pkg["mean"], pkg["std"]
        model = timm.create_model(pkg["backbone"], pretrained=False, num_classes=len(pkg["labels"]), img_size=size)
        model.load_state_dict(pkg["state_dict"])
        model.eval()

        fp32 = WORK / f"{spec['id']}_fp32.onnx"
        int8 = OUT / f"{spec['id']}.onnx"
        dummy = torch.randn(1, 3, size, size)
        try:
            torch.onnx.export(model, dummy, str(fp32), input_names=["pixels"], output_names=["logits"],
                              dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                              dynamo=False)
        except TypeError:  # older torch without the dynamo switch
            torch.onnx.export(model, dummy, str(fp32), input_names=["pixels"], output_names=["logits"],
                              dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        xs = [web_preprocess(img, size, mean, std) for img in photos]
        with torch.no_grad():
            ref = np.array([model(torch.from_numpy(x)).argmax(1).item() for x in xs])
        for bits in spec["bits"]:
            q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=bits, block_size=32, is_symmetric=True)
            q.process()
            q.model.save_model_to_file(str(int8), use_external_data_format=False)
            sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
            t = time.time()
            got = np.array([sess.run(None, {"pixels": x})[0].argmax(1).item() for x in xs])
            t_ort = time.time() - t
            agree = float(np.mean(got == ref))
            print(f"  {spec['id']} {bits}-bit weights: same answer as PyTorch on {agree:.1%}", flush=True)
            if agree >= MIN_AGREEMENT:
                break

        labels = [split_label(g) for g in pkg["labels"]]
        crop_classes = {}
        for i, (crop, _) in enumerate(labels):
            crop_classes.setdefault(crop, []).append(i)
        entry = {
            "id": spec["id"], "name": spec["name"], "file": f"models/{int8.name}", "img_size": size, "mean": mean,
            "std": std, "crop_pct": CROP_PCT, "labels": [{"crop": c, "label": l} for c, l in labels],
            "crop_classes": crop_classes, "size_mb": round(int8.stat().st_size / 1e6, 1), "sha256": sha256(int8),
            "weight_bits": bits, "quant_agreement": round(agree, 4), "cpu_ms_per_photo": round(1000 * t_ort / len(photos)),
        }
        if spec["metrics"] and Path(spec["metrics"]).exists():
            m = json.loads(Path(spec["metrics"]).read_text(encoding="utf-8"))
            entry["scores"] = {"clean_macro_f1": round(m["test_clean"]["macro_f1"], 3),
                               "crop_given_macro_f1": round(m["test_clean"]["crop_given"]["macro_f1"], 3),
                               "field_macro_f1": round(m["test_by_domain"]["field"]["macro_f1"], 3)}
        manifest.append(entry)
        print(f"{spec['id']}: {len(labels)} classes, {size}px, {bits}-bit {entry['size_mb']} MB, agrees with PyTorch on "
              f"{agree:.1%} of {len(photos)} photos, {entry['cpu_ms_per_photo']} ms/photo on this CPU, {time.time() - t0:.0f}s")
    (OUT / "models.json").write_text(json.dumps({"models": manifest}, ensure_ascii=False, indent=1) + "\n",
                                     encoding="utf-8")
    print(f"wrote {OUT / 'models.json'}")


if __name__ == "__main__":
    main()
