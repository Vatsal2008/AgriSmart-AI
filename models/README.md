# Models

All AgriSmart models in one place.

- **App models (ONNX)** are stored here in git. The Android APK and the Windows `run.bat` app run these.
- **PyTorch originals** are the models the ONNX files were exported from. Each is larger than GitHub's
  100 MB file limit, so they are downloads from this repo's releases.

| Model | Crops / classes | App model (ONNX, this folder) | PyTorch original (release) | Field macro-F1 |
|---|---|---|---|---|
| **India v2** (latest, 13 Sep 2026, 336 px) | 59 / 387 | [`india_v2.onnx`](india_v2.onnx), 56 MB | [`india-model-v2`](https://github.com/Vatsal2008/AgriSmart-AI/releases/tag/india-model-v2), 329 MB | 0.741 |
| **India v1** (default in the apps, 12 Sep 2026, 224 px) | 59 / 387 | [`india_v1.onnx`](india_v1.onnx), 55 MB | [`india-model-v1`](https://github.com/Vatsal2008/AgriSmart-AI/releases/tag/india-model-v1), 328 MB | 0.744 |
| **Core v2** (hackathon, 11 Sep 2026, 224 px) | 13 / 28 | [`core.onnx`](core.onnx), 25 MB | [`model-v2`](https://github.com/Vatsal2008/AgriSmart-AI/releases/tag/model-v2), 83 MB | 0.714 (PlantDoc) |

All three are DINOv2 vision transformers. The India models are ViT-B/14 and the core model is ViT-S/14.

- **[`models.json`](models.json)** describes the ONNX models: labels, input size, normalisation and scores.
  It is the same file the apps read.
- **Confusion matrices** for every model are in [`../confusion_matrix`](../confusion_matrix).

## The app models (ONNX)

The ONNX models were exported from the PyTorch weights by
[`offline/tools/export_onnx.py`](../offline/tools/export_onnx.py), with weight-only quantization. The India
models use 4-bit weights and the core model uses 8-bit weights. On 280 real leaf photos, each one gives the
same answer as its PyTorch original for:

| Model | Same answer |
|---|---|
| India v2 | 100% |
| India v1 | 99.6% |
| Core | 98.6% |

**Input:** one RGB image as float32, shape `1 x 3 x S x S`. `S` is 336 for India v2 and 224 for the others.
To prepare it:
1. Resize the shorter side to `S / 0.9`.
2. Centre-crop to `S x S`.
3. Scale the pixels to 0-1.
4. Normalise with the model's mean and std, which are in `models.json`.

**Output:** one logit per class, in the label order given in `models.json`.

## Downloading the PyTorch originals

```bash
gh release download india-model-v2 -R Vatsal2008/AgriSmart-AI -p model.pt -D models/pytorch/india_v2
gh release download india-model-v1 -R Vatsal2008/AgriSmart-AI -p model.pt -D model/india
gh release download model-v2 -R Vatsal2008/AgriSmart-AI -p model.pt -D model
```

The second and third commands put the files where the code expects them: `model/india/predict.py` and the
live-scan India page use `model/india/model.pt`, and `model/predict.py` uses `model/model.pt`.
`model/predict.py` also downloads its model by itself on first use.

## Checksums (SHA-256)

| File | SHA-256 |
|---|---|
| `india_v2.onnx` | `807cb6d4120d1f28757b3cc9b90386a3150204e031dbf6620804340bb8a43bf7` |
| `india_v1.onnx` | `f6b0ad51ea7175a7426d40a08b2f56bade93f4de5c89da9ebe2540d3c81a27b4` |
| `core.onnx` | `950a80eeac7982be5cd9965645f91aa8f78617455143996fe64cf5b98db78bdb` |
| India v2 `model.pt` | `748a3194df9d115aaa3a981cfed73364909bb2f38a20b7c3fb1b8b1b3469715d` |
| India v1 `model.pt` | `ffaffb1e3ca0a81831454e8268b11cf2d16b418838042b023ad188f79e61a08a` |
| Core v2 `model.pt` | `361daa8f299733046ec8c241107cfa3e9433737dc3abf7ed353c7cb97abf35cc` |

Some training datasets are non-commercial (PlantVillage, PlantWild), so these models are for
non-commercial research and education.
