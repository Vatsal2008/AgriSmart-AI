# Model

## Interface

[`predict.py`](predict.py) is the interface the organizers ask for:

```bash
python model/predict.py --image leaf.jpg            # prints one label, e.g. Tomato___Late_blight
python model/predict.py --image leaf.jpg --top 3    # three likeliest labels with probabilities
```

```python
from predict import predict, predict_proba
predict("leaf.jpg")          # -> "Tomato___Late_blight"
predict_proba("leaf.jpg")    # -> {"Apple___Apple_scab": 0.001, ...}
```

- Labels are PlantVillage folder names, listed in [`labels.json`](labels.json).
- The weights (`model.pt`, 87 MB) are not stored in git. On first use `predict.py` downloads them from the release
  named in [`weights.json`](weights.json) and checks the SHA-256. They load with `weights_only=True`, so the file can
  contain only tensors, numbers and strings.

## Training (Kaggle)

The notebooks run on Kaggle's free T4 GPUs. Each `.py` file is the source; the `.ipynb` next to it is generated with
`jupytext --to ipynb <file>.py`, and `kernel-metadata.json` says which datasets to attach.

| Folder | What it does | Time on a T4 |
|---|---|---|
| [`training/00_smoke_test`](training/00_smoke_test) | 1 epoch on 4 classes: checks the GPU, the dataset mounts and the push-run-download loop | ~2 min |
| [`training/01_data_prep`](training/01_data_prep) | Builds the training dataset: leaf-grouped PlantVillage split, PlantWild / PlantSeg field photos, PlantDoc validation set, leakage guard, backgrounds | ~14 min |
| [`training/02_train`](training/02_train) | Trains and tests the classifier; writes `model.pt`, `predict.py`, reports and figures | ~44 min |

To reproduce with the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) (signed in):

1. Change the `id` in each `kernel-metadata.json` to your own Kaggle username.
2. `kaggle kernels push -p model/training/01_data_prep` and wait for it to finish (`kaggle kernels status <id>`).
3. `kaggle kernels push -p model/training/02_train`. It attaches the data-prep output as its input.
4. `kaggle kernels output <id> -p <folder>` downloads `model.pt` and the reports.

To switch to the organizers' official class list, edit `KEEP` near the top of `01_data_prep` and re-run both notebooks.
