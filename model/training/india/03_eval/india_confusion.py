"""Confusion matrices for an India model (59 crops, 387 classes), from its saved test predictions.

    python model/training/india/03_eval/india_confusion.py --model v2    # the latest model (default)
    python model/training/india/03_eval/india_confusion.py --model v1    # the released model

Reads the run's test predictions (one row per test photo: crop, label, domain, near_train_cos, predicted; saved
by the Kaggle training run) - model/india/test_predictions_v2.csv for v2, model/india/test_predictions.csv for
v1 - and writes, with <m> = v2 or v1:
  confusion_matrix/india_<m>/india_<m>_confusion_crops.png        which crop the model thinks a photo shows (59 x 59), clean test set
  confusion_matrix/india_<m>/india_<m>_confusion_crops_field.png  the same on real field photos only
  confusion_matrix/india_<m>/india_<m>_confusion_full.png         all 387 classes, grouped by crop
  confusion_matrix/india_<m>/india_<m>_confusion_crops.csv        the counts behind the first figure
  confusion_matrix/india_<m>/india_<m>_top_confusions.csv         the most frequent class mix-ups, with counts and shares
The clean test set leaves out test photos with a near-identical twin in the training set (cosine >= 0.95), the
same filter as the reported clean scores.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[4]
PREDICTIONS = {"v2": ROOT / "model" / "india" / "test_predictions_v2.csv",   # 13 Sep 2026, 336 px (latest)
               "v1": ROOT / "model" / "india" / "test_predictions.csv"}      # 12 Sep 2026, 224 px (released)
NAMES = ROOT / "server" / "live_camera_india" / "india_translations.json"
OUT = ROOT / "confusion_matrix"            # one sub-folder per model: india_v2/, india_v1/
TWIN_COS = 0.95
MIN_FIELD = 10          # crops with fewer field test photos are left out of the field figure

# chart palette (light surface): one-hue sequential blue ramp (steps 100 -> 700), text inks, hairline rules
SURFACE, INK, INK_2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
CMAP.set_bad(SURFACE)                      # a cell nobody landed in stays the colour of the page
NORM = PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)   # stretches the small shares, where the mix-ups are
TICKS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": RULE, "savefig.facecolor": SURFACE})


def ink_on(value):
    """Dark text on light cells, white text on dark ones."""
    r, g, b, _ = CMAP(NORM(value))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def draw(counts, rows, cols, title, subtitle, out, size, tick_size, annotate=None, blocks=None):
    """Row-normalised heatmap: each row (true class) sums to 100%."""
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    shown = np.where(counts > 0, share, np.nan)
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(shown, cmap=CMAP, norm=NORM, interpolation="nearest", aspect="equal")
    ax.set_anchor("N")                     # the square plot sits right under its titles; spare space is cropped
    if blocks is None:
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=tick_size)
        ax.set_yticks(range(len(rows)), rows, fontsize=tick_size)
    else:                                  # one tick per crop block, hairlines between blocks
        edges, centres, names = blocks
        for e in edges[1:-1]:
            ax.axhline(e - 0.5, color=RULE, lw=0.4)
            ax.axvline(e - 0.5, color=RULE, lw=0.4)
        ax.set_xticks(centres, names, rotation=90, fontsize=tick_size)
        ax.set_yticks(centres, names, fontsize=tick_size)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_color(RULE)
        s.set_linewidth(0.6)
    ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
    ax.set_ylabel("True", fontsize=10, labelpad=6)
    if annotate:                           # selective labels: the notable cells only
        for i, j, text in annotate(share):
            ax.text(j, i, text, ha="center", va="center", fontsize=tick_size - 0.5, color=ink_on(share[i, j]))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, ticks=TICKS, shrink=0.6)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    cb.ax.tick_params(labelsize=8, color=RULE, length=3)
    cb.outline.set_edgecolor(RULE)
    cb.set_label("Share of the row's photos (each row sums to 100%)", fontsize=8.5, color=INK_2)
    # title and subtitle hang off the plot itself, left-aligned with it (a figure title floats far above)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=10)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points",
                ha="left", va="bottom", fontsize=15, fontweight="semibold", color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def crop_labels(share, weak=0.90, notable=0.05):
    """Label mix-ups of 5% or more, and correct shares below 90% (the weak crops)."""
    out = []
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share[i, j]
            if (i == j and v < weak) or (i != j and v >= notable):
                out.append((i, j, f"{v:.0%}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PREDICTIONS), default="v2")
    m = ap.parse_args().model
    tag, label = f"india_{m}", f"India model {m}"
    out = OUT / tag
    tr = json.loads(NAMES.read_text(encoding="utf-8"))
    name = lambda crop: (tr["crops"].get(crop) or {}).get("en") or crop.replace("_", " ").capitalize()
    d = pd.read_csv(PREDICTIONS[m])
    d["true"] = d["crop"] + "::" + d["label"]
    d["pcrop"] = d["predicted"].str.split("::").str[0]
    clean = d[d["near_train_cos"] < TWIN_COS]
    acc = (clean["true"] == clean["predicted"]).mean()
    crop_acc = (clean["crop"] == clean["pcrop"]).mean()
    print(f"clean test photos {len(clean):,}: exact class {acc:.2%}, right crop {crop_acc:.2%}")

    # 1. crops, clean test set
    crops = sorted(clean["crop"].unique(), key=name)
    cm = pd.crosstab(clean["crop"], clean["pcrop"]).reindex(index=crops, columns=crops, fill_value=0)
    out.mkdir(parents=True, exist_ok=True)
    cm.rename(index=name, columns=name).to_csv(out / f"{tag}_confusion_crops.csv", encoding="utf-8")
    draw(cm.to_numpy(), [name(c) for c in crops], [name(c) for c in crops],
         f"{label}: which crop does the model see?",
         f"{len(clean):,} test photos (near-copies of training photos left out) · right crop {crop_acc:.1%} · "
         "labels: mix-ups of 5% or more, and crops recognised less than 90% of the time",
         out / f"{tag}_confusion_crops.png", size=(13.5, 13.8), tick_size=7.5, annotate=crop_labels)

    # 2. crops, field photos only
    field = clean[clean["domain"] == "field"]
    fcount = field["crop"].value_counts()
    fcrops = sorted([c for c in fcount.index if fcount[c] >= MIN_FIELD], key=name)
    f = field[field["crop"].isin(fcrops)].copy()
    f["pc"] = np.where(f["pcrop"].isin(fcrops), f["pcrop"], "__other")
    fm = pd.crosstab(f["crop"], f["pc"]).reindex(index=fcrops, columns=fcrops + ["__other"], fill_value=0)
    f_acc = (f["crop"] == f["pcrop"]).mean()
    draw(fm.to_numpy(), [name(c) for c in fcrops], [name(c) for c in fcrops] + ["Other crops"],
         f"{label} on real field photos: which crop does the model see?",
         f"{len(f):,} field test photos of the {len(fcrops)} crops with at least {MIN_FIELD} of them · "
         f"right crop {f_acc:.1%} · same labels as the full test set figure",
         out / f"{tag}_confusion_crops_field.png", size=(11.5, 11.2), tick_size=8, annotate=crop_labels)

    # 3. all classes, grouped by crop
    classes = sorted(clean["true"].unique(), key=lambda g: (name(g.split("::")[0]), g.split("::")[1] != "Healthy", g))
    full = pd.crosstab(clean["true"], clean["predicted"]).reindex(index=classes, columns=classes, fill_value=0)
    owner = [g.split("::")[0] for g in classes]
    edges = [0] + [i for i in range(1, len(owner)) if owner[i] != owner[i - 1]] + [len(owner)]
    centres = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
    draw(full.to_numpy(), None, None,
         f"{label}: all {len(classes)} classes (crop and disease)",
         f"{len(clean):,} test photos · exact class {acc:.1%} · classes grouped by crop, healthy first; "
         "the dark diagonal is right answers, dots off it are mix-ups",
         out / f"{tag}_confusion_full.png", size=(17, 17.3), tick_size=6,
         blocks=(edges, centres, [name(owner[a]) for a in edges[:-1]]))

    # the table view: the most frequent mix-ups
    wrong = clean[clean["true"] != clean["predicted"]]
    per_class = clean["true"].value_counts()
    top = wrong.groupby(["true", "predicted"]).size().sort_values(ascending=False).head(30).reset_index(name="photos")
    top["share_of_true_class"] = (top["photos"] / top["true"].map(per_class)).round(3)
    top["same_crop"] = top["true"].str.split("::").str[0] == top["predicted"].str.split("::").str[0]
    top.to_csv(out / f"{tag}_top_confusions.csv", index=False, encoding="utf-8")
    print(f"wrote {(out / f'{tag}_top_confusions.csv').relative_to(ROOT)}; top 5:")
    print(top.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
